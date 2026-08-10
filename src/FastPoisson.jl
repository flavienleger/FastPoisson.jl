module FastPoisson

using LinearAlgebra
using FFTW

export CellCentered, NodeCentered
export PoissonSolver, PoissonState
export solve_poisson!, solve_poisson

# ==============================================================================
# 1. Topology / Algorithm Types
# ==============================================================================

"""
    CellCentered()

Unknowns at cell centres, `x_i = (i-1/2)dx`, `i = 1…N`, covering `[0, N·dx]`.
Mirror-Neumann boundaries are diagonalized exactly by DCT-II (`REDFT10`), with
eigenvalues `2(1-cos(kπ/N))/dx²` and uniform dual volumes `dx^D`.
"""
struct CellCentered end

"""
    NodeCentered()

Unknowns at nodes, `x_i = (i-1)dx`, `i = 1…N`, covering `[0, (N-1)dx]` — so the
two *endpoints* of the domain are unknowns.  Mirror-Neumann boundaries are
diagonalized exactly by DCT-I (`REDFT00`), with eigenvalues
`2(1-cos(kπ/(N-1)))/dx²`.

Two consequences that callers must respect, both verified in
`../validation/test_poisson_vertex.jl` (2D) and `test/test_poisson_3d.jl` (3D):

* the dual volumes are **trapezoid-weighted**, `∏dx_d · ∏w_{i_d}` with `w = 1`
  except `w_1 = w_N = 1/2`, so a face node owns half a cell, an edge node a
  quarter, and a **corner node in 3D owns an eighth**;
* therefore the compatibility condition this solve enforces is that the
  **trapezoid-weighted** mean of `rhs` vanish, not the plain mean.  A source with
  zero plain mean but nonzero weighted mean is silently projected, so
  `solve_poisson!` checks the weighted one.

Use this layout when a feature of the problem must sit exactly *on* the grid — in
the Rochet-Choné solver the outside option `y_∅ = 0`, whose displacement to
`(dx/2, …)` on a cell-centred grid costs a clean `O(1/N)` bias in the profit
(`reviews/phase2-ygrid.md` in the parent project; the 3D repeat is Phase 4.1
gate A, where the bias is predicted to be larger and the grids are coarser).
"""
struct NodeCentered end

# ==============================================================================
# 2. The Solver Context (Immutable Setup)
# ==============================================================================

"""
    PoissonSolver(N1, N2, dx1, dx2, topology;   flags = FFTW.MEASURE, nthreads = Threads.nthreads())
    PoissonSolver(N1, N2, N3, dx1, dx2, dx3, topology; flags, nthreads)
    PoissonSolver(N, dx, topology; flags, nthreads)                    # square,  2D
    PoissonSolver(N, dx, Val(3), topology; flags, nthreads)            # cubic,   3D

Build the FFTW plans and spectral eigenvalues for `-Δϕ = rhs` with mirror-Neumann
boundaries on the given `topology` ([`CellCentered`](@ref) or
[`NodeCentered`](@ref)), in **2 or 3 dimensions**.  The dimension is carried in
the type, so `solve_poisson!` is one implementation per topology rather than one
per topology per dimension.

# `flags` — planning effort, and the reproducibility tradeoff

`FFTW.MEASURE` (the default) chooses the transform algorithm by *timing*
candidates. `FFTW.ESTIMATE` chooses it from a cost model, without running
anything. The choice is a genuine tradeoff and it is deliberately a parameter
rather than a constant, because the right answer depends entirely on the problem:

* **Large grids, repeated solves, a cluster** — `MEASURE`. On big 3D transforms
  the payoff is large and it is paid once.  Note that 3D planning is not cheap:
  measured on this project's 257³ DCT-II, `MEASURE` costs ~20 s of planning.
* **Reproducibility across runs** — `ESTIMATE`. `MEASURE` times candidates on a
  machine whose timings jitter, so it does not always pick the same algorithm,
  and different algorithms round differently. Two processes running identical
  code can then produce results that differ in the last bits — measured in the
  caller of this module, where the difference is amplified by a chaotic
  iteration into the 6th significant figure. Within one process it is a
  non-issue: FFTW caches its wisdom, so every later plan of the same shape
  reuses the first choice.
* **Small or moderate 2D grids** — it does not matter for speed. Measured on
  REDFT00 at N = 129…513: identical transform times, while `MEASURE` costs
  0.2–1.2 s of planning per call.

A third option gets both: plan once with `MEASURE`, then
`FFTW.export_wisdom(path)`, and `FFTW.import_wisdom(path)` in later runs — plan
choice is then fixed *and* tuned. That is the right setup for a production
cluster run, and at 3D planning costs it is close to mandatory.

# `nthreads`

Forwarded to `FFTW.set_num_threads`, which is **global FFTW state**: it affects
every plan created afterwards anywhere in the session, not just this solver.
The default preserves the historical behaviour (all Julia threads). Pass
`nthreads = 1` for a single-threaded solver, or to avoid perturbing a caller
that manages FFTW threading itself.
"""
struct PoissonSolver{Topology, D, P_Fwd, P_Inv}
    dx::NTuple{D, Float64}
    dct_plan::P_Fwd
    idct_plan::P_Inv
    eigenvalues::Array{Float64, D}
end

# --- the two eigenvalue families -----------------------------------------------
# `m_d` is the mode period: N_d for DCT-II (cell-centered), N_d − 1 for DCT-I
# (node-centered).  `isotropic` groups the sum before the single division, which
# is a *different rounding* from dividing each term by its own dx² whenever dx²
# is not a power of two (measured: the two differ in 20892 of 65536 entries at
# N=256).  The grouped form is what `analysis/rc2d_variants.jl` uses and what the
# RC2D migration gate asserts bit-identity against, so the square/cubic
# constructors below must keep taking this branch.
function _eigenvalues(N::NTuple{D,Int}, m::NTuple{D,Int}, dx::NTuple{D,Float64},
                      isotropic::Bool) where {D}
    ev = zeros(Float64, N...)
    if isotropic
        h2 = dx[1]^2
        @inbounds for I in CartesianIndices(ev)
            s = 0.0
            for d in 1:D
                s += 2.0 * (1.0 - cos((I[d] - 1) * pi / m[d]))
            end
            ev[I] = s / h2
        end
    else
        @inbounds for I in CartesianIndices(ev)
            s = 0.0
            for d in 1:D
                s += 2.0 * (1.0 - cos((I[d] - 1) * pi / m[d])) / dx[d]^2
            end
            ev[I] = s
        end
    end
    return ev
end

function _build(::CellCentered, N::NTuple{D,Int}, dx::NTuple{D,Float64},
                isotropic::Bool, flags, nthreads::Int) where {D}
    all(>(0), N) || throw(ArgumentError("A cell-centered grid needs N ≥ 1 per axis; got $N."))
    FFTW.set_num_threads(nthreads)
    dummy = zeros(Float64, N...)
    # REDFT10 is DCT-II (forward), REDFT01 is DCT-III (inverse)
    fwd = FFTW.plan_r2r(dummy, FFTW.REDFT10, flags = flags)
    inv = FFTW.plan_r2r(dummy, FFTW.REDFT01, flags = flags)
    ev = _eigenvalues(N, N, dx, isotropic)
    return PoissonSolver{CellCentered, D, typeof(fwd), typeof(inv)}(dx, fwd, inv, ev)
end

function _build(::NodeCentered, N::NTuple{D,Int}, dx::NTuple{D,Float64},
                isotropic::Bool, flags, nthreads::Int) where {D}
    all(>(1), N) || throw(ArgumentError("A node-centered grid needs N ≥ 2 per axis; got $N."))
    FFTW.set_num_threads(nthreads)
    dummy = zeros(Float64, N...)
    # REDFT00 is DCT-I, its own inverse up to the factor ∏ 2(N_d − 1)
    fwd = FFTW.plan_r2r(dummy, FFTW.REDFT00, flags = flags)
    # DCT-I modes are cos(kπ(i-1)/(N-1)), so the period is N−1, not N
    ev = _eigenvalues(N, N .- 1, dx, isotropic)
    return PoissonSolver{NodeCentered, D, typeof(fwd), typeof(fwd)}(dx, fwd, fwd, ev)
end

# --- 2D ------------------------------------------------------------------------
PoissonSolver(N1::Int, N2::Int, dx1::Float64, dx2::Float64, alg;
              flags = FFTW.MEASURE, nthreads::Int = Threads.nthreads()) =
    _build(alg, (N1, N2), (dx1, dx2), false, flags, nthreads)

# --- 3D ------------------------------------------------------------------------
PoissonSolver(N1::Int, N2::Int, N3::Int, dx1::Float64, dx2::Float64, dx3::Float64, alg;
              flags = FFTW.MEASURE, nthreads::Int = Threads.nthreads()) =
    _build(alg, (N1, N2, N3), (dx1, dx2, dx3), false, flags, nthreads)

# --- square / cubic: one N, one dx ---------------------------------------------
# Whether the per-axis terms are summed before or after the division by dx² is
# **frozen per topology, deliberately and not on principle**: `NodeCentered`
# groups, `CellCentered` does not.  Both are equally valid discretizations and
# differ only in rounding, but each is the form some existing bit-identity gate
# was recorded against — `RC2DHybrid` (node) against
# `analysis/rc2d_variants.jl`'s grouped form, and the historical cell-centered
# layouts in that same file against the ungrouped one.  Changing either silently
# breaks a gate without changing any answer's correctness, which is the worst
# kind of change.  New topologies should pick one and say so here.
@inline _groups(::NodeCentered) = true
@inline _groups(::CellCentered) = false

PoissonSolver(N::Int, dx::Float64, alg;
              flags = FFTW.MEASURE, nthreads::Int = Threads.nthreads()) =
    _build(alg, (N, N), (dx, dx), _groups(alg), flags, nthreads)

"""
    PoissonSolver(N, dx, Val(D), alg; kw...)

Cubic `D`-dimensional grid, one `N` and one `dx`.  `Val(2)` is the square 2D
solver; `Val(3)` is the cube.
"""
PoissonSolver(N::Int, dx::Float64, ::Val{D}, alg;
              flags = FFTW.MEASURE, nthreads::Int = Threads.nthreads()) where {D} =
    _build(alg, ntuple(_ -> N, D), ntuple(_ -> dx, D), _groups(alg), flags, nthreads)

"""
    trapezoid_weight(i, N)

1D dual-cell weight of node `i` on a node-centered `N`-point grid: `1/2` at the
two endpoints, `1` inside.  The dual volume of a node is `∏dx_d` times the
product of its per-axis weights — so in 3D a face node owns `1/2` of a cell, an
edge node `1/4`, and a corner node `1/8`.
"""
@inline trapezoid_weight(i::Int, N::Int) = (i == 1 || i == N) ? 0.5 : 1.0

@inline function _trapezoid_weight(I::CartesianIndex{D}, N::NTuple{D,Int}) where {D}
    w = 1.0
    @inbounds for d in 1:D
        w *= trapezoid_weight(I[d], N[d])
    end
    return w
end

# ==============================================================================
# 3. The Mutable State (Workspace Only)
# ==============================================================================
struct PoissonState{D}
    freq_buffer::Array{Float64, D}
end

PoissonState(N1::Int, N2::Int) = PoissonState{2}(zeros(Float64, N1, N2))
PoissonState(N1::Int, N2::Int, N3::Int) = PoissonState{3}(zeros(Float64, N1, N2, N3))
PoissonState(N::Int) = PoissonState(N, N)
PoissonState(N::Int, ::Val{D}) where {D} = PoissonState{D}(zeros(Float64, ntuple(_ -> N, D)...))

# ==============================================================================
# 4. The HPC API (In-Place, Zero Allocation)
# ==============================================================================

"""
    solve_poisson!(phi, rhs, state::PoissonState, solver::PoissonSolver)

Solves `-Δϕ = rhs` strictly in-place with mirror-Neumann boundaries, in 2D or 3D.
The output is written to `phi`; `rhs` is left untouched.

`rhs` must satisfy the compatibility condition of the topology — zero plain mean
for [`CellCentered`](@ref), zero **trapezoid-weighted** mean for
[`NodeCentered`](@ref) — and is rejected rather than silently projected if it
does not.  The zero mode of `ϕ` is set to zero, i.e. the solution returned is the
mean-zero representative.
"""
function solve_poisson!(phi::AbstractArray{Float64,D}, rhs::AbstractArray{Float64,D},
                        state::PoissonState{D},
                        solver::PoissonSolver{CellCentered,D}) where {D}
    N = size(phi)
    _checksize(N, size(rhs), size(state.freq_buffer), size(solver.eigenvalues))

    # 1. Strict physical validation (Neumann compatibility: the plain mean, since
    #    every cell-centered dual volume is the same dx^D).
    mean_rhs = sum(rhs) / prod(N)
    if abs(mean_rhs) > 1e-10
        throw(ArgumentError("Ill-posed PDE: The source term `rhs` must integrate to zero for Neumann boundaries. Current mean: $mean_rhs"))
    end

    # 2. Forward transform to the frequency domain
    mul!(state.freq_buffer, solver.dct_plan, rhs)

    # 3. Spectral division by the eigenvalues.
    #    REDFT10 followed by REDFT01 scales the array by ∏ 2N_d.
    normalization = 1.0 / prod(2 * n for n in N)
    buf = state.freq_buffer
    ev = solver.eigenvalues
    @inbounds for I in CartesianIndices(buf)
        if I == first(CartesianIndices(buf))
            # The DC component (k=0) must be mapped to 0 to anchor the solution
            buf[I] = 0.0
        else
            buf[I] *= (normalization / ev[I])
        end
    end

    # 4. Inverse transform back to the physical domain
    mul!(phi, solver.idct_plan, buf)
    return nothing
end

function solve_poisson!(phi::AbstractArray{Float64,D}, rhs::AbstractArray{Float64,D},
                        state::PoissonState{D},
                        solver::PoissonSolver{NodeCentered,D}) where {D}
    N = size(phi)
    _checksize(N, size(rhs), size(state.freq_buffer), size(solver.eigenvalues))

    # 1. Strict physical validation — the *trapezoid-weighted* mean.  This is the
    #    compatibility condition the mirror-Neumann operator actually imposes
    #    here; a source with zero plain mean but nonzero weighted mean would be
    #    silently projected, so it is rejected instead.
    wsum = 0.0; wtot = 0.0
    @inbounds for I in CartesianIndices(rhs)
        w = _trapezoid_weight(I, N)
        wsum += w * rhs[I]
        wtot += w
    end
    mean_rhs = wsum / wtot
    if abs(mean_rhs) > 1e-10
        throw(ArgumentError("Ill-posed PDE: the source term `rhs` must have zero trapezoid-weighted mean for Neumann boundaries on a node-centered grid. Current weighted mean: $mean_rhs"))
    end

    # 2. Forward transform to the frequency domain
    mul!(state.freq_buffer, solver.dct_plan, rhs)

    # 3. Spectral division.  REDFT00 applied twice scales by ∏ 2(N_d − 1).
    #    NB the arithmetic here is `x * norm / eig`, not the `x * (norm/eig)` of
    #    the CellCentered branch — the two round differently, and this is the
    #    form the RC2D migration gate asserts bit-identity against.
    normalization = 1.0 / prod(2 * (n - 1) for n in N)
    buf = state.freq_buffer
    ev = solver.eigenvalues
    @inbounds for I in CartesianIndices(buf)
        if I == first(CartesianIndices(buf))
            buf[I] = 0.0
        else
            buf[I] = buf[I] * normalization / ev[I]
        end
    end

    # 4. Inverse transform back to the physical domain (DCT-I is its own inverse)
    mul!(phi, solver.idct_plan, buf)
    return nothing
end

function _checksize(nphi, nrhs, nbuf, nev)
    (nphi == nrhs == nbuf == nev) || throw(DimensionMismatch(
        "phi $nphi, rhs $nrhs, state $nbuf and solver $nev must all have the same shape."))
    return nothing
end

# ==============================================================================
# 5. The Casual API (Allocating, User-Friendly)
# ==============================================================================

"""
    solve_poisson(rhs, dx1, dx2; alg = CellCentered())
    solve_poisson(rhs, dx1, dx2, dx3; alg = CellCentered())
    solve_poisson(rhs, dx; alg = CellCentered())

Allocates the required FFTW plans and workspace, and solves `-Δϕ = rhs`.
Returns `ϕ`.  Convenient, but it plans a transform on every call — build a
[`PoissonSolver`](@ref) once and reuse it inside any loop, especially in 3D
where `MEASURE` planning runs to tens of seconds.
"""
function solve_poisson(rhs::AbstractArray{Float64,2}, dx1::Float64, dx2::Float64;
                       alg = CellCentered(), kw...)
    N1, N2 = size(rhs)
    solver = PoissonSolver(N1, N2, dx1, dx2, alg; kw...)
    phi = zeros(Float64, N1, N2)
    solve_poisson!(phi, rhs, PoissonState(N1, N2), solver)
    return phi
end

function solve_poisson(rhs::AbstractArray{Float64,3}, dx1::Float64, dx2::Float64,
                       dx3::Float64; alg = CellCentered(), kw...)
    N1, N2, N3 = size(rhs)
    solver = PoissonSolver(N1, N2, N3, dx1, dx2, dx3, alg; kw...)
    phi = zeros(Float64, N1, N2, N3)
    solve_poisson!(phi, rhs, PoissonState(N1, N2, N3), solver)
    return phi
end

# The convenience square/cubic API.  Routed through the *cubic* PoissonSolver
# constructor rather than the rectangular one, so that this and the in-place API
# agree to the last bit (they differ otherwise — see `_eigenvalues`).
function solve_poisson(rhs::AbstractArray{Float64,D}, dx::Float64;
                       alg = CellCentered(), kw...) where {D}
    N = size(rhs)
    allequal(N) || throw(ArgumentError(
        "Array is $(join(N, " x ")) but only one grid step 'dx' was provided. " *
        "Provide one dx per axis for non-cubic domains."))
    solver = PoissonSolver(N[1], dx, Val(D), alg; kw...)
    phi = zeros(Float64, N...)
    solve_poisson!(phi, rhs, PoissonState(N[1], Val(D)), solver)
    return phi
end

end # module FastPoisson
