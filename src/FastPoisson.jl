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
eigenvalues `2(1-cos(kπ/N))/dx²` and uniform dual volumes `dx²`.
"""
struct CellCentered end

"""
    NodeCentered()

Unknowns at nodes, `x_i = (i-1)dx`, `i = 1…N`, covering `[0, (N-1)dx]` — so the
two *endpoints* of the domain are unknowns.  Mirror-Neumann boundaries are
diagonalized exactly by DCT-I (`REDFT00`), with eigenvalues
`2(1-cos(kπ/(N-1)))/dx²`.

Two consequences that callers must respect, both verified in
`../validation/test_poisson_vertex.jl`:

* the dual volumes are **trapezoid-weighted**, `dx²·w_i·w_j` with `w = 1` except
  `w_1 = w_N = 1/2`, so a boundary node owns half a cell and a corner a quarter;
* therefore the compatibility condition this solve enforces is that the
  **trapezoid-weighted** mean of `rhs` vanish, not the plain mean.  A source with
  zero plain mean but nonzero weighted mean is silently projected, so
  `solve_poisson!` checks the weighted one.

Use this layout when a feature of the problem must sit exactly *on* the grid — in
the Rochet-Choné solver the outside option `y_∅ = (0,0)`, whose displacement to
`(dx/2, dx/2)` on a cell-centred grid costs a clean `O(1/N)` bias in the profit
(`reviews/phase2-ygrid.md` in the parent project).
"""
struct NodeCentered end

# ==============================================================================
# 2. The Solver Context (Immutable Setup)
# ==============================================================================

"""
    PoissonSolver(N1, N2, dx1, dx2, topology; flags = FFTW.MEASURE, nthreads = Threads.nthreads())
    PoissonSolver(N, dx, topology; flags, nthreads)

Build the FFTW plans and spectral eigenvalues for `-Δϕ = rhs` with mirror-Neumann
boundaries on the given `topology` ([`CellCentered`](@ref) or
[`NodeCentered`](@ref)).

# `flags` — planning effort, and the reproducibility tradeoff

`FFTW.MEASURE` (the default) chooses the transform algorithm by *timing*
candidates. `FFTW.ESTIMATE` chooses it from a cost model, without running
anything. The choice is a genuine tradeoff and it is deliberately a parameter
rather than a constant, because the right answer depends entirely on the problem:

* **Large grids, repeated solves, a cluster** — `MEASURE`. On big 3D transforms
  the payoff is large and it is paid once.
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
cluster run.

# `nthreads`

Forwarded to `FFTW.set_num_threads`, which is **global FFTW state**: it affects
every plan created afterwards anywhere in the session, not just this solver.
The default preserves the historical behaviour (all Julia threads). Pass
`nthreads = 1` for a single-threaded solver, or to avoid perturbing a caller
that manages FFTW threading itself.
"""
struct PoissonSolver{Topology, P_Fwd, P_Inv, M <: AbstractMatrix{Float64}}
    dx1::Float64
    dx2::Float64
    dx1_sq::Float64
    dx2_sq::Float64
    dct_plan::P_Fwd
    idct_plan::P_Inv
    eigenvalues::M
end

# Constructor for the 2D Cell-Centered Rectangular Grid 
function PoissonSolver(N1::Int, N2::Int, dx1::Float64, dx2::Float64, ::CellCentered;
                       flags = FFTW.MEASURE, nthreads::Int = Threads.nthreads())
    # FFTW requires a dummy array of the exact size and type to optimize the plan
    dummy = zeros(Float64, N1, N2)
    
    # REDFT10 is DCT-II (Forward), REDFT01 is DCT-III (Inverse)
    FFTW.set_num_threads(nthreads)
    dct_plan = FFTW.plan_r2r(dummy, FFTW.REDFT10, flags=flags)
    idct_plan = FFTW.plan_r2r(dummy, FFTW.REDFT01, flags=flags)
    
    # Precompute the spectral Laplacian eigenvalues for the asymmetric grid
    eigenvalues = zeros(Float64, N1, N2)
    for j in 1:N2, i in 1:N1
        k1 = i - 1
        k2 = j - 1
        # DCT-II staggered/cell-centered eigenvalues split by axis dimension
        eigenvalues[i, j] = 2.0 * (1.0 - cos(k1 * pi / N1)) / dx1^2 + 
                            2.0 * (1.0 - cos(k2 * pi / N2)) / dx2^2
    end
    
    return PoissonSolver{CellCentered, typeof(dct_plan), typeof(idct_plan), Matrix{Float64}}(
        dx1, dx2, dx1^2, dx2^2, dct_plan, idct_plan, eigenvalues
    )
end

# The Casual User Wrapper
PoissonSolver(N::Int, dx::Float64, alg::CellCentered; kw...) =
    PoissonSolver(N, N, dx, dx, alg; kw...)

# Constructor for the 2D Node-Centered Rectangular Grid
function PoissonSolver(N1::Int, N2::Int, dx1::Float64, dx2::Float64, ::NodeCentered;
                       flags = FFTW.MEASURE, nthreads::Int = Threads.nthreads())
    (N1 > 1 && N2 > 1) || throw(ArgumentError("A node-centered grid needs N ≥ 2 per axis; got ($N1, $N2)."))
    dummy = zeros(Float64, N1, N2)

    # REDFT00 is DCT-I, which is its own inverse up to the factor 4(N1-1)(N2-1)
    FFTW.set_num_threads(nthreads)
    dct_plan = FFTW.plan_r2r(dummy, FFTW.REDFT00, flags=flags)
    idct_plan = dct_plan

    # DCT-I modes are cos(kπ(i-1)/(N-1)), so the period is N-1, not N
    eigenvalues = zeros(Float64, N1, N2)
    for j in 1:N2, i in 1:N1
        eigenvalues[i, j] = 2.0 * (1.0 - cos((i - 1) * pi / (N1 - 1))) / dx1^2 +
                            2.0 * (1.0 - cos((j - 1) * pi / (N2 - 1))) / dx2^2
    end

    return PoissonSolver{NodeCentered, typeof(dct_plan), typeof(idct_plan), Matrix{Float64}}(
        dx1, dx2, dx1^2, dx2^2, dct_plan, idct_plan, eigenvalues
    )
end

# The Casual User Wrapper.  On a square grid both axes share `dx`, so the two
# second differences are summed *before* the single division — which is a
# different rounding from `A/dx² + B/dx²` above at any N for which dx² is not a
# power of two (measured: they differ in 20892 of 65536 entries at N=256).  The
# grouped form is the one `analysis/rc2d_variants.jl` uses, and the RC2D
# migration gate is bit-identity against it, so it is reproduced here exactly.
function PoissonSolver(N::Int, dx::Float64, ::NodeCentered;
                       flags = FFTW.MEASURE, nthreads::Int = Threads.nthreads())
    N > 1 || throw(ArgumentError("A node-centered grid needs N ≥ 2; got $N."))
    dummy = zeros(Float64, N, N)
    FFTW.set_num_threads(nthreads)
    dct_plan = FFTW.plan_r2r(dummy, FFTW.REDFT00, flags=flags)

    m = N - 1
    eigenvalues = zeros(Float64, N, N)
    for j in 1:N, i in 1:N
        eigenvalues[i, j] = (2.0 * (1.0 - cos((i - 1) * pi / m)) +
                             2.0 * (1.0 - cos((j - 1) * pi / m))) / dx^2
    end

    return PoissonSolver{NodeCentered, typeof(dct_plan), typeof(dct_plan), Matrix{Float64}}(
        dx, dx, dx^2, dx^2, dct_plan, dct_plan, eigenvalues
    )
end

"""
    trapezoid_weight(i, N)

1D dual-cell weight of node `i` on a node-centered `N`-point grid: `1/2` at the
two endpoints, `1` inside.  The 2D dual volume of `(i,j)` is `dx1·dx2·w_i·w_j`.
"""
@inline trapezoid_weight(i::Int, N::Int) = (i == 1 || i == N) ? 0.5 : 1.0

# ==============================================================================
# 3. The Mutable State (Workspace Only)
# ==============================================================================
struct PoissonState
    freq_buffer::Matrix{Float64}
    
    function PoissonState(N1::Int, N2::Int)
        new(zeros(Float64, N1, N2))
    end
end

# The Casual Outer Constructor
PoissonState(N::Int) = PoissonState(N, N)

# ==============================================================================
# 4. The HPC API (In-Place, Zero Allocation)
# ==============================================================================

"""
    solve_poisson!(phi::Matrix{Float64}, rhs::Matrix{Float64}, state::PoissonState, solver::PoissonSolver)

Solves the Poisson equation `-Δϕ = rhs` strictly in-place on a rectangular grid.
The output is written directly to `phi`. `rhs` is the source term.
"""
function solve_poisson!(phi::Matrix{Float64}, rhs::Matrix{Float64}, state::PoissonState, solver::PoissonSolver{CellCentered})
    N1, N2 = size(phi)
    
    # 1. Strict Physical Validation (Neumann Compatibility)
    mean_rhs = sum(rhs) / (N1 * N2)
    if abs(mean_rhs) > 1e-10
        throw(ArgumentError("Ill-posed PDE: The source term `rhs` must integrate to zero for Neumann boundaries. Current mean: $mean_rhs"))
    end
    
    # 2. Forward Transform to Frequency Domain
    mul!(state.freq_buffer, solver.dct_plan, rhs)
    
    # 3. Spectral Division by Eigenvalues
    # Note: Using REDFT10 followed by REDFT01 automatically scales the array by 4 * N1 * N2
    normalization = 1.0 / (4.0 * N1 * N2)
    
    @inbounds for j in 1:N2, i in 1:N1
        if i == 1 && j == 1
            # The DC component (k=0) must be mapped to 0 to anchor the solution
            state.freq_buffer[i, j] = 0.0 
        else
            # We are solving -Δϕ = f, hence the division by the positive eigenvalues
            state.freq_buffer[i, j] *= (normalization / solver.eigenvalues[i, j])
        end
    end
    
    # 4. Inverse Transform back to Physical Domain
    mul!(phi, solver.idct_plan, state.freq_buffer)
    
    return nothing
end

"""
    solve_poisson!(phi, rhs, state::PoissonState, solver::PoissonSolver{NodeCentered})

Solves `-Δϕ = rhs` in-place on a node-centered grid via DCT-I.  `rhs` must have
zero **trapezoid-weighted** mean — the discrete mass `Σ vol·rhs` — which is the
compatibility condition the mirror-Neumann operator actually imposes here; a
source with zero plain mean but nonzero weighted mean would be silently
projected, so it is rejected instead.
"""
function solve_poisson!(phi::Matrix{Float64}, rhs::Matrix{Float64}, state::PoissonState, solver::PoissonSolver{NodeCentered})
    N1, N2 = size(phi)

    # 1. Strict Physical Validation (Neumann Compatibility, trapezoid-weighted)
    wsum = 0.0
    wtot = 0.0
    @inbounds for j in 1:N2
        wj = trapezoid_weight(j, N2)
        for i in 1:N1
            w = trapezoid_weight(i, N1) * wj
            wsum += w * rhs[i, j]
            wtot += w
        end
    end
    mean_rhs = wsum / wtot
    if abs(mean_rhs) > 1e-10
        throw(ArgumentError("Ill-posed PDE: the source term `rhs` must have zero trapezoid-weighted mean for Neumann boundaries on a node-centered grid. Current weighted mean: $mean_rhs"))
    end

    # 2. Forward Transform to Frequency Domain
    mul!(state.freq_buffer, solver.dct_plan, rhs)

    # 3. Spectral Division by Eigenvalues.
    # REDFT00 applied twice scales the array by 4 * (N1-1) * (N2-1).
    normalization = 1.0 / (4.0 * (N1 - 1) * (N2 - 1))

    @inbounds for j in 1:N2, i in 1:N1
        if i == 1 && j == 1
            # The DC component (k=0) must be mapped to 0 to anchor the solution
            state.freq_buffer[i, j] = 0.0
        else
            state.freq_buffer[i, j] = state.freq_buffer[i, j] * normalization / solver.eigenvalues[i, j]
        end
    end

    # 4. Inverse Transform back to Physical Domain (DCT-I is its own inverse)
    mul!(phi, solver.idct_plan, state.freq_buffer)

    return nothing
end

# ==============================================================================
# 5. The Casual API (Allocating, User-Friendly)
# ==============================================================================

"""
    solve_poisson(rhs::Matrix{Float64}, dx1::Float64, dx2::Float64; alg=CellCentered())

Allocates the required FFTW plans and workspaces, and solves `-Δϕ = rhs`.
Returns the potential matrix `ϕ`. 
"""
function solve_poisson(rhs::Matrix{Float64}, dx1::Float64, dx2::Float64; alg=CellCentered(), kw...)
    N1, N2 = size(rhs)
    
    # Allocate Engine, Workspace, and Output
    solver = PoissonSolver(N1, N2, dx1, dx2, alg; kw...)
    state = PoissonState(N1, N2)
    phi = zeros(Float64, N1, N2)
    
    # Execute
    solve_poisson!(phi, rhs, state, solver)
    
    return phi
end

# The Convenience Casual API (Square).  Routed through the *square* PoissonSolver
# constructor rather than through the rectangular one, so that this and the
# in-place API agree to the last bit (they differ otherwise — see the note on the
# square NodeCentered constructor).
function solve_poisson(rhs::Matrix{Float64}, dx::Float64; alg=CellCentered(), kw...)
    N1, N2 = size(rhs)
    if N1 != N2
        throw(ArgumentError("Matrix is $N1 x $N2 but only one grid step 'dx' was provided. Provide dx1 and dx2 for rectangular domains."))
    end

    solver = PoissonSolver(N1, dx, alg; kw...)
    state = PoissonState(N1, N1)
    phi = zeros(Float64, N1, N1)
    solve_poisson!(phi, rhs, state, solver)
    return phi
end

end # module FastPoisson