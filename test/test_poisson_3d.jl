# Validation gate for the 3D Poisson solves (Phase 4.1), and a regression gate on
# the 2D ones that the N-dimensional refactor must not have moved.
#
# The load-bearing assertion is *not* a convergence order against a smooth exact
# solution — that would pass even with the wrong dual volumes or the wrong
# normalization, which is exactly the class of error that bit 2D in Phase 2
# (`reviews/phase2-poisson-bc.md`).  It is an **algebraic** one: the solver must
# invert the discrete mirror-Neumann Laplacian that the caller's finite-volume
# assembly actually builds, to machine precision, on random data.
#
#     apply the 7-point stencil to phi, get back rhs (mean-zero component).
#
# That is a statement about the operator, not about smoothness, so no
# discretization error hides in it.
#
# Run:  julia --project=. -t 4 test/test_poisson_3d.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)
using FastPoisson, FFTW, Printf, Random, Test
using FastPoisson: trapezoid_weight

# ------------------------------------------------------------------------------
# The discrete operator, written independently of the solver.
#
# Both topologies discretize -div(grad) by summing face fluxes over the dual cell
# and dividing by the dual volume, but the **boundary rows differ**, and getting
# this wrong is the single easiest way to write a Poisson test that certifies the
# wrong solver:
#
# * `CellCentered` — the mirror reflects about the boundary *face*, so the ghost
#   equals its neighbour, the flux cancels, and the missing term simply drops.
#   Dual volumes are uniform.
# * `NodeCentered` — the mirror reflects about the boundary *node*, so the ghost
#   equals the node's second neighbour and the surviving term is **doubled**.
#   Equivalently: the boundary node's dual cell is half-width, and dividing the
#   one-sided flux balance by that half volume produces exactly the same factor
#   of 2.  The two statements agreeing is the consistency of the layout.
#
# So `NodeCentered` is not `CellCentered` with different dual volumes bolted on.
# ------------------------------------------------------------------------------
function neumann_laplacian(phi::Array{Float64,D}, dx::NTuple{D,Float64}, topo) where {D}
    N = size(phi)
    mirror = topo isa NodeCentered      # boundary term doubles, rather than dropping
    out = zeros(Float64, N...)
    @inbounds for I in CartesianIndices(phi)
        acc = 0.0
        for d in 1:D
            h2 = dx[d]^2
            e = CartesianIndex(ntuple(k -> k == d ? 1 : 0, D))
            lo, hi = I[d] > 1, I[d] < N[d]
            w = (mirror && !(lo && hi)) ? 2.0 : 1.0
            if lo; acc += w * (phi[I] - phi[I - e]) / h2; end
            if hi; acc += w * (phi[I] - phi[I + e]) / h2; end
        end
        out[I] = acc
    end
    return out
end

weights(N::NTuple{D,Int}, ::NodeCentered) where {D} =
    [prod(d -> trapezoid_weight(I[d], N[d]), 1:D) for I in CartesianIndices(N)]
weights(N::NTuple{D,Int}, ::CellCentered) where {D} = ones(Float64, N...)

"Project onto the mean-zero subspace in the topology's own dual-volume inner product."
function project!(r, w)
    r .-= sum(r .* w) / sum(w)
    return r
end

function check(name, N::NTuple{D,Int}, dx::NTuple{D,Float64}, alg;
               solver, state) where {D}
    w = weights(N, alg)
    Random.seed!(4103)
    rhs = project!(randn(N...), w)

    phi = zeros(Float64, N...)
    solve_poisson!(phi, rhs, state, solver)

    # (a) the operator identity: -Δ_h phi == rhs, to machine precision
    back = neumann_laplacian(phi, dx, alg)
    resid = maximum(abs, back .- rhs) / maximum(abs, rhs)

    # (b) the returned solution is the mean-zero representative in the right
    #     inner product (plain for cell-centered, trapezoid for node-centered)
    gauge = abs(sum(phi .* w) / sum(w))

    # (c) rhs must be left untouched
    rhs2 = copy(rhs)
    solve_poisson!(phi, rhs, state, solver)
    untouched = rhs == rhs2

    @printf("%-34s  relative residual = %8.1e   gauge = %8.1e   rhs untouched: %s\n",
            name, resid, gauge, untouched ? "yes" : "NO")
    @test resid < 1e-11
    @test gauge < 1e-11
    @test untouched
    return phi
end

@testset "FastPoisson" begin

@testset "3D operator identity" begin
    for N in (17, 33)
        # NodeCentered is the layout RC3DHybrid runs: y_∅ = 0 is the node (1,1,1).
        h = 2.0 / (N - 1)
        check("NodeCentered cubic N=$N", (N, N, N), (h, h, h), NodeCentered();
              solver = PoissonSolver(N, h, Val(3), NodeCentered(); flags = FFTW.ESTIMATE),
              state  = PoissonState(N, Val(3)))
        d = 2.0 / N
        check("CellCentered cubic N=$N", (N, N, N), (d, d, d), CellCentered();
              solver = PoissonSolver(N, d, Val(3), CellCentered(); flags = FFTW.ESTIMATE),
              state  = PoissonState(N, Val(3)))
    end
    # Anisotropic and non-cubic: catches an axis-order or eigenvalue-pairing slip
    # that a cube cannot see.
    N = (9, 13, 17); dx = (0.25, 0.1, 0.0625)
    check("NodeCentered 9×13×17 aniso", N, dx, NodeCentered();
          solver = PoissonSolver(N..., dx..., NodeCentered(); flags = FFTW.ESTIMATE),
          state  = PoissonState(N...))
    check("CellCentered 9×13×17 aniso", N, dx, CellCentered();
          solver = PoissonSolver(N..., dx..., CellCentered(); flags = FFTW.ESTIMATE),
          state  = PoissonState(N...))
end

@testset "2D operator identity (regression)" begin
    for N in (33, 65)
        h = 2.0 / (N - 1)
        check("NodeCentered square N=$N", (N, N), (h, h), NodeCentered();
              solver = PoissonSolver(N, h, NodeCentered(); flags = FFTW.ESTIMATE),
              state  = PoissonState(N))
        d = 2.0 / N
        check("CellCentered square N=$N", (N, N), (d, d), CellCentered();
              solver = PoissonSolver(N, d, CellCentered(); flags = FFTW.ESTIMATE),
              state  = PoissonState(N))
    end
end

@testset "2D eigenvalue rounding is frozen per topology" begin
    # The square NodeCentered constructor GROUPS the per-axis second differences
    # before the single division by dx²; the square CellCentered one does not.
    # That asymmetry is historical, not principled, and RC2D's migration gate is
    # bit-identity against it — so it is pinned here rather than left to be
    # "cleaned up" by a later reader.  See `_groups` in the module.
    # dx must NOT be a power of two, or division by dx² is exact and the pin is
    # vacuous — at dx = 2/64 = 2⁻⁵ the two forms agree in all 4096 entries.  Use
    # the RC2D spacing h = (a+1)/(N−1) = 2/255 at which the module's comment
    # records the measurement.
    N = 256; dx = 2.0 / (N - 1); m = N - 1
    node = PoissonSolver(N, dx, NodeCentered(); flags = FFTW.ESTIMATE)
    cell = PoissonSolver(N, dx, CellCentered(); flags = FFTW.ESTIMATE)
    grouped   = [(2(1 - cos((i-1)π/m)) + 2(1 - cos((j-1)π/m))) / dx^2 for i in 1:N, j in 1:N]
    ungrouped = [2(1 - cos((i-1)π/N))/dx^2 + 2(1 - cos((j-1)π/N))/dx^2 for i in 1:N, j in 1:N]
    @test node.eigenvalues == grouped
    @test cell.eigenvalues == ungrouped
    # and the grouping genuinely matters at this N — otherwise the pin is vacuous
    alt = [2(1 - cos((i-1)π/m))/dx^2 + 2(1 - cos((j-1)π/m))/dx^2 for i in 1:N, j in 1:N]
    ndiff = count(node.eigenvalues .!= alt)
    @printf("%-34s  grouped vs ungrouped differ in %d of %d entries\n", "rounding pin", ndiff, N^2)
    @test ndiff > 0
end

@testset "compatibility conditions are enforced, not projected" begin
    N = 17
    h = 2.0 / (N - 1)
    sn = PoissonSolver(N, h, Val(3), NodeCentered(); flags = FFTW.ESTIMATE)
    st = PoissonState(N, Val(3))
    phi = zeros(N, N, N)

    # A source with zero PLAIN mean but nonzero trapezoid-weighted mean is the
    # exact trap the node-centered layout sets — the 3D corner node owns h³/8,
    # so a naive `sigma[1,1,1] -= sum(sigma)` dump leaves the solve incompatible.
    bad = randn(N, N, N); bad .-= sum(bad) / length(bad)
    w = weights((N, N, N), NodeCentered())
    @test abs(sum(bad) / length(bad)) < 1e-12          # plain mean is zero...
    @test abs(sum(bad .* w) / sum(w)) > 1e-3           # ...weighted mean is not
    @test_throws ArgumentError solve_poisson!(phi, bad, st, sn)

    # and the correctly dumped source is accepted
    good = copy(bad); good[1, 1, 1] -= sum(good .* w) / w[1, 1, 1]
    solve_poisson!(phi, good, st, sn)
    @test maximum(abs, neumann_laplacian(phi, (h, h, h), NodeCentered()) .- good) / maximum(abs, good) < 1e-11

    # corner dual volume in 3D is 1/8, not the 2D 1/4
    @test w[1, 1, 1] == 0.125
    @test w[1, 1, 2] == 0.25
    @test w[1, 2, 2] == 0.5
    @test w[2, 2, 2] == 1.0

    sc = PoissonSolver(N, 2.0/N, Val(3), CellCentered(); flags = FFTW.ESTIMATE)
    @test_throws ArgumentError solve_poisson!(phi, randn(N, N, N) .+ 1.0, st, sc)
end

@testset "casual API agrees with the in-place API bitwise" begin
    N = 17; h = 2.0 / (N - 1)
    w = weights((N, N, N), NodeCentered())
    Random.seed!(9); rhs = project!(randn(N, N, N), w)
    a = solve_poisson(rhs, h; alg = NodeCentered(), flags = FFTW.ESTIMATE)
    b = zeros(N, N, N)
    solve_poisson!(b, rhs, PoissonState(N, Val(3)),
                   PoissonSolver(N, h, Val(3), NodeCentered(); flags = FFTW.ESTIMATE))
    @test a == b
    @test_throws ArgumentError solve_poisson(zeros(4, 5, 6), h)
end

end
