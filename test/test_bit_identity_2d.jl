# Bit-identity gate for the N-dimensional refactor (Phase 4.1).
#
# Generalizing this module from 2D to N-D must not move a single bit in 2D.  That
# is a stronger requirement than "still correct", and it is the right one: the
# caller's RC2D migration gate (`validation/test_migration_equivalence.jl`) is
# itself an assertion of bit-identity, and the ascent it drives amplifies
# round-off by ×1.05 per step, so a last-bit change in the Poisson solve fails a
# downstream gate that has nothing obviously to do with FFTW.
#
# Compares against an arbitrary git ref, exported to a temp dir so the comparison
# runs against an immutable snapshot rather than a dirty tree.
#
# Run:  julia --project=. -t 4 test/test_bit_identity_2d.jl [ref]
#       (ref defaults to the commit before the refactor)
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)
using FFTW, Printf, Random, Test

const REF  = length(ARGS) ≥ 1 ? ARGS[1] : "e99b982"
const REPO = normpath(joinpath(@__DIR__, ".."))
const TMP  = mktempdir()

run(pipeline(`git -C $REPO archive $REF --prefix=old/ src/FastPoisson.jl`, `tar -x -C $TMP`))

module Old; end
module New; end
Base.include(Old, joinpath(TMP, "old", "src", "FastPoisson.jl"))
Base.include(New, joinpath(REPO, "src", "FastPoisson.jl"))
const O = Old.FastPoisson
const N_ = New.FastPoisson

function compare(name, so, sn, rhs)
    n1, n2 = size(rhs)
    a = zeros(n1, n2); b = zeros(n1, n2)
    O.solve_poisson!(a, rhs, O.PoissonState(n1, n2), so)
    N_.solve_poisson!(b, rhs, N_.PoissonState(n1, n2), sn)
    ev, ok = so.eigenvalues == sn.eigenvalues, a == b
    @printf("%-34s eigenvalues identical: %-3s   phi bit-identical: %-3s   maxdiff %.1e\n",
            name, ev ? "yes" : "NO", ok ? "yes" : "NO", maximum(abs, a .- b))
    @test ev
    @test ok
end

# ESTIMATE, not MEASURE: MEASURE picks its algorithm by timing, so two plans of
# the same shape in the same process are only identical because FFTW caches
# wisdom.  ESTIMATE makes the gate deterministic rather than incidentally so.
const FL = FFTW.ESTIMATE

@testset "2D bit-identity against $REF" begin
    for N in (64, 129, 256)
        Random.seed!(7)
        h = 2.0 / (N - 1)                     # node-centered: RC2D's spacing
        w = [((i == 1 || i == N) ? 0.5 : 1.0) * ((j == 1 || j == N) ? 0.5 : 1.0)
             for i in 1:N, j in 1:N]
        r = randn(N, N); r .-= sum(r .* w) / sum(w)
        compare("NodeCentered square N=$N",
                O.PoissonSolver(N, h, O.NodeCentered(); flags = FL),
                N_.PoissonSolver(N, h, N_.NodeCentered(); flags = FL), r)
        compare("NodeCentered rect   N=$N",
                O.PoissonSolver(N, N, h, h, O.NodeCentered(); flags = FL),
                N_.PoissonSolver(N, N, h, h, N_.NodeCentered(); flags = FL), r)

        d = 2.0 / N                           # cell-centered
        r2 = randn(N, N); r2 .-= sum(r2) / length(r2)
        compare("CellCentered square N=$N",
                O.PoissonSolver(N, d, O.CellCentered(); flags = FL),
                N_.PoissonSolver(N, d, N_.CellCentered(); flags = FL), r2)
        compare("CellCentered rect   N=$N",
                O.PoissonSolver(N, N, d, d, O.CellCentered(); flags = FL),
                N_.PoissonSolver(N, N, d, d, N_.CellCentered(); flags = FL), r2)
    end

    Random.seed!(11); r = randn(48, 80); r .-= sum(r) / length(r)
    compare("CellCentered 48×80 aniso",
            O.PoissonSolver(48, 80, 0.03, 0.017, O.CellCentered(); flags = FL),
            N_.PoissonSolver(48, 80, 0.03, 0.017, N_.CellCentered(); flags = FL), r)
end
