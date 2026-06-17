module FastPoisson

using LinearAlgebra
using FFTW

export CellCentered
export PoissonSolver, PoissonState
export solve_poisson!, solve_poisson

# ==========================================
# 1. Topology / Algorithm Types
# ==========================================
struct CellCentered end

# ==========================================
# 2. The Solver Context (Immutable Setup)
# ==========================================
struct PoissonSolver{Topology, P_Fwd, P_Inv, M <: AbstractMatrix{Float64}}
    dx::Float64
    dy::Float64
    dx_sq::Float64
    dy_sq::Float64
    dct_plan::P_Fwd
    idct_plan::P_Inv
    eigenvalues::M
end

# Constructor for the 2D Cell-Centered Grid 
function PoissonSolver(N::Int, dx::Float64, dy::Float64, ::CellCentered)
    # FFTW requires a dummy array of the exact size and type to optimize the plan
    dummy = zeros(Float64, N, N)
    
    # REDFT10 is DCT-II (Forward), REDFT01 is DCT-III (Inverse)
    FFTW.set_num_threads(Threads.nthreads())
    dct_plan = FFTW.plan_r2r(dummy, FFTW.REDFT10, flags=FFTW.MEASURE)
    idct_plan = FFTW.plan_r2r(dummy, FFTW.REDFT01, flags=FFTW.MEASURE)
    
    # Precompute the spectral Laplacian eigenvalues
    eigenvalues = zeros(Float64, N, N)
    for j in 1:N, i in 1:N
        kx = i - 1
        ky = j - 1
        # DCT-II staggered/cell-centered eigenvalues
        eigenvalues[i, j] = 2.0 * (1.0 - cos(kx * pi / N)) / dx^2 + 
                            2.0 * (1.0 - cos(ky * pi / N)) / dy^2
    end
    
    return PoissonSolver{CellCentered, typeof(dct_plan), typeof(idct_plan), Matrix{Float64}}(
        dx, dy, dx^2, dy^2, dct_plan, idct_plan, eigenvalues
    )
end

# ==========================================
# 3. The Mutable State (Workspace Only)
# ==========================================
struct PoissonState
    freq_buffer::Matrix{Float64}
    
    function PoissonState(N::Int)
        new(zeros(Float64, N, N))
    end
end

# ==========================================
# 4. The HPC API (In-Place, Zero Allocation)
# ==========================================

"""
    solve_poisson!(phi::Matrix{Float64}, rhs::Matrix{Float64}, state::PoissonState, solver::PoissonSolver)

Solves the Poisson equation `-Δϕ = rhs` strictly in-place.
The output is written directly to `phi`. `rhs` is the source term.
"""
function solve_poisson!(phi::Matrix{Float64}, rhs::Matrix{Float64}, state::PoissonState, solver::PoissonSolver{CellCentered})
    N = size(phi, 1)
    
    # 1. Strict Physical Validation (Neumann Compatibility)
    mean_rhs = sum(rhs) / (N * N)
    if abs(mean_rhs) > 1e-10
        throw(ArgumentError("Ill-posed PDE: The source term `rhs` must integrate to zero for Neumann boundaries. Current mean: $mean_rhs"))
    end
    
    # 2. Forward Transform to Frequency Domain
    mul!(state.freq_buffer, solver.dct_plan, rhs)
    
    # 3. Spectral Division by Eigenvalues
    # Note: Using REDFT10 followed by REDFT01 automatically scales the array by 4N^2
    normalization = 1.0 / (4.0 * N * N)
    
    @inbounds for j in 1:N, i in 1:N
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

# ==========================================
# 5. The Casual API (Allocating, User-Friendly)
# ==========================================

"""
    solve_poisson(rhs::Matrix{Float64}, dx::Float64, dy::Float64; alg=CellCentered())

Allocates the required FFTW plans and workspaces, and solves `-Δϕ = rhs`.
Returns the potential matrix `ϕ`. 
"""
function solve_poisson(rhs::Matrix{Float64}, dx::Float64, dy::Float64; alg=CellCentered())
    N = size(rhs, 1)
    if size(rhs, 2) != N
        throw(ArgumentError("FastPoisson currently assumes a square N x N grid."))
    end
    
    # Allocate Engine, Workspace, and Output
    solver = PoissonSolver(N, dx, dy, alg)
    state = PoissonState(N)
    phi = zeros(Float64, N, N)
    
    # Execute
    solve_poisson!(phi, rhs, state, solver)
    
    return phi
end

end # module FastPoisson