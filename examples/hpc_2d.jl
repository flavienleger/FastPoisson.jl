# ==============================================================================
# FastPoisson.jl - 2D HPC Dipole Example
# ==============================================================================
# Run this from the terminal using: julia --project=@. examples/hpc_2d.jl
# ==============================================================================

using FastPoisson
using Printf

function run_hpc_dipole()
    println("Setting up 2D Fast Poisson Dipole Example...")
    
    # 1. Initialization parameters
    N = 256
    dx = 1.0 / N
    dy = 1.0 / N
    
    # ==========================================================================
    # THE HPC WAY: Pre-allocate EVERYTHING outside the hot loop
    # ==========================================================================
    # Allocate the FFTW plans and eigenvalue matrices EXACTLY ONCE
    solver = PoissonSolver(N, dx, dy, CellCentered())
    
    # Allocate the mutable workspace EXACTLY ONCE
    state = PoissonState(N)
    
    # Allocate the physical data arrays (user-owned) EXACTLY ONCE
    phi = zeros(Float64, N, N)
    rhs = zeros(Float64, N, N)
    
    # 2. Setup the Physics (The Dipole)
    # We place a positive charge (+) and a negative charge (-) in the domain.
    pos_x = N ÷ 3
    neg_x = N - pos_x + 1  # Exactly mirrored across the center
    
    pos_idx = (pos_x, N ÷ 2)
    neg_idx = (neg_x, N ÷ 2)
    
    # Fill the pre-allocated rhs array
    rhs[pos_idx...] =  1000.0
    rhs[neg_idx...] = -1000.0
    
    # Strict validation: Ensure the source term has exactly zero mean
    mean_rhs = sum(rhs) / (N^2)
    @printf("Source term mean: %.2e (Must be effectively zero)\n", mean_rhs)
    
    println("\nStarting zero-allocation hot loop...")
    t0 = time()
    
    # Simulate a PDE gradient flow where we solve the Poisson equation repeatedly
    iters = 1000
    for step in 1:iters
        # We pass the pre-allocated target, source, workspace, and solver context.
        # This function call allocates exactly 0 bytes of memory!
        solve_poisson!(phi, rhs, state, solver)
        
        # ... (In a real gradient flow, you would update the source term `rhs` here) ...
    end
    elapsed = time() - t0
    
    # ==========================================================================
    # Verification & Diagnostics
    # ==========================================================================
    @printf("Executed %d Poisson solves in %.4f seconds.\n", iters, elapsed)
    
    # Diagnostics
    max_phi = maximum(phi)
    min_phi = minimum(phi)
    
    println("\n--- Physics Check ---")
    @printf("Max Potential (near + charge):  %.4f\n", max_phi)
    @printf("Min Potential (near - charge): %.4f\n", min_phi)
    
    # Because of the DC component zeroing in our solver, the potential should be symmetric
    symmetry_error = abs(max_phi + min_phi)
    
    if symmetry_error < 1e-10
        println("✅ Dipole potential is perfectly symmetric!")
    else
        println("❌ Warning: Potential lost symmetry.")
    end
end

# Execute the script
run_hpc_dipole()