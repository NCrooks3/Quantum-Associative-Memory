from multiprocessing import Pool, cpu_count
import numpy as np
import os


# Optimized simulateSystem to reduce I/O
def simulateSystemOptimized(args):
    # Unpack variables
    N, sweep, trans, J, b, h = args
    
    # Initialize spins randomly
    s = np.random.choice([-1, 1], size=N)
    m = np.sum(s)  # Initial magnetization
    
    # Data storage
    magnetizations = []
    
    # Main loop
    for mcs in range(sweep):
        s, m = MCstep(s, J, b, h, N, m)
        if mcs >= trans:  # Skip transient period
            magnetizations.append(m / N)
    
    # Calculate and return results in-memory
    avg_magnetization = np.mean(magnetizations)
    return N, h, avg_magnetization


# Optimized parallel execution
def mainParallelOptimized():
    # Lattice sizes and magnetic fields
    system_sizes = [2**2, 20**2, 50**2, 100**2]
    args_list = [(N, sweep, trans, J, b, h) for N in system_sizes for h in h_values]
    
    # Check number of cores
    num_cores = cpu_count()

    print(f"Using {num_cores} cores for simulation.")
    
    # Run parallel simulations
    with Pool(num_cores) as pool:
        results = pool.map(simulateSystemOptimized, args_list)
    
    # Save all results at once
    summary_file = "summary_results_optimized.txt"
    with open(summary_file, "w") as f:
        f.write("N\th\tAvg_Magnetization\n")
        for N, h, avg_m in results:
            f.write(f"{N}\t{h:.2f}\t{avg_m:.4f}\n")
    
    print(f"Monte Carlo simulations complete. Results saved to {summary_file}.")


if __name__ == "__main__":
    # Run parallel simulations
    mainParallelOptimized()
