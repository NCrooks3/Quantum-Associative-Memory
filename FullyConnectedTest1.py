import numpy as np
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool, cpu_count
import os

# Parameters
J = 1.0      # Coupling 
b = 1.5      # Inverse temperature
h_values = [0.0, 0.1, 0.5, 1.0, -1.0]  # Applied magnetic fields
sweep = 10000  # num of sweeps
trans = 1000   # Initial values are cut - let results stabelise


def calc_energy(s, N, h):
    m = np.sum(s)   #Sum Spins for magnetisation
    return -J * m**2 / (2 * N) - h * m  #See notes

def monte_carlo_step_optimized(s, J, b, h, N, m):
    flip_indices = np.random.randint(0, N, N)  # Random indices to flip
    for k in flip_indices:
        dE = 2 * s[k] * (J * m / N + h)  # Energy difference
        if dE < 0 or random.random() < np.exp(-b * dE):
            s[k] = -s[k]  # Accept flip
            m += 2 * s[k]  # Update magnetization
    return s, m

def simulate_system(args):
    # Unpack variables
    N, sweep, trans, J, b, h = args

    # Random initial spins
    s = np.random.choice([-1, 1], size=N)
    m = np.sum(s)  # Initial magnetization

    magnetizations = []
    energies = []

    # Main loop
    for mcs in range(sweep):
        s, m = monte_carlo_step_optimized(s, J, b, h, N, m)
        if mcs >= trans:  # Cut out early values - before stabelised
            magnetizations.append(m / N)  # Normalize 
            energies.append(calc_energy(s, N, h)) 

    # Save results
    base_name = f"N_{N}_h_{h:.2f}_data"
    np.savetxt(f"{base_name}_magnetizations.txt", magnetizations)
    np.savetxt(f"{base_name}_energies.txt", energies)

    return N, h, np.mean(magnetizations), np.mean(energies)

# Run for different N and h in paralell
def main_parallel():
    # lattice sizes used
    system_sizes = [2**2, 20**2, 50**2, 100**2]  # N = 4, 400, 2500, 10000
    args_list = [(N, sweep, trans, J, b, h) for N in system_sizes for h in h_values]

    # Check number of cores - for speed
    num_cores = cpu_count()

    with Pool(num_cores) as pool:
        results = pool.map(simulate_system, args_list)

    # Save summarised results
    summary_file = "summary_results.txt"
    with open(summary_file, "w") as f:
        f.write("N\th\tAvg_Magnetization\tAvg_Energy\n")
        for res in results:
            f.write(f"{res[0]}\t{res[1]:.2f}\t{res[2]:.4f}\t{res[3]:.4f}\n")

    print(f"Monte-Carlo complete. Results saved to {summary_file}.")

def plot_results():
    # Gather all files with data
    files = [f for f in os.listdir() if f.endswith("_magnetizations.txt")]

    # Group fby N
    grouped_files = {}
    for file in files:
        N = int(file.split("_")[1])  # Extract N from filename
        h = float(file.split("_h_")[1].split("_")[0])  # Extract h from filename

        if N not in grouped_files:
            grouped_files[N] = []
        
        grouped_files[N].append((h, file))

    # Plot magnetization for different h values at fixed N
    for N, file_group in grouped_files.items():
        plt.figure(figsize=(10, 6)) 
        for h, file in file_group:
            magnetizations = np.loadtxt(file)
            plt.plot(magnetizations, label=f"h={h:.2f}")

        plt.xlabel("Monte Carlo Steps", fontsize=12)
        plt.ylabel("Magnetization", fontsize=12)
        plt.title(f"Magnetization vs Monte Carlo Steps for N={N}", fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

        # Histogram of magnetizations for each system size at fixed N
        plt.figure(figsize=(10, 6))
        for h, file in file_group:
            magnetizations = np.loadtxt(file)
            plt.hist(magnetizations, bins=50, density=True, alpha=0.5, label=f"h={h:.2f}")

        plt.xlabel("Magnetization", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"Magnetization Distribution for N={N}", fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Run parallel simulations
    main_parallel()

    # Plot results
    plot_results()
