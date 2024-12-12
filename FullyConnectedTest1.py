import numpy as np
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool, cpu_count
import os

# Parameters
J = 1.0      # Coupling 
#b = 1.5      # Inverse temperature
b = 1.0
h_values = [0.0, 0.1, 0.5, 1.0, -1.0, -0.1, -0.5]  # Applied magnetic fields
sweep = 10000  # num of sweeps
trans = 0   # Initial values are cut - let results stabelise


def calcEnergy(s, N, h):
    m = np.sum(s)   #sum spins for net magnetisation
    return -J * m**2 / (2 * N) - h * m  #See Ising model notes


#Run a Monte-Carlo step
def MCstep(s, J, b, h, N, m):
    for k in range(N):  #Loop over sites
        # Choose a random spin index
        i = np.random.randint(0, N)

        # Calculate energy change if flip spin at site i
        dE = 2 * s[i] * (J * m / N + h)  

        # Metropolis: accept flip if energy decreases or with probability exp(-beta * dE)
        if dE < 0 or np.random.random() < np.exp(-b * dE):
            s[i] = -s[i]  # Flip the spin
            m += 2 * s[i]  # Update the magnetization incrementally
    return s, m

def simulateSystem(args):

    # Unpack variables
    N, sweep, trans, J, b, h = args

    # Define filenames for the data
    base = f"N_{N}_h_{h:.2f}_data"
    magnetizationFile = f"{base}_magnetizations.txt"
    energyFile = f"{base}_energies.txt"

    # Check if data files already exist - Don't need to run again if already done
    if os.path.exists(magnetizationFile) and os.path.exists(energyFile):
        print(f"Data for N={N}, h={h} already exists. Skipping simulation.")
        # If data exists, read from the files
        magnetizations = np.loadtxt(magnetizationFile)
        energies = np.loadtxt(energyFile)
        avgMagnetization = np.mean(magnetizations)
        return N, h, avgMagnetization

    # Initialize spins randomly (±1 with equal probability)
    s = np.random.choice([-1, 1], size=N)
    m = np.sum(s)  # Initial magnetization

    # Data storage
    magnetizations = []
    energies = []

    # Main loop
    for mcs in range(sweep):
        s, m = MCstep(s, J, b, h, N, m)
        if mcs >= trans:  # Skip transient period
            magnetizations.append(m / N)  # Normalize magnetization from sum
            energies.append(calcEnergy(s, N, h))  # Calculate and store energy

    # Save results to text files
    np.savetxt(magnetizationFile, magnetizations)
    np.savetxt(energyFile, energies)

    # Return average magnetization for this system
    return N, h, np.mean(magnetizations)

# Run for different N and h in paralell
def mainParallel():
    # lattice sizes used
    #system_sizes = [2**2, 20**2, 50**2, 100**2]  # N = 4, 400, 2500, 10000
    system_sizes = [4**2, 50**2, 75**2]  # N = 4, 400, 2500, 10000
    args_list = [(N, sweep, trans, J, b, h) for N in system_sizes for h in h_values]

    # Check number of cores - for speed
    num_cores = cpu_count()

    with Pool(num_cores) as pool:
        results = pool.map(simulateSystem, args_list)

    # Save summarised results
    summaryFile = "summary_results.txt"
    with open(summaryFile, "w") as f:
        f.write("N\th\tAvg_Magnetization\tAvg_Energy\n")
        for res in results:
            f.write(f"{res[0]}\t{res[1]:.2f}\t{res[2]:.4f}\n")

    print(f"Monte-Carlo complete. Results saved to {summaryFile}.")

def plotResults():
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
        print(N)
        plt.figure(figsize=(10, 6)) 
        for h, file in file_group:
            magnetizations = np.loadtxt(file)
            plt.plot(np.linspace(0, len(magnetizations), len(magnetizations)) / N, magnetizations, label=f"h={h:.2f}")

        plt.xlabel("Monte Carlo Sweep", fontsize=12)
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
    mainParallel()

    # Plot results
    plotResults()
