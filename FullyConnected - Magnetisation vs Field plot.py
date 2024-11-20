import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count

# Parameters
J = 1.0       # Coupling 
b = 1.5       # Inverse temperature (for example, b = 1/T - using natural units so k_B = 1)
h_values = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.075, -0.05, -0.025, -0.01, -0.001, -0.0001, 0.0001, 0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Different external fields
sweep = 80000  # Number of sweeps
trans = 15000  # Will ignore values from sweep index below this


def calc_energy(s, N, h):
    m = np.sum(s)   #sum spins for net magnetisation
    return -J * m**2 / (2 * N) - h * m  #See Ising model notes


def monte_carlo_step_optimized(s, J, b, h, N, m):
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

# Runs for each N and h
def simulate_system(args):
    # Unpack variables
    N, sweep, trans, J, b, h = args

    # Define filenames for the data
    base_name = f"N_{N}_h_{h:.2f}_data"
    magnetization_file = f"{base_name}_magnetizations.txt"
    energy_file = f"{base_name}_energies.txt"

    # Check if data files already exist - Don't need to run again if already done
    if os.path.exists(magnetization_file) and os.path.exists(energy_file):
        print(f"Data for N={N}, h={h} already exists. Skipping simulation.")
        # If data exists, read from the files
        magnetizations = np.loadtxt(magnetization_file)
        energies = np.loadtxt(energy_file)
        avg_magnetization = np.mean(magnetizations)
        return N, h, avg_magnetization

    # Initialize spins randomly (±1 with equal probability)
    s = np.random.choice([-1, 1], size=N)
    m = np.sum(s)  # Initial magnetization

    # Data storage
    magnetizations = []
    energies = []

    # Main loop
    for mcs in range(sweep):
        s, m = monte_carlo_step_optimized(s, J, b, h, N, m)
        if mcs >= trans:  # Skip transient period
            magnetizations.append(m / N)  # Normalize magnetization from sum
            energies.append(calc_energy(s, N, h))  # Calculate and store energy

    # Save results to text files
    np.savetxt(magnetization_file, magnetizations)
    np.savetxt(energy_file, energies)

    # Return average magnetization for this system
    return N, h, np.mean(magnetizations)

# Function to calculate average magnetization for various external field strengths
def calculate_average_magnetization_for_h():
    # Fix system size
    #N = 20**2  # System size (for example, 400 sites)
    N = 50**2  #10000 sites

    # Run for different field strengths h
    args_list = [(N, sweep, trans, J, b, h) for h in h_values]

    # No of available cores - for speed
    num_cores = cpu_count()

    #Run using all available cores :)
    with Pool(num_cores) as pool:
        results = pool.map(simulate_system, args_list)

    # Extract average magnetizations and corresponding h values
    h_values_sorted = [result[1] for result in results]
    avg_magnetizations = [result[2] for result in results]

    return h_values_sorted, avg_magnetizations


# Plot magnetization vs. applied field
def plot_magnetization_vs_field(h_values, avg_magnetizations, b):
    plt.figure(figsize=(8, 6))
    
    # Plot the Monte Carlo data
    plt.plot(h_values, avg_magnetizations, marker='o', linestyle='-', color='b', label="Average Magnetization (MC)")
    
    # Customize the plot
    plt.xlabel("External Field (h)", fontsize=12)
    plt.ylabel("Average Magnetization", fontsize=12)
    plt.title("Magnetization vs Applied External Field", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

# So multiprocessing doesn't break - can only run if definitely in main
if __name__ == "__main__":
    # Average magnetization for different field strengths
    h_values_sorted, avg_magnetizations = calculate_average_magnetization_for_h()

    # Plot results - field curve
    plot_magnetization_vs_field(h_values_sorted, avg_magnetizations, b)
