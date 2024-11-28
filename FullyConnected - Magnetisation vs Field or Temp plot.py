import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count

#Fix random seed so reproducable
np.random.seed(1234)

# Parameters
J = 1.0       # Coupling 
b = 1.5       # Inverse temperature 
hVals = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.075, -0.05, -0.025, -0.01, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Different external fields
T_vals = np.linspace(0.5, 4.0, 20)
sweep = 10000  # Number of sweeps
trans = 5000  # Will ignore values from sweep index below this

  


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

def simulateSystemTemperature(args):
    # Unpack variables
    N, sweep, trans, J, T, h = args
    b = 1.0 / T  # Compute beta from temperature

    # Initialize spins randomly (±1 with equal probability)
    s = np.random.choice([-1, 1], size=N)
    m = np.sum(s)  # Initial magnetization

    # Data storage
    magnetizations = []

    # Main loop
    for mcs in range(sweep):
        s, m = MCstep(s, J, b, h, N, m)
        if mcs >= trans:  # Skip transient period
            magnetizations.append(m / N)  # Normalize magnetization from sum

    # Return average magnetization for this temperature
    return T, np.mean(magnetizations)


def temperatureSweepMagnetisations(h=0.1):
    # Set system size and external field (choose h = 0 or another fixed value)
    N = 30**2
    #h = 0.1  # Example: Fix external field for this sweep

    # Define output file for temperature sweep data
    output_file = f"temperature_sweep_N_{N}_h_{h:.2f}_J_{J:.2f}.txt"

    # Check if the data file already exists
    if os.path.exists(output_file):
        print(f"Temperature sweep data for h={h:.2f}, N={N}, h={h} already exists. Loading from file.")
        data = np.loadtxt(output_file, comments='#')  # Ignore comment lines
        temperatures = data[:, 0]  # First column is temperature
        avgMagnetizations = data[:, 1]  # Second column is average magnetization
        return temperatures, avgMagnetizations

    print(f"Temperature sweep data for h={h:.2f}, N={N} not found. Running simulations...")

    # Run for different temperatures
    argsList = [(N, sweep, trans, J, T, h) for T in T_vals]

    # No of available cores - for speed
    num_cores = cpu_count()

    # Run simulations in parallel
    with Pool(num_cores) as pool:
        results = pool.map(simulateSystemTemperature, argsList)

    # Extract temperatures and average magnetizations
    temperatures = [result[0] for result in results]
    avgMagnetizations = [result[1] for result in results]

    # Save results to a text file
    with open(output_file, 'w') as file:
        file.write("# Temperature (T)   Average Magnetization\n")
        for T, m in zip(temperatures, avgMagnetizations):
            file.write(f"{T:.6f}   {m:.6f}\n")

    print(f"Temperature sweep data saved to {output_file}")

    return temperatures, avgMagnetizations

def plot_magnetization_vs_temperature(T_vals, avgMagnetizations, h):
    plt.figure(figsize=(8, 6))
    
    # Plot the Monte Carlo data
    plt.plot(T_vals, avgMagnetizations, marker='o', linestyle='-', color='r', label="Average Magnetization (MC)")
    
    # Customize the plot
    plt.xlabel("Temperature (T)", fontsize=12)
    plt.ylabel("Average Magnetization", fontsize=12)
    plt.title("Magnetization vs Temperature for h = %s" % str(h), fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()


# Runs for each N and h
def simulateSystem(args):

    # Unpack variables
    N, sweep, trans, J, b, h = args

    # Define filenames for the data
    base = f"N_{N}_h_{h:.2f}_b{b:.2f}_J_{J:.2f}_data"
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

# Function to calculate average magnetization for various external field strengths
def hSweepMagnetisations():
    # Set system size
    #N = 20**2  # System size (for example, 400 sites)
    N = 30**2  

    # Run for different field strengths h
    argsList = [(N, sweep, trans, J, b, h) for h in hVals]

    # No of available cores - for speed
    num_cores = cpu_count()

    #Run using all available cores :)
    with Pool(num_cores) as pool:
        results = pool.map(simulateSystem, argsList)

    # Extract average magnetizations and corresponding h values
    sortedFields = [result[1] for result in results]
    avgMagnetizations = [result[2] for result in results]

    return sortedFields, avgMagnetizations

# Plot magnetization vs. applied field
def plot_magnetization_vs_field(hVals, avgMagnetizations, b):
    plt.figure(figsize=(8, 6))
    
    # Plot the Monte Carlo data
    plt.plot(hVals, avgMagnetizations, marker='o', linestyle='-', color='b', label="Average Magnetization (MC)")
    
    # Customize the plot
    plt.xlabel("External Field (h)", fontsize=12)
    plt.ylabel("Average Magnetization", fontsize=12)
    plt.title("Magnetization vs Applied External Field", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

# So multiprocessing doesn't break - can only run if definitely in main
if __name__ == "__main__":
    
<<<<<<< Updated upstream
    h = 0.1
    print("Starting Field Sweep")
=======
    h = 20.0
    #print("Starting Field Sweep")
>>>>>>> Stashed changes
    # Field sweep
    # Average magnetization for different field strengths
    sortedFields, magnetizations = hSweepMagnetisations()
    # Plot results - field curve
    plot_magnetization_vs_field(sortedFields, magnetizations, b)

    print("Starting Temp Sweep")
    # Temperature sweep
    temperatures, magnetizations_temp = temperatureSweepMagnetisations(0.1)
    plot_magnetization_vs_temperature(temperatures, magnetizations_temp, h)
