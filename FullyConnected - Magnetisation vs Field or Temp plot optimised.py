import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count

# Fix random seed for reproducibility
np.random.seed(1234)

# Parameters
J = 1.0  # Coupling constant
b = 1.0  # Inverse temperature
hVals = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0, 0.05,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#hVals = [-1.0, -0.5, 0, 0.5, 1.0]
T_vals = np.linspace(0.00000001, 4.0, 50)
sweep = 10000  # Number of sweeps
trans = 5000  # Ignore values below this sweep index

def calcEnergy(s, N, h):
    m = np.sum(s)  # Net magnetization
    return -J * m**2 / (2 * N) - h * m  # Energy calculation

def MCsweep(s, J, b, h, N, m):
    for k in range(N):
        i = np.random.randint(0, N)
        dE = 2 * s[i] * (J * m / N + h)
        if dE < 0 or np.random.random() < np.exp(-b * dE):
            s[i] = -s[i]
            m += 2 * s[i]
    return s, m

def simulateSystemTemperature(args):
    N, sweep, trans, J, T, h = args
    b = 1.0 / T
    s = np.random.choice([-1, 1], size=N)
    m = np.sum(s)
    magnetizations = []
    for mcs in range(sweep):
        s, m = MCsweep(s, J, b, h, N, m)
        if mcs >= trans:
            magnetizations.append(m / N)
    return T, np.mean(magnetizations)

def temperatureSweepMagnetisations(h):
    N = 20**2
    output_file = f"temperature_sweep_N_{N}_h_{h:.2f}_J_{J:.2f}.txt"
    if os.path.exists(output_file):
        print(f"Loading existing data for h={h:.2f}, N={N}")
        data = np.loadtxt(output_file, comments='#')
        temperatures = data[:, 0]
        avgMagnetizations = data[:, 1]
        return temperatures, avgMagnetizations

    print(f"Simulating for h={h:.2f}, N={N}")
    argsList = [(N, sweep, trans, J, T, h) for T in T_vals]

    with Pool(cpu_count() - 1) as pool:
        results = pool.map(simulateSystemTemperature, argsList)

    temperatures = [result[0] for result in results]
    avgMagnetizations = [result[1] for result in results]

    with open(output_file, 'w') as file:
        file.write("# Temperature (T)   Average Magnetization\n")
        for T, m in zip(temperatures, avgMagnetizations):
            file.write(f"{T:.6f}   {m:.6f}\n")

    return temperatures, avgMagnetizations

def hSweepMagnetisations():
    N = 30**2
    argsList = [(N, sweep, trans, J, b, h) for h in hVals]

    results = []
    for h in hVals:
        with Pool(cpu_count() - 1) as pool:
            results += pool.map(simulateSystem, [arg for arg in argsList if arg[-1] == h])

    sortedFields = [result[1] for result in results]
    avgMagnetizations = [result[2] for result in results]
    std = [result[3] for result in results]
    return sortedFields, avgMagnetizations, b, std

def simulateSystem(args):
    N, sweep, trans, J, b, h = args
    base = f"N_{N}_h_{h:.2f}_b{b:.2f}_J_{J:.2f}_data"
    magnetizationFile = f"{base}_magnetizations.txt"
    energyFile = f"{base}_energies.txt"
    if os.path.exists(magnetizationFile) and os.path.exists(energyFile):
        print(f"Loading existing data for N={N}, h={h:.2f}")
        magnetizations = np.loadtxt(magnetizationFile)
        energies = np.loadtxt(energyFile)
        avgMagnetization = np.mean(magnetizations)
        std = np.std(magnetizations)
        return N, h, avgMagnetization, std

    s = np.random.choice([-1, 1], size=N)
    m = np.sum(s)
    magnetizations = []
    energies = []
    for mcs in range(sweep):
        s, m = MCsweep(s, J, b, h, N, m)
        if mcs >= trans:
            magnetizations.append(m / N)
            energies.append(calcEnergy(s, N, h))

    np.savetxt(magnetizationFile, magnetizations)
    np.savetxt(energyFile, energies)
    return N, h, np.mean(magnetizations), np.std(magnetizations)

def plot_magnetization_vs_temperature(T_vals, avgMagnetizations, h, std):

    plt.figure(figsize=(8, 6))
    plt.plot(T_vals, avgMagnetizations, marker='o', color='r', label="Average Magnetization (MC)")
    plt.errorbar(T_vals, avgMagnetizations, marker="o", yerr = std)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Magnetization")
    plt.title(f"Magnetization vs Temperature for h={h}, J={J}")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_magnetization_vs_field(hVals, avgMagnetizations, b, std):

    plt.figure(figsize=(8, 6))
    #plt.plot(hVals, avgMagnetizations, marker='o', yerr = std, color='b', label="Average Magnetization (MC)")

    print(T_vals)
    print(avgMagnetizations)
    print(std)

    #plt.plot(hVals, avgMagnetizations, marker='o', color='royalblue', label="Average Magnetization (MC)")
    plt.errorbar(hVals, avgMagnetizations, fmt=".", yerr = std, color="royalblue")
    plt.xlabel("External Field (h)")
    plt.ylabel("Average Magnetization")
    plt.title(f"Magnetization vs Field for $\\beta J$={b * J}, N = $30^2$")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    h = 7.5
    #print("Starting Field Sweep")
    # Field sweep
    # Average magnetization for different field strengths
    sortedFields, avgMagnetizations, b, std = hSweepMagnetisations()
    # Plot results - field curve
    plot_magnetization_vs_field(sortedFields, avgMagnetizations, b, std)
    #print("Starting Temp Sweep")
    #temperatures, magnetizations_temp = temperatureSweepMagnetisations(h)
    #plot_magnetization_vs_temperature(temperatures, magnetizations_temp, h)
