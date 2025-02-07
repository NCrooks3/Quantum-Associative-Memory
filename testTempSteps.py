import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os
from scipy.optimize import fsolve 

# Fix random seed for reproducibility
np.random.seed(1234)

# Parameters
J = 1.0     # Coupling constant
#b = 1.0     # Inverse temperature (beta)
b=1.0
N = 10   # System size

hVals = [0.8]

#temperature values to sweep
T_vals = np.linspace(0.00000001, 4.0, 50)
sweep = 100000  # Number of sweeps
trans = 0  # Ignore values below this sweep index

# Monte Carlo Functions 
def calcEnergy(s, N, h):
    m = np.sum(s)  # Net magnetization (M)
    return -J * m**2 / (2 * N) - h * m  # Energy calculation


def MCsweep(s, J, b, h, N, m):
    for k in range(N):
        i = np.random.randint(0, N) #Select random site
        dE = 2 * s[i] * (J * m / N + h)     #Energy difference if i th spin is flipped.
        #dE = ((-J * (2 * s[i] * (m) ) / (2 * N)) - (h * (2 * s[i])))
        dE = dE * s[i]
        if dE < 0 or np.random.random() < np.exp(-b * dE):  #Flip if reduces energy, or with probability e^{-beta dE}
            s[i] = -s[i]    #Flip i th spin
            m += 2 * s[i]   #Update net magnetisation (M)
    return s, m

#For when plotitng temp dependance at fixed field h
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

    plt.plot(np.linspace(0, sweep / N, len(magnetizations)), magnetizations)
    plt.show()
    return T, np.mean(magnetizations)


def hSweepMagnetisations():
    global N
    argsList = [(N, sweep, trans, J, b, h) for h in hVals]

    results = []

    for h in hVals:
        #Process pool for efficiency
        with Pool(cpu_count() - 1) as pool:
            results += pool.map(simulateSystem, [arg for arg in argsList if arg[-1] == h])

    sortedFields = [result[1] for result in results]
    avgMagnetizations = [result[2] for result in results]
    std = [result[3] for result in results]
    return sortedFields, avgMagnetizations, b, std


def simulateSystem(args):
    N, sweep, trans, J, b, h = args


    #Initial random spin configuration
    s = np.random.choice([-1, 1], size=N)
    m = np.sum(s)   #Net magnetisation (M)
    magnetizations = []
    energies = []
    for mcs in range(sweep):
        s, m = MCsweep(s, J, b, h, N, m)
        if mcs >= trans:
            magnetizations.append(m / N)
            energies.append(calcEnergy(s, N, h))

    #np.savetxt(magnetizationFile, magnetizations)
    #np.savetxt(energyFile, energies)

    plt.plot(np.linspace(0, sweep/N, len(magnetizations)), magnetizations)
    plt.show()
    
    return N, h, np.mean(magnetizations), np.std(magnetizations)

def simulateOneSystem(N, sweep, trans, J, b, h):

    
    #Initial random spin configuration
    s = np.random.choice([-1, 1], size=N)
    m = np.sum(s)   #Net magnetisation (M)
    magnetizations = []
    energies = []
    for mcs in range(sweep):
        s, m = MCsweep(s, J, b, h, N, m)
        if mcs >= trans:
            magnetizations.append(m / N)
            energies.append(calcEnergy(s, N, h))

    plt.plot(np.linspace(0, sweep/N, len(magnetizations)), magnetizations)

# Solve Self-Consistency Equation
def solve_self_consistency(b, J, hVals):
    #m = tanh(b * J * m + b * h)
    def equation(m, b, J, h):
        return m - np.tanh(b * J * m + b * h)

    solutions = []
    for h in hVals:
        m_initial_guess = 0.0
        m_solution, = fsolve(equation, m_initial_guess, args=(b, J, h))     #Finds m that makes equation function 0
        solutions.append(m_solution)
    return solutions

# Plot Comparison of Results
def plot_magnetization_comparison(hVals, avgMagnetizations_MC, b, std):
    self_consistent_m = solve_self_consistency(b, J, hVals)

    plt.figure(figsize=(8, 6))
    plt.errorbar(hVals, avgMagnetizations_MC, yerr=std, fmt="o", label="Monte-Carlo ", color="royalblue")
    plt.plot(hVals, self_consistent_m, '-', color="firebrick", label="Self-Consistency Equation", linewidth=2)
    plt.xlabel("External Field (h)")
    plt.ylabel("Average Magnetization (m)")
    plt.title(f"Magnetization vs Field for $\\beta J$ = {b * J}")
    plt.grid(True)
    plt.legend()
    plt.show()

#Runs the program 
#Must be in main for paralell processing to work.
if __name__ == "__main__":
    h = 0.6
    temps = np.linspace(0.1, 2.2, 3)
    betaVals = 1/temps

    for b in betaVals:
        simulateOneSystem(N, sweep, trans, J, b, h)

    plt.show()

    #print("Starting Field Sweep")
    # Field sweep
    #sortedFields, avgMagnetizations, b, std = hSweepMagnetisations()
    # Plot results - comparison with self-consistency
    #plot_magnetization_comparison(sortedFields, avgMagnetizations, b, std)

    #print("Starting Temp Sweep")
    #temperatures, magnetizations_temp = temperatureSweepMagnetisations(h)
    #plot_magnetization_vs_temperature(temperatures, magnetizations_temp, h)