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
N = 100   # System size

hVals = [0.8]

#temperature values to sweep
T_vals = np.linspace(0.00000001, 4.0, 50)
sweep = 20000  # Number of sweeps
trans = 0  # Ignore values below this sweep index

# Monte Carlo Functions 
def calcEnergy(s, N, h):
    m = np.sum(s)  # Net magnetization (M)
    return -J * m**2 / (2 * N) - h * m  # Energy calculation



def MCstep(s, J, b, h, N, m):
    i = np.random.randint(0, N)  # Select random site
    sum_s = np.sum(s)  # Net magnetization
    dE = 2 * s[i] * (J * sum_s / N + h)  # Energy difference if i-th spin is flipped
        
    if dE < 0 or np.random.random() < np.exp(-b * dE):  # Flip if reduces energy, or with probability exp(-beta * dE)
        s[i] = -s[i]  # Flip i-th spin
        m += 2 * s[i]  # Update net magnetization (M)
    return s, m


def simulateSystemOld(args):
    N, sweep, trans, J, b, h = args
    threshold = 0.2
    
    #Initial random spin configuration
    s = np.random.choice([-1, 1], size=N)
    m = np.sum(s)   #Net magnetisation (M)
    magnetizations = []
    energies = []
    for mcs in range(sweep * N):
        s, m = MCstep(s, J, b, h, N, m)
        if mcs >= trans:
            magnetizations.append(m / N)
            energies.append(calcEnergy(s, N, h))

            if abs(m/N) < threshold:
                times += [s]

    timescales = np.zeros(len(times))
    timescales[0] = times[0]

    for i in range(1, len(times)):
        timescales[i] = times[i] - times[i-1]

    #np.savetxt(magnetizationFile, magnetizations)
    #np.savetxt(energyFile, energies)

    plt.plot(np.linspace(0, sweep, len(magnetizations)), magnetizations)
    plt.show()
    
    return N, h, np.mean(magnetizations), np.std(magnetizations)

def simulateOneSystem(N, sweep, trans, J, b, h, color):
    threshold = 0.0  # Look for sign changes instead of a threshold
    s = np.random.choice([-1, 1], size=N)  # Initial random spin configuration
    m = np.sum(s)   # Net magnetization (M)
    previousFlipped = 0

    magnetizations = []
    energies = []
    flip_times = []  # Track when flips occur

    previous_sign = np.sign(m)

    for sweep_count in range(sweep):  # Loop over the number of sweeps
        for _ in range(N):  # Perform N spin-flip attempts in one sweep
            s, m = MCstep(s, J, b, h, N, m)

        if sweep_count >= trans:  # Start recording data after the transient period
            magnetizations.append(m / N)
            energies.append(calcEnergy(s, N, h))

            current_sign = np.sign(m)
            if current_sign != previous_sign:  # Detect sign change
                if previousFlipped > 10:    #So must have remained flipped for a few steps - to try
                    flip_times.append(sweep_count)  # Record the sweep index of the flip
                    previous_sign = current_sign  # Update for next iteration
                else:
                    previousFlipped += 1
            else:
                previousFlipped = 0
    # Calculate timescales between flips
    timescales = np.diff(flip_times) if len(flip_times) > 1 else [0]

    plt.plot(np.linspace(0, sweep, len(magnetizations)), magnetizations, label=f"T = {round(1 / b, 2)}", color=color)
    return timescales

# Solve Self-Consistency Equation
def solve_self_consistency(b, J, hVals):

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

systemSizes = np.arange(100, 500, 50)

for N in systemSizes:
    print(N)
    h = 0
    temps = np.array([0.9])
    #temps = np.array([0.85, 0.9, 0.95])
    betaVals = 1/temps

    colours = ["blue", "green", "orange"]

    for i in range(0, len(betaVals)):
        b = betaVals[i]
        color = colours[i]
        timescales = simulateOneSystem(N, sweep, trans, J, b, h, color)

        timescale = np.mean(timescales)
        var = np.var(timescales)

        print("timescale")
        if os.path.exists("Timescales for T = %s" % str(temps[i])):
            f = open("Timescales for T = %s" % str(temps[i]), "a")
            np.savetxt(f, [timescale])
        else:
            np.savetxt("Timescales for T = %s" % str(temps[i]), [timescale])

        if os.path.exists("Sizes for T = %s" % str(temps[i])):
            f = open("Sizes for T = %s" % str(temps[i]), "a")
            np.savetxt(f, [N])
        else:
            np.savetxt("Sizes for T = %s" % str(temps[i]), [N])

        if os.path.exists("sem for T = %s" % str(temps[i])):
            f = open("sem for T = %s" % str(temps[i]), "a")
            np.savetxt(f, [var/len(timescales)])
        else:
            np.savetxt("sem for T = %s" % str(temps[i]), [var/len(timescales)])

        print(timescale)
   