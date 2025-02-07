import numpy as np
import matplotlib.pyplot as plt
import os

# System sizes and temperatures
systemSizes = np.arange(100, 1000000, 1)
temps = [0.9]

i = 0  # Index for temperature
for N in systemSizes:
    timescales_file = f"Timescales for T = {temps[i]}"
    sizes_file = f"Sizes for T = {temps[i]}"
    errors_file = f"sem for T = {temps[i]}"

    # Check if all files exist
    if os.path.exists(timescales_file) and os.path.exists(sizes_file) and os.path.exists(errors_file):
        # Load data if files exist
        timescales = np.loadtxt(timescales_file)
        sizes = np.loadtxt(sizes_file)
        errors = np.loadtxt(errors_file)

        # Plot the error bars
        plt.errorbar(sizes, timescales, errors, label=f"T = {temps[i]}")
    else:
        print(f"Skipping files for T = {temps[i]} because one or more files are missing.")

plt.xlabel("System Sizes")
plt.ylabel("Timescales")
plt.title("Timescales vs System Sizes")
plt.legend()
plt.show()
