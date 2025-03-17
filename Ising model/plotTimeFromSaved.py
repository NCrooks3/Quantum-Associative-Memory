import numpy as np
import matplotlib.pyplot as plt
import os

# System sizes and temperatures
systemSizes = np.arange(100, 500, 1)
temps = [0.9]

i = 0  # Index for temperature
for N in [0]:
    timescales_file = f"Timescales for T = {temps[i]}"
    sizes_file = f"Sizes for T = {temps[i]}"
    errors_file = f"sem for T = {temps[i]}"

    # Check if all files exist
    if os.path.exists(timescales_file) and os.path.exists(sizes_file) and os.path.exists(errors_file):
        # Load data if files exist
        timescales = np.loadtxt(timescales_file)
        sizes = np.loadtxt(sizes_file)
        errors = np.loadtxt(errors_file)
        print(sizes)

        # Plot the error bars
        plt.plot(sizes, np.log(timescales), "*", label=f"T = {temps[i]}")
    else:
        print(f"Skipping files for T = {temps[i]} because one or more files are missing.")

yerr = np.log(yerr) / np.sqrt(N)

# Ensure all y values are positive before applying log
if np.any(y <= 0):
    raise ValueError("All values in y must be positive for logarithm transformation.")

# Log-transform y and error bars
y = y / np.sqrt(10)
log_y = np.log(y)
log_yerr = yerr / y  # Propagation of error for log transform

# Fit a weighted straight line (linear regression with errors)
coeffs, cov = np.polyfit(N, log_y, 1, w=1/log_yerr, cov=True)
slope, intercept = coeffs
slope_err = np.sqrt(cov[0, 0])  # Standard deviation of the slope

# Generate fitted values
fitted_y = np.polyval(coeffs, N)

# Plot with error bars
plt.errorbar(N, log_y, yerr=yerr, fmt='o', color='red', label="Data Points", capsize=5)
plt.plot(N, fitted_y, label=f"Fitted Line (slope={slope:.4f} ± {slope_err:.4f})", linestyle="--")

plt.xlabel("$N$")
plt.ylabel("$\log(\\tau)$")
plt.legend()
plt.show()

# Output the gradient with error
print(f"Gradient (slope) of the fitted line: {slope:.4f} ± {slope_err:.4f}")