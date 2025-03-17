import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def coupled_eqs(t, y, gamma1, gamma_m, m, n, eta, theta, Delta):
    R, phi = y
    
    dR_dt = (-0.5 * gamma1 * R) - (0.5 * m * gamma_m * (R ** ((2 * m) - 1))) - (n * eta * (R**(n-1)) * np.cos(n * (phi + theta)))
    dphi_dt = -Delta + (n * eta * (R**(n-2)) * np.sin(n * (phi + theta)))
    
    return [dR_dt, dphi_dt]

# Parameters
gamma1 = 1.0
gamma_m = 1.0
m = 2
n = 3
eta = 0.5
theta = 0
Delta = 0.1

# Define initial conditions grid
x_vals = np.linspace(-2, 2, 50)
y_vals = np.linspace(-2, 2, 50)

X_grid, Y_grid = np.meshgrid(x_vals, y_vals)  # Create meshgrid

U = np.zeros_like(X_grid)
V = np.zeros_like(Y_grid)

t_span = (0, 1000)
t_eval = np.linspace(*t_span, 1000)

# Solve system for each (x0, y0)
for i in range(X_grid.shape[0]):
    for j in range(X_grid.shape[1]):
        x_0 = X_grid[i, j]
        y_0 = Y_grid[i, j]

        R0 = np.sqrt(x_0**2 + y_0**2)
        phi0 = np.arctan2(y_0, x_0)

        y0 = [R0, phi0]
        sol = solve_ivp(coupled_eqs, t_span, y0, args=(gamma1, gamma_m, m, n, eta, theta, Delta), t_eval=t_eval)
        
        Rf, phif = sol.y[:, -1]  # Final values
        
        alpha0 = R0 * np.exp(1j * phi0)  # Convert to complex plane
        alphaf = Rf * np.exp(1j * phif)
        
        # Compute vector field components
        U[i, j] = alphaf.real - alpha0.real
        V[i, j] = alphaf.imag - alpha0.imag

# Plot streamplot
plt.figure(figsize=(8, 6))
plt.streamplot(X_grid, Y_grid, U, V, color=np.hypot(U, V), cmap='inferno', density=3)
plt.xlabel('Re(alpha)')
plt.ylabel('Im(alpha)')
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.grid()
plt.colorbar(label="gradient Magnitude")
plt.show()