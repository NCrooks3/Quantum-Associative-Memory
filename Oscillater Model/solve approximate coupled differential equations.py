import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def coupled_eqs(t, y, gamma1, gamma_m, m, n, eta, theta, Delta):
    R, phi = y
    
    dR_dt = -gamma1 * R + 2 * R - m / (2 * gamma_m * R**(2*m-1)) - n * eta * R**(n-1) * np.cos(n * (phi - theta))
    dphi_dt = -Delta + n * eta * R**(n-2) * np.sin(n * (phi - theta))
    
    return [dR_dt, dphi_dt]

# Parameters
gamma1 = 1.0  # Example value
gamma_m = 1.0  # Example value
m = 2  # Example value
n = 3  # Example value
eta = 0.5  # Example value
theta = np.pi / 4  # Example value
Delta = 0.1  # Example value

# Initial conditions
R0 = 1.0  # Initial R
phi0 = 1.0  # Initial phi
y0 = [R0, phi0]

# Time span
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# Solve the system
sol = solve_ivp(coupled_eqs, t_span, y0, args=(gamma1, gamma_m, m, n, eta, theta, Delta), t_eval=t_eval)

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[0], label='R(t)')
plt.xlabel('Time')
plt.ylabel('R')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sol.t, sol.y[1], label='phi(t)', color='r')
plt.xlabel('Time')
plt.ylabel('phi')
plt.legend()

plt.tight_layout()
plt.show()
