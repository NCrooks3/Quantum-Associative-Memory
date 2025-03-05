import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 41               # Grid size
N = n**2             # Number of spins
steps = 1000       # Number of Monte Carlo steps (each step consists of N spin updates)
J = 1.0              # Coupling constant
beta = 0.002          # Inverse temperature  

# Create Hopfield patterns
def create_swiss_flag(n):
    flag = -np.ones((n, n), dtype=int)
    cross_width = n // 5
    vertical_length = n // 2
    horizontal_length = n * 3 // 5

    vertical_start = (n - vertical_length) // 2
    vertical_end = vertical_start + vertical_length
    horizontal_start = (n - horizontal_length) // 2
    horizontal_end = horizontal_start + horizontal_length

    flag[vertical_start:vertical_end, (n - cross_width) // 2 : (n + cross_width) // 2] = 1
    flag[(n - cross_width) // 2 : (n + cross_width) // 2, horizontal_start:horizontal_end] = 1

    return flag

def create_vertical_lines(n, line_width=3, spacing=3):
    pattern = -np.ones((n, n), dtype=int)
    for x in range(0, n, line_width + spacing):
        pattern[:, x:x + line_width] = 1
    return pattern

# Define two patterns
pattern_1 = create_swiss_flag(n)
pattern_2 = create_vertical_lines(n)
patterns = [pattern_1, pattern_2]

# Compute Hopfield weight matrix using Hebbian learning rule
W = np.zeros((N, N))
for p in patterns:
    p_flat = p.flatten()
    W += np.outer(p_flat, p_flat)
W /= len(patterns)  # Normalize by the number of stored patterns
np.fill_diagonal(W, 0)  # No self-connections

# Initialize spins randomly
s = np.random.choice([-1, 1], size=(n, n))

# Calculate initial magnetisation (average spin)
m = np.mean(s)

def monte_carlo_step(s, beta):

    #Attempt to flip a spin, using metrapolis rule

    s_flat = s.flatten()
    idx = np.random.randint(N)
    delta_E = s_flat[idx] * np.dot(W[idx], s_flat)
    flip_occurred = False
    delta_m = 0.0
    if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
        old_val = s_flat[idx]
        s_flat[idx] *= -1  # Flip the spin
        flip_occurred = True
        delta_m = -2 * old_val / N  # Change in mean magnetisation
    return s_flat.reshape(n, n), flip_occurred, delta_m

# Lists to store the magnetisation values and corresponding time steps
magnetisations = []
time_steps = []

# Monte Carlo simulation
for step in range(steps):
    for _ in range(N):  # N spin updates per Monte Carlo step
        s, flip_occurred, dm = monte_carlo_step(s, beta)
        if flip_occurred:
            m += dm  # Update magnetisation incrementally
    
    if step % 500 == 0:  # Print progress every 500 steps
        print(f"Step {step} of {steps}")
        print(f"Step {step}: Δm = {dm:.4f}, Magnetization = {m:.4f}")

    # Store magnetisation once per Monte Carlo step
    magnetisations.append(m)
    time_steps.append(step)

# Convert lists to numpy arrays
magnetisations = np.array(magnetisations, dtype=np.float32)
time_steps = np.array(time_steps, dtype=np.float32)

# Downsample for plotting if too large
#downsample_factor = max(1, len(time_steps) // 10000)  # Keep max 10000 points
plt.figure(figsize=(10, 6))
#plt.plot(time_steps[::downsample_factor], magnetisations[::downsample_factor], marker='o', markersize=2)
plt.plot(time_steps, magnetisations)
plt.title(f'$\\beta J = {beta * J}$')
plt.xlabel('Time ($\\frac{\\#}{N}$)')
plt.ylabel('Magnetisation (m)')
plt.grid(True)
plt.show()

# Save the results to a text file
filename = f'magnetisation_vs_time_steps_{steps}_beta_{beta}_J_{J}.txt'
output_data = np.column_stack((time_steps, magnetisations))
np.savetxt(filename, output_data, header='TimeStep Magnetisation')
print(f"Results saved to {filename}")
