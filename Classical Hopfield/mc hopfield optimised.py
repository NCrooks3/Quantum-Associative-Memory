#
#This generates a folder of images - need other program to turn into a movie.
#
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Parameters
n = 41  # Grid size
N = n**2  # Number of spins
steps = 50000000  # Monte Carlo steps
J = 1.0  # Coupling constant
beta = 1.0  # Inverse temperature  
save_every = 1000  # Save every nth step

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

pattern_1 = create_swiss_flag(n)
pattern_2 = create_vertical_lines(n)
patterns = [pattern_1, pattern_2]

# Compute Hopfield weight matrix
W = np.zeros((N, N))
for p in patterns:
    p_flat = p.flatten()
    W += np.outer(p_flat, p_flat)

W /= len(patterns)  # Normalize by the number of stored patterns
np.fill_diagonal(W, 0)  # No self-connections

# Initialize spins randomly
s = np.random.choice([-1, 1], size=(n, n))

def monte_carlo_step(s, beta):
    """Perform a single Monte Carlo step by flipping one spin based on Metropolis rule."""
    s_flat = s.flatten()
    
    # Select a random spin
    idx = np.random.randint(N)
    
    # Compute energy change using vectorized weight lookup
    delta_E = 2 * s_flat[idx] * np.dot(W[idx], s_flat)
    
    # Update
    if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
        s_flat[idx] *= -1  # Flip spin
    
    return s_flat.reshape(n, n)

# Create a temporary directory for frames
temp_dir = r'C:\\Users\\scfro\\OneDrive\\Documents\\tempFilesbeta=1Long'
os.makedirs(temp_dir, exist_ok=True)

# Run simulation
for step in range(steps * N):  # Total frames = steps * N (one spin update per frame)
    if step % 10 == 0:
        print("Step %s of %s" % (str(step), str(steps * N)))
    
    s = monte_carlo_step(s, beta)  # Flip exactly one spin
    
    if step % save_every == 0:

        # Save frame immediately to disk
        fig, ax = plt.subplots(figsize=(5, 5))  # Adjust the size if needed
        ax.imshow(s, cmap='gray')
        ax.axis('off')

        plt.savefig(f'{temp_dir}/temp_frame_{step}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)



