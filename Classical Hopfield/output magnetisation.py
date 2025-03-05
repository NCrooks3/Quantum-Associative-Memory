import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 41               # Grid size
N = n**2             # Number of spins
steps = 100000       # Number of Monte Carlo steps (each step consists of N spin updates)
J = 1.0              # Coupling constant
beta = 0.0025          # Inverse temperature  

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

m1 = np.nanmean(pattern_1)
print("magnetisation - swiss flag")
print(m1)
pattern_2 = create_vertical_lines(n)
patterns = [pattern_1, pattern_2]

print("magnetisation - vertical lines")
m2 = np.nanmean(pattern_2)
print(m2)
