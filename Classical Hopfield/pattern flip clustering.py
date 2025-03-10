import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.mixture import GaussianMixture
from scipy.ndimage import uniform_filter1d

# Directory containing magnetization data files
data_dir = r'C:\Users\scfro\OneDrive\Pictures\mc hopfield gifs\varying N\\'
file_pattern = data_dir + 'magnetisation_vs_time_steps_*.txt'

# Load all files
files = glob.glob(file_pattern)
files.sort()  # Sort for consistency

flip_times_all = []  # Store all flip intervals

for file in files:
    # Load data
    data = np.loadtxt(file, skiprows=1)
    time_steps = data[:, 0]
    magnetisations = data[:, 1]

    # Apply moving average filter to smooth noise
    window_size = 500  # Adjust for smoothing strength
    smooth_magnetisations = uniform_filter1d(magnetisations, size=window_size)

    # Use Gaussian Mixture Model to find 4 clusters (2 patterns on each side of m=0)
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm.fit(smooth_magnetisations.reshape(-1, 1))
    cluster_means = np.sort(gmm.means_.flatten())  # Sort cluster centers

    # Assign cluster labels to magnetization values
    labels = gmm.predict(smooth_magnetisations.reshape(-1, 1))

    # Detect pattern flips: When labels change
    flip_indices = np.where(np.diff(labels) != 0)[0] + 1  # Get time indices of flips
    flip_times = time_steps[flip_indices]
    
    # Store intervals between flips
    if len(flip_times) > 1:
        flip_intervals = np.diff(flip_times)
        flip_times_all.extend(flip_intervals)

    # Plot magnetization with detected clusters
    plt.figure(figsize=(10, 5))
    plt.scatter(time_steps, smooth_magnetisations, c=labels, cmap='viridis', s=2)
    plt.xlabel("Time Steps")
    plt.ylabel("Magnetisation (m)")
    plt.title(f"Magnetisation Dynamics: {file.split('_')[-1]}")
    plt.colorbar(label="Detected Cluster")
    plt.show()

# Plot histogram of time intervals between flips
plt.figure(figsize=(8, 6))
plt.hist(flip_times_all, bins=30, color='b', alpha=0.7, edgecolor='black')
plt.xlabel("Time Between Pattern Flips")
plt.ylabel("Frequency")
plt.title("Distribution of Time Intervals Between Pattern Flips")
plt.grid(True)
plt.show()
