import numpy as np
import matplotlib.pyplot as plt
import random

def normalize_vectors(vectors):
    """Normalizes a field of vectors."""
    # Calculate the norm (magnitude) of each vector, adding a small epsilon to avoid division by zero
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    epsilon = 1e-8
    
    # Normalize the vectors by dividing by their norm
    normalized_vectors = np.divide(vectors, norms + epsilon, where=(norms != 0))
    return normalized_vectors

def generate_bending_fibers(shape=(50, 100, 100)): # default shape 50,100,100
    """
    Generates a block of fibers bending from the Y-axis to the X-axis.
    The bend occurs along the X-axis.
    """
    grid = np.zeros(shape + (3,))
    nx, ny, nz = shape # touple unpacking. assign the vaules in shape accordingly

    # Create coordinate arrays
    x = np.arange(nx)
    
    # The angle of bend is a function of the x-coordinate
    # It goes from 0 (pointing along Y) to pi/2 (pointing along X)
    angle = (np.pi / 2.0) * (x / (nx - 1))
    
    # Create vectors based on the angle
    vx = np.sin(angle)
    vy = np.cos(angle)
    
    # Assign the vectors to the grid, broadcasting across Y and Z
    grid[:, :, :, 0] = vx[:, np.newaxis, np.newaxis]
    grid[:, :, :, 1] = vy[:, np.newaxis, np.newaxis]
    grid[:, :, :, 2] = 0 

    return grid

def generate_fanning_fibers(shape=(50, 100, 100)):
    """
    Generates a block of fibers fanning out from a central line.
    The main direction is along the X-axis, and fanning occurs in the YZ-plane.
    """
    grid = np.zeros(shape + (3,))
    nx, ny, nz = shape

    # Define the center of the fan
    center_y, center_z = ny // 2, nz // 2

    # Create coordinate arrays
    y = np.arange(ny)
    z = np.arange(nz)

    # Calculate distance from the fan's center line
    dy = y - center_y
    dz = z - center_z

    # Calculate the angle of fanning in the YZ plane
    angle = np.arctan2(dz[np.newaxis, :], dy[:, np.newaxis])

    # Create vectors. The primary component is along X.
    # The spread of the fan is controlled by the magnitude of the Y and Z components.
    grid[:, :, :, 0] = 1.0 # Main direction along X
    grid[:, :, :, 1] = 0.4 * np.cos(angle)[np.newaxis, :, :] # Fanning in Y
    grid[:, :, :, 2] = 0.4 * np.sin(angle)[np.newaxis, :, :] # Fanning in Z

    return normalize_vectors(grid)

def generate_crossing_fibers(shape=(40, 40, 40)):
    """
    Generates a block with two crossing fiber populations using a checkerboard pattern.
    """
    grid = np.zeros(shape + (3,))
    nx, ny, nz = shape
    
    # Define the two fiber orientation vectors
    vec1 = np.array([1, 1, 0]) / np.sqrt(2)
    vec2 = np.array([1, -1, 0]) / np.sqrt(2)

    # Create a 3D checkerboard pattern to assign vectors
    i, j, k = np.indices(shape)
    checkerboard = (i + j + k) % 2 == 0
    
    # Assign vectors based on the checkerboard pattern
    grid[checkerboard] = vec1
    grid[~checkerboard] = vec2
    
    return grid

def generate_straight_fibers(shape, direction):
    """Generates a block of straight, parallel fibers."""
    grid = np.zeros(shape + (3,))
    
    # Ensure the direction vector is a unit vector
    unit_direction = np.array(direction) / np.linalg.norm(direction)
    
    grid[:] = unit_direction
    return grid

def generate_twisting_fibers(shape=(100, 100, 100)):
    """
    Generates a smoothly varying field of fibers that twist around the Z-axis.
    """
    grid = np.zeros(shape + (3,))
    nx, ny, nz = shape
    
    # Create 3D coordinate grids
    x, y, z = np.indices(shape)
    
    # Center the coordinates
    x_c = x - nx / 2
    y_c = y - ny / 2
    
    # Define the angle of twist based on the Z coordinate (more twist as you go up)
    # The 'twist_rate' controls how fast it twists
    twist_rate = np.pi / nz
    angle = z * twist_rate
    
    # Define the main vector field (e.g., pointing along Y)
    vx = -np.sin(angle)
    vy = np.cos(angle)
    vz = 0.2 # Give a slight upward component
    
    # Assign to the grid (requires stacking and transposing)
    vectors = np.stack([vx, vy, vz], axis=-1)
    
    return normalize_vectors(vectors)

def add_vector_noise(grid, noise_level=0.3):
    """
    Adds random noise to a vector field and re-normalizes.
    
    Args:
        grid (np.array): The input vector field.
        noise_level (float): The strength of the noise (0=none, 1=high).
    """
    # Create random noise vectors with the same shape as the grid
    noise = np.random.normal(0, 1, grid.shape)
    
    # Add the scaled noise to the original vectors
    noisy_grid = grid + noise * noise_level
    
    # Re-normalize the vectors to ensure they remain unit vectors
    return normalize_vectors(noisy_grid)

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

print("--- Assembling a varied volume with random and noisy tiling ---")

master_shape = (100, 100, 100)
master_grid = np.zeros(master_shape + (3,))

# --- Create an expanded list of fiber pattern generators ---
block_size = 20 # Use smaller blocks
sub_shape = (block_size, block_size, block_size)

pattern_generators = [
    # --- Original "Clean" Patterns ---
    lambda: generate_straight_fibers(sub_shape, direction=[1, 0, 0]),
    lambda: generate_straight_fibers(sub_shape, direction=[0, 1, 0]),
    lambda: generate_bending_fibers(sub_shape),
    lambda: generate_fanning_fibers(sub_shape),
    lambda: generate_crossing_fibers(sub_shape),
    
    # --- NEW: "Noisy" Versions of the Patterns ---
    lambda: add_vector_noise(generate_straight_fibers(sub_shape, direction=[1, 0, 0]), noise_level=0.4),
    lambda: add_vector_noise(generate_straight_fibers(sub_shape, direction=[0, 1, 0]), noise_level=0.4),
    lambda: add_vector_noise(generate_bending_fibers(sub_shape), noise_level=0.3),
    lambda: add_vector_noise(generate_fanning_fibers(sub_shape), noise_level=0.5)
]
print(f"Created a pool of {len(pattern_generators)} different pattern types (clean and noisy).")


# --- Loop through the master grid and place random blocks ---
print(f"Tiling the {master_shape} grid with {sub_shape} blocks...")
for x in range(0, master_shape[0], block_size):
    for y in range(0, master_shape[1], block_size):
        for z in range(0, master_shape[2], block_size):
            # Choose a random pattern generator from the expanded list
            random_generator = random.choice(pattern_generators)
            
            # Generate the block
            new_block = random_generator()
            
            # Place it in the master grid
            master_grid[x:x+block_size, y:y+block_size, z:z+block_size] = new_block

print("✅ Composite volume with random and noisy tiling assembled!")

# You can now run your FOD analysis script on this new 'master_grid' to see the result.


# --- Step 1: Reshape the data and filter out zero vectors ---
all_vectors = master_grid.reshape(-1, 3)
all_vectors = all_vectors[np.any(all_vectors != 0, axis=1)]
print(f"Found {all_vectors.shape[0]} non-zero vectors.")

# --- Step 2: Convert Cartesian vectors (x, y, z) to Spherical angles ---
x, y, z = all_vectors[:, 0], all_vectors[:, 1], all_vectors[:, 2]

# Azimuth (phi) is the angle in the xy-plane (from -pi to pi)
azimuth = np.arctan2(y, x)

# Elevation (theta) is the angle from the z-axis (from 0 to pi)
# Note: We need to clip the z value to avoid domain errors in arccos
elevation = np.arccos(np.clip(z, -1.0, 1.0))

print("Converted vector directions to spherical coordinates (azimuth, elevation).")

# --- Step 3: Create and visualize the 2D histogram ---
print("Generating 2D histogram of orientations...")

plt.figure(figsize=(10, 7))

# hist2d creates a 2D histogram (a heatmap) of the angles
# The bins define the resolution of our orientation histogram
plt.hist2d(np.degrees(azimuth), np.degrees(elevation), bins=90, cmap='inferno')

plt.colorbar(label='Voxel Count')
plt.title('Simulated Fiber Orientation Distribution (FOD)', fontsize=16)
plt.xlabel('Azimuthal Angle (φ) [degrees]', fontsize=12)
plt.ylabel('Polar Angle (θ) [degrees]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)


print("✅ FOD visualization complete.")
plt.show()