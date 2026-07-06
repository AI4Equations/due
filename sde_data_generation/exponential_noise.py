import numpy as np
from scipy.io import savemat
import os
import time

# =======================================================================
# CONFIGURATION PARAMETERS (Edit these to swap/change the dataset)
# =======================================================================

MU = -2     # Linear coefficient
SIGMA = 0.1      # Volatility (noise scale)

# Simulation Horizon & Volume
N_TRAJECTORIES = 100000  # Total number of independent training trajectories (H)
N_TEST_TRAJECTORIES = 50000 # Total number of independent test trajectories
T_TRAIN_HORIZON = 1          # Total time to simulate
T_TEST_HORIZON = 5
DT = 0.01                # Time step size

# Initial State Distribution (Uniform Sampling)
X0_MIN = 0.2         # Lower bound for initial state
X0_MAX = 0.9             # Upper bound for initial state

# Output Configuration
OUTPUT_TRAIN_FILENAME = "1d_exp_noise_process_train.mat"
OUTPUT_TEST_FILENAME = "1d_exp_noise_process_test.mat"
# =======================================================================

def generate_exponential_trajectories(num_trajectories, dataset_type="Train"):
    print(f"--- Generating 1D Exponential Noise Process ({dataset_type}) ---")
    print(f"Trajectories: {num_trajectories}")
    if dataset_type == "Train":
        print(f"Time Horizon: {T_TRAIN_HORIZON} (dt={DT})")
    if dataset_type == "Test":
        print(f"Time Horizon: {T_TEST_HORIZON} (dt={DT})")
    print(f"Parameters: mu={MU}, sigma={SIGMA}")
    
    start_time = time.time()
    
    # Calculate the total number of time steps (including t=0)
    if dataset_type == "Train":
        num_steps = int(T_TRAIN_HORIZON / DT) + 1
    if dataset_type == "Test":
        num_steps = int(T_TEST_HORIZON / DT) + 1
    
    # Initialize the data array. 
    # Shape: (N_trajectories, D_variables, T_steps) -> (N, 1, T)
    # This matches the shape expected by your ODE dataset loaders and evaluation scripts.
    trajectories = np.zeros((num_trajectories, 1, num_steps))
    
    # Sample initial states uniformly from [X0_MIN, X0_MAX]
    # Shape: (N_trajectories, 1)
    x0 = np.random.uniform(X0_MIN, X0_MAX, size=(num_trajectories, 1))
    trajectories[:, :, 0] = x0
    
    # Pre-calculate the square root of dt for the Wiener process (Brownian motion)
    sqrt_dt = np.sqrt(DT)
    
    # =======================================================================
    # VECTORIZED EULER-MARUYAMA SCHEME
    # =======================================================================
    # We loop through time, but compute ALL trajectories simultaneously 
    # at each time step using NumPy arrays. This is extremely fast.
    for t in range(1, num_steps):
        # Current state of all trajectories: shape (N, 1)
        x_prev = trajectories[:, :, t-1]
        
        # Generate Gaussian noise for all trajectories: shape (N, 1)
        noise = np.random.exponential(scale=1, size=(num_trajectories, 1))
        
        # Calculate the drift 
        drift = MU * x_prev * DT
        
        # Calculate the diffusion (stochastic noise)
        diffusion = SIGMA * sqrt_dt * noise
        
        # Update the state
        trajectories[:, :, t] = x_prev + drift + diffusion
        
    end_time = time.time()
    print(f"Simulation complete in {end_time - start_time:.3f} seconds.")
    
    return trajectories

def save_to_mat(trajectories, filename):
    # Package the data exactly how the pde_dataset / ode_dataset loader expects it
    mat_dictionary = {
        "trajectories": trajectories,
        "dt": np.array([[DT]]), # Saved as a 2D array for MATLAB compatibility
        "metadata": {
            "mu" : MU,
            "sigma": SIGMA,
            "description": "1D Exponential Noise Process"
        }
    }
    
    # Save the .mat file
    savemat(filename, mat_dictionary)
    print(f"Successfully saved to {os.path.abspath(filename)}")
    print(f"Saved Array Shape: {trajectories.shape} (N, D, T)")
    print("-" * 40)

if __name__ == "__main__":
    # Fix the random seed for perfect reproducibility
    np.random.seed(42)
    
    # Run the generator for Training Data
    train_data = generate_exponential_trajectories(N_TRAJECTORIES, "Train")
    save_to_mat(train_data, OUTPUT_TRAIN_FILENAME)
    
    # Change seed slightly so test set is independent, but still reproducible
    np.random.seed(999) 
    
    # Run the generator for Test Data
    test_data = generate_exponential_trajectories(N_TEST_TRAJECTORIES, "Test")
    save_to_mat(test_data, OUTPUT_TEST_FILENAME)