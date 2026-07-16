import numpy as np
from scipy.io import savemat
import os
import time

# SDE parameters: dX_t = theta * (mu - X_t) dt + sigma * dW_t
THETA = 1.0
MU = 1.2
SIGMA = 0.3

N_TRAJECTORIES = 15000
N_TEST_TRAJECTORIES = 50000
T__TRAIN_HORIZON = 1.0
T_TEST_HORIZON = 5.0
DT = 0.01

X0_MIN = 0.0
X0_MAX = 2.5

OUTPUT_TRAIN_FILENAME = "1d_OU_process_train.mat"
OUTPUT_TEST_FILENAME = "1d_OU_process_test.mat"

def generate_ou_trajectories(num_trajectories, dataset_type="Train"):
    print(f"--- Generating 1D OU Process ({dataset_type}) ---")
    print(f"Trajectories: {num_trajectories}")
    if dataset_type == "Train":
        print(f"Time Horizon: {T__TRAIN_HORIZON} (dt={DT})")
    if dataset_type == "Test":
        print(f"Time Horizon: {T_TEST_HORIZON} (dt={DT})")

    print(f"Parameters: theta={THETA}, mu={MU}, sigma={SIGMA}")

    start_time = time.time()

    if dataset_type == "Train":
        num_steps = int(T__TRAIN_HORIZON / DT) + 1
    if dataset_type == "Test":
        num_steps = int(T_TEST_HORIZON / DT) + 1

    trajectories = np.zeros((num_trajectories, 1, num_steps))

    x0 = np.random.uniform(X0_MIN, X0_MAX, size=(num_trajectories, 1))
    trajectories[:, :, 0] = x0

    sqrt_dt = np.sqrt(DT)

    for t in range(1, num_steps):
        x_prev = trajectories[:, :, t-1]
        noise = np.random.normal(0, 1, size=(num_trajectories, 1))
        drift = THETA * (MU - x_prev) * DT
        diffusion = SIGMA * sqrt_dt * noise
        trajectories[:, :, t] = x_prev + drift + diffusion

    end_time = time.time()
    print(f"Simulation complete in {end_time - start_time:.3f} seconds.")

    return trajectories

def save_to_mat(trajectories, filename):
    mat_dictionary = {
        "trajectories": trajectories,
        "dt": np.array([[DT]]),
        "metadata": {
            "theta": THETA,
            "mu": MU,
            "sigma": SIGMA,
            "description": "1D Ornstein-Uhlenbeck Process"
        }
    }

    savemat(filename, mat_dictionary)
    print(f"Successfully saved to {os.path.abspath(filename)}")
    print(f"Saved Array Shape: {trajectories.shape} (N, D, T)")
    print("-" * 40)

if __name__ == "__main__":
    np.random.seed(42)

    train_data = generate_ou_trajectories(N_TRAJECTORIES, "Train")
    save_to_mat(train_data, OUTPUT_TRAIN_FILENAME)

    # Different seed so the test set is independent, but still reproducible.
    np.random.seed(999)

    test_data = generate_ou_trajectories(N_TEST_TRAJECTORIES, "Test")
    save_to_mat(test_data, OUTPUT_TEST_FILENAME)
