import numpy as np
np.random.seed(0)
from scipy.io import loadmat

class sde_dataset():
    """
    A class representing an SDE dataset, matching the API of due.datasets.ode.ode_dataset
    (same load() contract) but tailored to stochastic flow-map learning:
    trainY is the raw observed X_{t+dt}, not yet the diffusion-generated label -- 
    that label is produced later by due.models.sde_diffusion.generate_labeled_data().

    Args:
        config (dict): the "data" config block (see due.utils.read_config).
            - problem_dim (int): number of state variables.
            - memory (int): number of past steps in the condition (SDEs are Markovian, so 0 is typical).
            - multi_steps (int): must be 1 -- the flow map only defines a single-step transition.
            - nbursts (int): number of (x, dx) pairs sampled per trajectory.
            - dtype (str): "single" or "double".
            - raw_latents (bool): if True and the .mat file has a "latents" key, also
              return those pre-baked Gaussian draws (rarely used; unrelated to the
              cached reverse-ODE latents used by the diffusion engine).
    """

    def __init__(self, config):
        self.problem_dim = config["problem_dim"]
        self.memory_steps = config.get("memory", 0)
        self.multi_steps = config.get("multi_steps", 0)
        self.prediction_steps = self.multi_steps + 1
        self.nbursts = config["nbursts"]
        self.dtype = config["dtype"]
        self.raw_latents = config.get("raw_latents", False)

    def load(self, file_path_train, file_path_test=None):
        """
        Loads the dataset and slices it into the exact (X, Y) pairs
        needed for supervised training of the conditional diffusion model.

        Unlike due.datasets.ode.ode_dataset, trainX/trainY are returned in RAW
        physical units, not pre-normalized to [-1,1]. The training-free
        diffusion engine (due.models.sde_diffusion) operates on the raw
        state, keeping the derivation written directly in terms of X_t.
        vmin/vmax are still computed and returned (needed to construct the
        network and to normalize its inputs/targets right before training --
        see due.models.sde_diffusion.generate_labeled_data), just not
        applied to trainX/trainY here.

        Returns (trainX, trainY, vmin, vmax), plus test_data if file_path_test
        is given, plus z_target if raw_latents is set and present.
        """
        try:
            data_dict = loadmat(file_path_train)
        except NotImplementedError:
            print("mat file too large, falling back to h5py/mat73...")
            import mat73
            data_dict = mat73.loadmat(file_path_train)

        try:
            data = data_dict["trajectories"]
        except KeyError:
            raise ValueError("Please name your SDE dataset as 'trajectories'.")

        N = data.shape[0]
        T = data.shape[2]

        if data.shape[1] != self.problem_dim:
            raise ValueError(f'Expected shape (N, {self.problem_dim}, T). Got {data.shape}')

        print(f"SDE Dataset loaded: {N} trajectories, {self.problem_dim} variables, {T} time steps")

        slice_len = self.memory_steps + 1 + self.prediction_steps
        max_windows = T - slice_len + 1
        if max_windows <= 0:
            raise ValueError(
                f"Trajectory length T={T} is too short for "
                f"memory={self.memory_steps}, multi_steps={self.multi_steps}."
            )
        if self.nbursts > max_windows:
            self.nbursts = max_windows

        target = np.zeros((N * self.nbursts, self.problem_dim, slice_len))

        if self.raw_latents and "latents" in data_dict:
            latents = data_dict["latents"]
            target_latents = np.zeros((N * self.nbursts, self.problem_dim, self.prediction_steps))

        for i in range(N):
            inits = np.random.choice(max_windows, self.nbursts, replace=False)

            selected = np.asarray([data[i, :, init : init + slice_len] for init in inits])
            target[i * self.nbursts:(i + 1) * self.nbursts, ...] = selected

            if self.raw_latents and "latents" in data_dict:
                selected_z = np.asarray([latents[i, :, init + self.memory_steps + 1 : init + slice_len] for init in inits])
                target_latents[i * self.nbursts:(i + 1) * self.nbursts, ...] = selected_z

        print(f"Dataset regrouped into {target.shape[0]} training bursts.")

        # trainX/trainY themselves are NOT transformed here (see docstring above).
        vmax = np.max(target, axis=(0, 2), keepdims=True)
        vmin = np.min(target, axis=(0, 2), keepdims=True)

        trainX = target[..., :self.memory_steps + 1].transpose(0, 2, 1).reshape(target.shape[0], -1)
        trainY = target[..., self.memory_steps + 1:]

        if self.prediction_steps == 1:
            trainY = trainY.squeeze(-1)

        print(f"Final Input (X) shape: {trainX.shape}")
        print(f"Final Target (Y) shape: {trainY.shape}")

        if self.raw_latents and "latents" in data_dict:
            if self.prediction_steps == 1:
                z_target = target_latents.squeeze(-1)
            else:
                z_target = target_latents.reshape(target_latents.shape[0], -1)

            print(f"Final Latents (Z) shape: {z_target.shape}")

            if file_path_test is None:
                return trainX.astype(self.dtype), trainY.astype(self.dtype), z_target.astype(self.dtype), np.asarray(vmin).astype(self.dtype), np.asarray(vmax).astype(self.dtype)
            else:
                test_data_dict = loadmat(file_path_test)
                test_data = test_data_dict["trajectories"]
                return trainX.astype(self.dtype), trainY.astype(self.dtype), z_target.astype(self.dtype), test_data.astype(self.dtype), np.asarray(vmin).astype(self.dtype), np.asarray(vmax).astype(self.dtype)

        else:
            if file_path_test is None:
                return trainX.astype(self.dtype), trainY.astype(self.dtype), np.asarray(vmin).astype(self.dtype), np.asarray(vmax).astype(self.dtype)
            else:
                test_data_dict = loadmat(file_path_test)
                test_data = test_data_dict["trajectories"]
                return trainX.astype(self.dtype), trainY.astype(self.dtype), test_data.astype(self.dtype), np.asarray(vmin).astype(self.dtype), np.asarray(vmax).astype(self.dtype)
