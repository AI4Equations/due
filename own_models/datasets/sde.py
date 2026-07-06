import numpy as np
np.random.seed(0)
from scipy.io import loadmat

class sde_dataset():
    """
    A class representing an SDE dataset, matching the API of due.datasets.ode.ode_dataset
    (same load()/normalize() contract) but tailored to stochastic flow-map learning:
    trainY is the raw observed increment X_{t+dt} (Eq. 2.7 in the paper), not yet
    the diffusion-generated label -- that label is produced later by
    own_models.models.sde_diffusion.generate_labeled_data().

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

        # For SDEs (which are usually Markovian), memory is often 0 (we only look at X_t).
        self.memory_steps = config.get("memory", 0)
        self.multi_steps = config.get("multi_steps", 1)
        self.nbursts = config["nbursts"]
        self.dtype = config["dtype"]

        # If True, also slice pre-baked Gaussian "latents" stored inside the .mat file itself.
        self.raw_latents = config.get("raw_latents", False)

    def load(self, file_path_train, file_path_test=None):
        """
        Loads the dataset and slices it into the exact (X, Y) pairs
        needed for supervised training of the conditional diffusion model.

        Unlike due.datasets.ode.ode_dataset, trainX/trainY are returned in RAW
        physical units, not pre-normalized to [-1,1]. The training-free
        diffusion engine (own_models.models.sde_diffusion) is meant to operate
        on the raw state, matching how the paper's Section 3 derivation is
        written directly in terms of X_t: normalizing here first would distort
        the score estimator's spatial bandwidth (nu) for heavy-tailed
        processes (e.g. geometric Brownian motion), since a fixed nu means
        something very different once outliers have compressed the bulk of
        the data into a sliver of [-1,1]. vmin/vmax are still computed and
        returned (needed to construct the network and to normalize its
        inputs/targets right before training -- see
        own_models.models.sde_diffusion.generate_labeled_data), just not
        applied to trainX/trainY here.
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

        # ==========================================
        # THE SLICING LOGIC (Creating the Bursts)
        # ==========================================
        max_windows = T - self.multi_steps - self.memory_steps - 1
        if self.nbursts > max_windows:
            self.nbursts = max_windows

        # We need to slice chunks of time.
        # Total slice length = Memory (Past) + 1 (Current) + Multi_steps (Future)
        slice_len = self.memory_steps + self.multi_steps + 1
        target = np.zeros((N * self.nbursts, self.problem_dim, slice_len))

        # If the user also wants to load pre-baked Gaussian latents stored in the .mat file
        if self.raw_latents and "latents" in data_dict:
            latents = data_dict["latents"]
            target_latents = np.zeros((N * self.nbursts, self.problem_dim, self.multi_steps))

        # Randomly sample 'bursts' from the trajectories
        for i in range(N):
            inits = np.random.choice(max_windows, self.nbursts, replace=False)

            selected = np.asarray([data[i, :, init : init + slice_len] for init in inits])
            target[i * self.nbursts:(i + 1) * self.nbursts, ...] = selected

            if self.raw_latents and "latents" in data_dict:
                # Latents match the future steps
                selected_z = np.asarray([latents[i, :, init + self.memory_steps + 1 : init + slice_len + 1] for init in inits])
                target_latents[i * self.nbursts:(i + 1) * self.nbursts, ...] = selected_z

        print(f"Dataset regrouped into {target.shape[0]} training bursts.")

        # ==========================================
        # NORMALIZATION BOUNDS (computed here, applied later)
        # ==========================================
        # trainX/trainY themselves are NOT transformed -- see the load()
        # docstring for why. vmin/vmax use the same (0, 2)-axis reduction (over
        # trajectories and the time/slice axis, keeping the state-variable
        # axis) as due.datasets.ode.ode_dataset, so they broadcast the same way
        # against the network's (N, dim, memory+1)-shaped inputs.
        vmax = np.max(target, axis=(0, 2), keepdims=True)
        vmin = np.min(target, axis=(0, 2), keepdims=True)

        # ==========================================
        # INPUT / OUTPUT PAIRING (on the RAW target)
        # ==========================================
        # trainX is the condition (the current physical state X_t)
        # trainY is the observed future state X_{t+dt} (Eq. 2.7); the diffusion engine
        # turns (trainX, trainY) into the actual training labels via the reverse ODE.
        trainX = target[..., :self.memory_steps + 1].transpose(0, 2, 1).reshape(target.shape[0], -1)
        trainY = target[..., self.memory_steps + 1:]

        # Fix trainY shape to be cleanly (Batch, Dim) for single step
        if self.multi_steps == 1:
            trainY = trainY.squeeze(-1)

        print(f"Final Input (X) shape: {trainX.shape}")
        print(f"Final Target (Y) shape: {trainY.shape}")

        # ==========================================
        # RETURNS
        # ==========================================
        if self.raw_latents and "latents" in data_dict:
            # Flatten latents if multi_steps > 1, otherwise squeeze
            if self.multi_steps == 1:
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
            # Standard return identical to due.datasets.ode.ode_dataset so loaders don't break
            if file_path_test is None:
                return trainX.astype(self.dtype), trainY.astype(self.dtype), np.asarray(vmin).astype(self.dtype), np.asarray(vmax).astype(self.dtype)
            else:
                test_data_dict = loadmat(file_path_test)
                test_data = test_data_dict["trajectories"]
                return trainX.astype(self.dtype), trainY.astype(self.dtype), test_data.astype(self.dtype), np.asarray(vmin).astype(self.dtype), np.asarray(vmax).astype(self.dtype)
