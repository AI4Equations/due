"""
Training-free conditional diffusion model for learning stochastic dynamical
systems (SDEs): estimates the score function of a probability-flow ODE
analytically from trajectory data via Monte Carlo neighbor weighting,
avoiding the need to train a score network, then generates supervised labels
for a flow-map network by solving the resulting reverse ODE.

Reference: Y. Liu, Y. Chen, D. Xiu, and G. Zhang, "A Training-Free
Conditional Diffusion Model for Learning Stochastic Dynamical Systems,"
SIAM J. Sci. Comput., 47(5):C1144-C1171, 2025.
https://doi.org/10.1137/24M1699589
"""

import os
import time
import torch
import numpy as np
from scipy.spatial import cKDTree

from ..utils import normalize_state

_TORCH_DTYPE = {"single": torch.float32, "double": torch.float64}
_NUMPY_DTYPE = {"single": "float32", "double": "float64"}


def select_neighbor_subset(x_targets, X_obs, dX_obs, nu, chunk_size=1000,
                            subsample_ratio=1.0, max_subset_size=None, verbose=False):
    """
    One-time nearest-neighbor subset selection: for each query x in x_targets,
    find the closest ~1% of the observed {x_m} via a KD-tree (scipy.spatial.cKDTree),
    and precompute the resulting spatial log-weight exp{-||x-x_m||^2/(2*nu^2)}.
    Independent of the reverse-ODE pseudotime tau, so it is computed once per x
    and reused for every one of the K reverse-ODE steps (see score_from_neighbors).

    x_targets: (N, ...) query points.
    X_obs, dX_obs: (M, ...) observed states and increments (the neighbor database).
    nu: spatial kernel bandwidth.
    chunk_size: query-axis batch size.
    subsample_ratio: fraction of X_obs/dX_obs to search against.
    max_subset_size: hard cap on the neighbor subset size.

    Returns (dX_sub, log_weight_spatial), both indexed by target and then by
    neighbor: (N, subset_size, dim) and (N, subset_size).
    """
    N = x_targets.shape[0]
    device = x_targets.device
    dtype = x_targets.dtype

    x_targets_flat = x_targets.reshape(N, -1).cpu().numpy()
    X_obs_flat = X_obs.reshape(X_obs.shape[0], -1).cpu().numpy()
    dX_obs_flat = dX_obs.reshape(dX_obs.shape[0], -1).cpu().numpy()
    dim = dX_obs_flat.shape[1]

    if subsample_ratio < 1.0:
        M_total = X_obs_flat.shape[0]
        sub_size = max(int(M_total * subsample_ratio), 100)
        sub_size = min(sub_size, M_total)

        perm = np.random.permutation(M_total)[:sub_size]
        X_obs_flat = X_obs_flat[perm]
        dX_obs_flat = dX_obs_flat[perm]

    M = X_obs_flat.shape[0]
    subset_size = max(int(0.01 * M), 100)
    subset_size = min(subset_size, M)
    if max_subset_size is not None:
        subset_size = min(subset_size, max_subset_size)

    tree = cKDTree(X_obs_flat)

    total_chunks = (N + chunk_size - 1) // chunk_size
    dX_sub = torch.empty((N, subset_size, dim), dtype=dtype, device=device)
    log_weight_spatial = torch.empty((N, subset_size), dtype=dtype, device=device)

    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        x_chunk = x_targets_flat[i:end_idx]

        chunk_idx = (i // chunk_size) + 1
        should_print = False

        if isinstance(verbose, bool):
            if verbose:
                print_interval = max(1, total_chunks // 10)
                if chunk_idx == 1 or chunk_idx == total_chunks or chunk_idx % print_interval == 0:
                    should_print = True
        elif isinstance(verbose, int) and verbose > 0:
            if chunk_idx == 1 or chunk_idx == total_chunks or chunk_idx % verbose == 0:
                should_print = True

        if should_print:
            print(f"    [Neighbor Search] Chunk {chunk_idx}/{total_chunks} ({chunk_idx/total_chunks*100:.1f}%) | "
                  f"Active slice: {i}:{end_idx}")

        dist, idx = tree.query(x_chunk, k=subset_size, eps=0, workers=-1)
        if subset_size == 1:
            # scipy returns 1D arrays when k=1; keep the (chunk, 1) convention used elsewhere.
            dist = dist[:, None]
            idx = idx[:, None]

        local_dX = dX_obs_flat[idx]  # (chunk, subset_size, dim)
        local_dist_sq = dist ** 2

        dX_sub[i:end_idx] = torch.from_numpy(local_dX).to(device=device, dtype=dtype)
        log_weight_spatial[i:end_idx] = torch.from_numpy(-local_dist_sq / (2 * nu**2)).to(device=device, dtype=dtype)

    return dX_sub, log_weight_spatial


def score_from_neighbors(z_current, dX_sub, dX_sq, log_weight_spatial, tau, chunk_size=100):
    """
    Training-free Monte Carlo score estimator, given an already-selected
    neighbor subset (dX_sub, log_weight_spatial) from select_neighbor_subset.
    Only the tau-dependent diffusion weight is computed here; the spatial
    weight is reused unchanged since it does not depend on tau.

    Avoids forming any (N, subset, dim) difference tensor: the softmax
    argument is expanded algebraically (dropping the per-row-constant ||z||^2
    term), and sum_j w_j = 1 lets the weighted score collapse into a weighted
    mean of the neighbor increments.

    Returns the score, same shape as z_current.
    """
    N = z_current.shape[0]
    z_current_flat = z_current.view(N, -1)

    alpha_tau = 1.0 - tau
    beta_tau_sq = max(tau, 1e-6)
    c_lin = alpha_tau / beta_tau_sq
    c_sq = alpha_tau * alpha_tau / (2.0 * beta_tau_sq)

    scores = []
    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        z_chunk = z_current_flat[i:end_idx]                       # (c, d)
        dX_sub_chunk = dX_sub[i:end_idx]                          # (c, s, d)
        dX_sq_chunk = dX_sq[i:end_idx]                            # (c, s)
        log_weight_spatial_chunk = log_weight_spatial[i:end_idx]  # (c, s)

        zdot = torch.einsum("nd,nsd->ns", z_chunk, dX_sub_chunk)  # (c, s)

        log_weights = log_weight_spatial_chunk + c_lin * zdot - c_sq * dX_sq_chunk
        w = torch.softmax(log_weights, dim=1)                     # (c, s)

        wdX = torch.einsum("ns,nsd->nd", w, dX_sub_chunk)

        score_chunk = -(z_chunk - alpha_tau * wdX) / beta_tau_sq
        scores.append(score_chunk)

    return torch.cat(scores, dim=0).view_as(z_current)


def build_tau_schedule(num_timesteps, schedule="uniform", tau_min=1e-5):
    """
    Build the descending sequence of pseudotime nodes (tau_1 > tau_2 > ... > 0)
    at which the reverse ODE is evaluated, together with the signed step size
    d_tau_k = tau_k - tau_{k+1} > 0 taken from each node.

    schedule="uniform" (default): the original evenly-spaced grid,
    tau_k = k/K for k = K, K-1, ..., 1.
    schedule="geometric": nodes log-spaced from tau=1 down to tau_min,
    clustering where the score sharpens near tau=0; reaches the same
    accuracy at a much smaller num_timesteps.

    Returns (taus, d_taus), both length-num_timesteps 1-D numpy arrays. The
    final step lands exactly on tau=0 in both schedules, so the RHS is never
    evaluated at tau=0 itself.
    """
    K = num_timesteps
    if schedule == "uniform":
        taus = np.arange(K, 0, -1, dtype=np.float64) / K
        d_taus = np.full(K, 1.0 / K, dtype=np.float64)
        return taus, d_taus

    if schedule == "geometric":
        nodes = np.geomspace(1.0, tau_min, K, dtype=np.float64)
        nodes = np.append(nodes, 0.0)
        taus = nodes[:-1]
        d_taus = nodes[:-1] - nodes[1:]
        return taus, d_taus

    raise ValueError(f"Unknown tau schedule '{schedule}'. Choose 'geometric' or 'uniform'.")


@torch.no_grad()
def extract_latents(x_targets, X_obs, Y_obs, config, verbose=True):
    """
    Reverse-ODE solver that samples z_1 ~ N(0, I) and integrates backward in
    pseudotime tau=1 -> 0 using the training-free score estimator, producing
    the paired latents (Z1, Z0): Z1 is the initial Gaussian noise and Z0 is
    the corresponding label (Z0 = X_dt - x) needed to supervise the flow-map
    network.

    x_targets: (N, ...) states to generate labels for.
    X_obs, Y_obs: (M, ...) the full observation database (current/next state).
    config: diffusion config block (diffusion_timesteps, tau_schedule, tau_min,
        nu, chunk_size, subsample_ratio, max_subset_size, increment_scale,
        ode_solver, score_chunk_size).

    Returns (z_initial, z_current), both same shape as x_targets.
    """
    num_timesteps = config.get("diffusion_timesteps", 100)
    tau_schedule = config.get("tau_schedule", "uniform")
    tau_min = config.get("tau_min", 1e-5)
    nu = config.get("nu", 1.0)
    chunk_size = config.get("chunk_size", 1000)
    subsample_ratio = config.get("subsample_ratio", 1.0)
    max_subset_size = config.get("max_subset_size", 1000)
    increment_scale = float(config.get("increment_scale", 1.0))
    ode_solver = str(config.get("ode_solver", "euler")).lower()
    if not np.isfinite(increment_scale) or increment_scale <= 0:
        raise ValueError("increment_scale must be a positive finite number.")
    if ode_solver not in {"euler", "heun"}:
        raise ValueError("ode_solver must be either 'euler' or 'heun'.")

    device = x_targets.device
    X_obs = X_obs.to(device)
    Y_obs = Y_obs.to(device)
    # increment_scale improves conditioning near tau=0; converted back to
    # physical units before returning below.
    dX_obs = (Y_obs - X_obs) * increment_scale

    N = x_targets.shape[0]

    z_initial = torch.randn_like(x_targets)
    z_current = z_initial.clone()

    if verbose:
        db_size = X_obs.shape[0]
        db_note = f" (subsample_ratio={subsample_ratio} -> ~{int(db_size * subsample_ratio)} actually searched)" if subsample_ratio < 1.0 else ""
        print(f"Selecting nearest-neighbor subsets for {N} targets against a "
              f"{db_size}-point database{db_note}...", flush=True)
        start_time = time.time()

    dX_sub, log_weight_spatial = select_neighbor_subset(
        x_targets=x_targets,
        X_obs=X_obs,
        dX_obs=dX_obs,
        nu=nu,
        chunk_size=chunk_size,
        subsample_ratio=subsample_ratio,
        max_subset_size=max_subset_size,
        verbose=verbose,
    )

    # tau-independent; precomputed once and chunked to bound peak memory.
    dX_sq = torch.empty(dX_sub.shape[:2], dtype=dX_sub.dtype, device=dX_sub.device)
    _sq_chunk = max(1, min(N, 2_000_000 // max(dX_sub.shape[1], 1)))
    for i in range(0, N, _sq_chunk):
        e = min(i + _sq_chunk, N)
        dX_sq[i:e] = (dX_sub[i:e] ** 2).sum(dim=2)

    # Batch size tuned for CPU cache throughput; override via score_chunk_size.
    subset_size = dX_sub.shape[1]
    score_chunk = config.get("score_chunk_size", 0) or max(1, min(N, 20_000_000 // max(subset_size, 1)))

    taus, d_taus = build_tau_schedule(num_timesteps, schedule=tau_schedule, tau_min=tau_min)
    n_nodes = len(taus)

    if verbose:
        print(f"Neighbor search complete in {time.time() - start_time:.2f} seconds.", flush=True)
        print(f"Starting reverse-ODE integration for {N} targets (score chunk={score_chunk}, "
              f"solver={ode_solver}, increment scale={increment_scale:g}, tau_schedule={tau_schedule}"
              f"{f', tau_min={tau_min:g}' if tau_schedule == 'geometric' else ''}) using the "
              f"training-free score estimator...", flush=True)
        start_time = time.time()

    for i, (tau, d_tau) in enumerate(zip(taus, d_taus)):
        is_last_node = (i == n_nodes - 1)

        score = score_from_neighbors(
            z_current=z_current,
            dX_sub=dX_sub,
            dX_sq=dX_sq,
            log_weight_spatial=log_weight_spatial,
            tau=tau,
            chunk_size=score_chunk,
        )

        if verbose and (i % max(1, n_nodes // 10) == 0 or is_last_node):
            print(f"  ODE Step {i}, tau = {tau:.4g} completed.")

        # Reverse probability-flow ODE drift.
        tau_safe = min(tau, 1.0 - 1e-5)
        b_tau = -1.0 / (1.0 - tau_safe)
        sigma_sq_tau = (1.0 + tau_safe) / (1.0 - tau_safe)
        drift = b_tau * z_current - 0.5 * sigma_sq_tau * score

        if ode_solver == "euler":
            z_current = z_current - drift * d_tau
        else:
            # Heun step; falls back to the Euler predictor on the last node
            # since the empirical score is singular at tau=0.
            z_predict = z_current - drift * d_tau
            if is_last_node:
                z_current = z_predict
                continue
            tau_next = max(tau - d_tau, 0.0)
            score_next = score_from_neighbors(
                z_current=z_predict,
                dX_sub=dX_sub,
                dX_sq=dX_sq,
                log_weight_spatial=log_weight_spatial,
                tau=tau_next,
                chunk_size=score_chunk,
            )
            b_next = -1.0 / (1.0 - tau_next)
            sigma_sq_next = (1.0 + tau_next) / (1.0 - tau_next)
            drift_next = b_next * z_predict - 0.5 * sigma_sq_next * score_next
            z_current = z_current - 0.5 * (drift + drift_next) * d_tau

    if verbose:
        print(f"Extraction complete in {time.time() - start_time:.2f} seconds.", flush=True)

    return z_initial, z_current / increment_scale


def generate_labeled_data(trainX, trainY, config, vmin, vmax):
    """
    Data-generation step of the training-free diffusion pipeline: turns the
    observation dataset D_obs = {(x_m, dx_m)} (as produced by
    due.datasets.sde.sde_dataset, in raw physical units) into the labeled
    dataset D_label = {(x_j, z_j, y_j)} needed for supervised training of the
    flow-map network, by sampling z_j ~ N(0, I) and solving the reverse ODE
    for each x_j = x_m.

    Runs entirely on the raw (unnormalized) trainX/trainY; normalization to
    [-1,1] is applied only here, right before the network sees the data,
    using the same vmin/vmax the data was measured against
    (see due.datasets.sde.sde_dataset.load()).

    trainX, trainY: raw (unnormalized) arrays from sde_dataset.load().
    config: merged training/diffusion config (see due.utils.read_sde_config);
        config["label_fraction"] (default 1.0) searches neighbors over the
        full D_obs but only solves the reverse ODE for a random fraction of
        those points, since labeling cost scales with the number of targets
        rather than the database size.
    vmin, vmax: normalization bounds from sde_dataset.load().

    Returns:
        trainX_augmented: (N, input_dim + output_dim), flattened [normalize(x), z]
        trainY_synthetic: (N, output_dim, 1), normalize(x + z0)
    """
    if config.get("multi_steps", 1) != 1:
        raise ValueError(
            "The training-free conditional diffusion model only supports "
            "single-step supervised training (multi_steps=1): each label is "
            "generated independently by the reverse ODE for a fixed Delta t. "
            "Multi-step rollout happens only at inference via "
            "SDEResNet.predict(), not during training."
        )

    # Optional CPU thread-count override; unset avoids oversubscribing shared machines.
    num_threads = config.get("num_threads", 0)
    if num_threads:
        torch.set_num_threads(int(num_threads))

    device = torch.device(config.get("device", "cpu"))
    dtype = _TORCH_DTYPE[config["dtype"]]

    X = torch.as_tensor(trainX, dtype=dtype, device=device)
    Y = torch.as_tensor(trainY, dtype=dtype, device=device)
    if Y.dim() == X.dim() + 1:
        Y = Y.squeeze(-1)

    # X/Y remain the full neighbor database; X_targets is the label_fraction subset.
    label_fraction = float(config.get("label_fraction", 1.0))
    N_obs = X.shape[0]
    if label_fraction < 1.0:
        n_labels = max(1, int(round(label_fraction * N_obs)))
        label_idx = torch.randperm(N_obs, device=X.device)[:n_labels]
        X_targets = X[label_idx]
        print(f"Labeling a random {n_labels}/{N_obs} (label_fraction={label_fraction}) subset of "
              f"the loaded data; the full {N_obs} points remain the neighbor-search database.")
    else:
        X_targets = X

    save_path = config.get("save_path", "./sde_diffusion_output")
    latent_file = os.path.join(save_path, "extracted_latents.pt")

    if config.get("cache_latents", False) and os.path.exists(latent_file):
        print(f"Loading pre-computed latent variables from {latent_file}...")
        Z1, Z0 = torch.load(latent_file, map_location=device)
    else:
        print("Extracting latent variables using the training-free reverse ODE...")
        Z1, Z0 = extract_latents(X_targets, X, Y, config, verbose=True)

        print(f"Saving extracted latents to {latent_file} to speed up future runs...")
        os.makedirs(save_path, exist_ok=True)
        torch.save((Z1, Z0), latent_file)

    np_dtype = _NUMPY_DTYPE[config["dtype"]]

    X_norm = normalize_state(X_targets, vmin, vmax)

    trainX_augmented = torch.cat([X_norm, Z1], dim=-1).cpu().numpy().astype(np_dtype)

    trainY_synthetic = normalize_state(X_targets + Z0, vmin, vmax).cpu().numpy().astype(np_dtype)
    trainY_synthetic = np.expand_dims(trainY_synthetic, axis=-1)

    print(f"Augmented input shape: {trainX_augmented.shape} ([normalize(x), z])")
    print(f"Synthetic target shape: {trainY_synthetic.shape} (normalize(x + z0))")

    return trainX_augmented, trainY_synthetic
