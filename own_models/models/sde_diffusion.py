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
    One-time nearest-neighbor subset selection (Section 3.2, footnote 1 in the
    paper): for each query x in x_targets, find the closest ~1% of the
    observed {x_m} by Euclidean distance, and precompute the resulting
    spatial log-weight exp{-||x-x_m||^2/(2*nu^2)}.

    This does not depend on the reverse-ODE pseudotime tau, so the paper
    computes it once per x and reuses it for every one of the K reverse-ODE
    steps rather than re-searching neighbors at each step -- that reuse is
    exactly what makes this worth splitting out of the per-step score
    computation (see score_from_neighbors).

    Implemented with a KD-tree (scipy.spatial.cKDTree) rather than brute-force
    pairwise distances. Querying k exact nearest neighbors from a tree built
    over M points costs roughly O(log M) per query instead of brute force's
    O(M) -- this matters enormously once M reaches into the hundreds of
    thousands: measured 22-56x faster than the previous chunked cdist+topk
    implementation at N=M=50,000-100,000 (the speedup *grows* with N/M, since
    it's a log-M vs. M-scaling difference, not a constant factor), and brute
    force becomes multi-hour-impractical at the ~500,000+ scale some of the
    datasets in sde_examples/ (e.g. DoubleWell, ExpNoise) actually need,
    where the KD-tree still finishes in ~18 seconds.

    The search is EXACT (eps=0, scipy's default) -- it returns the same
    nearest neighbors as brute force, verified to floating-point precision,
    not an approximation. cKDTree is numpy/CPU-only, so tensors are moved off
    their original device for the search and moved back for the return; there
    is no GPU-accelerated equivalent, but nothing in this codebase currently
    runs the diffusion engine on GPU (all configs use device: "cpu").

    The database axis no longer needs manual chunking (the tree handles it
    internally); the query axis (x_targets) is still processed in
    `chunk_size`-sized batches, both to bound the size of the per-call
    (chunk_size, subset_size) intermediate distance/index arrays and to
    report progress on very large N.

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

    # OPTIMIZATION: Database Subsampling
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

    # Build once over the (possibly subsampled) database and reuse for every
    # query chunk below -- this is the one-time cost that replaces a fresh
    # brute-force distance computation per chunk.
    tree = cKDTree(X_obs_flat)

    total_chunks = (N + chunk_size - 1) // chunk_size
    # Preallocate the full outputs and write each query chunk into its slice
    # (avoids holding both per-chunk pieces and a final concatenated copy at
    # once, which would double the peak for these (N, subset_size, ...)
    # tensors -- multiple GB at large N).
    dX_sub = torch.empty((N, subset_size, dim), dtype=dtype, device=device)
    log_weight_spatial = torch.empty((N, subset_size), dtype=dtype, device=device)

    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        x_chunk = x_targets_flat[i:end_idx]

        # Progress Logging
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

        # Exact k-NN query (eps=0), parallelized across all available cores.
        dist, idx = tree.query(x_chunk, k=subset_size, eps=0, workers=-1)
        if subset_size == 1:
            # scipy returns 1D arrays when k=1; keep the (chunk, 1) convention
            # used everywhere else.
            dist = dist[:, None]
            idx = idx[:, None]

        local_dX = dX_obs_flat[idx]        # (chunk, subset_size, dim)
        local_dist_sq = dist ** 2          # cKDTree returns raw distance; our
                                            # weight uses squared distance, as
                                            # the paper's Eq. 3.13/3.19 do.

        dX_sub[i:end_idx] = torch.from_numpy(local_dX).to(device=device, dtype=dtype)
        log_weight_spatial[i:end_idx] = torch.from_numpy(-local_dist_sq / (2 * nu**2)).to(device=device, dtype=dtype)

    return dX_sub, log_weight_spatial


def score_from_neighbors(z_current, dX_sub, dX_sq, log_weight_spatial, tau, chunk_size=100):
    """
    Training-free Monte Carlo score estimator (Eqs. 3.18-3.19 in the paper),
    given an already-selected neighbor subset (dX_sub, log_weight_spatial)
    from select_neighbor_subset. Only the tau-dependent diffusion weight
    (Eq. 3.19's second exponential) is computed here; the spatial weight is
    reused unchanged since it does not depend on tau.

    This is called once per reverse-ODE step (K=diffusion_timesteps times), so
    it is the dominant cost of latent extraction. It is written to be
    mathematically identical to the naive per-neighbor formulation while
    avoiding both (N, subset, dim) temporaries:

      * Softmax argument via the expanded squared distance
        ||z - a dX_j||^2 = ||z||^2 - 2a<z,dX_j> + a^2 ||dX_j||^2. The ||z||^2
        term is constant across neighbors j within a row, so softmax drops it
        exactly (shift invariance); ||dX_j||^2 (dX_sq) is tau-independent and
        precomputed once. This avoids ever forming diff = z - a dX_j.

      * The score sum_j w_j * (-(z - a dX_j)/b2) collapses to
        -(z - a * sum_j w_j dX_j)/b2 because sum_j w_j = 1 (softmax), so only
        the weighted mean of dX is needed, not a full (N, subset, dim)
        weighted sum of per-neighbor vectors.
    """
    N = z_current.shape[0]
    z_current_flat = z_current.view(N, -1)

    alpha_tau = 1.0 - tau
    beta_tau_sq = max(tau, 1e-6)
    c_lin = alpha_tau / beta_tau_sq                       # coeff on <z, dX_j>
    c_sq = alpha_tau * alpha_tau / (2.0 * beta_tau_sq)    # coeff on ||dX_j||^2

    scores = []
    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        z_chunk = z_current_flat[i:end_idx]                       # (c, d)
        dX_sub_chunk = dX_sub[i:end_idx]                          # (c, s, d)
        dX_sq_chunk = dX_sq[i:end_idx]                            # (c, s)
        log_weight_spatial_chunk = log_weight_spatial[i:end_idx]  # (c, s)

        # <z, dX_j> for every neighbor j (contract over the state dimension)
        zdot = torch.einsum("nd,nsd->ns", z_chunk, dX_sub_chunk)  # (c, s)

        # Softmax argument; the row-constant ||z||^2/(2 b2) term is omitted
        # because softmax is invariant to a per-row additive constant.
        log_weights = log_weight_spatial_chunk + c_lin * zdot - c_sq * dX_sq_chunk
        w = torch.softmax(log_weights, dim=1)                     # (c, s)

        # Weighted mean of dX over neighbors -> (c, d)
        wdX = torch.einsum("ns,nsd->nd", w, dX_sub_chunk)

        # score = -(z - a * E_w[dX]) / b2   (uses sum_j w_j = 1)
        score_chunk = -(z_chunk - alpha_tau * wdX) / beta_tau_sq
        scores.append(score_chunk)

    return torch.cat(scores, dim=0).view_as(z_current)


def build_tau_schedule(num_timesteps, schedule="geometric", tau_min=1e-5):
    """
    Build the descending sequence of pseudotime nodes (tau_1 > tau_2 > ... > 0)
    at which the reverse ODE (Eq. 3.24) is evaluated, together with the signed
    step size d_tau_k = tau_k - tau_{k+1} > 0 taken from each node.

    The reverse ODE's right-hand side is smooth near tau=1 (the b(tau) and
    sigma^2(tau) singularities there cancel once the score is substituted in)
    but sharpens rapidly as tau -> 0: the score's softmax weights harden into a
    near-nearest-neighbor selector, with logit coefficients ~1/tau. All of the
    integration error therefore accumulates in a thin layer near tau=0, which a
    uniform grid (whose smallest node is 1/K) resolves only by making K huge.

    schedule="geometric" instead spaces the nodes logarithmically, from tau=1
    down to tau_min, so they cluster where the action is. Empirically this
    reaches the estimator's own Monte-Carlo noise floor at K ~ 100-200,
    where a uniform grid needs K ~ 5,000-10,000 for the same accuracy (a
    ~50-100x reduction). tau_min ~ 1e-5 is the sweet spot (1e-4 doesn't reach
    deep enough; 1e-6 spends nodes below where the score still varies) but is
    exposed as a knob since the ideal depth is mildly dataset-dependent.

    schedule="uniform" reproduces the paper's original evenly-spaced grid
    (tau_k = k/K, for k = K, K-1, ..., 1), for reproducing prior results.

    Returns (taus, d_taus), both length-num_timesteps 1-D numpy arrays, where
    taus[k] is the evaluation node and d_taus[k] the step taken from it. The
    final step lands exactly on tau=0 in both schedules, so the RHS is never
    evaluated at tau=0 itself (its 1/tau terms stay finite).
    """
    K = num_timesteps
    if schedule == "uniform":
        # Original scheme: evaluate at tau = K/K, ..., 1/K; each step is 1/K,
        # and the last one (from tau=1/K) lands on tau=0.
        taus = np.arange(K, 0, -1, dtype=np.float64) / K
        d_taus = np.full(K, 1.0 / K, dtype=np.float64)
        return taus, d_taus

    if schedule == "geometric":
        # Nodes log-spaced from 1 down to tau_min, then a closing node at 0 so
        # the last step integrates the remaining [0, tau_min] layer.
        nodes = np.geomspace(1.0, tau_min, K, dtype=np.float64)  # length K, descending
        nodes = np.append(nodes, 0.0)                            # length K+1
        taus = nodes[:-1]                                        # evaluation nodes (tau > 0)
        d_taus = nodes[:-1] - nodes[1:]                          # positive step from each node
        return taus, d_taus

    raise ValueError(f"Unknown tau schedule '{schedule}'. Choose 'geometric' or 'uniform'.")


@torch.no_grad()
def extract_latents(x_targets, X_obs, Y_obs, config, verbose=True):
    """
    Reverse-ODE solver (Eq. 3.23-3.24, Algorithm 3.1) that samples
    z_1 ~ N(0, I) and integrates backward in pseudotime tau=1 -> 0 using the
    training-free score estimator, producing the paired latents (Z1, Z0):
    Z1 is the initial Gaussian noise and Z0 is the corresponding label
    (Z0 = X_dt - x) needed to supervise the flow-map network.
    """
    num_timesteps = config.get("diffusion_timesteps", 100)
    tau_schedule = config.get("tau_schedule", "geometric")
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
    # Scaling only the physical increment improves conditioning near tau=0.
    # Z1 and the conditioning state x stay unchanged; the result is converted
    # back to physical units before returning below.
    dX_obs = (Y_obs - X_obs) * increment_scale

    N = x_targets.shape[0]

    # 1. Spawn pure standard Gaussian noise (Z1)
    z_initial = torch.randn_like(x_targets)
    z_current = z_initial.clone()

    # The nearest-neighbor subset (and its spatial weight) depends only on x,
    # not on the reverse-ODE pseudotime tau, so it is searched for once here
    # and reused at every step below -- matching the paper's stated
    # optimization (Section 3.2, footnote 1) instead of re-searching at each
    # of the num_timesteps steps.
    if verbose:
        print(f"Selecting nearest-neighbor subsets for {N} targets...", flush=True)
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

    # ||dX_j||^2 is independent of the reverse-ODE pseudotime tau, so precompute
    # it once here rather than reconstructing the distance from scratch at each
    # of the num_timesteps steps (see score_from_neighbors). Computed in chunks
    # so the transient dX_sub**2 copy is bounded to one chunk rather than a full
    # (N, subset_size, dim) allocation (several GB at large N).
    dX_sq = torch.empty(dX_sub.shape[:2], dtype=dX_sub.dtype, device=dX_sub.device)
    _sq_chunk = max(1, min(N, 2_000_000 // max(dX_sub.shape[1], 1)))
    for i in range(0, N, _sq_chunk):
        e = min(i + _sq_chunk, N)
        dX_sq[i:e] = (dX_sub[i:e] ** 2).sum(dim=2)

    # The score step's cost/throughput is governed by the size of its working
    # tensors, which are (score_chunk, subset_size). Benchmarks show a clear
    # cache sweet spot around ~2e7 elements per chunk on CPU -- both much
    # smaller (Python/loop overhead dominates) and much larger (cache
    # thrashing) are markedly slower. So size the score chunk to that budget
    # rather than reusing the neighbor-search chunk_size (which is tuned for a
    # different, cdist-bound working set). This is purely a batching choice and
    # does not change the result. An explicit score_chunk_size config overrides.
    subset_size = dX_sub.shape[1]
    score_chunk = config.get("score_chunk_size", 0) or max(1, min(N, 20_000_000 // max(subset_size, 1)))

    # Sequence of pseudotime nodes and (per-step) step sizes for the reverse
    # ODE. "geometric" (the default) clusters nodes near tau=0 where the score
    # sharpens, reaching full accuracy at far smaller diffusion_timesteps than
    # the evenly-spaced "uniform" grid (the paper's original scheme). See
    # build_tau_schedule. Both ode_solver options are schedule-agnostic: they
    # index nodes/step sizes from these arrays rather than assuming a fixed
    # d_tau, so "heun" works identically on either schedule.
    taus, d_taus = build_tau_schedule(num_timesteps, schedule=tau_schedule, tau_min=tau_min)
    n_nodes = len(taus)

    if verbose:
        print(f"Neighbor search complete in {time.time() - start_time:.2f} seconds.", flush=True)
        print(f"Starting reverse-ODE integration for {N} targets (score chunk={score_chunk}, "
              f"solver={ode_solver}, increment scale={increment_scale:g}, tau_schedule={tau_schedule}"
              f"{f', tau_min={tau_min:g}' if tau_schedule == 'geometric' else ''}) using the "
              f"training-free score estimator...", flush=True)
        start_time = time.time()

    # 2. Reverse ODE Loop
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
            print(f"  [ODE Step] tau = {tau:.4g} completed.")

        # Reverse probability-flow ODE drift (Eq. 3.23).
        tau_safe = min(tau, 1.0 - 1e-5)
        b_tau = -1.0 / (1.0 - tau_safe)
        sigma_sq_tau = (1.0 + tau_safe) / (1.0 - tau_safe)
        drift = b_tau * z_current - 0.5 * sigma_sq_tau * score

        if ode_solver == "euler":
            z_current = z_current - drift * d_tau
        else:
            # Explicit trapezoidal / Heun step. The endpoint score is evaluated
            # at the Euler predictor. On the final node of the schedule (tau
            # would land on the tau_next=0 endpoint) we keep the Euler
            # predictor instead, because the empirical score becomes singular
            # at tau=0 -- this check is by node index, not a hardcoded step
            # count, so it lands correctly on the last node of either schedule.
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
    Data-generation half of Algorithm 3.1: turns the observation dataset
    D_obs = {(x_m, dx_m)} (as produced by own_models.datasets.sde.sde_dataset,
    in RAW physical units) into the labeled dataset D_label = {(x_j, z_j, y_j)}
    needed for supervised training of the flow-map network, by sampling
    z_j ~ N(0, I) and solving the reverse ODE for each x_j = x_m.

    The reverse-ODE score estimator (extract_latents) runs entirely on the raw
    trainX/trainY, matching how the paper's Section 3 derivation is written
    directly in terms of X_t -- crucially, this keeps the spatial bandwidth
    `nu` meaningful across datasets of very different physical scale (a fixed
    nu calibrated against min-max-normalized data means something very
    different for a heavy-tailed process like geometric Brownian motion than
    for a bounded one like an OU process). Normalization to [-1,1] is applied
    only here, right before the network sees anything: to the condition x and
    to the synthetic target y = x + z0 (both physical-state quantities, using
    the same vmin/vmax the raw training data was measured against -- see
    own_models.datasets.sde.sde_dataset.load()). The noise z is standard
    normal by construction (Eq. 2.4) and is deliberately left un-normalized.

    Returns DUE-shaped arrays ready to hand straight to due.models.ODE:
        trainX_augmented: (N, input_dim + output_dim), the flattened
                          [normalize(x), z]
        trainY_synthetic: (N, output_dim, 1), normalize(x + z0), matching the
                          multi_steps=1 convention of
                          due.datasets.ode/due.models.ODE.

    Note: the flow map only defines a single-step transition (each label is
    generated independently for a fixed Delta t); multi-step rollout only
    happens at inference time via SDEResNet.predict(), which draws a fresh z
    at every step. Training with multi_steps > 1 is therefore not supported.
    """
    if config.get("multi_steps", 1) != 1:
        raise ValueError(
            "The training-free conditional diffusion model only supports "
            "single-step supervised training (multi_steps=1): each label is "
            "generated independently by the reverse ODE for a fixed Delta t. "
            "Multi-step rollout happens only at inference via "
            "SDEResNet.predict(), not during training."
        )

    # Optional CPU thread-count override. torch defaults to the physical core
    # count; on this workload benchmarks showed a ~1.3x gain from using the
    # logical core count instead. Only applied when explicitly configured, to
    # avoid oversubscribing shared machines.
    num_threads = config.get("num_threads", 0)
    if num_threads:
        torch.set_num_threads(int(num_threads))

    device = torch.device(config.get("device", "cpu"))
    dtype = _TORCH_DTYPE[config["dtype"]]

    X = torch.as_tensor(trainX, dtype=dtype, device=device)
    Y = torch.as_tensor(trainY, dtype=dtype, device=device)
    if Y.dim() == X.dim() + 1:
        Y = Y.squeeze(-1)
    save_path = config.get("save_path", "./sde_diffusion_output")
    latent_file = os.path.join(save_path, "extracted_latents.pt")

    if config.get("cache_latents", False) and os.path.exists(latent_file):
        print(f"Loading pre-computed latent variables from {latent_file}...")
        Z1, Z0 = torch.load(latent_file, map_location=device)
    else:
        print("Extracting latent variables using the training-free reverse ODE (Algorithm 3.1)...")
        Z1, Z0 = extract_latents(X, X, Y, config, verbose=True)

        print(f"Saving extracted latents to {latent_file} to speed up future runs...")
        os.makedirs(save_path, exist_ok=True)
        torch.save((Z1, Z0), latent_file)

    np_dtype = _NUMPY_DTYPE[config["dtype"]]

    # Normalization happens here, as late as possible in the pipeline: X and
    # (X + Z0) are physical-state quantities and are normalized to [-1,1]
    # using the same bounds the raw training data was measured against; Z1
    # (noise) is standard normal already and is left untouched.
    X_norm = normalize_state(X, vmin, vmax)

    # Flattened [normalize(x), z] condition, matching SDEResNet's expected input layout.
    trainX_augmented = torch.cat([X_norm, Z1], dim=-1).cpu().numpy().astype(np_dtype)

    # Synthetic label y = x + z_0 (Eq. 3.25), normalized like x above, reshaped
    # to (N, output_dim, multi_steps=1) to match due.datasets.ode/due.models.ODE's convention.
    trainY_synthetic = normalize_state(X + Z0, vmin, vmax).cpu().numpy().astype(np_dtype)
    trainY_synthetic = np.expand_dims(trainY_synthetic, axis=-1)

    print(f"Augmented input shape: {trainX_augmented.shape} ([normalize(x), z])")
    print(f"Synthetic target shape: {trainY_synthetic.shape} (normalize(x + z0))")

    return trainX_augmented, trainY_synthetic
