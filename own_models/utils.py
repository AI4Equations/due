import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from yaml import safe_load

import due


def normalize_state(x, vmin, vmax):
    """
    Applies the same [-1,1] min-max normalization own_models.datasets.sde.sde_dataset
    used to apply internally, but against already-known bounds rather than
    computing them from x. Used by own_models.models.sde_diffusion.generate_labeled_data
    to normalize the diffusion engine's raw-unit outputs right before they
    reach the network -- the reverse-ODE score estimator itself runs on raw
    data (see sde_dataset.load()), so normalization happens only here, as late
    as possible in the pipeline.

    x: (..., dim) tensor or ndarray, raw physical units.
    vmin, vmax: (1, dim, 1) ndarrays, as returned by sde_dataset.load();
        reshaped here to broadcast against x's last axis.

    Returns the same type as x (tensor in, tensor out; ndarray in, ndarray out).
    """
    vmin_flat = np.asarray(vmin).reshape(-1)
    vmax_flat = np.asarray(vmax).reshape(-1)
    range_flat = vmax_flat - vmin_flat
    range_flat = np.where(range_flat == 0, 1.0, range_flat)
    center = 0.5 * (vmax_flat + vmin_flat)

    if isinstance(x, torch.Tensor):
        center_t = torch.as_tensor(center, dtype=x.dtype, device=x.device)
        range_t = torch.as_tensor(range_flat, dtype=x.dtype, device=x.device)
        return torch.clamp(2 * (x - center_t) / range_t, -1.0, 1.0)
    else:
        return np.clip(2 * (x - center) / range_flat, -1.0, 1.0)


def read_sde_config(config_path):
    """
    Thin wrapper around due.utils.read_config that additionally extracts the
    top-level "diffusion" block (nu, diffusion_timesteps, subsample_ratio,
    chunk_size, cache_latents) used by the training-free score estimator,
    merging it into the training config so the result can be passed straight
    to own_models.models.sde_diffusion.generate_labeled_data.
    """
    conf_data, conf_net, conf_train = due.utils.read_config(config_path)

    raw = safe_load(Path(config_path).read_text())
    conf_train.update(raw.get("diffusion", {}))

    return conf_data, conf_net, conf_train


def sde_evaluate(prediction, truth, save_path, dt=0.01, n_paths=30):
    """
    Evaluates a batch of generated SDE trajectories against the ground truth.

    Unlike due.utils.ode_evaluate (pointwise error, appropriate for
    deterministic ODEs), SDE trajectories diverge pathwise even for a perfect
    model, so the comparison is distributional rather than sample-by-sample.
    Produces a 2x2 figure:
      - top row: true vs. generated "spaghetti" plots with many sample paths,
        so the ensemble spread is visible directly, not just summarized;
      - bottom-left: ensemble mean with both a +-1 StdDev band and a 5th-95th
        percentile band overlaid for true vs. generated, capturing spread
        beyond the Gaussian assumption (relevant for the paper's non-Gaussian
        noise cases too);
      - bottom-right: absolute error in mean/std over time, showing how well
        the spread/mean evolution is tracked across the prediction horizon
        (useful since autoregressive rollout error tends to grow with time).

    prediction, truth: (N, dim, T) unnormalized trajectories.
    n_paths: number of individual sample paths to draw in the spaghetti plots.
    """
    assert prediction.shape[1] == truth.shape[1]
    os.makedirs(save_path, exist_ok=True)

    t_steps = truth.shape[2]
    time_axis = np.arange(t_steps) * dt
    n_paths = min(n_paths, prediction.shape[0], truth.shape[0])

    true_mean = np.mean(truth[:, 0, :], axis=0)
    true_std = np.std(truth[:, 0, :], axis=0)
    pred_mean = np.mean(prediction[:, 0, :], axis=0)
    pred_std = np.std(prediction[:, 0, :], axis=0)
    true_p05, true_p95 = np.percentile(truth[:, 0, :], [5, 95], axis=0)
    pred_p05, pred_p95 = np.percentile(prediction[:, 0, :], [5, 95], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].set_title(f"True Trajectories ({n_paths} samples)")
    for i in range(n_paths):
        axes[0, 0].plot(time_axis, truth[i, 0, :], color='blue', alpha=0.25, linewidth=0.8)
    axes[0, 0].plot(time_axis, true_mean, color='black', linewidth=2, label='Mean')
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Physical State X")
    axes[0, 0].legend()

    axes[0, 1].set_title(f"Generated Trajectories ({n_paths} samples)")
    for i in range(n_paths):
        axes[0, 1].plot(time_axis, prediction[i, 0, :], color='red', alpha=0.25, linewidth=0.8)
    axes[0, 1].plot(time_axis, pred_mean, color='black', linewidth=2, label='Mean')
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Physical State X")
    axes[0, 1].legend()

    # Match y-limits across the two spaghetti panels for a fair visual comparison
    ylim = (min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0]),
            max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1]))
    axes[0, 0].set_ylim(ylim)
    axes[0, 1].set_ylim(ylim)

    axes[1, 0].set_title("Ensemble Spread (±1 StdDev, 5th-95th pct.)")
    axes[1, 0].fill_between(time_axis, true_p05, true_p95, color='blue', alpha=0.10, label='True 5th-95th pct.')
    axes[1, 0].fill_between(time_axis, true_mean - true_std, true_mean + true_std, color='blue', alpha=0.25)
    axes[1, 0].plot(time_axis, true_mean, color='blue', label='True Mean')

    axes[1, 0].fill_between(time_axis, pred_p05, pred_p95, color='red', alpha=0.10, label='Generated 5th-95th pct.')
    axes[1, 0].fill_between(time_axis, pred_mean - pred_std, pred_mean + pred_std, color='red', alpha=0.25)
    axes[1, 0].plot(time_axis, pred_mean, color='red', linestyle='--', label='Generated Mean')
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("Absolute Error in Mean & StdDev Over Time")
    axes[1, 1].plot(time_axis, np.abs(true_mean - pred_mean), color='purple', label='|Mean error|')
    axes[1, 1].plot(time_axis, np.abs(true_std - pred_std), color='darkorange', label='|StdDev error|')
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/sde_evaluation.png", dpi=150)
    plt.close()
    print(f"Saved Generative Evaluation Plot to: {save_path}/sde_evaluation.png")

    mean_err = np.mean(np.abs(true_mean - pred_mean))
    std_err = np.mean(np.abs(true_std - pred_std))
    np.savetxt(f"{save_path}/distribution_error.csv", np.stack([time_axis, np.abs(true_mean - pred_mean), np.abs(true_std - pred_std)], axis=1),
               header="time,abs_mean_error,abs_std_error", delimiter=",", comments="")
    print(f"Mean absolute error in ensemble mean: {mean_err:.6f} | in ensemble std: {std_err:.6f}")

    # Returned so batch runners can collect metrics; standalone example scripts
    # simply ignore the return value.
    return {
        "mean_abs_error_mean": float(mean_err),
        "mean_abs_error_std": float(std_err),
        "final_abs_error_mean": float(np.abs(true_mean - pred_mean)[-1]),
        "final_abs_error_std": float(np.abs(true_std - pred_std)[-1]),
    }


def sde_evaluate_multidim(prediction, truth, save_path, dt=0.01, pair_phase_space=True,
                          max_pairs=6, psd_dim=0):
    """
    Multi-dimensional analogue of sde_evaluate, for SDEs with problem_dim > 1
    (e.g. the coupled Van der Pol system in sde_data_generation). Overlaid 1D
    spaghetti plots are uninformative once the state is multi-dimensional, so
    this compares the true vs generated ensembles through four dimension-honest
    lenses and writes three figures under save_path:

      1. phase_portraits.png -- 2D histogram of each oscillator's (position,
         velocity) plane, true (top row) vs generated (bottom row). Assumes the
         even-dim, interleaved (x_i, y_i) layout the coupled-VdP generator
         emits, so consecutive coordinate pairs ARE phase-space pairs. This is
         where a limit cycle appears -- as a ring -- so it shows directly
         whether the learned dynamics reproduce the attractor geometry. Skipped
         (with a note) if the dimension is odd.

      2. marginals.png -- per-coordinate stationary histograms, true vs
         generated overlaid (aggregated over all trajectories AND time steps).
         Checks each coordinate's marginal spread.

      3. dynamics.png -- three panels: (a) the power spectral density of one
         coordinate (psd_dim), averaged over trajectories, true vs generated --
         for an oscillator this peaks at the limit-cycle frequency, so it tests
         whether the *temporal* structure is captured, which no static density
         can reveal; (b) per-coordinate ensemble mean vs time; (c) per-
         coordinate ensemble std vs time.

    prediction, truth: (N, dim, T) unnormalized trajectories.

    Returns the same metric keys as sde_evaluate (mean_abs_error_mean/std,
    final_abs_error_mean/std), averaged over coordinates, so run_experiments'
    summary table works unchanged.
    """
    assert prediction.shape[1] == truth.shape[1]
    os.makedirs(save_path, exist_ok=True)

    dim = truth.shape[1]
    T = min(prediction.shape[2], truth.shape[2])
    prediction = prediction[:, :, :T]
    truth = truth[:, :, :T]
    time_axis = np.arange(T) * dt

    # ---- 1. Phase-space density (only meaningful for interleaved (x,y) pairs) ----
    if pair_phase_space and dim % 2 == 0:
        n_pairs = min(dim // 2, max_pairs)
        fig, axes = plt.subplots(2, n_pairs, figsize=(4 * n_pairs, 8), squeeze=False)
        for j in range(n_pairs):
            xi, yi = 2 * j, 2 * j + 1
            tx, ty = truth[:, xi, :].ravel(), truth[:, yi, :].ravel()
            px, py = prediction[:, xi, :].ravel(), prediction[:, yi, :].ravel()
            # Shared extent + bins so the two rows are directly comparable.
            xr = [min(tx.min(), px.min()), max(tx.max(), px.max())]
            yr = [min(ty.min(), py.min()), max(ty.max(), py.max())]
            axes[0, j].hist2d(tx, ty, bins=120, range=[xr, yr], cmap="viridis")
            axes[0, j].set_title(f"Osc {j}: TRUE  (x{xi}, x{yi})")
            axes[1, j].hist2d(px, py, bins=120, range=[xr, yr], cmap="viridis")
            axes[1, j].set_title(f"Osc {j}: GENERATED")
            for r in (0, 1):
                axes[r, j].set_xlabel(f"position x{xi}")
                axes[r, j].set_ylabel(f"velocity x{yi}")
        fig.suptitle("Phase-space density (invariant measure): true vs generated", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_path}/phase_portraits.png", dpi=150)
        plt.close()
    else:
        print(f"[sde_evaluate_multidim] dim={dim} is odd (or pairing disabled); "
              f"skipping phase_portraits.png.")

    # ---- 2. Per-coordinate marginals ----
    ncols = min(3, dim)
    nrows = int(np.ceil(dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
    for c in range(dim):
        ax = axes[c // ncols][c % ncols]
        tvals, pvals = truth[:, c, :].ravel(), prediction[:, c, :].ravel()
        rng = [min(tvals.min(), pvals.min()), max(tvals.max(), pvals.max())]
        ax.hist(tvals, bins=80, range=rng, density=True, color="blue", alpha=0.45, label="True")
        ax.hist(pvals, bins=80, range=rng, density=True, color="red", alpha=0.45, label="Generated")
        ax.set_title(f"coord x{c} marginal")
        ax.legend(fontsize=7)
    for c in range(dim, nrows * ncols):
        axes[c // ncols][c % ncols].axis("off")
    fig.suptitle("Per-coordinate stationary marginals: true vs generated", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_path}/marginals.png", dpi=150)
    plt.close()

    # ---- 3. Dynamics: PSD + mean/std over time ----
    true_mean, pred_mean = truth.mean(axis=0), prediction.mean(axis=0)   # (dim, T)
    true_std, pred_std = truth.std(axis=0), prediction.std(axis=0)       # (dim, T)

    # PSD of one coordinate, averaged over trajectories (demeaned per series).
    freqs = np.fft.rfftfreq(T, d=dt)
    def _avg_psd(arr):  # arr: (N, T)
        a = arr - arr.mean(axis=1, keepdims=True)
        return (np.abs(np.fft.rfft(a, axis=1)) ** 2).mean(axis=0)
    true_psd = _avg_psd(truth[:, psd_dim, :])
    pred_psd = _avg_psd(prediction[:, psd_dim, :])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].semilogy(freqs, true_psd, color="blue", label="True")
    axes[0].semilogy(freqs, pred_psd, color="red", linestyle="--", label="Generated")
    axes[0].set_title(f"Power spectral density (coord x{psd_dim})")
    axes[0].set_xlabel("frequency"); axes[0].set_ylabel("power"); axes[0].legend()

    for c in range(dim):
        axes[1].plot(time_axis, true_mean[c], color="blue", alpha=0.6)
        axes[1].plot(time_axis, pred_mean[c], color="red", linestyle="--", alpha=0.6)
        axes[2].plot(time_axis, true_std[c], color="blue", alpha=0.6)
        axes[2].plot(time_axis, pred_std[c], color="red", linestyle="--", alpha=0.6)
    axes[1].set_title("Ensemble mean vs time (all coords)\nblue=true, red=generated")
    axes[1].set_xlabel("time")
    axes[2].set_title("Ensemble std vs time (all coords)\nblue=true, red=generated")
    axes[2].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(f"{save_path}/dynamics.png", dpi=150)
    plt.close()

    print(f"Saved multi-dim evaluation plots (phase_portraits/marginals/dynamics) to {save_path}")

    # ---- Metrics (averaged over coordinates, same keys as sde_evaluate) ----
    mean_err = float(np.mean(np.abs(true_mean - pred_mean)))
    std_err = float(np.mean(np.abs(true_std - pred_std)))
    np.savetxt(f"{save_path}/distribution_error.csv",
               np.column_stack([time_axis,
                                np.abs(true_mean - pred_mean).mean(axis=0),
                                np.abs(true_std - pred_std).mean(axis=0)]),
               header="time,mean_abs_mean_error_over_dims,mean_abs_std_error_over_dims",
               delimiter=",", comments="")
    print(f"Mean abs error in ensemble mean: {mean_err:.6f} | in ensemble std: {std_err:.6f} (avg over {dim} coords)")

    return {
        "mean_abs_error_mean": mean_err,
        "mean_abs_error_std": std_err,
        "final_abs_error_mean": float(np.mean(np.abs(true_mean - pred_mean)[:, -1])),
        "final_abs_error_std": float(np.mean(np.abs(true_std - pred_std)[:, -1])),
    }
