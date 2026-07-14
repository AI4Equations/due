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
