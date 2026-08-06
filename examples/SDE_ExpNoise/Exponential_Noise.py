"""
Standalone end-to-end example for the 1D SDE with exponentially-distributed
noise: load data -> generate labeled data via the training-free reverse
ODE -> train the flow-map network -> roll out predictions and evaluate
them -> compare the model's effective drift/diffusion against the exact
closed-form coefficients. All settings come from config.yaml; this script
takes no other input.

Run from this directory (matches every other example in examples/):
    python Exponential_Noise.py
"""

import os
import sys

import numpy as np
import torch
from pathlib import Path
from yaml import safe_load

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import due

# Exact closed-form drift a(x) and diffusion b(x) for dX = mu*X dt + sigma*sqrt(dt)*eta_t,
# eta_t ~ Exp(1), matching generate_1d_exponential_noise.py's mu=-2, sigma=0.1.
# a(x) includes +sigma/sqrt(dt) because Exp(1) has mean 1, not 0, so the raw
# increment carries a constant offset that a zero-mean noise term wouldn't.
MU, SIGMA = -2.0, 0.1
def a_true(x, dt):
    return MU * x + SIGMA / np.sqrt(dt)
def b_true(x, dt):
    return np.full_like(x, SIGMA)


@torch.no_grad()
def estimate_model_drift_diffusion(net, x_grid, dt, n_samples, device):
    """Monte Carlo estimate of the model's effective drift/diffusion: one
    predict() call per grid point, with n_samples replicated copies of x
    as the batch dimension so each gets an independent z draw in one
    batched forward pass."""
    dim = net.output_dim
    a_hat = np.zeros_like(x_grid)
    b_hat = np.zeros_like(x_grid)
    for i, x in enumerate(x_grid):
        x_input = np.full((n_samples, dim, 1), x, dtype=np.float32)
        pred = net.predict(x_input, steps=1, device=device)  # (n_samples, dim, 2)
        increments = pred[:, 0, 1] - x
        a_hat[i] = increments.mean() / dt
        b_hat[i] = increments.std() / np.sqrt(dt)
    return a_hat, b_hat


def plot_drift_diffusion(net, dt, save_path, n_samples, n_grid,
                         x_range=None, drift_ylim=None, diffusion_ylim=None):
    """x_range: [x_min, x_max] to sweep (default: the model's own [vmin, vmax]).
    drift_ylim / diffusion_ylim: [lo, hi] y-limits for the drift / diffusion
    subplot respectively (default: matplotlib auto-scale)."""
    import matplotlib.pyplot as plt

    vmin, vmax = float(net.vmin.ravel()[0]), float(net.vmax.ravel()[0])
    x_lo, x_hi = (x_range[0], x_range[1]) if x_range else (vmin, vmax)
    x_grid = np.linspace(x_lo, x_hi, n_grid, dtype=np.float32)
    a_hat, b_hat = estimate_model_drift_diffusion(net, x_grid, dt, n_samples, device="cpu")
    a_ref, b_ref = a_true(x_grid, dt), b_true(x_grid, dt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x_grid, a_ref, color="blue", label="True a(x)")
    axes[0].plot(x_grid, a_hat, color="red", linestyle="--", marker=".", label="Model a_hat(x)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("drift"); axes[0].set_title("Drift a(x)"); axes[0].legend()
    if drift_ylim is not None:
        axes[0].set_ylim(drift_ylim[0], drift_ylim[1])

    axes[1].plot(x_grid, b_ref, color="blue", label="True b(x)")
    axes[1].plot(x_grid, b_hat, color="red", linestyle="--", marker=".", label="Model b_hat(x)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("diffusion"); axes[1].set_title("Diffusion b(x)"); axes[1].legend()
    if diffusion_ylim is not None:
        axes[1].set_ylim(diffusion_ylim[0], diffusion_ylim[1])

    fig.suptitle("dX = mu*X dt + sigma*sqrt(dt)*eta_t,  eta~Exp(1), mu=-2.0, sigma=0.1", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(save_path, "drift_diffusion.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    rel_a_err = np.linalg.norm(a_hat - a_ref) / (np.linalg.norm(a_ref) + 1e-12)
    rel_b_err = np.linalg.norm(b_hat - b_ref) / (np.linalg.norm(b_ref) + 1e-12)
    print(f"Drift/diffusion relative L2 error: drift={rel_a_err:.3e}  diffusion={rel_b_err:.3e}")
    print(f"Saved {out_path}")


def main():
    conf_data, conf_net, conf_train = due.utils.read_sde_config("config.yaml")
    # "plotting" is a top-level config.yaml block not returned by read_sde_config.
    conf_plot = safe_load(Path("config.yaml").read_text()).get("plotting", {})

    # train_file/test_file resolve relative to this script's own directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, conf_data["train_file"])
    test_file = os.path.join(script_dir, conf_data["test_file"])
    loader = due.datasets.sde.sde_dataset(conf_data)
    trainX, trainY, test_set, vmin, vmax = loader.load(train_file, test_file)
    print(f"Loaded {trainX.shape[0]} training bursts and {test_set.shape[0]} test trajectories.")

    # Generate labeled data via the reverse ODE
    trainX_aug, trainY_synth = due.models.sde_diffusion.generate_labeled_data(
        trainX, trainY, conf_train, vmin, vmax)

    # Build + train the flow-map network
    mynet = due.networks.fcn.SDEResNet(vmin, vmax, conf_net)
    model = due.models.ODE(trainX_aug, trainY_synth, mynet, conf_train)
    model.train()
    model.save_hist()

    # Reload the best (not necessarily last-epoch) checkpoint before evaluating.
    hist = model.hist
    selection_col = 1 if model.do_validation else 0
    best_idx = int(torch.argmin(hist[:, selection_col]).item())
    checkpoint_path = os.path.join(conf_train["save_path"], "model")
    mynet = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mynet.eval()
    selection_name = "validation" if model.do_validation else "training"
    print(f"Evaluating best checkpoint from epoch {best_idx + 1} "
          f"({selection_name} loss={float(hist[best_idx, selection_col]):.6e}).")

    # Predict + evaluate (mean/std/percentile comparison vs. ground truth)
    steps_to_predict = test_set.shape[-1] - (conf_data["memory"] + 1)
    pred = mynet.predict(test_set[..., :conf_data["memory"] + 1], steps_to_predict, device=conf_train["device"])
    truth = test_set[..., :steps_to_predict + conf_data["memory"] + 1]
    due.utils.sde_evaluate(pred, truth, save_path=conf_train["save_path"],
                                  dt=conf_data.get("dt", 0.01), n_paths=conf_plot.get("n_paths", 30))

    # Compare effective drift/diffusion against the exact closed-form coefficients
    plot_drift_diffusion(mynet, dt=conf_data.get("dt", 0.01), save_path=conf_train["save_path"],
                          n_samples=conf_plot.get("drift_diffusion_n_samples", 5000),
                          n_grid=conf_plot.get("drift_diffusion_n_grid", 40),
                          x_range=conf_plot.get("drift_diffusion_x_range"),
                          drift_ylim=conf_plot.get("drift_ylim"),
                          diffusion_ylim=conf_plot.get("diffusion_ylim"))

    print(f"\nAll outputs written to: {conf_train['save_path']}")


if __name__ == "__main__":
    main()
