"""
Plot predicted (model) vs. true drift a(x) and diffusion b(x) coefficient
functions for a trained SDEResNet checkpoint -- matching the paper's own
evaluation method (Figs. 2/5/8/14/17/19: "Comparison of effective drift and
diffusion functions obtained by the simulated trajectories using the
generative model and the exact SDE").

For a grid of x values spanning the model's own training range (its stored
vmin/vmax), the model's effective drift/diffusion are estimated via Monte
Carlo (Eq. 4.8): at each fixed x, draw many z ~ N(0, I) (via
SDEResNet.predict(steps=1), which already draws independent noise per batch
row, so one call per grid point handles all samples at once) and take the
sample mean/std of the resulting one-step increment:
    a_hat(x) = E_z[G_theta(x, z) - x] / dt
    b_hat(x) = Std_z[G_theta(x, z) - x] / sqrt(dt)

The "true" a(x), b(x) are each SDE's exact closed-form drift/diffusion
functions, hardcoded below to match sde_data_generation/*.py exactly (rather
than re-estimated from binned simulated data as the paper does -- we know
these in closed form, which is strictly more precise).

Only 1D SDEs are supported (all four datasets in sde_examples/ are 1D); a
multi-dimensional drift vector / diffusion matrix comparison would need a
different visualization and isn't implemented here.

Usage:
    python plot_drift_diffusion.py --results-dir results/OU
    python plot_drift_diffusion.py --results-dir results/GBM --dataset GBM --n-samples 20000

    # Batch mode: point at a root containing multiple named experiment folders
    # (e.g. results/OU, results/GBM, ...), each gets its own plot:
    python plot_drift_diffusion.py --results-dir results
    python plot_drift_diffusion.py --results-dir ../sde_results/good_results
"""

import os
import re
import sys
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import own_models  # noqa: F401 (needed to unpickle SDEResNet checkpoints)


# Exact closed-form drift a(x) and diffusion b(x), matching
# sde_data_generation/*.py's parameters exactly. dt matches what all four
# datasets were generated with (Delta t = 0.01).
SDE_DEFS = {
    "OU": dict(
        a=lambda x, dt: 1.0 * (1.2 - x),
        b=lambda x, dt: np.full_like(x, 0.3),
        dt=0.01,
        eq_label="dX = theta(mu - X)dt + sigma dW,  theta=1.0, mu=1.2, sigma=0.3",
    ),
    "GBM": dict(
        a=lambda x, dt: 2.0 * x,
        b=lambda x, dt: 1.0 * x,
        dt=0.01,
        eq_label="dX = mu*X dt + sigma*X dW,  mu=2.0, sigma=1.0",
    ),
    "DoubleWell": dict(
        a=lambda x, dt: 1.0 * x - 1.0 * x**3,
        b=lambda x, dt: np.full_like(x, 0.5),
        dt=0.01,
        eq_label="dX = (alpha*X - beta*X^3)dt + sigma dW,  alpha=beta=1, sigma=0.5",
    ),
    "ExpNoise": dict(
        # a(x) includes + sigma/sqrt(dt) because eta_t ~ Exp(1) has E[eta]=1,
        # not 0 -- see paper Eq. 4.7. b(x) = sigma (Std[Exp(1)] = 1).
        a=lambda x, dt: -2.0 * x + 0.1 / np.sqrt(dt),
        b=lambda x, dt: np.full_like(x, 0.1),
        dt=0.01,
        eq_label="dX = mu*X dt + sigma*sqrt(dt)*eta_t,  eta~Exp(1), mu=-2.0, sigma=0.1",
    ),
}


def resolve_dataset_name(folder_name, override=None):
    """
    Match a known SDE_DEFS key against a whole underscore/non-alphanumeric-
    separated token in folder_name (e.g. "kinda_good_GBM" -> "GBM"), not a
    naive substring search -- "OU" is a substring of "DoubleWell" ("d-OU-
    blewell"), so plain `in` matching would misfire there.
    """
    if override:
        return override
    tokens = {t.lower() for t in re.split(r"[^A-Za-z0-9]+", folder_name) if t}
    matches = [key for key in SDE_DEFS if key.lower() in tokens]
    return matches[0] if len(matches) == 1 else None


def find_experiments(results_dir, dataset_override=None):
    """If results_dir itself holds a 'model' file, treat it as one experiment.
    Otherwise, treat it as a root and return every subfolder that has one."""
    if os.path.isfile(os.path.join(results_dir, "model")):
        name = os.path.basename(os.path.normpath(results_dir))
        return [(name, results_dir, resolve_dataset_name(name, dataset_override))]

    found = []
    for name in sorted(os.listdir(results_dir)):
        sub = os.path.join(results_dir, name)
        if os.path.isfile(os.path.join(sub, "model")):
            found.append((name, sub, resolve_dataset_name(name, dataset_override)))
    return found


@torch.no_grad()
def estimate_model_drift_diffusion(net, x_grid, dt, n_samples, device):
    """Monte Carlo estimate of the model's effective drift/diffusion at each
    x in x_grid (Eq. 4.8): one predict() call per grid point, with n_samples
    replicated copies of x as the batch dimension so each gets an independent
    z draw in a single batched forward pass."""
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


def plot_one(name, experiment_dir, dataset, n_samples, n_grid, device, dt_override):
    sde = SDE_DEFS[dataset]
    dt = dt_override or sde["dt"]

    model_path = os.path.join(experiment_dir, "model")
    net = torch.load(model_path, weights_only=False)
    net.eval()
    vmin, vmax = float(net.vmin.ravel()[0]), float(net.vmax.ravel()[0])
    if net.output_dim != 1:
        print(f"[{name}] SKIPPED: output_dim={net.output_dim} -- only 1D SDEs are supported.")
        return

    x_grid = np.linspace(0.6, 2, n_grid, dtype=np.float32)
    a_hat, b_hat = estimate_model_drift_diffusion(net, x_grid, dt, n_samples, device)
    a_true = sde["a"](x_grid, dt)
    b_true = sde["b"](x_grid, dt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x_grid, a_true, color="blue", label="True a(x)")
    axes[0].plot(x_grid, a_hat, color="red", linestyle="--", marker=".", label="Model a_hat(x)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("drift")
    axes[0].set_title(f"{name}: Drift a(x)")
    axes[0].legend()
    axes[0].set_ylim(-0.8, 0.6)

    axes[1].plot(x_grid, b_true, color="blue", label="True b(x)")
    axes[1].plot(x_grid, b_hat, color="red", linestyle="--", marker=".", label="Model b_hat(x)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("diffusion")
    axes[1].set_title(f"{name}: Diffusion b(x)")
    axes[1].legend()
    axes[1].set_ylim(0.27, 0.33)


    fig.suptitle(sde["eq_label"], fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(experiment_dir, "drift_diffusion.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    rel_a_err = np.linalg.norm(a_hat - a_true) / (np.linalg.norm(a_true) + 1e-12)
    rel_b_err = np.linalg.norm(b_hat - b_true) / (np.linalg.norm(b_true) + 1e-12)
    print(f"[{name}] dataset={dataset}  x range=[{vmin:.3f}, {vmax:.3f}]  "
          f"relative L2 error: drift={rel_a_err:.3e}  diffusion={rel_b_err:.3e}")
    print(f"[{name}] Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True,
                         help="A single experiment folder (containing a 'model' file), or a root "
                              "directory containing multiple named experiment subfolders.")
    parser.add_argument("--dataset", default=None, choices=list(SDE_DEFS.keys()),
                         help="Which SDE's true a(x)/b(x) to compare against. Auto-inferred from "
                              "the folder name if it matches one of: " + ", ".join(SDE_DEFS.keys()) +
                              " (case-insensitive, '_continued' suffix stripped). Required if it "
                              "can't be inferred, and only usable when --results-dir is a single "
                              "experiment (not a root with multiple subfolders).")
    parser.add_argument("--n-samples", type=int, default=5000,
                         help="Monte Carlo samples (fresh z draws) per x grid point.")
    parser.add_argument("--n-grid", type=int, default=40,
                         help="Number of x grid points spanning the model's own [vmin, vmax].")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dt", type=float, default=None,
                         help="Override the SDE's default Delta t (all four datasets use 0.01).")
    args = parser.parse_args()

    experiments = find_experiments(args.results_dir, dataset_override=args.dataset)
    if not experiments:
        print(f"No experiment folders with a 'model' file found under {args.results_dir}")
        return

    for name, sub_dir, dataset in experiments:
        if dataset is None:
            print(f"[{name}] SKIPPED: couldn't infer which SDE this is from the folder name "
                  f"'{name}'. Pass --dataset explicitly (only works when --results-dir points "
                  f"directly at this one experiment, not a root with multiple subfolders).")
            continue
        plot_one(name, sub_dir, dataset, args.n_samples, args.n_grid, args.device, args.dt)


if __name__ == "__main__":
    main()
