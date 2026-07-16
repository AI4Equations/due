"""
Plot predicted (model) vs. true drift a(x) and diffusion b(x) coefficient
functions for a trained SDEResNet checkpoint -- matching the paper's own
evaluation method (Figs. 2/5/8/14/17/19: "Comparison of effective drift and
diffusion functions obtained by the simulated trajectories using the
generative model and the exact SDE").

For a fixed state x, the model's effective drift/diffusion are estimated via
Monte Carlo (Eq. 4.8): draw many z ~ N(0, I) (via SDEResNet.predict(steps=1),
which already draws independent noise per batch row, so one call handles all
samples at once) and take the sample mean/std of the resulting one-step
increment:
    a_hat(x) = E_z[G_theta(x, z) - x] / dt
    b_hat(x) = Std_z[G_theta(x, z) - x] / sqrt(dt)          (per coordinate)

The "true" a, b are each SDE's exact closed-form drift/diffusion, hardcoded
below to match sde_data_generation/*.py exactly.

Dimensionality:
  * 1D SDEs (OU, GBM, DoubleWell, ExpNoise): the classic a(x) vs x / b(x) vs x
    curves, in drift_diffusion.png.
  * Multi-D SDEs (e.g. CoupledVdP): a full d-dimensional drift field / diffusion
    can't be drawn directly, so two complementary views are produced:
      - drift_diffusion_slices.png: coordinate slices (partial dependence). For
        each coordinate j, coordinate j is swept over its [vmin_j, vmax_j] range
        while the others are held at a reference state, and the j-th drift and
        diffusion components are compared (model vs true) along that line. This
        is the direct generalization of the 1D curve.
      - drift_field.png (even-dim, interleaved (x_i, y_i) pairs only, as the
        coupled-VdP generator emits): a phase-plane quiver of the drift vector
        over each oscillator's (position, velocity) plane, model vs true. This
        shows the rotational, limit-cycle-driving flow field the density plots
        (own_models.utils.sde_evaluate_multidim) only show the *result* of.

Usage:
    python plot_drift_diffusion.py --results-dir results/OU
    python plot_drift_diffusion.py --results-dir results/CoupledVdP
    python plot_drift_diffusion.py --results-dir results            # batch over subfolders
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


def coupled_vdp_drift(X, p):
    """Exact drift of a ring of coupled Van der Pol oscillators (matches
    sde_data_generation/coupled_van_der_pol.py). X: (n, d) interleaved
    [x0,y0,x1,y1,...]; returns (n, d)."""
    x = X[:, 0::2]   # (n, N) positions
    y = X[:, 1::2]   # (n, N) velocities
    coupling = np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1) - 2.0 * x
    dx = y
    dy = p["mu"] * (1.0 - x**2) * y - (p["omega"]**2) * x + p["k"] * coupling
    a = np.empty_like(X)
    a[:, 0::2] = dx
    a[:, 1::2] = dy
    return a


def coupled_vdp_diffusion(X, p):
    """Exact (state-independent) per-coordinate diffusion of the coupled-VdP
    system: sigma_x on positions (even dims), sigma_y on velocities (odd dims).
    Returns (n, d)."""
    n, d = X.shape
    b = np.empty(d)
    b[0::2] = p["sigma_x"]
    b[1::2] = p["sigma_y"]
    return np.broadcast_to(b, (n, d)).copy()


# Exact closed-form drift/diffusion, matching sde_data_generation/*.py. 1D
# entries expose scalar a(x, dt)/b(x, dt); multi-D entries set multidim=True and
# expose vector drift_vec(X, params)/diffusion_vec(X, params) plus the generator
# params (edit these if you regenerate the dataset with different parameters).
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
    "CoupledVdP": dict(
        multidim=True,
        drift_vec=coupled_vdp_drift,
        diffusion_vec=coupled_vdp_diffusion,
        # MUST match sde_data_generation/coupled_van_der_pol.py's defaults; edit
        # if you regenerated with different --mu/--omega/--k/--sigma-*.
        params=dict(mu=2.0, omega=1.0, k=0.5, sigma_x=0.1, sigma_y=0.3),
        dt=0.01,
        eq_label="ring of coupled Van der Pol oscillators (mu=2, omega=1, k=0.5, sigma_x=0.1, sigma_y=0.3)",
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
    """1D Monte Carlo estimate of the model's effective drift/diffusion at each
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


@torch.no_grad()
def estimate_model_drift_diffusion_multidim(net, X_points, dt, n_samples, device):
    """Multi-D generalization: at each state row of X_points (n_points, d),
    replicate it n_samples times, take one model step, and return the per-
    coordinate mean-increment drift a_hat (n_points, d) and std-increment
    diffusion b_hat (n_points, d)."""
    d = net.output_dim
    n_points = X_points.shape[0]
    a_hat = np.zeros((n_points, d))
    b_hat = np.zeros((n_points, d))
    for i in range(n_points):
        x = X_points[i]
        x_input = np.tile(x.reshape(1, d, 1), (n_samples, 1, 1)).astype(np.float32)
        pred = net.predict(x_input, steps=1, device=device)   # (n_samples, d, 2)
        inc = pred[:, :, 1] - x.reshape(1, d)                  # (n_samples, d)
        a_hat[i] = inc.mean(axis=0) / dt
        b_hat[i] = inc.std(axis=0) / np.sqrt(dt)
    return a_hat, b_hat


def _plot_1d(name, experiment_dir, net, sde, dt, n_samples, n_grid, device):
    vmin, vmax = float(net.vmin.ravel()[0]), float(net.vmax.ravel()[0])

    # x_grid = np.linspace(0.5, 2, n_grid, dtype=np.float32)
    x_grid = np.linspace(0.3, 0.9, n_grid, dtype=np.float32)
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
    # axes[0].set_ylim(-0.85, 0.75)

    axes[1].plot(x_grid, b_true, color="blue", label="True b(x)")
    axes[1].plot(x_grid, b_hat, color="red", linestyle="--", marker=".", label="Model b_hat(x)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("diffusion")
    axes[1].set_title(f"{name}: Diffusion b(x)")
    axes[1].legend()
    # axes[1].set_ylim(0.27, 0.33)
    axes[1].set_ylim(0.09, 0.11)

    fig.suptitle(sde["eq_label"], fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(experiment_dir, "drift_diffusion.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    rel_a_err = np.linalg.norm(a_hat - a_true) / (np.linalg.norm(a_true) + 1e-12)
    rel_b_err = np.linalg.norm(b_hat - b_true) / (np.linalg.norm(b_true) + 1e-12)
    print(f"[{name}] x range=[{vmin:.3f}, {vmax:.3f}]  "
          f"relative L2 error: drift={rel_a_err:.3e}  diffusion={rel_b_err:.3e}")
    print(f"[{name}] Saved plot to {out_path}")


def _plot_multidim(name, experiment_dir, net, sde, dt, n_samples, n_grid, device,
                   field_grid, field_samples, max_pairs):
    d = net.output_dim
    params = sde["params"]
    vmin = net.vmin.detach().cpu().numpy().ravel().astype(np.float64)
    vmax = net.vmax.detach().cpu().numpy().ravel().astype(np.float64)
    # Reference state (held fixed while one coordinate/pair is swept): the
    # midpoint of each coordinate's training range.
    x_ref = 0.5 * (vmin + vmax)

    # ---- View 1: coordinate slices (partial dependence) ----
    fig, axes = plt.subplots(d, 2, figsize=(11, 2.3 * d), squeeze=False)
    rel_a_errs, rel_b_errs = [], []
    for j in range(d):
        grid_j = np.linspace(vmin[j], vmax[j], n_grid)
        X_slice = np.tile(x_ref, (n_grid, 1))
        X_slice[:, j] = grid_j
        a_hat, b_hat = estimate_model_drift_diffusion_multidim(net, X_slice, dt, n_samples, device)
        a_true = sde["drift_vec"](X_slice, params)
        b_true = sde["diffusion_vec"](X_slice, params)

        axes[j][0].plot(grid_j, a_true[:, j], color="blue", label="True")
        axes[j][0].plot(grid_j, a_hat[:, j], color="red", linestyle="--", marker=".", ms=3, label="Model")
        axes[j][0].set_ylabel(f"drift a_{j}")
        axes[j][0].set_xlabel(f"x_{j}")
        if j == 0:
            axes[j][0].set_title("Drift component (sweep coord, others fixed at ref)")
            axes[j][0].legend(fontsize=8)

        axes[j][1].plot(grid_j, b_true[:, j], color="blue", label="True")
        axes[j][1].plot(grid_j, b_hat[:, j], color="red", linestyle="--", marker=".", ms=3, label="Model")
        axes[j][1].set_ylabel(f"diffusion b_{j}")
        axes[j][1].set_xlabel(f"x_{j}")
        if j == 0:
            axes[j][1].set_title("Diffusion component")
            axes[j][1].legend(fontsize=8)

        rel_a_errs.append(np.linalg.norm(a_hat[:, j] - a_true[:, j]) / (np.linalg.norm(a_true[:, j]) + 1e-12))
        rel_b_errs.append(np.linalg.norm(b_hat[:, j] - b_true[:, j]) / (np.linalg.norm(b_true[:, j]) + 1e-12))

    fig.suptitle(f"{name}: coordinate-slice drift/diffusion  ({sde['eq_label']})", fontsize=10)
    plt.tight_layout()
    slices_path = os.path.join(experiment_dir, "drift_diffusion_slices.png")
    plt.savefig(slices_path, dpi=150)
    plt.close()

    # ---- View 2: phase-plane drift-field quiver (even-dim interleaved pairs) ----
    field_path = None
    if d % 2 == 0:
        n_pairs = min(d // 2, max_pairs)
        fig, axes = plt.subplots(n_pairs, 2, figsize=(10, 4.5 * n_pairs), squeeze=False)
        for pi in range(n_pairs):
            xi, yi = 2 * pi, 2 * pi + 1
            gx = np.linspace(vmin[xi], vmax[xi], field_grid)
            gy = np.linspace(vmin[yi], vmax[yi], field_grid)
            XX, YY = np.meshgrid(gx, gy)
            X_field = np.tile(x_ref, (field_grid * field_grid, 1))
            X_field[:, xi] = XX.ravel()
            X_field[:, yi] = YY.ravel()

            a_hat, _ = estimate_model_drift_diffusion_multidim(net, X_field, dt, field_samples, device)
            a_true = sde["drift_vec"](X_field, params)

            for col, (a, tag) in enumerate([(a_hat, "MODEL"), (a_true, "TRUE")]):
                u = a[:, xi].reshape(field_grid, field_grid)
                v = a[:, yi].reshape(field_grid, field_grid)
                mag = np.hypot(u, v)
                axes[pi][col].quiver(XX, YY, u, v, mag, cmap="viridis", pivot="mid", scale_units="xy")
                axes[pi][col].set_title(f"Osc {pi} drift field ({tag})")
                axes[pi][col].set_xlabel(f"position x_{xi}")
                axes[pi][col].set_ylabel(f"velocity x_{yi}")
        fig.suptitle(f"{name}: phase-plane drift field (others held at reference state)", fontsize=11)
        plt.tight_layout()
        field_path = os.path.join(experiment_dir, "drift_field.png")
        plt.savefig(field_path, dpi=150)
        plt.close()

    print(f"[{name}] dim={d}  mean relative L2 error over coords: "
          f"drift={np.mean(rel_a_errs):.3e}  diffusion={np.mean(rel_b_errs):.3e}")
    print(f"[{name}] Saved {slices_path}" + (f" and {field_path}" if field_path else ""))


def plot_one(name, experiment_dir, dataset, n_samples, n_grid, device, dt_override,
             field_grid=14, field_samples=800, max_pairs=6):
    sde = SDE_DEFS[dataset]
    dt = dt_override or sde["dt"]

    net = torch.load(os.path.join(experiment_dir, "model"), weights_only=False)
    net.eval()

    is_multidim = sde.get("multidim", False)
    if net.output_dim > 1 and not is_multidim:
        print(f"[{name}] SKIPPED: checkpoint is {net.output_dim}D but '{dataset}' is a 1D SDE def.")
        return
    if net.output_dim == 1 and is_multidim:
        print(f"[{name}] SKIPPED: checkpoint is 1D but '{dataset}' is a multi-D SDE def.")
        return

    if is_multidim:
        _plot_multidim(name, experiment_dir, net, sde, dt, n_samples, n_grid, device,
                       field_grid, field_samples, max_pairs)
    else:
        _plot_1d(name, experiment_dir, net, sde, dt, n_samples, n_grid, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True,
                         help="A single experiment folder (containing a 'model' file), or a root "
                              "directory containing multiple named experiment subfolders.")
    parser.add_argument("--dataset", default=None, choices=list(SDE_DEFS.keys()),
                         help="Which SDE's true a/b to compare against. Auto-inferred from the "
                              "folder name if it matches one of: " + ", ".join(SDE_DEFS.keys()) +
                              " (case-insensitive). Required if it can't be inferred, and only "
                              "usable when --results-dir is a single experiment.")
    parser.add_argument("--n-samples", type=int, default=100000,
                         help="Monte Carlo samples (fresh z draws) per evaluated state.")
    parser.add_argument("--n-grid", type=int, default=100,
                         help="Number of grid points per coordinate sweep.")
    parser.add_argument("--field-grid", type=int, default=14,
                         help="(multi-D) grid resolution per axis for the phase-plane drift quiver.")
    parser.add_argument("--field-samples", type=int, default=800,
                         help="(multi-D) MC samples per quiver grid point (drift-only, so fewer needed).")
    parser.add_argument("--max-pairs", type=int, default=6,
                         help="(multi-D) max number of oscillator (x,y) pairs to draw quivers for.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dt", type=float, default=None,
                         help="Override the SDE's default Delta t (all datasets use 0.01).")
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
        plot_one(name, sub_dir, dataset, args.n_samples, args.n_grid, args.device, args.dt,
                 field_grid=args.field_grid, field_samples=args.field_samples, max_pairs=args.max_pairs)


if __name__ == "__main__":
    main()
