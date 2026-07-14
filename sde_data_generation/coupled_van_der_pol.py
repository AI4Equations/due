"""
Generate a ring of diffusively-coupled *stochastic Van der Pol oscillators* --
a genuinely multi-dimensional, nonlinear, limit-cycle SDE, as a harder test
bed than the (mostly 1D, and only-linear-in-multi-D) examples in the paper.

Model (N_OSC oscillators on a ring, i = 0 .. N-1, indices mod N):

    dx_i = y_i dt                                              + sigma_x dW^x_i
    dy_i = [ mu (1 - x_i^2) y_i - omega^2 x_i
             + k ( x_{i-1} - 2 x_i + x_{i+1} ) ] dt            + sigma_y dW^y_i

Each oscillator is a classic Van der Pol unit (nonlinear damping mu(1-x^2)y
gives a stable limit cycle); the k-term is nearest-neighbour diffusive
coupling on the ring, which drives (partial) synchronization. Both the
position and velocity equations carry independent additive noise, so every
one of the 2*N_OSC state coordinates has a well-defined diffusion coefficient
(useful for evaluation).

Why 2*N_OSC dims (even): Van der Pol is a 2nd-order oscillator, so each unit
needs a (position, velocity) pair to be Markovian -- which the training-free
diffusion method requires (it learns a one-step Markov transition map). A
literal odd 5D would need an arbitrary extra mode that breaks that structure,
so we use N_OSC oscillators -> 2*N_OSC dims. N_OSC=3 gives 6D (the shipped
default, closest genuine-VdP system to a "~5D complicated" target); N_OSC=2
gives 4D (safer for the neighbour-search-based score estimator, which -- being
local kernel regression in state space -- degrades in high dimension). Note
the invariant measure concentrates near a low-dimensional (torus-like)
attractor, so the *effective* dimension the neighbour search sees is well
below 2*N_OSC -- which is exactly what makes 6D tractable here.

State layout: interleaved [x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}], so
oscillator i occupies dims (2i, 2i+1) -- consecutive pairs, which the
multi-dimensional evaluation (own_models.utils.sde_evaluate_multidim) reads as
(position, velocity) phase-space pairs.

Output matches the same (N_trajectories, D, T_steps) .mat "trajectories"
convention as the 1D generators, so own_models.datasets.sde.sde_dataset loads
it unchanged.

Usage:
    python coupled_van_der_pol.py                 # defaults (3 oscillators -> 6D)
    python coupled_van_der_pol.py --n-osc 2       # 4D
    python coupled_van_der_pol.py --n-osc 3 --mu 3.0 --k 0.8
"""

import os
import time
import argparse

import numpy as np
from scipy.io import savemat


# =======================================================================
# DEFAULT CONFIGURATION (override any via the CLI; see argparse below)
# =======================================================================
N_OSC = 3            # number of oscillators -> state dimension is 2 * N_OSC (default 6D)
MU = 2.0             # Van der Pol nonlinearity (mu>>1 -> relaxation oscillations)
OMEGA = 1.0          # natural angular frequency of each oscillator
K_COUPLE = 0.5       # ring nearest-neighbour diffusive coupling strength
SIGMA_X = 0.1        # noise scale on the position equations
SIGMA_Y = 0.3        # noise scale on the velocity equations

N_TRAJECTORIES = 20000     # independent training trajectories
N_TEST_TRAJECTORIES = 2000 # independent test trajectories
T_TRAIN_HORIZON = 2.0      # short: training only needs the one-step transition, sampled densely
T_TEST_HORIZON = 20.0      # long: the rollout must cover many limit-cycle periods to compare the attractor
DT = 0.01                  # time step (matches the other datasets)

X0_RANGE = 3.0             # initial state sampled uniformly in [-X0_RANGE, X0_RANGE]^(2N),
                           # wide enough to cover both the transient approach and the cycle itself

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulated_data")
# =======================================================================


def _drift(state, mu, omega, k):
    """Vectorized drift for all trajectories at once.

    state: (n_traj, 2*N) interleaved [x0,y0,x1,y1,...].
    Returns the same-shaped drift vector.
    """
    x = state[:, 0::2]                                   # (n_traj, N) positions
    y = state[:, 1::2]                                   # (n_traj, N) velocities

    # Ring Laplacian: x_{i-1} - 2 x_i + x_{i+1}, periodic.
    coupling = np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1) - 2.0 * x

    dx = y
    dy = mu * (1.0 - x**2) * y - (omega**2) * x + k * coupling

    drift = np.empty_like(state)
    drift[:, 0::2] = dx
    drift[:, 1::2] = dy
    return drift


def generate_trajectories(num_trajectories, num_steps, n_osc, mu, omega, k,
                          sigma_x, sigma_y, dt, x0_range, label):
    dim = 2 * n_osc
    print(f"--- Generating Coupled Van der Pol ({label}) ---")
    print(f"Oscillators: {n_osc}  ->  state dimension: {dim}")
    print(f"Trajectories: {num_trajectories} | steps: {num_steps} (dt={dt}, horizon={num_steps*dt:.1f})")
    print(f"Params: mu={mu}, omega={omega}, k={k}, sigma_x={sigma_x}, sigma_y={sigma_y}")
    start_time = time.time()

    trajectories = np.zeros((num_trajectories, dim, num_steps))
    state = np.random.uniform(-x0_range, x0_range, size=(num_trajectories, dim))
    trajectories[:, :, 0] = state

    # Per-coordinate noise scale: sigma_x on positions (even dims), sigma_y on velocities (odd dims).
    noise_scale = np.empty(dim)
    noise_scale[0::2] = sigma_x
    noise_scale[1::2] = sigma_y
    sqrt_dt = np.sqrt(dt)

    for t in range(1, num_steps):
        drift = _drift(state, mu, omega, k)
        noise = np.random.normal(0.0, 1.0, size=state.shape) * noise_scale * sqrt_dt
        state = state + drift * dt + noise
        trajectories[:, :, t] = state

    print(f"Simulation complete in {time.time() - start_time:.2f} seconds.")
    return trajectories


def save_to_mat(trajectories, filename, meta):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    savemat(filename, {
        "trajectories": trajectories,
        "dt": np.array([[meta["dt"]]]),
        "metadata": {k: v for k, v in meta.items()},
    })
    print(f"Saved to {os.path.abspath(filename)}")
    print(f"Array shape: {trajectories.shape} (N, D, T)")
    print("-" * 50)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-osc", type=int, default=N_OSC, help="number of oscillators; state dim = 2*n_osc")
    p.add_argument("--mu", type=float, default=MU)
    p.add_argument("--omega", type=float, default=OMEGA)
    p.add_argument("--k", type=float, default=K_COUPLE, help="ring coupling strength")
    p.add_argument("--sigma-x", type=float, default=SIGMA_X)
    p.add_argument("--sigma-y", type=float, default=SIGMA_Y)
    p.add_argument("--n-train", type=int, default=N_TRAJECTORIES)
    p.add_argument("--n-test", type=int, default=N_TEST_TRAJECTORIES)
    p.add_argument("--t-train", type=float, default=T_TRAIN_HORIZON)
    p.add_argument("--t-test", type=float, default=T_TEST_HORIZON)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--x0-range", type=float, default=X0_RANGE)
    p.add_argument("--out-dir", default=OUTPUT_DIR, help="directory for the .mat files")
    args = p.parse_args()

    dim = 2 * args.n_osc
    train_name = os.path.join(args.out_dir, f"coupled_vdp_{dim}d_train.mat")
    test_name = os.path.join(args.out_dir, f"coupled_vdp_{dim}d_test.mat")

    meta = dict(n_osc=args.n_osc, mu=args.mu, omega=args.omega, k=args.k,
                sigma_x=args.sigma_x, sigma_y=args.sigma_y, dt=args.dt,
                description=f"{args.n_osc} ring-coupled stochastic Van der Pol oscillators ({dim}D)")

    np.random.seed(42)
    n_steps_train = int(round(args.t_train / args.dt)) + 1
    train = generate_trajectories(args.n_train, n_steps_train, args.n_osc, args.mu, args.omega,
                                  args.k, args.sigma_x, args.sigma_y, args.dt, args.x0_range, "Train")
    save_to_mat(train, train_name, meta)

    np.random.seed(999)
    n_steps_test = int(round(args.t_test / args.dt)) + 1
    test = generate_trajectories(args.n_test, n_steps_test, args.n_osc, args.mu, args.omega,
                                 args.k, args.sigma_x, args.sigma_y, args.dt, args.x0_range, "Test")
    save_to_mat(test, test_name, meta)

    print(f"\nDone. Point the experiment config's train_file/test_file at:")
    print(f"  simulated_data/coupled_vdp_{dim}d_train.mat")
    print(f"  simulated_data/coupled_vdp_{dim}d_test.mat")


if __name__ == "__main__":
    main()
