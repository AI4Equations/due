"""
Batch runner for the training-free conditional diffusion SDE model.

Iterates over a directory of experiment folders -- each one a self-contained
config.yaml in the same style as the standalone sde_examples/ folders -- and,
for every model/dataset pair, runs the full pipeline: load data, (optionally)
subsample it, generate labeled data via the reverse ODE, train + validate the
flow-map network, roll out a prediction, and evaluate the distributional error.
A summary table across all experiments is printed and written to results/summary.csv.

Each experiment's config.yaml adds a few batch-runner-specific keys under `data`:
    train_file / test_file : dataset paths, relative to the repo root
    train_fraction         : fraction of the sampled training bursts to train on
    test_fraction          : fraction of test trajectories to evaluate/plot on
    max_predict_steps      : cap on autoregressive rollout length (0 = full horizon)

These let you dial each dataset's cost independently -- important since some of
the .mat files are very large. Any of them can also be overridden globally on
the command line; see `python run_experiments.py --help`.

Examples:
    python run_experiments.py                       # run every experiment folder
    python run_experiments.py --only GBM,OU         # run a subset
    python run_experiments.py --train-fraction 0.02 # override all train fractions
    python run_experiments.py --epochs 10           # quick smoke run
"""

import os
import sys
import time
import shutil
import argparse
import traceback

import numpy as np
import torch

# own_models is not pip-installed (unlike due); make the repo root importable
# so `import own_models` resolves regardless of the current working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import due
import own_models


def discover_experiments(experiments_dir, only=None):
    """Return sorted (name, config_path) for every subfolder holding a config.yaml."""
    found = []
    for name in sorted(os.listdir(experiments_dir)):
        cfg = os.path.join(experiments_dir, name, "config.yaml")
        if os.path.isfile(cfg):
            if only is None or name in only:
                found.append((name, cfg))
    return found


def run_one(name, config_path, results_dir, overrides, latent_cache_root=None):
    """Run the full pipeline for a single experiment and return a summary dict."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    conf_data, conf_net, conf_train = own_models.utils.read_sde_config(config_path)

    # Per-experiment output directory (also where latents are cached).
    save_path = os.path.join(results_dir, name)
    os.makedirs(save_path, exist_ok=True)
    conf_train["save_path"] = save_path

    # Apply command-line overrides (None means "leave the config value alone").
    train_fraction = overrides["train_fraction"] if overrides["train_fraction"] is not None else conf_data.get("train_fraction", 1.0)
    test_fraction = overrides["test_fraction"] if overrides["test_fraction"] is not None else conf_data.get("test_fraction", 1.0)
    max_predict_steps = overrides["max_predict_steps"] if overrides["max_predict_steps"] is not None else conf_data.get("max_predict_steps", 0)
    if overrides["epochs"] is not None:
        conf_train["epochs"] = overrides["epochs"]

    # --- Load data ---
    train_file = os.path.join(_REPO_ROOT, conf_data["train_file"])
    test_file = os.path.join(_REPO_ROOT, conf_data["test_file"])
    loader = own_models.datasets.sde.sde_dataset(conf_data)
    trainX, trainY, test_set, vmin, vmax = loader.load(train_file, test_file)

    # --- Subsample training bursts ---
    total_bursts = trainX.shape[0]
    n_train = max(1, int(train_fraction * total_bursts))
    trainX, trainY = trainX[:n_train], trainY[:n_train]
    print(f"[{name}] Using {n_train}/{total_bursts} (fraction={train_fraction}) training bursts.")

    # --- Subsample test trajectories ---
    n_test = max(1, int(test_fraction * test_set.shape[0]))
    test_set = test_set[:n_test]
    print(f"[{name}] Evaluating on {n_test} test trajectories (fraction={test_fraction}).")

    # generate_labeled_data only ever looks for extracted_latents.pt inside this
    # experiment's own save_path (results_dir/name), which starts empty every time
    # results_dir is fresh -- e.g. after archiving a previous results/ directory
    # elsewhere (say to ../sde_results/resultsN/) to keep multiple runs around. Look
    # in latent_cache_root/name/ for a cache to reuse instead, if given, validating
    # the row count before trusting it (a cache from a different train_fraction or
    # dataset won't match trainX here). generate_labeled_data validates this too, but
    # checking here as well lets us report it per-experiment before the fact.
    if latent_cache_root is not None:
        resolved_latent_file = os.path.join(save_path, "extracted_latents.pt")
        candidate = os.path.join(latent_cache_root, name, "extracted_latents.pt")
        if not os.path.exists(resolved_latent_file) and os.path.exists(candidate):
            cached_Z1, _ = torch.load(candidate, map_location="cpu")
            if cached_Z1.shape[0] == trainX.shape[0]:
                print(f"[{name}] Reusing cached latents from {candidate} ({cached_Z1.shape[0]} rows, matches).")
                shutil.copy(candidate, resolved_latent_file)
                conf_train["cache_latents"] = True
            else:
                print(f"[{name}] Found a latent cache at {candidate}, but it has {cached_Z1.shape[0]} rows "
                      f"vs. {trainX.shape[0]} in the current training data (likely a different "
                      f"train_fraction or dataset) -- ignoring it and re-extracting.")

    # --- Generate labeled data via the reverse ODE (Algorithm 3.1) ---
    t0 = time.time()
    trainX_aug, trainY_synth = own_models.models.sde_diffusion.generate_labeled_data(trainX, trainY, conf_train, vmin, vmax)
    gen_time = time.time() - t0

    # --- Build + train the flow-map network ---
    mynet = own_models.networks.sde.SDEResNet(vmin, vmax, conf_net)
    model = due.models.ODE(trainX_aug, trainY_synth, mynet, conf_train)
    t0 = time.time()
    model.train()
    train_time = time.time() - t0
    model.save_hist()

    # Reload the checkpoint selected by due.models.ODE so every downstream
    # evaluation uses the same validation-best (or train-best) network that was
    # saved to disk, rather than the in-memory weights from the final epoch.
    hist = model.hist
    selection_col = 1 if model.do_validation else 0
    best_idx = int(torch.argmin(hist[:, selection_col]).item())
    train_loss = float(hist[best_idx, 0])
    val_loss = float(hist[best_idx, 1]) if model.do_validation else float("nan")
    checkpoint_path = os.path.join(save_path, "model")
    # Load on CPU because SDEResNet normalizes CPU inputs with its stored
    # vmin/vmax before predict() moves the network/input to the requested device.
    mynet = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mynet.eval()
    selection_name = "validation" if model.do_validation else "training"
    print(f"[{name}] Evaluating best checkpoint from epoch {best_idx + 1} "
          f"({selection_name} loss={float(hist[best_idx, selection_col]):.6e}).")

    # --- Predict + evaluate ---
    full_steps = test_set.shape[-1] - (conf_data["memory"] + 1)
    steps_to_predict = full_steps if not max_predict_steps else min(full_steps, int(max_predict_steps))
    eval_seed = int(conf_train.get("seed", 0))
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)
    pred = mynet.predict(test_set[..., :conf_data["memory"] + 1], steps_to_predict, device=conf_train["device"])

    # Truncate the ground truth to the same horizon so the evaluation aligns.
    truth = test_set[..., :steps_to_predict + conf_data["memory"] + 1]
    metrics = own_models.utils.sde_evaluate(pred, truth, save_path=save_path, dt=conf_data.get("dt", 0.01))

    return {
        "name": name,
        "status": "ok",
        "n_train": n_train,
        "n_test": n_test,
        "predict_steps": steps_to_predict,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "mean_err": metrics["mean_abs_error_mean"],
        "std_err": metrics["mean_abs_error_std"],
        "gen_time_s": gen_time,
        "train_time_s": train_time,
    }


def write_summary(rows, results_dir):
    """Print a table and write results/summary.csv."""
    cols = ["name", "status", "n_train", "n_test", "predict_steps",
            "train_loss", "val_loss", "mean_err", "std_err", "gen_time_s", "train_time_s"]

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'name':<12} {'status':<8} {'n_train':>9} {'n_test':>7} {'steps':>6} " \
             f"{'train_loss':>12} {'val_loss':>12} {'mean_err':>10} {'std_err':>10} {'gen_s':>8} {'train_s':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["status"] == "ok":
            print(f"{r['name']:<12} {r['status']:<8} {r['n_train']:>9} {r['n_test']:>7} {r['predict_steps']:>6} "
                  f"{r['train_loss']:>12.3e} {r['val_loss']:>12.3e} {r['mean_err']:>10.4f} {r['std_err']:>10.4f} "
                  f"{r['gen_time_s']:>8.1f} {r['train_time_s']:>8.1f}")
        else:
            print(f"{r['name']:<12} {r['status']:<8} {r.get('error', '')}")
    print(f"\nWrote {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments-dir", default=os.path.join(os.path.dirname(__file__), "experiments"),
                        help="Directory holding one subfolder (with config.yaml) per experiment.")
    parser.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "results"),
                        help="Directory to write per-experiment outputs and summary.csv.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated experiment names to run (default: all discovered).")
    parser.add_argument("--train-fraction", type=float, default=None,
                        help="Override train_fraction for all experiments.")
    parser.add_argument("--test-fraction", type=float, default=None,
                        help="Override test_fraction for all experiments.")
    parser.add_argument("--max-predict-steps", type=int, default=None,
                        help="Override max_predict_steps for all experiments (0 = full horizon).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs for all experiments (handy for smoke runs).")
    parser.add_argument("--latent-cache-root", default=None,
                        help="Root directory containing archived <name>/extracted_latents.pt files "
                             "to reuse per experiment (e.g. an old results/ directory you renamed "
                             "elsewhere after a previous run, since a fresh --results-dir starts "
                             "empty). Row count must match the current training data or it's "
                             "ignored and re-extracted fresh.")
    args = parser.parse_args()

    only = set(s.strip() for s in args.only.split(",")) if args.only else None
    experiments = discover_experiments(args.experiments_dir, only=only)
    if not experiments:
        print(f"No experiments found in {args.experiments_dir}"
              + (f" matching {sorted(only)}" if only else ""))
        return

    overrides = {
        "train_fraction": args.train_fraction,
        "test_fraction": args.test_fraction,
        "max_predict_steps": args.max_predict_steps,
        "epochs": args.epochs,
    }

    print(f"Running {len(experiments)} experiment(s): {[n for n, _ in experiments]}")
    rows = []
    for name, cfg in experiments:
        try:
            rows.append(run_one(name, cfg, args.results_dir, overrides, latent_cache_root=args.latent_cache_root))
        except Exception as exc:  # keep going so one failure doesn't sink the batch
            print(f"\n[{name}] FAILED: {exc}")
            traceback.print_exc()
            rows.append({"name": name, "status": "FAILED", "error": repr(exc)})

    write_summary(rows, args.results_dir)


if __name__ == "__main__":
    main()
