"""
Continue training a previously-saved flow-map network (own_models.networks.sde.SDEResNet)
on a dataset -- either more of the same data it was originally trained on, or a
different dataset entirely (fine-tuning).

due.models.ODE saves the network as a full pickled object (torch.save(self.mynet, ...)),
whenever validation (or training, if no validation split) loss improves -- so the
loaded checkpoint is the BEST epoch seen, not necessarily the last one. Loading it and
handing it to a fresh due.models.ODE(...) picks optimization back up from those learned
weights; the optimizer and LR scheduler are *not* resumed (due.models.ODE always builds
them fresh), so treat this as "start from pretrained weights", not "resume an
interrupted run bit-for-bit". model.save_hist() also only reflects the epochs run in
THIS continuation, not the original training history.

Usage:
    python continue_training.py --model ../sde_experiments/results/OU/model --config experiments/OU/config.yaml
    python continue_training.py --model results/OU/model --config experiments/OU/config.yaml \
        --epochs 20 --learning-rate 0.0001 --save-path results/OU_continued

    # Fine-tune on a different dataset than the model was originally trained on:
    python continue_training.py --model results/OU/model --config experiments/GBM/config.yaml \
        --save-path results/OU_finetuned_on_GBM

    # Point at a standalone sde_examples config (which has no data.train_file/test_file):
    python continue_training.py --model ../sde_examples/OU/sde_diffusion_output/model \
        --config ../sde_examples/OU/config.yaml \
        --train-file simulated_data/1d_OU_process_train.mat \
        --test-file simulated_data/1d_OU_process_test.mat
"""

import os
import sys
import shutil
import argparse

import numpy as np
import torch

# own_models is not pip-installed (unlike due); make the repo root importable
# so `import own_models` resolves regardless of the current working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import due
import own_models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                         help="Path to a saved model checkpoint (e.g. results/OU/model).")
    parser.add_argument("--config", required=True,
                         help="Path to a config.yaml (same format as sde_examples/sde_experiments configs).")
    parser.add_argument("--train-file", default=None,
                         help="Training .mat path, relative to repo root. Overrides/supplies data.train_file "
                              "if the config doesn't define one (e.g. standalone sde_examples configs).")
    parser.add_argument("--test-file", default=None,
                         help="Test .mat path, relative to repo root. Overrides/supplies data.test_file.")
    parser.add_argument("--save-path", default=None,
                         help="Where to write continued-training outputs. Defaults to the "
                              "--model checkpoint's own directory + '_continued', so a diverging "
                              "continuation run can't overwrite the loaded checkpoint. Pass "
                              "--overwrite to write back into that same directory instead.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Write outputs into the --model checkpoint's own directory instead "
                              "of a new '_continued' directory (may overwrite the loaded checkpoint).")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override training epochs for this continuation run.")
    parser.add_argument("--learning-rate", type=float, default=None,
                         help="Override learning rate for this continuation run (fine-tuning "
                              "commonly uses a lower LR than the original training run).")
    parser.add_argument("--train-fraction", type=float, default=None,
                         help="Use only this fraction of the (re-sampled) training bursts. "
                              "Defaults to the config's own data.train_fraction (same as "
                              "run_experiments.py), NOT to the full dataset -- so a checkpoint "
                              "produced by run_experiments.py from this same config lines up "
                              "with its cached latents (see --cache-latents) without having to "
                              "repeat the value manually.")
    parser.add_argument("--cache-latents", action="store_true",
                         help="Reuse a previously cached extracted_latents.pt instead of "
                              "re-solving the reverse ODE. Looked up in --latent-cache if given, "
                              "else the --model checkpoint's own directory (where the original "
                              "training run's cache lives), falling back to a fresh extraction "
                              "if none is found there or if its row count doesn't match the "
                              "current training data (e.g. after a different --train-fraction).")
    parser.add_argument("--latent-cache", default=None,
                         help="Directory OR direct path to an extracted_latents.pt file to reuse, "
                              "overriding the --cache-latents default of the --model checkpoint's "
                              "directory. Passing this implies --cache-latents (no need to pass both).")
    args = parser.parse_args()

    if args.latent_cache is not None:
        args.cache_latents = True

    conf_data, conf_net, conf_train = own_models.utils.read_sde_config(args.config)

    checkpoint_dir = os.path.dirname(os.path.abspath(args.model))
    if args.overwrite:
        save_path = checkpoint_dir
    else:
        save_path = args.save_path or (checkpoint_dir.rstrip("/\\") + "_continued")
    os.makedirs(save_path, exist_ok=True)
    conf_train["save_path"] = save_path

    if args.epochs is not None:
        conf_train["epochs"] = args.epochs
    if args.learning_rate is not None:
        conf_train["learning_rate"] = args.learning_rate
    elif conf_train.get("scheduler") in ("cosine", "Cosine"):
        # due.models.ODE always builds a fresh optimizer AND a fresh LR scheduler --
        # neither is restored from the loaded checkpoint. With a cosine schedule, the
        # original run's LR had annealed from this config's learning_rate down toward 0
        # by its final (best) epoch -- restarting that curve from the top on an
        # already-converged model reliably causes a large but temporary loss spike in
        # the first few epochs (recovers on its own, but --learning-rate with a much
        # smaller value avoids it; verified with real runs: full LR gave a 28x spike at
        # epoch 1, 50x-lower LR gave none).
        print(f"NOTE: no --learning-rate override given, so this run starts at the config's "
              f"full learning_rate ({conf_train['learning_rate']}) with a freshly-restarted "
              f"cosine schedule -- since the original training's LR had annealed down from "
              f"there, expect a temporary loss spike in the first few epochs. Pass "
              f"--learning-rate with a much smaller value to avoid it.")

    # Accept either the checkpoint file itself or the experiment folder that
    # contains it (due.models.ODE saves the checkpoint as "<save_path>/model"),
    # so `--model results/Foo` and `--model results/Foo/model` both work.
    model_path = args.model
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model")
    print(f"Loading model from {model_path} ...")
    mynet = torch.load(model_path, weights_only=False)
    # The model's OWN vmin/vmax (learned from whatever data it was originally trained
    # on) are reused below, not recomputed from the freshly loaded data -- the network's
    # weights and residual connection are calibrated to that specific normalization, so
    # changing it would silently invalidate everything already learned.
    vmin = mynet.vmin.cpu().numpy()
    vmax = mynet.vmax.cpu().numpy()
    print(f"Loaded network with {mynet.count_params()} trainable parameters.")

    train_file_rel = args.train_file or conf_data.get("train_file")
    test_file_rel = args.test_file or conf_data.get("test_file")
    if train_file_rel is None or test_file_rel is None:
        raise ValueError(
            "No train/test file resolved. Either use a config.yaml with data.train_file/"
            "data.test_file (see sde_experiments/experiments/*/config.yaml), or pass "
            "--train-file/--test-file explicitly."
        )
    train_file = os.path.join(_REPO_ROOT, train_file_rel)
    test_file = os.path.join(_REPO_ROOT, test_file_rel)

    loader = own_models.datasets.sde.sde_dataset(conf_data)
    trainX, trainY, test_set, fresh_vmin, fresh_vmax = loader.load(train_file, test_file)

    # Sanity check: warn if the newly loaded data's own scale differs substantially from
    # the model's stored normalization bounds, since the model's bounds -- not these
    # freshly computed ones -- are what actually get used below.
    rel_diff = np.abs(fresh_vmax - vmax) / (np.abs(vmax) + 1e-8)
    if np.any(rel_diff > 0.1):
        print("WARNING: the freshly loaded data's min/max differs from the model's stored "
              "normalization bounds by more than 10%. Continuing training will still use the "
              "model's ORIGINAL vmin/vmax (required to stay consistent with what it already "
              "learned), so values outside that original range will be clipped to [-1, 1] "
              "during normalization -- some of the new data may be poorly represented.")

    # Default to the config's own data.train_fraction (matching run_experiments.py's
    # behavior) rather than the full dataset, so a checkpoint produced by
    # run_experiments.py from this same config uses the same amount of data here --
    # otherwise its cached latents would never match (see --cache-latents below).
    train_fraction = args.train_fraction if args.train_fraction is not None else conf_data.get("train_fraction", 1.0)
    n_train = max(1, int(train_fraction * trainX.shape[0]))
    trainX, trainY = trainX[:n_train], trainY[:n_train]
    print(f"Using {n_train}/{trainX.shape[0]} training bursts (train_fraction={train_fraction}).")

    # generate_labeled_data only ever looks for extracted_latents.pt inside conf_train's
    # save_path -- which by default is a fresh '_continued' directory with nothing in it
    # yet, so --cache-latents would silently find nothing on a first run even though the
    # original training run's cache exists right next to the checkpoint. Look there (or
    # --latent-cache, if given) instead, and validate the row count before trusting it,
    # since a cache from a different --train-fraction or dataset won't match trainX here.
    if args.cache_latents:
        resolved_latent_file = os.path.join(save_path, "extracted_latents.pt")
        if not os.path.exists(resolved_latent_file):
            if args.latent_cache and os.path.isfile(args.latent_cache):
                # --latent-cache pointed directly at the .pt file, not its directory.
                candidate = args.latent_cache
            else:
                latent_cache_dir = args.latent_cache or checkpoint_dir
                candidate = os.path.join(latent_cache_dir, "extracted_latents.pt")
            if os.path.exists(candidate):
                cached_Z1, _ = torch.load(candidate, map_location="cpu")
                if cached_Z1.shape[0] == trainX.shape[0]:
                    print(f"Reusing cached latents from {candidate} ({cached_Z1.shape[0]} rows, matches).")
                    shutil.copy(candidate, resolved_latent_file)
                else:
                    print(f"Found a latent cache at {candidate}, but it has {cached_Z1.shape[0]} rows "
                          f"vs. {trainX.shape[0]} in the current training data (likely a different "
                          f"--train-fraction or dataset) -- ignoring it and re-extracting.")
                    args.cache_latents = False
            else:
                print(f"--cache-latents was set but no extracted_latents.pt found at {candidate} -- "
                      f"extracting fresh (it will be cached at {save_path} for next time).")
                args.cache_latents = False
    conf_train["cache_latents"] = args.cache_latents

    # Generate labeled data using the MODEL's own vmin/vmax (see above), not the freshly
    # computed fresh_vmin/fresh_vmax.
    trainX_aug, trainY_synth = own_models.models.sde_diffusion.generate_labeled_data(
        trainX, trainY, conf_train, vmin, vmax)

    # Continue training the loaded network: fresh optimizer/scheduler state, but starting
    # from its already-learned weights.
    model = due.models.ODE(trainX_aug, trainY_synth, mynet, conf_train)
    model.train()
    model.save_hist()

    # Predict + evaluate, exactly like the standalone examples. 1D spaghetti plots
    # are uninformative once the state is multi-dimensional, so dispatch to the
    # phase-space/marginal/spectral evaluation for problem_dim > 1 (see run_experiments.py).
    steps_to_predict = test_set.shape[-1] - (conf_data.get("memory", 0) + 1)
    pred = mynet.predict(test_set[..., :conf_data.get("memory", 0) + 1], steps_to_predict,
                          device=conf_train["device"])
    if pred.shape[1] > 1:
        own_models.utils.sde_evaluate_multidim(pred, test_set, save_path=save_path, dt=conf_data.get("dt", 0.01))
    else:
        own_models.utils.sde_evaluate(pred, test_set, save_path=save_path, dt=conf_data.get("dt", 0.01))

    print(f"\nContinued-training outputs written to: {save_path}")


if __name__ == "__main__":
    main()
