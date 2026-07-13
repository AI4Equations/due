# sde_experiments

Batch runner for the training-free conditional diffusion SDE model. Trains and
validates a sequence of model/dataset pairs and produces a comparison summary.

## Layout

```
sde_experiments/
├── run_experiments.py     # the runner
├── experiments/           # one self-contained config.yaml per model/dataset pair
│   ├── OU/config.yaml
│   ├── GBM/config.yaml
│   ├── DoubleWell/config.yaml
│   └── ExpNoise/config.yaml
└── results/               # created at runtime: per-experiment outputs + summary.csv
```

Each `experiments/<name>/config.yaml` is the same format as the standalone
`sde_examples/` configs, plus a few batch-runner keys under `data`:

| key                 | meaning                                                        |
|---------------------|----------------------------------------------------------------|
| `train_file`        | training `.mat`, path relative to the repo root                |
| `test_file`         | test `.mat`, path relative to the repo root                    |
| `train_fraction`    | fraction of sampled training bursts to actually train on       |
| `test_fraction`     | fraction of test trajectories to evaluate/plot on              |
| `max_predict_steps` | cap on autoregressive rollout length (0 = full test horizon)   |

`train_fraction`, `test_fraction`, and `max_predict_steps` let you dial each
dataset's cost independently — important because some `.mat` files are large
(DoubleWell/ExpNoise are ~80–200 MB with very long test horizons).

## Usage

Run from inside this folder:

```bash
python run_experiments.py                        # every experiment folder
python run_experiments.py --only GBM,OU          # a subset
python run_experiments.py --train-fraction 0.02  # override all train fractions
python run_experiments.py --epochs 10            # quick smoke run
```

To add a new pair, drop a new folder under `experiments/` with a `config.yaml`.

## Output

Per experiment, under `results/<name>/`: `sde_evaluation.png`, training-history
plot/CSV, `distribution_error.csv`, the saved model, and `extracted_latents.pt`.
Across experiments: `results/summary.csv` plus a printed table (final train/val
loss, ensemble mean/std error, and wall-clock timings).
