"""Run the selected models on both horizons and write Results/<tag>/aggregated_ddnn_comparison.pkl.

Usage (repository root, inside .venv):
    python run_comparative_experiments.py --fast-debug --tag smoke_test   # minutes, numbers not meaningful
    python run_comparative_experiments.py                                 # full run: DNN + DDNN-JSU, ~10-11 h
    python run_comparative_experiments.py --models DNN DDNN-JSU LEAR      # later: add LEAR to the same tag

Defaults: naive + DNN-1/2 (DNN-Ens, DNN-QRA) + DDNN-JSU-1/2 (pEns, qEns); NN re-fit every 7 days
on the trailing 1,449 usable days with daily inference; QRA re-fit daily; 4 workers x 1 thread.
The run is resumable: re-launching the same command skips finished blocks (Ctrl+C is safe).
Hyper-parameters: best trials of runs 1-2 of the paper's Optuna studies in trialfiles/ (no re-tuning).
"""
from __future__ import annotations

import os

# Thread caps must be set before numpy / TensorFlow are imported (workers inherit them).
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse      # noqa: E402
import math          # noqa: E402
import pickle        # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402

from ddnn_compare import aggregate, hyperparams, runner                 # noqa: E402
from ddnn_compare.config import (AGGREGATE_NAME, DEFAULT_GROUPS, HORIZONS, MODEL_GROUPS, N_INSAMPLE,  # noqa: E402
                                 N_OOS, RunSettings, results_dir)


class _Tee:
    """Mirror stdout into Results/<tag>/run.log (appended across resumed runs)."""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast-debug", action="store_true",
                    help="cheap settings (sparse retraining, 5 epochs) with full output shapes")
    ap.add_argument("--tag", default=None, help="results folder Results/<tag>/ (default: neural_w7, or fast_debug)")
    ap.add_argument("--horizon", choices=list(HORIZONS), action="append", help="default: both")
    ap.add_argument("--models", nargs="+", choices=MODEL_GROUPS, default=list(DEFAULT_GROUPS),
                    help="model groups (default: DNN DDNN-JSU; add LEAR to include LEAR + LEAR-QRA)")
    ap.add_argument("--max-workers", "--n-jobs", dest="n_jobs", type=int, default=None,
                    help="worker processes in the shared queue (default 4)")
    ap.add_argument("--tf-threads", type=int, default=None, help="threads per worker (default 1)")
    ap.add_argument("--retrain-every", type=int, default=None, help="NN re-fit frequency in days (default 7)")
    ap.add_argument("--lear-retrain-every", type=int, default=None, help="LEAR re-fit frequency in days (default 1)")
    ap.add_argument("--qra-refit-every", type=int, default=None, help="QRA re-fit frequency in days (default 1)")
    ap.add_argument("--lear-method", choices=("ic", "lassocv"), default=None,
                    help="LEAR estimator: LassoLarsIC (default) or LassoCV as in lear.py")
    ap.add_argument("--chunk-days", type=int, default=None,
                    help="checkpoint block size (default: lcm of the re-fit frequencies)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--reset", action="store_true", help="delete existing checkpoints of this tag first")
    return ap.parse_args(argv)


def make_settings(args) -> RunSettings:
    s = RunSettings.fast_debug() if args.fast_debug else RunSettings()
    for name in ("n_jobs", "tf_threads", "retrain_every", "lear_retrain_every", "qra_refit_every",
                 "lear_method", "chunk_days", "epochs", "patience", "tag"):
        v = getattr(args, name)
        if v is not None:
            setattr(s, name, v)
    if args.chunk_days is None and not args.fast_debug:
        s.chunk_days = math.lcm(s.retrain_every, s.lear_retrain_every, s.qra_refit_every)
    s.validate()
    return s


def main(argv=None) -> None:
    args = parse_args(argv)
    settings = make_settings(args)
    groups = [g for g in MODEL_GROUPS if g in args.models]
    horizons = args.horizon or list(HORIZONS)
    out_dir = results_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / "run.log")
    print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    print(f"Tag: {settings.tag} | models: {groups} | horizons: {horizons} | output: {out_dir}")
    print(f"Settings: {settings.to_dict()}", flush=True)
    runner.check_settings(settings, args.reset)
    t0 = time.time()

    hp = hyperparams.load_all()
    for fam, runs in hp.items():
        print(f"  {fam}: best validation losses " +
              ", ".join(f"run{r}={v['best_value']:.3f}" for r, v in runs.items()))

    failed = runner.run_tasks(runner.stage1_tasks(horizons, hp, settings, groups), settings, "stage1 NN/LEAR")
    if failed:
        sys.exit(f"{len(failed)} stage-1 tasks failed; fix and re-run (finished blocks are kept).")

    stage2 = []
    for h in horizons:
        point_oos = aggregate.oos_point_forecasts(settings, h, groups)
        actuals = runner.get_data(h).Y[N_INSAMPLE:N_INSAMPLE + N_OOS]
        stage2 += runner.stage2_tasks(h, point_oos, actuals, settings, groups)
    failed = runner.run_tasks(stage2, settings, "stage2 QRA")
    if failed:
        sys.exit(f"{len(failed)} stage-2 tasks failed; fix and re-run.")

    print("Aggregating ...", flush=True)
    artifact = aggregate.build_artifact(settings, horizons, hp, groups)
    path = out_dir / AGGREGATE_NAME
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB) in {(time.time() - t0) / 60:.1f} min")
    for h, res in artifact["horizons"].items():
        print(f"\n=== {h} ===")
        print(res["metrics_table"].round(3).to_string())
        d = res["fit_diagnostics"]
        if len(d):
            print("\nNN fits: " + ", ".join(
                f"{m}: {len(g)} fits, epochs median {g['epochs'].median():.0f}, hit cap {g['hit_cap'].mean():.0%}"
                for m, g in d.groupby("model")))


if __name__ == "__main__":
    main()
