"""Stage 1: run all models on both horizons and write Results/aggregated_ddnn_comparison.pkl.

Usage (repository root, inside .venv):
    python run_comparative_experiments.py --fast-debug          # minutes; writes Results/fast_debug/
    python run_comparative_experiments.py --n-jobs 6            # full run (days of CPU time)
    python run_comparative_experiments.py --retrain-every 7     # cheaper NN rolling scheme

The run is resumable: re-launching the same command skips finished chunks.
Hyper-parameters: best trials of the paper's 12 Optuna studies in trialfiles/
(no re-tuning), applied to both horizons.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from ddnn_compare import aggregate, hyperparams, runner                 # noqa: E402
from ddnn_compare.config import AGGREGATE_NAME, HORIZONS, N_INSAMPLE, N_OOS, RunSettings, results_dir  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast-debug", action="store_true",
                    help="cheap settings (sparse retraining, few epochs) with full output shapes")
    ap.add_argument("--horizon", choices=list(HORIZONS), action="append", help="default: both")
    ap.add_argument("--n-jobs", type=int, default=None, help="worker processes (default 4)")
    ap.add_argument("--tf-threads", type=int, default=None, help="TensorFlow threads per worker (default 2)")
    ap.add_argument("--retrain-every", type=int, default=None, help="NN refit frequency in days")
    ap.add_argument("--lear-retrain-every", type=int, default=None, help="LEAR refit frequency in days")
    ap.add_argument("--qra-refit-every", type=int, default=None, help="QRA/QRM refit frequency in days")
    ap.add_argument("--chunk-days", type=int, default=None, help="checkpoint granularity in days")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--reset", action="store_true", help="delete existing checkpoints of this mode first")
    return ap.parse_args(argv)


def make_settings(args) -> RunSettings:
    s = RunSettings.fast_debug() if args.fast_debug else RunSettings()
    for name in ("n_jobs", "tf_threads", "retrain_every", "lear_retrain_every", "qra_refit_every",
                 "chunk_days", "epochs"):
        v = getattr(args, name)
        if v is not None:
            setattr(s, name, v)
    return s


def main(argv=None) -> None:
    args = parse_args(argv)
    settings = make_settings(args)
    horizons = args.horizon or list(HORIZONS)
    out_dir = results_dir(settings)
    print(f"Mode: {settings.tag} | horizons: {horizons} | output: {out_dir}")
    print(f"Settings: {settings.to_dict()}", flush=True)
    runner.check_settings(settings, args.reset)
    t0 = time.time()

    hp = hyperparams.load_all()
    for fam, runs in hp.items():
        print(f"  {fam}: best validation losses " +
              ", ".join(f"run{r}={v['best_value']:.3f}" for r, v in runs.items()))

    failed = runner.run_tasks(runner.stage1_tasks(horizons, hp, settings), settings, "stage1 LEAR+NN")
    if failed:
        sys.exit(f"{len(failed)} stage-1 tasks failed; fix and re-run (finished chunks are kept).")

    stage2 = []
    for h in horizons:
        point_oos = aggregate.oos_point_forecasts(settings, h)
        actuals = runner.get_data(h).Y[N_INSAMPLE:N_INSAMPLE + N_OOS]
        stage2 += runner.stage2_tasks(h, point_oos, actuals, settings)
    failed = runner.run_tasks(stage2, settings, "stage2 QRA/QRM")
    if failed:
        sys.exit(f"{len(failed)} stage-2 tasks failed; fix and re-run.")

    print("Aggregating ...", flush=True)
    artifact = aggregate.build_artifact(settings, horizons, hp)
    path = out_dir / AGGREGATE_NAME
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB) in {(time.time() - t0) / 60:.1f} min")
    for h, res in artifact["horizons"].items():
        print(f"\n=== {h} ===")
        print(res["metrics_table"].round(3).to_string())


if __name__ == "__main__":
    main()
