"""Task orchestration with on-disk checkpoints (resume after interruption).

Work is cut into small tasks (one model/run x one block of OOS days). Blocks are
aligned to the re-fit calendar, so with chunk_days == retrain_every every NN task
is exactly one fit. Every finished task writes
`Results/<tag>/checkpoints/<horizon>/<model>/chunk_XXXX.npz` atomically; on
restart, existing chunks are skipped. All tasks share one process pool.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import models
from .config import (FAMILY_DAYS, HORIZONS, LEAR_WINDOWS, N_OOS, N_TEST, NN_FAMILIES, RUNS, RunSettings,
                     results_dir)
from .features import DailyData, build_daily, load_hourly

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")

_DATA: dict[str, DailyData] = {}


def limit_threads(n: int = 1) -> None:
    """Cap BLAS/OpenMP/TF threads. Must run before numpy/TF are imported to be fully effective."""
    for k in THREAD_ENV:
        os.environ[k] = str(n)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"


def _worker_init(threads: int) -> None:
    limit_threads(threads)


def get_data(horizon: str) -> DailyData:
    if horizon not in _DATA:
        _DATA[horizon] = build_daily(load_hourly(HORIZONS[horizon]))
    return _DATA[horizon]


@dataclass
class Task:
    kind: str           # "nn" | "lear" | "qra"
    horizon: str
    model: str          # checkpoint key, e.g. "DDNN-JSU-2", "LEAR-84", "DNN-QRA"
    days: list          # OOS indices (nn/lear) or test indices (qra)
    payload: dict       # kind-specific arguments

    @property
    def start(self) -> int:
        return self.days[0]


def checkpoint_dir(settings: RunSettings) -> Path:
    return results_dir(settings) / "checkpoints"


def chunk_path(settings: RunSettings, task: Task) -> Path:
    return checkpoint_dir(settings) / task.horizon / task.model / f"chunk_{task.start:04d}.npz"


def _save_atomic(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def execute(task: Task, settings: RunSettings) -> tuple[str, float]:
    """Worker entry point (top-level so it pickles under Windows spawn)."""
    t0 = time.time()
    path = chunk_path(settings, task)
    if path.exists():
        return str(path), 0.0
    data = get_data(task.horizon)
    if task.kind == "lear":
        arrays = {"point": models.lear_forecast(data, task.payload["cal"], task.days, settings)}
    elif task.kind == "nn":
        arrays = models.nn_forecast(data, task.payload["params"], task.payload["dist"], task.days, settings,
                                    task.payload["seed"])
    elif task.kind == "qra":
        p = task.payload
        arrays = {"quantiles": models.qra_forecast(p["forecasts"], p["actuals"], task.days, settings)}
    else:
        raise ValueError(task.kind)
    arrays["days"] = np.asarray(task.days)
    _save_atomic(path, arrays)
    return str(path), time.time() - t0


def _blocks(lo: int, hi: int, size: int) -> list[list[int]]:
    """Days lo..hi-1 cut on the global grid 0, size, 2*size, ... (blocks never straddle a re-fit)."""
    out = []
    for s in range(0, hi, size):
        days = [d for d in range(s, min(s + size, hi)) if d >= lo]
        if days:
            out.append(days)
    return out


def nn_seed(settings: RunSettings, horizon: str, family: str, run: int) -> int:
    """Stable seed: independent of which horizons/models are selected in a given call."""
    h_i = list(HORIZONS).index(horizon)
    f_i = list(NN_FAMILIES).index(family)
    return settings.base_seed + 1_000_000 * h_i + 100_000 * f_i + 10_000 * run


def stage1_tasks(horizons, hp: dict, settings: RunSettings, groups) -> list[Task]:
    tasks = []
    for horizon in horizons:
        for family, (_, dist) in NN_FAMILIES.items():
            if family not in groups:
                continue
            lo, hi = FAMILY_DAYS[family]
            for run in RUNS:
                for days in _blocks(lo, hi, settings.chunk_days):
                    tasks.append(Task("nn", horizon, f"{family}-{run}", days,
                                      {"params": hp[family][run]["params"], "dist": dist,
                                       "seed": nn_seed(settings, horizon, family, run)}))
        if "LEAR" in groups:
            lo, hi = FAMILY_DAYS["LEAR"]
            for cal in LEAR_WINDOWS:
                for days in _blocks(lo, hi, settings.chunk_days):
                    tasks.append(Task("lear", horizon, f"LEAR-{cal}", days, {"cal": cal}))
    # longest tasks first (NN before LEAR, 2020-2026 needs more epochs) -> better packing
    order = {"nn": 0, "lear": 1}
    tasks.sort(key=lambda t: (order[t.kind], t.horizon != "2020-2026"))
    return tasks


def qra_inputs(groups) -> list[tuple[str, list[str]]]:
    """(QRA model, point-forecast inputs) for the selected groups."""
    out = []
    if "LEAR" in groups:
        out.append(("LEAR-QRA", [f"LEAR-{c}" for c in LEAR_WINDOWS]))
    if "DNN" in groups:
        out.append(("DNN-QRA", [f"DNN-{r}" for r in RUNS]))
    return out


def stage2_tasks(horizon: str, point_oos: dict, actuals_oos: np.ndarray, settings: RunSettings, groups) -> list[Task]:
    """QRA on LEAR and/or DNN point forecasts (needs stage-1 output)."""
    tasks = []
    for name, keys in qra_inputs(groups):
        F = np.stack([point_oos[k] for k in keys], axis=-1)
        for days in _blocks(0, N_TEST, settings.chunk_days):
            tasks.append(Task("qra", horizon, name, days, {"forecasts": F, "actuals": actuals_oos}))
    return tasks


def run_tasks(tasks: list[Task], settings: RunSettings, label: str) -> list[Task]:
    """Run pending tasks in one shared process pool; returns the list of failed tasks."""
    pending = [t for t in tasks if not chunk_path(settings, t).exists()]
    print(f"[{label}] {len(tasks)} tasks, {len(tasks) - len(pending)} already checkpointed, "
          f"{len(pending)} to run on {settings.n_jobs} worker(s) x {settings.tf_threads} thread(s)", flush=True)
    if not pending:
        return []
    failed, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=settings.n_jobs, initializer=_worker_init,
                             initargs=(settings.tf_threads,)) as pool:
        futures = {pool.submit(execute, t, settings): t for t in pending}
        for fut in as_completed(futures):
            t = futures[fut]
            done += 1
            try:
                _, secs = fut.result()
                elapsed = time.time() - t0
                eta = elapsed / done * (len(pending) - done)
                print(f"[{label}] {done}/{len(pending)} {t.horizon} {t.model} days {t.days[0]}-{t.days[-1]} "
                      f"({secs:.0f}s) | elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min", flush=True)
            except Exception:  # keep going; report at the end
                failed.append(t)
                print(f"[{label}] FAILED {t.horizon} {t.model} days {t.days[0]}-{t.days[-1]}\n"
                      f"{traceback.format_exc()}", flush=True)
    return failed


def load_model(settings: RunSettings, horizon: str, model: str, expected_days) -> dict[str, np.ndarray]:
    """Concatenate all chunks of one model into arrays ordered by day.

    Keys starting with 'fit_' are per-fit diagnostics (one row per fit, ordered by anchor day).
    """
    expected = np.asarray(expected_days)
    folder = checkpoint_dir(settings) / horizon / model
    files = sorted(folder.glob("chunk_*.npz"))
    parts = [dict(np.load(f)) for f in files if not f.name.endswith(".tmp.npz")]
    if not parts:
        raise FileNotFoundError(f"No checkpoints for {horizon}/{model} in {folder}")
    days = np.concatenate([p.pop("days") for p in parts])
    order = np.argsort(days)
    if not np.array_equal(days[order], expected):
        raise RuntimeError(f"{horizon}/{model}: incomplete chunks ({len(days)}/{len(expected)} days)")
    out = {}
    for k in parts[0]:
        if k.startswith("fit_"):
            continue
        out[k] = np.concatenate([p[k] for p in parts])[order]
    fit_keys = [k for k in parts[0] if k.startswith("fit_")]
    if fit_keys:
        fit = {k: np.concatenate([p[k] for p in parts]) for k in fit_keys}
        o = np.argsort(fit["fit_anchor"], kind="stable")
        out.update({k: v[o] for k, v in fit.items()})
    return out


STAMP_EXCLUDE = ("n_jobs", "tf_threads")   # do not change results


def check_settings(settings: RunSettings, reset: bool) -> None:
    """Refuse to mix checkpoints produced with different settings in the same tag."""
    root = checkpoint_dir(settings)
    if reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = root / "settings.json"
    current = {k: v for k, v in settings.to_dict().items() if k not in STAMP_EXCLUDE}
    if stamp.exists():
        previous = json.loads(stamp.read_text())
        if previous != current:
            diff = {k: (previous.get(k), v) for k, v in current.items() if previous.get(k) != v}
            raise SystemExit(f"Checkpoints in {root} were made with different settings {diff}. "
                             "Use the same flags, another --tag, or --reset to discard them.")
    else:
        stamp.write_text(json.dumps(current, indent=2))
