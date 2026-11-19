"""Task orchestration with on-disk checkpoints (resume after interruption).

Work is cut into small tasks (one model/run x one chunk of OOS days). Every
finished task writes `<results>/checkpoints/<horizon>/<model>/chunk_XXXX.npz`
atomically; on restart, existing chunks are skipped.
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
from .config import (HORIZONS, LEAR_WINDOWS, N_OOS, N_RUNS, N_TEST, NN_FAMILIES, RunSettings,
                     results_dir)
from .features import DailyData, build_daily, load_hourly

_DATA: dict[str, DailyData] = {}


def get_data(horizon: str) -> DailyData:
    if horizon not in _DATA:
        _DATA[horizon] = build_daily(load_hourly(HORIZONS[horizon]))
    return _DATA[horizon]


@dataclass
class Task:
    kind: str           # "nn" | "lear" | "qra"
    horizon: str
    model: str          # checkpoint key, e.g. "DDNN-JSU-3", "LEAR-84", "DNN-QRM"
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
        arrays = {"quantiles": models.qra_forecast(p["forecasts"], p["actuals"], task.days, settings,
                                                   p["average"])}
    else:
        raise ValueError(task.kind)
    arrays["days"] = np.asarray(task.days)
    _save_atomic(path, arrays)
    return str(path), time.time() - t0


def _chunks(n: int, size: int) -> list[list[int]]:
    return [list(range(s, min(s + size, n))) for s in range(0, n, size)]


def stage1_tasks(horizons, hp: dict, settings: RunSettings) -> list[Task]:
    tasks = []
    for h_i, horizon in enumerate(horizons):
        for f_i, (family, (_, dist)) in enumerate(NN_FAMILIES.items()):
            for run in range(1, N_RUNS + 1):
                seed = settings.base_seed + 1_000_000 * h_i + 100_000 * f_i + 10_000 * run
                for days in _chunks(N_OOS, settings.chunk_days):
                    tasks.append(Task("nn", horizon, f"{family}-{run}", days,
                                      {"params": hp[family][run]["params"], "dist": dist, "seed": seed}))
        for cal in LEAR_WINDOWS:
            for days in _chunks(N_OOS, settings.chunk_days):
                tasks.append(Task("lear", horizon, f"LEAR-{cal}", days, {"cal": cal}))
    return tasks


def stage2_tasks(horizon: str, point_oos: dict, actuals_oos: np.ndarray, settings: RunSettings) -> list[Task]:
    """QRA/QRM on LEAR and DNN point forecasts (needs stage-1 output)."""
    tasks = []
    chunk = max(settings.chunk_days, settings.qra_refit_every)
    for base, keys in (("LEAR", [f"LEAR-{c}" for c in LEAR_WINDOWS]), ("DNN", [f"DNN-{r}" for r in range(1, 5)])):
        F = np.stack([point_oos[k] for k in keys], axis=-1)
        for average, suffix in ((False, "QRA"), (True, "QRM")):
            for days in _chunks(N_TEST, chunk):
                tasks.append(Task("qra", horizon, f"{base}-{suffix}", days,
                                  {"forecasts": F, "actuals": actuals_oos, "average": average}))
    return tasks


def run_tasks(tasks: list[Task], settings: RunSettings, label: str) -> list[Task]:
    """Run pending tasks in a process pool; returns the list of failed tasks."""
    pending = [t for t in tasks if not chunk_path(settings, t).exists()]
    print(f"[{label}] {len(tasks)} tasks, {len(tasks) - len(pending)} already checkpointed, "
          f"{len(pending)} to run on {settings.n_jobs} worker(s)", flush=True)
    if not pending:
        return []
    failed, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=settings.n_jobs) as pool:
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


def load_model(settings: RunSettings, horizon: str, model: str, n_expected: int) -> dict[str, np.ndarray]:
    """Concatenate all chunks of one model into arrays ordered by day."""
    folder = checkpoint_dir(settings) / horizon / model
    files = sorted(folder.glob("chunk_*.npz"))
    parts = [dict(np.load(f)) for f in files if not f.name.endswith(".tmp.npz")]
    if not parts:
        raise FileNotFoundError(f"No checkpoints for {horizon}/{model}")
    days = np.concatenate([p.pop("days") for p in parts])
    order = np.argsort(days)
    if not np.array_equal(days[order], np.arange(n_expected)):
        raise RuntimeError(f"{horizon}/{model}: incomplete chunks ({len(days)}/{n_expected} days)")
    return {k: np.concatenate([p[k] for p in parts])[order] for k in parts[0]}


def check_settings(settings: RunSettings, reset: bool) -> None:
    """Refuse to mix checkpoints produced with different settings."""
    root = checkpoint_dir(settings)
    if reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = root / "settings.json"
    current = {k: v for k, v in settings.to_dict().items() if k not in ("n_jobs", "tf_threads")}
    if stamp.exists():
        previous = json.loads(stamp.read_text())
        if previous != current:
            diff = {k: (previous.get(k), v) for k, v in current.items() if previous.get(k) != v}
            raise SystemExit(f"Checkpoints in {root} were made with different settings {diff}. "
                             "Use the same flags, or --reset to discard them.")
    else:
        stamp.write_text(json.dumps(current, indent=2))
