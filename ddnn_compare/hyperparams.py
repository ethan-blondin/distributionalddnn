"""Read the paper's best Optuna trials directly from trialfiles/ (read-only SQLite).

The attached studies were written by Optuna 2.8; recent Optuna refuses to open
them without an in-place schema upgrade, so we decode the tables ourselves and
never modify the files.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config import NN_FAMILIES, RUNS, TRIAL_DIR
from .features import FEATURE_GROUPS


def _decode(value: float, dist_json: str):
    dist = json.loads(dist_json)
    name, attrs = dist["name"], dist["attributes"]
    if name == "CategoricalDistribution":
        return attrs["choices"][int(value)]
    if name.startswith("Int"):
        return int(value)
    return float(value)


def best_trial(path: Path) -> dict:
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        direction = con.execute("SELECT direction FROM study_directions").fetchone()[0]
        order = "ASC" if direction == "MINIMIZE" else "DESC"
        trial_id, number, value = con.execute(
            "SELECT t.trial_id, t.number, v.value FROM trials t JOIN trial_values v ON t.trial_id = v.trial_id "
            f"WHERE t.state = 'COMPLETE' ORDER BY v.value {order} LIMIT 1").fetchone()
        n_complete = con.execute("SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'").fetchone()[0]
        params = {name: _decode(val, dj) for name, val, dj in con.execute(
            "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id = ?", (trial_id,))}
    finally:
        con.close()
    return {"params": params, "best_value": value, "trial_number": number, "n_complete_trials": n_complete,
            "source": path.name}


def load_all(trial_dir: Path = TRIAL_DIR) -> dict:
    """{family: {run: best_trial_dict}} for the families in NN_FAMILIES and the runs in RUNS."""
    out = {}
    for family, (stem, _) in NN_FAMILIES.items():
        out[family] = {}
        for run in RUNS:
            info = best_trial(Path(trial_dir) / f"FINAL_DE_selection_prob_{stem}{run}")
            info["feature_mask"] = {g: bool(info["params"].get(g, False)) for g in FEATURE_GROUPS}
            out[family][run] = info
    return out
