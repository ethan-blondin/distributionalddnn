"""Experiment configuration: horizons, window splits, model registry, run settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Datasets"
TRIAL_DIR = REPO_ROOT / "trialfiles"
RESULTS_DIR = REPO_ROOT / "Results"
AGGREGATE_NAME = "aggregated_ddnn_comparison.pkl"

# --------------------------------------------------------------------------- #
# Horizons (2,192 days = 52,608 h each)
# --------------------------------------------------------------------------- #
HORIZONS = {
    "2015-2020": DATA_DIR / "DE_dataset_2015-2020.csv",
    "2020-2026": DATA_DIR / "DE_dataset_2020-2026.csv",
}

# --------------------------------------------------------------------------- #
# Window splits (Marcjasz et al., 2022) - identical for both horizons
# --------------------------------------------------------------------------- #
N_DAYS = 2192
N_TRAIN = 1092            # in-sample training days (hyper-parameter tuning)
N_VAL = 364               # in-sample validation days
N_INSAMPLE = N_TRAIN + N_VAL   # 1,456 = rolling calibration window length
N_OOS = N_DAYS - N_INSAMPLE    # 736 out-of-sample days
N_QRA = 182               # rolling QRA/QRM calibration window (first 182 OOS days)
N_TEST = N_OOS - N_QRA    # 554 evaluation days
MAX_LAG = 7               # longest lag (D-7): first 7 days of a window have no features
NN_TRAIN_DAYS = N_INSAMPLE - MAX_LAG   # 1,449 training samples per rolling NN fit

TAUS = np.arange(1, 100) / 100.0          # 99 percentiles
LEAR_WINDOWS = (56, 84, 1092, 1456)
N_RUNS = 4

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
NN_FAMILIES = {            # family -> (trialfile stem, output distribution)
    "DNN": ("point", None),
    "DDNN-N": ("normal", "Normal"),
    "DDNN-JSU": ("jsu", "JSU"),
}
LEAR_MODELS = [f"LEAR-{w}" for w in LEAR_WINDOWS]
POINT_MODELS = LEAR_MODELS + ["LEAR-Ens"] + [f"DNN-{r}" for r in range(1, N_RUNS + 1)] + ["DNN-Ens"]
PROB_MODELS = (["naive", "LEAR-QRA", "LEAR-QRM", "DNN-QRA", "DNN-QRM"]
               + [f"DDNN-N-{r}" for r in range(1, N_RUNS + 1)] + ["DDNN-N-pEns", "DDNN-N-qEns"]
               + [f"DDNN-JSU-{r}" for r in range(1, N_RUNS + 1)] + ["DDNN-JSU-pEns", "DDNN-JSU-qEns"])
ALL_MODELS = POINT_MODELS + PROB_MODELS
DM_MODELS = ["naive", "LEAR-QRA", "LEAR-QRM", "DNN-QRA", "DNN-QRM",
             "DDNN-N-pEns", "DDNN-N-qEns", "DDNN-JSU-pEns", "DDNN-JSU-qEns"]
TABLE1_MODELS = ["LEAR-Ens", "DNN-Ens", "naive", "LEAR-QRA", "LEAR-QRM", "DNN-QRA", "DNN-QRM",
                 "DDNN-N-pEns", "DDNN-N-qEns", "DDNN-JSU-pEns", "DDNN-JSU-qEns"]

# Documented deviations from the published code in Python/ (all fixed here).
ORIGINAL_CODE_FLAWS = {
    "weekday_index": "X[d, 220] = data.index[d].weekday() uses hour d of the first day, not day d "
                     "(should be data.index[d*24]); the weekday feature was essentially constant/wrong.",
    "brent_rolling_slice": "lear.py / nn_rolling_pred.py read Brent from the full `data` frame instead of the "
                           "rolling window `df`, i.e. Brent always came from the first days of the dataset.",
    "dow_encoding": "a single integer weekday (0..6) was used instead of 7 one-hot day-of-week dummies "
                    "as described in the paper; inputs now 227 instead of 221.",
}


@dataclass
class RunSettings:
    """Compute settings. `fast_debug()` returns a cheap variant with identical output shapes."""
    retrain_every: int = 1         # NN: refit every k OOS days (paper: 1)
    lear_retrain_every: int = 1    # LEAR: refit every k OOS days (paper: 1)
    qra_refit_every: int = 1       # QRA/QRM: refit every k test days (paper: 1)
    chunk_days: int = 16           # checkpoint granularity (OOS days per task)
    epochs: int = 1500
    patience: int = 50
    batch_size: int = 32
    val_fraction: float = 0.2      # random early-stopping split, as in nn_rolling_pred.py
    lear_n_alphas: int = 100
    lear_cv: int = 7
    lear_standardize: bool = False  # original code: no scaling
    n_jobs: int = 4
    tf_threads: int = 2
    base_seed: int = 12345
    tag: str = "full"

    @classmethod
    def fast_debug(cls, **overrides) -> "RunSettings":
        s = cls(retrain_every=92, lear_retrain_every=92, qra_refit_every=46, chunk_days=92,
                epochs=15, patience=3, lear_n_alphas=10, lear_cv=3, tag="fast_debug")
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def to_dict(self) -> dict:
        return asdict(self)


def results_dir(settings: RunSettings) -> Path:
    """Production -> Results/, debug -> Results/fast_debug/ (never mixed)."""
    return RESULTS_DIR if settings.tag == "full" else RESULTS_DIR / settings.tag
