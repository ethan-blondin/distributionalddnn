"""Experiment configuration: horizons, window splits, model registry, run settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Datasets"
TRIAL_DIR = REPO_ROOT / "trialfiles"
RESULTS_DIR = REPO_ROOT / "Results"
AGGREGATE_NAME = "aggregated_ddnn_comparison.pkl"

# --------------------------------------------------------------------------- #
# Horizons (2,192 days = 52,608 h each), evaluated separately
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
N_QRA = 182               # rolling QRA calibration window (first 182 OOS days)
N_TEST = N_OOS - N_QRA    # 554 evaluation days
MAX_LAG = 7               # longest lag (D-7): first 7 days of a window have no features
NN_TRAIN_DAYS = N_INSAMPLE - MAX_LAG   # 1,449 training samples per rolling NN fit

TAUS = np.arange(1, 100) / 100.0          # 99 percentiles
LEAR_WINDOWS = (56, 84, 1092, 1456)
RUNS = (1, 2)             # hyper-parameter runs taken from trialfiles/ (paper has 1..4)
N_RUNS = len(RUNS)

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
NN_FAMILIES = {            # family -> (trialfile stem, output distribution)
    "DNN": ("point", None),
    "DDNN-JSU": ("jsu", "JSU"),
}
MODEL_GROUPS = ("DNN", "DDNN-JSU", "LEAR")   # selectable with --models
DEFAULT_GROUPS = ("DNN", "DDNN-JSU")         # LEAR is wired in but off by default
# Days each family must forecast (OOS indices 0..735). DNN and LEAR feed QRA, which
# needs the 182 calibration days; DDNN-JSU is only evaluated on the 554 test days.
FAMILY_DAYS = {"DNN": (0, N_OOS), "DDNN-JSU": (N_QRA, N_OOS), "LEAR": (0, N_OOS)}


def model_lists(groups) -> dict:
    """Model names present in a run with the given model groups."""
    g = set(groups)
    lear, dnn, jsu = "LEAR" in g, "DNN" in g, "DDNN-JSU" in g
    jsu_runs = [f"DDNN-JSU-{r}" for r in RUNS]
    point = ([f"LEAR-{w}" for w in LEAR_WINDOWS] + ["LEAR-Ens"] if lear else []) + \
            ([f"DNN-{r}" for r in RUNS] + ["DNN-Ens"] if dnn else [])
    prob = ["naive"] + (["LEAR-QRA"] if lear else []) + (["DNN-QRA"] if dnn else []) + \
           (jsu_runs + ["DDNN-JSU-pEns", "DDNN-JSU-qEns"] if jsu else [])
    dm = ["naive"] + (["LEAR-QRA"] if lear else []) + (["DNN-QRA"] if dnn else []) + \
         (jsu_runs + ["DDNN-JSU-pEns", "DDNN-JSU-qEns"] if jsu else [])
    table1 = (["LEAR-Ens"] if lear else []) + (["DNN-Ens"] if dnn else []) + dm
    pi = (["LEAR-QRA"] if lear else []) + (["DNN-QRA"] if dnn else []) + \
         (["DDNN-JSU-pEns", "DDNN-JSU-qEns"] if jsu else [])
    return {"point": point, "probabilistic": prob, "dm": dm, "table1": table1, "pi": pi}


# Documented deviations from the published code in Python/ (all fixed here).
ORIGINAL_CODE_FLAWS = {
    "weekday_index": "X[d, 220] = data.index[d].weekday() uses hour d of the first day, not day d "
                     "(should be data.index[d*24]); the weekday feature was essentially constant/wrong.",
    "brent_rolling_slice": "lear.py / nn_rolling_pred.py read Brent from the full `data` frame instead of the "
                           "rolling window `df`, i.e. Brent always came from the first days of the dataset.",
    "dow_encoding": "a single integer weekday (0..6) was used instead of 7 one-hot day-of-week dummies "
                    "as described in the paper; inputs now 227 instead of 221.",
}

LIMITATIONS = [
    "Hyper-parameters and input selections are the paper's best Optuna trials (runs 1-2 of 4), tuned on "
    "2015-2018 with the buggy features of the published code; re-used unchanged on both horizons. "
    "2015-2020 is therefore evaluated with settings tuned on its own in-sample period, 2020-2026 is not.",
    "Neural networks are re-trained every `retrain_every` days (paper: daily); weights are reused in "
    "between, with daily inference on fresh features. Adaptation to abrupt regime changes is understated.",
    "Ensembles average 2 runs instead of 4, so pEns/qEns gain less from diversity than in the paper.",
    "Training is stochastic (random 20 % early-stopping split, weight init): the gap between run 1 and "
    "run 2 is the noise floor - differences smaller than that are not findings.",
    "Price levels and volatility differ strongly between the two test periods (2019-06 -> 2020-12 incl. COVID "
    "vs 2025-05 -> 2026-11 with frequent negative prices): compare horizons only through relative metrics "
    "(Rel. CRPS, rankings, Kupiec, DM within a horizon), not absolute EUR/MWh scores.",
    "The 2015-2020 power data come from the paper's DE.csv, the 2020-2026 power data from SMARD: "
    "same definitions, but load differs by ~1 % on the overlap days.",
]


@dataclass
class RunSettings:
    """Compute settings. `fast_debug()` returns a cheap variant with identical output shapes."""
    retrain_every: int = 7         # NN: re-fit every k OOS days (paper: 1)
    lear_retrain_every: int = 1    # LEAR: re-fit every k OOS days (paper: 1)
    qra_refit_every: int = 1       # QRA: re-fit every k test days (paper: 1)
    chunk_days: int = 7            # checkpoint granularity (one NN fit per block)
    epochs: int = 1500
    patience: int = 50
    batch_size: int = 32
    val_fraction: float = 0.2      # random early-stopping split, as in nn_rolling_pred.py
    lear_method: str = "ic"        # "ic": LassoLarsIC(AIC) + Lasso refit, "lassocv": LassoCV as in lear.py
    lear_n_alphas: int = 100       # lassocv only
    lear_cv: int = 7               # lassocv only
    n_jobs: int = 4                # worker processes (shared task queue)
    tf_threads: int = 1            # threads per worker
    base_seed: int = 12345
    tag: str = "neural_w7"         # results go to Results/<tag>/
    debug: bool = False            # True -> numbers are not meaningful

    @classmethod
    def fast_debug(cls, **overrides) -> "RunSettings":
        s = cls(retrain_every=184, lear_retrain_every=184, qra_refit_every=92, chunk_days=184,
                epochs=5, patience=2, lear_n_alphas=10, lear_cv=3, tag="fast_debug", debug=True)
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def validate(self) -> None:
        for name in ("retrain_every", "lear_retrain_every", "qra_refit_every"):
            k = getattr(self, name)
            if k < 1 or self.chunk_days % k:
                raise ValueError(f"chunk_days ({self.chunk_days}) must be a multiple of {name} ({k}), "
                                 "otherwise blocks would repeat fits")
        if self.lear_method not in ("ic", "lassocv"):
            raise ValueError(f"unknown lear_method {self.lear_method!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def results_dir(settings: RunSettings) -> Path:
    """Every run tag has its own folder: Results/<tag>/ (never mixed, never overwritten)."""
    return RESULTS_DIR / settings.tag
