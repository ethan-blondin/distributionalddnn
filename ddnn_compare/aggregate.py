"""Turn checkpoints into the single aggregated results artifact."""
from __future__ import annotations

import platform
from datetime import datetime

import numpy as np
import pandas as pd

from . import distributions as D
from .config import (DM_MODELS, LEAR_WINDOWS, N_INSAMPLE, N_OOS, N_QRA, N_RUNS, N_TEST, N_TRAIN,
                     NN_FAMILIES, ORIGINAL_CODE_FLAWS, POINT_MODELS, PROB_MODELS, TAUS, RunSettings)
from .evaluation import crps_hourly, dm_pvalues, metrics_table
from .models import PARAM_NAMES, naive_quantiles
from .runner import get_data, load_model

TEST = slice(N_QRA, N_OOS)     # test part of the OOS arrays


def oos_point_forecasts(settings: RunSettings, horizon: str) -> dict[str, np.ndarray]:
    out = {f"LEAR-{c}": load_model(settings, horizon, f"LEAR-{c}", N_OOS)["point"] for c in LEAR_WINDOWS}
    for r in range(1, N_RUNS + 1):
        out[f"DNN-{r}"] = load_model(settings, horizon, f"DNN-{r}", N_OOS)["point"]
    return out


def split_dates(dates: pd.DatetimeIndex) -> dict:
    """First day of each period (+ last day) for plots."""
    return {"train_start": dates[0], "val_start": dates[N_TRAIN], "qra_start": dates[N_INSAMPLE],
            "test_start": dates[N_INSAMPLE + N_QRA], "end": dates[-1]}


def aggregate_horizon(settings: RunSettings, horizon: str, hp: dict) -> dict:
    data = get_data(horizon)
    oos_days = np.arange(N_INSAMPLE, N_INSAMPLE + N_OOS)
    test_days = oos_days[TEST]
    actuals = data.Y[test_days]
    test_dates = data.dates[test_days]
    test_index = pd.date_range(test_dates[0], periods=N_TEST * 24, freq="h")

    point_oos = oos_point_forecasts(settings, horizon)
    point_oos["LEAR-Ens"] = np.mean([point_oos[f"LEAR-{c}"] for c in LEAR_WINDOWS], axis=0)
    point_oos["DNN-Ens"] = np.mean([point_oos[f"DNN-{r}"] for r in range(1, N_RUNS + 1)], axis=0)

    quant, point = {}, {}
    for m in POINT_MODELS:
        f = point_oos[m][TEST]
        point[m] = {"median": f, "mean": f}

    # naive
    _, quant["naive"] = naive_quantiles(data, list(range(N_QRA, N_OOS)))
    # QRA / QRM
    for m in ("LEAR-QRA", "LEAR-QRM", "DNN-QRA", "DNN-QRM"):
        quant[m] = load_model(settings, horizon, m, N_TEST)["quantiles"]
    # DDNN runs and ensembles
    dist_params, means = {}, {}
    for family, (_, dist) in NN_FAMILIES.items():
        if dist is None:
            continue
        runs = []
        for r in range(1, N_RUNS + 1):
            name = f"{family}-{r}"
            p = load_model(settings, horizon, name, N_OOS)
            p = {k: p[k][TEST] for k in PARAM_NAMES[dist]}
            dist_params[name] = p
            runs.append(p)
            quant[name] = D.quantiles(dist, p)
            means[name] = D.mean(dist, p)
        quant[f"{family}-qEns"] = np.mean([quant[f"{family}-{r}"] for r in range(1, N_RUNS + 1)], axis=0)
        quant[f"{family}-pEns"] = D.mixture_quantiles(dist, runs)
        ens_mean = np.mean([means[f"{family}-{r}"] for r in range(1, N_RUNS + 1)], axis=0)
        means[f"{family}-qEns"] = means[f"{family}-pEns"] = ens_mean

    for m in PROB_MODELS:
        q = quant[m]
        point[m] = {"median": q[..., 49], "mean": means.get(m, q.mean(-1))}  # QRA/naive: mean of quantiles

    quant = {m: quant[m].astype("float32") for m in PROB_MODELS}
    models = POINT_MODELS + PROB_MODELS
    table = metrics_table(actuals, point, quant, models)
    daily_crps = {m: crps_hourly(quant[m].astype("float64"), actuals).sum(-1) for m in PROB_MODELS}

    return {
        "dates": data.dates,
        "splits": split_dates(data.dates),
        "test_dates": test_dates,
        "test_index": test_index,
        "actuals": actuals,
        "quantiles": quant,
        "point_forecasts": point,
        "oos_point_forecasts": {k: v for k, v in point_oos.items()},   # 736 days incl. QRA calibration
        "dist_params": dist_params,
        "metrics_table": table,
        "daily_crps_l1": pd.DataFrame(daily_crps, index=test_dates),
        "dm_test": dm_pvalues(daily_crps, DM_MODELS),
        "selected_hyperparams": hp,
    }


def build_artifact(settings: RunSettings, horizons, hp: dict) -> dict:
    import scipy
    import sklearn
    return {
        "meta": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "settings": settings.to_dict(),
            "taus": TAUS,
            "windows": {"N_TRAIN": N_TRAIN, "N_INSAMPLE": N_INSAMPLE, "N_OOS": N_OOS, "N_QRA": N_QRA,
                        "N_TEST": N_TEST},
            "models": {"point": POINT_MODELS, "probabilistic": PROB_MODELS, "dm": DM_MODELS},
            "hyperparameter_source": "best trials of trialfiles/FINAL_DE_selection_prob_{point,normal,jsu}{1..4} "
                                     "(paper), reused for both horizons; no re-tuning",
            "original_code_flaws_fixed": ORIGINAL_CODE_FLAWS,
            "definitions": {
                "MAE": "from the median (q=0.50); point models: the point forecast",
                "RMSE": "from the mean (DDNN: analytic; naive/QRA/QRM: mean of the 99 quantiles)",
                "CRPS": "mean pinball loss over 99 percentiles, averaged over days and hours",
                "Rel_CRPS": "100 * (CRPS_ref - CRPS) / CRPS_ref, positive = better than reference",
                "Kupiec": "per-hour unconditional coverage LR test at 5% for the 50% (q25-q75) and 90% (q05-q95) PI",
                "DM": "one-sided DM on daily L1 norm of hourly CRPS; p[row, col] small => column better than row",
                "pEns": "equal-weight mixture of the 4 runs' distributions (vertical averaging)",
                "qEns": "average of the 4 runs' quantiles (horizontal averaging)",
                "naive": "similar-day (D-7 Mon/Sat/Sun, else D-1) + hourly empirical residual quantiles, 1,456-day window",
            },
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        },
        "horizons": {h: aggregate_horizon(settings, h, hp) for h in horizons},
    }
