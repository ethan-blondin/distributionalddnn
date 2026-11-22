"""Turn checkpoints into the single aggregated results artifact."""
from __future__ import annotations

import platform
from datetime import datetime

import numpy as np
import pandas as pd

from . import distributions as D
from .config import (FAMILY_DAYS, LEAR_WINDOWS, LIMITATIONS, N_INSAMPLE, N_OOS, N_QRA, N_TEST, N_TRAIN,
                     NN_FAMILIES, ORIGINAL_CODE_FLAWS, RUNS, TAUS, RunSettings, model_lists)
from .evaluation import crps_hourly, dm_pvalues, metrics_table
from .models import FIT_KEYS, PARAM_NAMES, naive_quantiles
from .runner import get_data, load_model, qra_inputs

TEST = slice(N_QRA, N_OOS)     # test part of the OOS arrays


def _expected(family: str) -> np.ndarray:
    lo, hi = FAMILY_DAYS[family]
    return np.arange(lo, hi)


def oos_point_forecasts(settings: RunSettings, horizon: str, groups) -> dict[str, np.ndarray]:
    """All 736 OOS point forecasts of the models that feed QRA."""
    out = {}
    if "LEAR" in groups:
        for c in LEAR_WINDOWS:
            out[f"LEAR-{c}"] = load_model(settings, horizon, f"LEAR-{c}", _expected("LEAR"))["point"]
    if "DNN" in groups:
        for r in RUNS:
            out[f"DNN-{r}"] = load_model(settings, horizon, f"DNN-{r}", _expected("DNN"))["point"]
    return out


def split_dates(dates: pd.DatetimeIndex) -> dict:
    """First day of each period (+ last day) for plots."""
    return {"train_start": dates[0], "val_start": dates[N_TRAIN], "qra_start": dates[N_INSAMPLE],
            "test_start": dates[N_INSAMPLE + N_QRA], "end": dates[-1]}


def _diagnostics(name: str, p: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({k[4:]: p[k] for k in FIT_KEYS})
    df.insert(0, "fit_date", dates[N_INSAMPLE + df["anchor"].to_numpy()])
    df.insert(0, "model", name)
    return df


def aggregate_horizon(settings: RunSettings, horizon: str, hp: dict, groups) -> dict:
    names = model_lists(groups)
    data = get_data(horizon)
    oos_days = np.arange(N_INSAMPLE, N_INSAMPLE + N_OOS)
    test_days = oos_days[TEST]
    actuals = data.Y[test_days]
    test_dates = data.dates[test_days]
    test_index = pd.date_range(test_dates[0], periods=N_TEST * 24, freq="h")

    point_oos = oos_point_forecasts(settings, horizon, groups)
    if "LEAR" in groups:
        point_oos["LEAR-Ens"] = np.mean([point_oos[f"LEAR-{c}"] for c in LEAR_WINDOWS], axis=0)
    if "DNN" in groups:
        point_oos["DNN-Ens"] = np.mean([point_oos[f"DNN-{r}"] for r in RUNS], axis=0)

    quant, point, diags = {}, {}, []
    for m in names["point"]:
        f = point_oos[m][TEST]
        point[m] = {"median": f, "mean": f}

    _, quant["naive"] = naive_quantiles(data, list(range(N_QRA, N_OOS)))
    for m, _ in qra_inputs(groups):
        quant[m] = load_model(settings, horizon, m, np.arange(N_TEST))["quantiles"]

    if "DNN" in groups:
        for r in RUNS:
            diags.append(_diagnostics(f"DNN-{r}", load_model(settings, horizon, f"DNN-{r}", _expected("DNN")),
                                      data.dates))

    dist_params, means = {}, {}
    for family, (_, dist) in NN_FAMILIES.items():
        if dist is None or family not in groups:
            continue
        runs = []
        for r in RUNS:
            name = f"{family}-{r}"
            raw = load_model(settings, horizon, name, _expected(family))
            diags.append(_diagnostics(name, raw, data.dates))
            lo = FAMILY_DAYS[family][0]
            sl = slice(N_QRA - lo, N_OOS - lo)            # test part of what was computed
            p = {k: raw[k][sl] for k in PARAM_NAMES[dist]}
            dist_params[name] = p
            runs.append(p)
            quant[name] = D.quantiles(dist, p)
            means[name] = D.mean(dist, p)
        quant[f"{family}-qEns"] = np.mean([quant[f"{family}-{r}"] for r in RUNS], axis=0)
        quant[f"{family}-pEns"] = D.mixture_quantiles(dist, runs)
        ens_mean = np.mean([means[f"{family}-{r}"] for r in RUNS], axis=0)
        means[f"{family}-qEns"] = means[f"{family}-pEns"] = ens_mean

    for m in names["probabilistic"]:
        q = quant[m]
        point[m] = {"median": q[..., 49], "mean": means.get(m, q.mean(-1))}  # QRA/naive: mean of quantiles

    quant = {m: quant[m].astype("float32") for m in names["probabilistic"]}
    all_models = names["point"] + names["probabilistic"]
    table = metrics_table(actuals, point, quant, all_models)
    daily_crps = {m: crps_hourly(quant[m].astype("float64"), actuals).sum(-1) for m in names["probabilistic"]}

    return {
        "dates": data.dates,
        "splits": split_dates(data.dates),
        "test_dates": test_dates,
        "test_index": test_index,
        "actuals": actuals,
        "quantiles": quant,
        "point_forecasts": point,
        "oos_point_forecasts": point_oos,   # 736 days incl. QRA calibration
        "dist_params": dist_params,
        "metrics_table": table,
        "daily_crps_l1": pd.DataFrame(daily_crps, index=test_dates),
        "dm_test": dm_pvalues(daily_crps, names["dm"]),
        "fit_diagnostics": pd.concat(diags, ignore_index=True) if diags else pd.DataFrame(),
        "selected_hyperparams": hp,
    }


def build_artifact(settings: RunSettings, horizons, hp: dict, groups) -> dict:
    import scipy
    import sklearn
    names = model_lists(groups)
    return {
        "meta": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "settings": settings.to_dict(),
            "groups": list(groups),
            "taus": TAUS,
            "windows": {"N_TRAIN": N_TRAIN, "N_INSAMPLE": N_INSAMPLE, "N_OOS": N_OOS, "N_QRA": N_QRA,
                        "N_TEST": N_TEST},
            "models": names,
            "runs": list(RUNS),
            "hyperparameter_source": f"best trials of trialfiles/FINAL_DE_selection_prob_{{point,jsu}}{{{','.join(map(str, RUNS))}}} "
                                     "(paper), reused for both horizons; no re-tuning",
            "original_code_flaws_fixed": ORIGINAL_CODE_FLAWS,
            "limitations": LIMITATIONS,
            "definitions": {
                "MAE": "from the median (q=0.50); point models: the point forecast",
                "RMSE": "from the mean (DDNN: analytic; naive/QRA: mean of the 99 quantiles)",
                "CRPS": "mean pinball loss over 99 percentiles, averaged over days and hours",
                "Rel_CRPS": "100 * (CRPS_ref - CRPS) / CRPS_ref, positive = better than reference",
                "Kupiec": "per-hour unconditional coverage LR test at 5% for the 50% (q25-q75) and 90% (q05-q95) PI",
                "DM": "one-sided DM on daily L1 norm of hourly CRPS; p[row, col] small => column better than row",
                "pEns": f"equal-weight mixture of the {len(RUNS)} runs' distributions (vertical averaging)",
                "qEns": f"average of the {len(RUNS)} runs' quantiles (horizontal averaging)",
                "QRA": "linear quantile regression of the price on the point forecasts + intercept, "
                       "182-day rolling calibration (LEAR-QRA: 4 LEAR windows, DNN-QRA: 2 DNN runs)",
                "naive": "similar-day (D-7 Mon/Sat/Sun, else D-1) + hourly empirical residual quantiles, 1,456-day window",
                "LEAR": "24 hourly LASSO models on all 227 inputs; method " + (
                    "LassoLarsIC (AIC) + Lasso refit" if settings.lear_method == "ic" else "LassoCV as in lear.py"),
            },
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        },
        "horizons": {h: aggregate_horizon(settings, h, hp, groups) for h in horizons},
    }
