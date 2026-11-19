"""Point and probabilistic scores, Kupiec coverage test, Diebold-Mariano test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import TAUS


def pinball(q: np.ndarray, y: np.ndarray, taus: np.ndarray = TAUS) -> np.ndarray:
    """Pinball loss per quantile: q (..., T), y (...) -> (..., T)."""
    diff = y[..., None] - q
    return np.maximum(taus * diff, (taus - 1) * diff)


def crps_hourly(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS approximated by the mean pinball loss over the 99 percentiles -> (days, 24)."""
    return pinball(q, y).mean(-1)


def kupiec(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, coverage: float) -> np.ndarray:
    """Kupiec (1995) unconditional coverage LR test per hour -> p-values (24,)."""
    hits = (y < lo) | (y > hi)                  # (days, 24) exceedances
    n = hits.shape[0]
    x = hits.sum(0).astype(float)
    p = 1 - coverage
    pi = x / n
    with np.errstate(divide="ignore", invalid="ignore"):
        ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
        ll1 = np.where(x > 0, x * np.log(np.where(pi > 0, pi, 1)), 0) + \
              np.where(n - x > 0, (n - x) * np.log(np.where(pi < 1, 1 - pi, 1)), 0)
    lr = -2 * (ll0 - ll1)
    return 1 - stats.chi2.cdf(lr, df=1)


def dm_pvalues(daily_losses: dict[str, np.ndarray], models: list[str]) -> pd.DataFrame:
    """One-sided multivariate DM test (Ziel & Weron, 2018).

    daily loss = L1 norm of the 24 hourly CRPS of a day. Entry [row i, col j] is the
    p-value of H1: forecasts of model j (column) are significantly better than
    those of model i (row), i.e. E[L_i - L_j] > 0.
    """
    out = pd.DataFrame(np.nan, index=models, columns=models)
    for i in models:
        for j in models:
            if i == j:
                continue
            d = daily_losses[i] - daily_losses[j]
            sd = d.std(ddof=1)
            stat = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else 0.0
            out.loc[i, j] = 1 - stats.norm.cdf(stat)
    out.index.name = "row (worse?)"
    out.columns.name = "column (better?)"
    return out


def metrics_table(actuals: np.ndarray, point: dict, quant: dict, models: list[str],
                  reference_naive: str = "naive", reference_qra: str = "LEAR-QRA") -> pd.DataFrame:
    rows, crps = {}, {}
    for m in models:
        med, mu = point[m]["median"], point[m]["mean"]
        r = {"MAE": np.abs(med - actuals).mean(), "RMSE": np.sqrt(((mu - actuals) ** 2).mean())}
        if m in quant:
            q = quant[m]
            crps[m] = crps_hourly(q, actuals).mean()
            r["CRPS"] = crps[m]
            r["Kupiec_50_passed_hours"] = int((kupiec(actuals, q[..., 24], q[..., 74], 0.50) > 0.05).sum())
            r["Kupiec_90_passed_hours"] = int((kupiec(actuals, q[..., 4], q[..., 94], 0.90) > 0.05).sum())
        rows[m] = r
    df = pd.DataFrame(rows).T
    # relative improvement in % (positive = better than the reference)
    for ref, col in ((reference_naive, "Rel_CRPS_vs_Naive"), (reference_qra, "Rel_CRPS_vs_LEAR_QRA")):
        if ref in crps:
            df[col] = 100 * (crps[ref] - df["CRPS"]) / crps[ref]
    cols = ["MAE", "RMSE", "CRPS", "Rel_CRPS_vs_Naive", "Rel_CRPS_vs_LEAR_QRA",
            "Kupiec_50_passed_hours", "Kupiec_90_passed_hours"]
    return df.reindex(columns=cols)
