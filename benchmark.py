"""Benchmark of a finished run, following Lago et al. (2021) and Nowotarski & Weron (2018).

Reads Results/<tag>/aggregated_ddnn_comparison.pkl (no model is re-fitted) and writes
Results/<tag>/benchmark.md plus one CSV per table.

    python benchmark.py [--tag neural_w7]

Point: MAE, rMAE (vs the similar-day naive of the paper and vs the weekly naive Y_{T-7}), RMSE
(reported, not ranked on), no MAPE; multivariate DM on daily L1 absolute errors.
Probabilistic: CRPS (pinball over 99 percentiles), relative CRPS vs naive, reliability (Kupiec,
empirical coverage) and sharpness (mean interval width); multivariate DM on daily CRPS.
Computation time from the per-fit timings (NN) and the run log (LEAR, QRA).
"""
from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from ddnn_compare.config import N_INSAMPLE, N_QRA, N_TEST, RESULTS_DIR, AGGREGATE_NAME
from ddnn_compare.evaluation import crps_hourly, dm_pvalues, kupiec
from ddnn_compare.runner import get_data

ALPHA = 0.05
INTERVALS = {50: (24, 74), 90: (4, 94)}     # quantile indices of the central intervals


def weekly_naive(horizon: str, actuals: np.ndarray) -> np.ndarray:
    data = get_data(horizon)
    days = np.arange(N_INSAMPLE + N_QRA, N_INSAMPLE + N_QRA + N_TEST)
    assert np.allclose(data.Y[days], actuals), "pickle and dataset disagree on the test prices"
    return data.Y[days - 7]


def dm_wins(p: pd.DataFrame) -> pd.Series:
    """Number of other models each model beats significantly (column better than row)."""
    return (p < ALPHA).sum(axis=0).rename("DM_wins")


def point_table(res: dict, horizon: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = res["actuals"]
    pf = res["point_forecasts"]
    mae_naive = np.abs(pf["naive"]["median"] - y).mean()
    mae_weekly = np.abs(weekly_naive(horizon, y) - y).mean()
    rows, daily = {}, {}
    for m, f in pf.items():
        err = f["median"] - y
        rows[m] = {"MAE": np.abs(err).mean(),
                   "rMAE": np.abs(err).mean() / mae_naive,
                   "rMAE_weekly": np.abs(err).mean() / mae_weekly,
                   "RMSE": np.sqrt(((f["mean"] - y) ** 2).mean())}
        daily[m] = np.abs(err).sum(-1)
    df = pd.DataFrame(rows).T
    dm = dm_pvalues(daily, list(df.index))
    df = df.join(dm_wins(dm)).sort_values("rMAE")
    df["rank_rMAE"] = np.arange(1, len(df) + 1)
    df["rank_rMAE_weekly"] = df["rMAE_weekly"].rank().astype(int)
    return df, dm


def prob_table(res: dict) -> pd.DataFrame:
    y = res["actuals"]
    rows = {}
    for m, q in res["quantiles"].items():
        q = q.astype("float64")
        r = {"CRPS": crps_hourly(q, y).mean()}
        for cov, (lo, hi) in INTERVALS.items():
            inside = (y >= q[..., lo]) & (y <= q[..., hi])
            r[f"cover_{cov}"] = 100 * inside.mean()
            r[f"width_{cov}"] = (q[..., hi] - q[..., lo]).mean()
            r[f"kupiec_{cov}_ok"] = int((kupiec(y, q[..., lo], q[..., hi], cov / 100) > ALPHA).sum())
        rows[m] = r
    df = pd.DataFrame(rows).T
    df.insert(1, "rel_CRPS_vs_naive_%", 100 * (df.loc["naive", "CRPS"] - df["CRPS"]) / df.loc["naive", "CRPS"])
    df = df.join(dm_wins(res["dm_test"])).sort_values("CRPS")
    return df


def compute_time(res_by_h: dict, tag: str) -> pd.DataFrame:
    """Single-worker seconds summed over all fits (NN: per-fit timer, LEAR/QRA: run log)."""
    secs = defaultdict(float)
    for h, res in res_by_h.items():
        d = res["fit_diagnostics"]
        for m, g in d.groupby("model"):
            secs[(h, m)] += g["seconds"].sum()
    log = RESULTS_DIR / tag / "run.log"
    if log.exists():
        pat = re.compile(r"\] \d+/\d+ (\S+) (LEAR-\S+) days \S+ \((\d+)s\)")
        for line in log.read_text().splitlines():
            if mt := pat.search(line):
                secs[(mt[1], mt[2])] += float(mt[3])
    s = pd.Series(secs, name="hours") / 3600
    df = s.unstack(0).round(2)
    df["total_h"] = df.sum(axis=1)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="neural_w7")
    tag = ap.parse_args().tag
    out = RESULTS_DIR / tag
    with open(out / AGGREGATE_NAME, "rb") as fh:
        art = pickle.load(fh)

    md = [f"# Benchmark — run `{tag}`",
          f"Test: {N_TEST} days per horizon. DM at {int(ALPHA * 100)} %: `DM_wins` = number of models "
          "beaten significantly (multivariate DM, Ziel & Weron 2018).",
          "rMAE = MAE / MAE(naive). `naive` = similar-day naive of the paper (Y_{T-7} Mon/Sat/Sun, "
          "Y_{T-1} otherwise); `rMAE_weekly` uses Y_{T-7} for every day.", ""]
    summary = {}
    for h, res in art["horizons"].items():
        pt, dm_pt = point_table(res, h)
        pr = prob_table(res)
        pt.to_csv(out / f"benchmark_point_{h}.csv")
        pr.to_csv(out / f"benchmark_prob_{h}.csv")
        dm_pt.to_csv(out / f"benchmark_dm_point_{h}.csv")
        summary[h] = pt["rMAE"], pr["rel_CRPS_vs_naive_%"]
        md += [f"## {h}", "", "### Point forecasts (ranked by rMAE)", "", pt.round(3).to_markdown(), "",
               "### Probabilistic forecasts (ranked by CRPS)", "",
               "Nominal coverage: 50 % and 90 %; `kupiec_*_ok` = hours (of 24) where Kupiec does not "
               "reject; `width_*` = sharpness (EUR/MWh, lower is sharper).", "", pr.round(3).to_markdown(), ""]
        print(f"\n=== {h} ===\n{pt.round(3).to_string()}\n\n{pr.round(3).to_string()}")

    cross = pd.concat({f"rMAE {h}": s[0] for h, s in summary.items()} |
                      {f"rel CRPS % {h}": s[1] for h, s in summary.items()}, axis=1)
    ct = compute_time(art["horizons"], tag)
    cross.to_csv(out / "benchmark_cross_period.csv")
    ct.to_csv(out / "benchmark_compute_time.csv")
    md += ["## Cross-period comparison (scale-free metrics)", "", cross.round(3).to_markdown(), "",
           "## Computation time (single-worker hours, summed over fits)", "",
           "NN: per-fit timer (machine of the NN run). LEAR: run log of the LEAR run. "
           "Wall time = total / number of parallel workers. Caveat: a few NN fits last several hours "
           "(machine asleep or overloaded), e.g. DNN-1 2020-2026 has 4 fits > 1 h; medians are more reliable.",
           "", ct.to_markdown(), ""]
    (out / "benchmark.md").write_text("\n".join(md))
    print(f"\n{cross.round(3).to_string()}\n\n{ct.to_string()}\n\nWrote {out / 'benchmark.md'}")


if __name__ == "__main__":
    main()
