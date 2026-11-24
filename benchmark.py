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
from scipy import stats

from ddnn_compare.config import N_INSAMPLE, N_QRA, N_TEST, RESULTS_DIR, TAUS, AGGREGATE_NAME
from ddnn_compare.evaluation import crps_hourly, dm_pvalues, kupiec, pinball
from ddnn_compare.runner import get_data

ALPHA = 0.05
INTERVALS = {50: (24, 74), 90: (4, 94), 98: (0, 98)}     # quantile indices of the central intervals
NW_LAG = 7                                                # Newey-West lag (weekly autocorrelation)
# Quantile weights of the weighted CRPS (Gneiting & Ranjan, 2011), normalised to mean 1
TAIL_WEIGHTS = {"both": (2 * TAUS - 1) ** 2, "left": (1 - TAUS) ** 2, "right": TAUS ** 2}


def christoffersen(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, coverage: float) -> tuple[np.ndarray, np.ndarray]:
    """Christoffersen (1998) per hour on the daily hit sequence -> (p_independence, p_conditional_coverage).

    LR_ind: first-order Markov alternative vs independent hits (chi2(1));
    LR_cc = LR_uc (Kupiec) + LR_ind (chi2(2)).
    """
    hits = ((y < lo) | (y > hi)).astype(int)            # (days, 24)
    prev, cur = hits[:-1], hits[1:]
    n00 = ((prev == 0) & (cur == 0)).sum(0); n01 = ((prev == 0) & (cur == 1)).sum(0)
    n10 = ((prev == 1) & (cur == 0)).sum(0); n11 = ((prev == 1) & (cur == 1)).sum(0)

    def ll(n_stay, n_go, p):                             # sum of n*log(p) with 0*log(0) = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(n_stay > 0, n_stay * np.log(np.where(p < 1, 1 - p, 1)), 0) + \
                   np.where(n_go > 0, n_go * np.log(np.where(p > 0, p, 1)), 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        p01 = np.where(n00 + n01 > 0, n01 / (n00 + n01), 0)
        p11 = np.where(n10 + n11 > 0, n11 / (n10 + n11), 0)
        p1 = (n01 + n11) / (n00 + n01 + n10 + n11)
    lr_ind = -2 * (ll(n00 + n10, n01 + n11, p1) - ll(n00, n01, p01) - ll(n10, n11, p11))
    p_uc = kupiec(y, lo, hi, coverage)
    lr_uc = stats.chi2.isf(np.clip(p_uc, 1e-300, 1), df=1)
    return stats.chi2.sf(lr_ind, df=1), stats.chi2.sf(lr_uc + lr_ind, df=2)


def dm_matrix(loss: dict[str, np.ndarray], models: list[str], variant: str) -> pd.DataFrame:
    """One-sided DM p-values, [row i, col j] = P-value of H1 'column j better than row i'.

    loss[m]: (days, 24) hourly losses. Variants:
      L1      multivariate, daily loss = sum of the 24 hourly losses (Ziel & Weron, 2018)
      L2      multivariate, daily loss = Euclidean norm of the 24 hourly losses
      L1-NW   L1 with Newey-West variance (lag NW_LAG) and HLN small-sample t(n-1) reference
      hourly  univariate DM per hour; entry = number of hours (of 24) where j is better at ALPHA
    """
    out = pd.DataFrame(np.nan, index=models, columns=models)
    for i in models:
        for j in models:
            if i == j:
                continue
            if variant == "hourly":
                d = loss[i] - loss[j]
                stat = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(len(d)))
                out.loc[i, j] = int((stats.norm.sf(stat) < ALPHA).sum())
                continue
            if variant == "L2":
                d = np.sqrt((loss[i] ** 2).sum(1)) - np.sqrt((loss[j] ** 2).sum(1))
            else:
                d = loss[i].sum(1) - loss[j].sum(1)
            n = len(d)
            if variant == "L1-NW":
                e = d - d.mean()
                var = e @ e / n + 2 * sum((1 - k / (NW_LAG + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, NW_LAG + 1))
                stat = d.mean() / np.sqrt(var / n) * np.sqrt((n - 1) / n)      # HLN factor for h = 1
                out.loc[i, j] = stats.t.sf(stat, df=n - 1)
            else:
                sd = d.std(ddof=1)
                out.loc[i, j] = stats.norm.sf(d.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.5
    out.index.name, out.columns.name = "row (worse?)", "column (better?)"
    return out


def dm_summary(loss: dict[str, np.ndarray], models: list[str], tag: str, prefix: str, out_dir) -> pd.DataFrame:
    """DM_wins per variant (+ CSV of every p-value matrix)."""
    cols = {}
    for v in ("L1", "L2", "L1-NW"):
        p = dm_matrix(loss, models, v)
        p.to_csv(out_dir / f"benchmark_dm_{prefix}_{v}_{tag}.csv")
        cols[f"wins_{v}"] = dm_wins(p)
    h = dm_matrix(loss, models, "hourly")
    h.to_csv(out_dir / f"benchmark_dm_{prefix}_hourly_{tag}.csv")
    cols["hours_better_total"] = h.sum(axis=0).astype(int)     # hours (summed over opponents) significantly better
    return pd.DataFrame(cols)


def tail_table(res: dict) -> pd.DataFrame:
    """Errors in the tails: of the predictive distribution (weighted CRPS, extreme quantiles)
    and of the price distribution (scores on the hours with the 5 % lowest / highest prices)."""
    y = res["actuals"]
    lo5, hi5 = np.percentile(y, [5, 95])
    low, high, neg = y <= lo5, y >= hi5, y < 0
    rows = {}
    for m, q in res["quantiles"].items():
        pl = pinball(q.astype("float64"), y)               # (days, 24, 99)
        r = {f"twCRPS_{k}": (pl * w / w.mean()).mean() for k, w in TAIL_WEIGHTS.items()}
        r["pinball_q01-05"] = pl[..., :5].mean()
        r["pinball_q95-99"] = pl[..., 94:].mean()
        crps = pl.mean(-1)
        med = res["point_forecasts"][m]["median"]
        for name, mask in (("low5%", low), ("high5%", high), ("neg", neg)):
            if mask.any():
                r[f"MAE_{name}"] = np.abs(med - y)[mask].mean()
                r[f"CRPS_{name}"] = crps[mask].mean()
        rows[m] = r
    df = pd.DataFrame(rows).T.sort_values("twCRPS_both")
    df.attrs["thresholds"] = (lo5, hi5, int(neg.sum()))
    return df


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
            p_ind, p_cc = christoffersen(y, q[..., lo], q[..., hi], cov / 100)
            r[f"indep_{cov}_ok"] = int((p_ind > ALPHA).sum())
            r[f"christ_{cov}_ok"] = int((p_cc > ALPHA).sum())
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

        y = res["actuals"]
        crps_loss = {m: crps_hourly(q.astype("float64"), y) for m, q in res["quantiles"].items()}
        prob_models = list(res["dm_test"].index)
        # self-check: the L1 variant must reproduce the DM matrix stored by the pipeline
        assert np.allclose(dm_matrix(crps_loss, prob_models, "L1").to_numpy(float),
                           res["dm_test"].loc[prob_models, prob_models].to_numpy(float), equal_nan=True)
        dm_prob = dm_summary(crps_loss, prob_models, h, "prob", out)
        ae_loss = {m: np.abs(f["median"] - y) for m, f in res["point_forecasts"].items()}
        dm_point = dm_summary(ae_loss, list(pt.index), h, "point", out)
        tails = tail_table(res)
        lo5, hi5, nneg = tails.attrs["thresholds"]
        for name, df in (("tails", tails), ("dm_variants_prob", dm_prob), ("dm_variants_point", dm_point)):
            df.to_csv(out / f"benchmark_{name}_{h}.csv")

        md += [f"## {h}", "", "### Point forecasts (ranked by rMAE)", "", pt.round(3).to_markdown(), "",
               "### Probabilistic forecasts (ranked by CRPS)", "",
               "Central intervals 50 % (q25-q75), 90 % (q05-q95), 98 % (q01-q99). Per hour (of 24): "
               "`kupiec_*_ok` = unconditional coverage not rejected; `indep_*_ok` = Christoffersen independence "
               "not rejected; `christ_*_ok` = Christoffersen conditional coverage (chi2(2)) not rejected. "
               "`width_*` = sharpness (EUR/MWh, lower is sharper).", "", pr.round(3).to_markdown(), "",
               "### Errors in the tails", "",
               "`twCRPS_*`: quantile-weighted CRPS (Gneiting & Ranjan, 2011), weights (2τ-1)² / (1-τ)² / τ² "
               "normalised to mean 1. `pinball_q01-05` / `q95-99`: mean pinball loss of the 5 lowest / highest "
               f"percentiles. `*_low5%` / `*_high5%`: hours with price ≤ {lo5:.2f} / ≥ {hi5:.2f} EUR/MWh "
               f"(5 % / 95 % percentiles of the test prices); `*_neg`: the {nneg} hours with negative prices.",
               "", tails.round(3).to_markdown(), "",
               "### Diebold-Mariano variants", "",
               f"`wins_*` = number of models beaten at {int(ALPHA * 100)} %. L1: multivariate, daily sum of the "
               "24 hourly losses (Ziel & Weron, 2018, as in the pipeline). L2: daily Euclidean norm. L1-NW: L1 with "
               f"Newey-West variance (lag {NW_LAG}) and Harvey-Leybourne-Newbold t(n-1) reference. "
               "`hours_better_total`: univariate DM per hour, number of (hour, opponent) pairs where the model "
               "is significantly better. Full p-value matrices in `benchmark_dm_*.csv`.", "",
               "Probabilistic (loss = hourly CRPS):", "", dm_prob.to_markdown(), "",
               "Point (loss = absolute error of the median):", "", dm_point.to_markdown(), ""]
        print(f"\n=== {h} ===\n{pt.round(3).to_string()}\n\n{pr.round(3).to_string()}\n\n"
              f"{tails.round(3).to_string()}\n\n{dm_prob.to_string()}\n\n{dm_point.to_string()}")

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
