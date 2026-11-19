"""Generate DDNN_Comparative_Analysis.ipynb (run: python -m ddnn_compare.make_notebook)."""
from __future__ import annotations

import nbformat as nbf

from .config import REPO_ROOT

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))   # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))    # noqa: E731

md(r"""
# DDNN comparative analysis — 2015–2020 vs 2020–2026

Stage 2 of the comparison of the distributional deep neural network of
Marcjasz, Narajewski, Weron & Ziel (2022) on the original window (**2015-01-01 → 2020-12-31**)
and an updated window (**2020-11-19 → 2026-11-19**), each 2,192 days.

This notebook **does not train anything**. It only reads

* `Datasets/DE_dataset_2015-2020.csv`, `Datasets/DE_dataset_2020-2026.csv`
* `Results/aggregated_ddnn_comparison.pkl` (produced by `run_comparative_experiments.py`;
  falls back to `Results/fast_debug/…` if the full run has not been made yet)

**Splits (both horizons):** 1,092 training days → 364 validation days (in-sample = 1,456 days) →
182 QRA/QRM calibration days → 554 test days.

**Set-up choices:** hyper-parameters and input selections are the best trials of the paper's
12 Optuna studies (`trialfiles/`), re-used unchanged on both horizons (no re-tuning), so the
2020–2026 results measure the *out-of-sample generalisation* of the paper's architecture.
Three flaws of the published code are fixed for both horizons (see the appendix).
""")

code(r"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = Path.cwd()
DATA = {"2015-2020": ROOT / "Datasets" / "DE_dataset_2015-2020.csv",
        "2020-2026": ROOT / "Datasets" / "DE_dataset_2020-2026.csv"}
CANDIDATES = [ROOT / "Results" / "aggregated_ddnn_comparison.pkl",
              ROOT / "Results" / "fast_debug" / "aggregated_ddnn_comparison.pkl"]
RESULTS_PATH = next(p for p in CANDIDATES if p.exists())

with open(RESULTS_PATH, "rb") as fh:
    R = pickle.load(fh)
META, H = R["meta"], R["horizons"]
frames = {h: pd.read_csv(p, index_col=0, parse_dates=True) for h, p in DATA.items()}

print(f"Results : {RESULTS_PATH.relative_to(ROOT)}  (created {META['created']}, mode = {META['settings']['tag']})")
if META["settings"]["tag"] != "full":
    print("WARNING: fast-debug artifact - sparse retraining / few epochs, numbers are NOT meaningful.")
for h, df in frames.items():
    print(f"{h}: {df.index[0]} -> {df.index[-1]}  ({len(df):,} rows)")
""")

code(r"""
# ---- shared style (reference palette: categorical slots, blue sequential ramp) ----
C_HOR = {"2015-2020": "#2a78d6", "2020-2026": "#eb6834"}          # slot 1 / slot 2
C_FUEL = {"API2_Coal": "#2a78d6", "Brent_oil": "#eb6834", "TTF_Gas": "#1baf7a"}
C_BENCH = {"LEAR-QRA": "#2a78d6", "LEAR-QRM": "#eb6834", "DNN-QRA": "#1baf7a", "DNN-QRM": "#eda100"}
PI_FILL = {98: "#cde2fb", 90: "#86b6ef", 50: "#2a78d6"}             # sequential blue, light -> dark
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OVERLAP = (pd.Timestamp("2020-11-19"), pd.Timestamp("2020-12-31 23:00"))
LABEL = {"2015-2020": "2015–2020", "2020-2026": "2020–2026"}

plt.rcParams.update({
    "figure.dpi": 110, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 9,
    "lines.linewidth": 1.0,
})

def boundaries(h):
    s = H[h]["splits"]
    return [("Validation", s["val_start"]), ("QRA calibration", s["qra_start"]), ("Out-of-sample test", s["test_start"])]

print(pd.DataFrame({h: {k: v.date() for k, v in H[h]["splits"].items()} for h in H}))
""")

# --------------------------------------------------------------------------- #
md(r"""
## 1. Feature & regime overview (adaptation of Figure 2)

Both datasets on one continuous time axis, 2015-01-01 → 2026-11-19. The grey band is the
43-day overlap (2020-11-19 → 2020-12-31), where both series are drawn. Dashed lines mark the
start of validation, QRA calibration and out-of-sample testing for each horizon (horizon colour).
""")

code(r"""
PANELS = [("Price", "DA Price (EUR/MWh)", 1.0),
          ("Load_DA_Forecast", "DA Load Forecast (GWh)", 1e-3),
          ("Renewables_DA_Forecast", "DA RES Forecast (GWh)", 1e-3),
          ("EUA", "EUA Price (EUR/tCO2)", 1.0)]
FUELS = [("API2_Coal", "API2 Coal (EUR/t)"), ("Brent_oil", "Brent Oil (EUR/bbl.)"), ("TTF_Gas", "TTF Gas (EUR/MWh)")]

def overview(start=None, end=None, hourly_lw=0.25):
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    for ax, (col, label, k) in zip(axes, PANELS):
        for h, df in frames.items():
            s = df[col].loc[start:end] * k
            ax.plot(s.index, s.values, color=C_HOR[h], lw=hourly_lw if col != "EUA" else 1.0,
                    label=f"{LABEL[h]} dataset")
        ax.set_ylabel(label)
    ax = axes[4]
    for col, label in FUELS:
        for h, df in frames.items():
            s = df[col].loc[start:end]
            ax.plot(s.index, s.values, color=C_FUEL[col], lw=1.0, ls="-" if h == "2015-2020" else "--",
                    label=f"{label} – {LABEL[h]}")
    ax.set_ylabel("Fuel prices")
    lo = pd.Timestamp(start) if start else min(df.index[0] for df in frames.values())
    hi = pd.Timestamp(end) if end else max(df.index[-1] for df in frames.values())
    inside = lambda d: lo <= d <= hi                                   # noqa: E731
    for ax in axes:
        ax.axvspan(*OVERLAP, color="#bdbcb6", alpha=0.45, lw=0)
        for h in H:
            for name, d in boundaries(h):
                if inside(d):
                    ax.axvline(d, color=C_HOR[h], ls="--", lw=0.9)
        ax.set_xlim(lo, hi)
    top = axes[0]
    for h in H:
        for name, d in [("Training", H[h]["splits"]["train_start"])] + boundaries(h):
            if inside(d):
                top.text(d, 0.98, f" {name} ({LABEL[h]}) ", rotation=90, va="top", ha="left", fontsize=7,
                         color=C_HOR[h], transform=top.get_xaxis_transform(),
                         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    for ax, ncols, size, loc in ((axes[0], 2, 8, "lower left"), (axes[4], 3, 7, "upper left")):
        leg = ax.legend(loc=loc, ncols=ncols, fontsize=size, frameon=True, facecolor="white",
                        edgecolor="none", framealpha=0.85)
        for line in leg.get_lines():
            line.set_linewidth(2)
    if start is None:
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig

fig = overview()
plt.show()
""")

md(r"""
### 1b. Zoom on the overlap window

Both datasets drawn over the 43 overlap days (±3 weeks). Price is identical by construction for
the shared days; load/RES differ slightly (2015–2020 power data come from `DE.csv`, 2020–2026 from
SMARD) and the daily commodity series differ by the source of the raw quotes.
""")

code(r"""
fig = overview("2020-10-29", "2021-01-21", hourly_lw=0.9)
plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""
## 2. In-sample price distributions & fitted densities (adaptation of Figure 3)

In-sample = first 1,456 days of each horizon (training + validation):
**2015–2018** = 2015-01-01 → 2018-12-26, **2020–2024** = 2020-11-19 → 2024-11-13.
Normal and Johnson's SU densities are fitted by maximum likelihood for each era and hour.
Histograms are clipped to the 0.5–99.5 % range of the pooled data for readability (fits use all points).
""")

code(r"""
HOURS = [0, 6, 12, 18]
ERA = {"2015-2020": "2015–2018", "2020-2026": "2020–2024"}
N_IN = META["windows"]["N_INSAMPLE"]
insample = {h: frames[h]["Price"].to_numpy()[: N_IN * 24].reshape(N_IN, 24) for h in frames}

fits, fig, axes = [], *plt.subplots(1, 4, figsize=(15, 3.8), sharey=False)
for ax, hr in zip(axes, HOURS):
    pooled = np.concatenate([insample[h][:, hr] for h in insample])
    lo, hi = np.percentile(pooled, [0.5, 99.5])
    grid = np.linspace(lo, hi, 400)
    for h, era in ERA.items():
        y = insample[h][:, hr]
        ax.hist(y, bins=np.linspace(lo, hi, 70), density=True, alpha=0.35, color=C_HOR[h], label=f"{era} In-Sample")
        mu, sd = stats.norm.fit(y)
        a, b, loc, scale = stats.johnsonsu.fit(y)
        ls = "--" if h == "2015-2020" else "-"
        ax.plot(grid, stats.norm.pdf(grid, mu, sd), color=C_HOR[h], ls=ls, lw=1.4, alpha=0.6, label=f"Normal ({era})")
        ax.plot(grid, stats.johnsonsu.pdf(grid, a, b, loc, scale), color=INK if h == "2015-2020" else "#6250d6",
                ls=ls, lw=1.4, label=f"Johnson's SU ({era})")
        fits += [{"era": era, "hour": hr, "distribution": "Normal", "mu (location)": mu, "sigma (scale)": sd,
                  "nu (skewness)": np.nan, "tau (tail weight)": np.nan},
                 {"era": era, "hour": hr, "distribution": "Johnson's SU", "mu (location)": loc, "sigma (scale)": scale,
                  "nu (skewness)": a, "tau (tail weight)": b}]
    ax.set_title(f"Hour {hr}")
    ax.set_xlabel("Price (EUR/MWh)")
axes[0].set_ylabel("Density")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncols=6, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
plt.show()

fit_table = pd.DataFrame(fits).set_index(["hour", "distribution", "era"]).sort_index()
fit_table.style.format("{:.3f}", na_rep="–")
""")

md(r"""
Johnson's SU uses SciPy's parameterisation: location μ = `loc` (ξ), scale σ = `scale` (λ),
skewness ν = `a` (γ), tail weight τ = `b` (δ) — the same as `tfp.distributions.JohnsonSU`
used in the network.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 3. Accuracy & coverage of benchmarks and ensembles (adaptation of Table 1)

Test period: last 554 days of each horizon. MAE from the median, RMSE from the mean, CRPS = mean
pinball loss over 99 percentiles, Rel. CRPS = % improvement over LEAR-QRA (positive = better),
Kupiec = number of hours (out of 24) for which the 50 % / 90 % PI passes the coverage test at 5 %.
Point models (LEAR-Ens, DNN-Ens) have no CRPS / Kupiec. **Bold** = best value in each column.
""")

code(r"""
TABLE1 = ["LEAR-Ens", "DNN-Ens", "naive", "LEAR-QRA", "LEAR-QRM", "DNN-QRA", "DNN-QRM",
          "DDNN-N-pEns", "DDNN-N-qEns", "DDNN-JSU-pEns", "DDNN-JSU-qEns"]
COLS = {"MAE": "MAE", "RMSE": "RMSE", "CRPS": "CRPS (Pinball)", "Rel_CRPS_vs_LEAR_QRA": "Rel. CRPS (% vs LEAR-QRA)",
        "Kupiec_50_passed_hours": "Kupiec 50% (hrs/24)", "Kupiec_90_passed_hours": "Kupiec 90% (hrs/24)"}
table1 = pd.concat({LABEL[h]: H[h]["metrics_table"].loc[TABLE1, list(COLS)].rename(columns=COLS) for h in H}, axis=1)

def bold_best(col):
    best = col.max() if ("Rel." in col.name[1] or "Kupiec" in col.name[1]) else col.min()
    return ["font-weight: bold" if np.isfinite(v) and np.isclose(v, best) else "" for v in col]

fmt = {c: ("{:.0f}" if "Kupiec" in c[1] else "{:.2f}") for c in table1.columns}
table1.style.apply(bold_best, axis=0).format(fmt, na_rep="")
""")

code(r"""
# all models (incl. individual runs) for reference
pd.concat({LABEL[h]: H[h]["metrics_table"] for h in H}, axis=1).style.format("{:.2f}", na_rep="")
""")

# --------------------------------------------------------------------------- #
md(r"""
## 4. CRPS dispersion & Diebold–Mariano tests (adaptations of Figures 5 & 6)

**Left/right:** pinball score of the 4 individual DDNN runs (×), their probability mixture
pEns (X) and quantile average qEns (O), for the Normal and JSU variants, against the
QRA/QRM benchmarks (horizontal lines). Each horizon has its own y-scale (price levels differ
strongly between eras).
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, h in zip(axes, H):
    mt = H[h]["metrics_table"]["CRPS"]
    for x, fam in enumerate(["DDNN-N", "DDNN-JSU"]):
        runs = [mt[f"{fam}-{r}"] for r in range(1, 5)]
        ax.scatter([x - 0.12] * 4, runs, marker="x", s=45, color=INK, label="individual runs" if x == 0 else None)
        ax.scatter([x + 0.05], [mt[f"{fam}-pEns"]], marker="X", s=110, color="#6250d6",
                   edgecolor="white", linewidth=1.5, label="pEns" if x == 0 else None, zorder=3)
        ax.scatter([x + 0.18], [mt[f"{fam}-qEns"]], marker="o", s=80, facecolor="none",
                   edgecolor="#e34948", linewidth=2, label="qEns" if x == 0 else None, zorder=3)
    for m, c in C_BENCH.items():
        ax.axhline(mt[m], color=c, lw=1.6, ls="-" if "LEAR" in m else "--", label=m)
    ax.set_xticks([0, 1], ["Normal", "JSU"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_title(LABEL[h])
    ax.set_ylabel("CRPS (pinball, EUR/MWh)")
axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
plt.show()
""")

md(r"""
**Diebold–Mariano p-values** (one-sided, daily L1-norm of the 24 hourly CRPS). A small p-value in
cell (row *i*, column *j*) means the **column** model's forecasts are significantly better than the
**row** model's. Colour: 0 % dark green → 5 % yellow → ≥ 10 % black.
""")

code(r"""
DM_CMAP = LinearSegmentedColormap.from_list("dm", [(0, "#0a5c0a"), (0.5, "#f5e400"), (1, "#000000")])
DM_CMAP.set_bad("#ffffff")
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)
for ax, h in zip(axes, H):
    dm = H[h]["dm_test"]
    im = ax.imshow(dm.to_numpy(dtype=float), cmap=DM_CMAP, vmin=0, vmax=0.10)
    ax.set_xticks(range(len(dm)), dm.columns, rotation=90)
    ax.set_yticks(range(len(dm)), dm.index)
    ax.grid(False)
    ax.set_title(LABEL[h])
    ax.set_xlabel("column model (better?)")
axes[0].set_ylabel("row model")
cb = fig.colorbar(im, ax=axes, shrink=0.8, ticks=[0, 0.025, 0.05, 0.075, 0.10])
cb.ax.set_yticklabels(["0%", "2.5%", "5%", "7.5%", "≥10%"])
plt.show()
""")

code(r"""
# table view of the DM p-values
pd.concat({LABEL[h]: H[h]["dm_test"] for h in H}).style.format("{:.3f}", na_rep="–")
""")

# --------------------------------------------------------------------------- #
md(r"""
## 5. Prediction-interval strips (adaptation of Figure 4)

`plot_pi_week(horizon, start_date, days=7)` shows the realised price (black) with the 50 %, 90 %
and 98 % central prediction intervals (q25–q75, q05–q95, q01–q99) of LEAR-QRA, DNN-QRA,
DDNN-JSU-qEns and DDNN-N-qEns, plus the day-ahead load and RES forecasts. Any window inside the
554-day test period of the horizon can be used.
""")

code(r"""
PI_MODELS = ["LEAR-QRA", "DNN-QRA", "DDNN-JSU-qEns", "DDNN-N-qEns"]
PI_BANDS = [(98, 0, 98), (90, 4, 94), (50, 24, 74)]       # (level, lower idx, upper idx) in the 99 quantiles

def plot_pi_week(horizon, start_date, days=7, models=PI_MODELS):
    res = H[horizon]
    dates = pd.DatetimeIndex(res["test_dates"])
    start = pd.Timestamp(start_date).normalize()
    if start not in dates or dates.get_loc(start) + days > len(dates):
        raise ValueError(f"{start.date()} + {days} d is outside the test period "
                         f"{dates[0].date()} -> {dates[-1].date()} of {horizon}")
    i0 = dates.get_loc(start)
    sl = slice(i0, i0 + days)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    real = res["actuals"][sl].reshape(-1)
    fig, axes = plt.subplots(len(models) + 1, 1, figsize=(12, 2.1 * (len(models) + 1)), sharex=True)
    for ax, m in zip(axes, models):
        q = res["quantiles"][m][sl].reshape(-1, 99)
        for level, lo, hi in PI_BANDS:
            ax.fill_between(idx, q[:, lo], q[:, hi], color=PI_FILL[level], lw=0, step="mid", label=f"{level}% PI")
        ax.plot(idx, real, color=INK, lw=1.2, drawstyle="steps-mid", label="real")
        ax.set_ylabel("EUR/MWh")
        ax.set_title(m, loc="left", fontsize=9)
    ax = axes[-1]
    ext = frames[horizon].loc[idx]
    ax.plot(idx, ext["Load_DA_Forecast"] / 1e3, color="#2a78d6", label="Load forecast")
    ax.plot(idx, ext["Renewables_DA_Forecast"] / 1e3, color="#1baf7a", label="RES forecast")
    ax.set_ylabel("GWh")
    ax.set_title("External factors", loc="left", fontsize=9)
    ax.legend(loc="upper left", ncols=2)
    axes[0].legend(loc="upper left", ncols=4)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b %Y"))
    fig.suptitle(f"{LABEL[horizon]}: {start.date()} → {(start + pd.Timedelta(days=days - 1)).date()}", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout()
    return fig

def extreme_week(horizon, kind, days=7):
    # window of `days` test days centred on the day with the lowest ('negative') / highest ('spike') hourly price
    a = H[horizon]["actuals"]
    day = int(np.argmin(a.min(1)) if kind == "negative" else np.argmax(a.max(1)))
    start = int(np.clip(day - days // 2, 0, len(a) - days))
    return pd.Timestamp(H[horizon]["test_dates"][start])
""")

md("### 5a. Paper benchmark week (sanity check): 2020-09-12 → 2020-09-18, 2015–2020 horizon")
code(r"""
fig = plot_pi_week("2015-2020", "2020-09-12"); plt.show()
""")

md("### 5b. 2020–2026: week around the most negative test-period price")
code(r"""
wk = extreme_week("2020-2026", "negative"); print("start:", wk.date())
fig = plot_pi_week("2020-2026", wk); plt.show()
""")

md("### 5c. 2020–2026: week around the highest test-period price spike")
code(r"""
wk = extreme_week("2020-2026", "spike"); print("start:", wk.date())
fig = plot_pi_week("2020-2026", wk); plt.show()
""")

md("### 5d. Custom window\nChange `HORIZON`, `START` and `DAYS` and re-run the cell.")
code(r"""
HORIZON, DAYS = "2020-2026", 7
START = pd.Timestamp(H[HORIZON]["test_dates"][100])     # e.g. "2025-12-22"
fig = plot_pi_week(HORIZON, START, DAYS); plt.show()
""")

# --------------------------------------------------------------------------- #
md("## Appendix A. Hyper-parameters re-used from the paper's Optuna studies")
code(r"""
hp = H["2015-2020"]["selected_hyperparams"]
rows = []
for fam, runs in hp.items():
    for r, info in runs.items():
        p = info["params"]
        rows.append({"family": fam, "run": r, "trial": info["trial_number"], "val. loss": info["best_value"],
                     "neurons": f"{p['neurons_1']}/{p['neurons_2']}", "activations": f"{p['activation_1']}/{p['activation_2']}",
                     "learning rate": p["learning_rate"], "dropout": p.get("dropout_rate") if p.get("dropout") else 0.0,
                     "inputs": ", ".join(g for g, on in info["feature_mask"].items() if on)})
pd.DataFrame(rows).set_index(["family", "run"])
""")

md("## Appendix B. Flaws of the published code (fixed for both horizons) and run metadata")
code(r"""
for k, v in META["original_code_flaws_fixed"].items():
    print(f"- {k}: {v}")
print()
for k, v in META["definitions"].items():
    print(f"- {k}: {v}")
print("\nsettings:", META["settings"])
print("versions:", META["versions"])
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}

if __name__ == "__main__":
    path = REPO_ROOT / "DDNN_Comparative_Analysis.ipynb"
    nbf.write(nb, path)
    print("wrote", path)
