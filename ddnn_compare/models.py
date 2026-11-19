"""Forecasting models: LEAR, (D)DNN, naive, QRA / QRM.

Each rolling model works on *OOS day indices* i = 0..735, i.e. absolute day
t = N_INSAMPLE + i. Nothing here touches the disk; see runner.py.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
from scipy.optimize import linprog

from .config import MAX_LAG, N_INSAMPLE, N_QRA, NN_TRAIN_DAYS, TAUS, RunSettings
from .features import DailyData, column_mask

# --------------------------------------------------------------------------- #
# LEAR (lear.py): 24 hourly LassoCV models on all inputs, window `cal` days
# --------------------------------------------------------------------------- #
def lear_forecast(data: DailyData, cal: int, oos_days: list[int], s: RunSettings) -> np.ndarray:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    out = np.zeros((len(oos_days), 24))
    models = None
    for n, i in enumerate(oos_days):
        t = N_INSAMPLE + i
        if models is None or i % s.lear_retrain_every == 0:
            rows = np.arange(t - cal + MAX_LAG, t)        # same sample as lear.py (cal - 7 days)
            X, Y = data.X[rows], data.Y[rows]
            scaler = StandardScaler().fit(X) if s.lear_standardize else None
            Xs = scaler.transform(X) if scaler else X
            models = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                for h in range(24):
                    models.append(LassoCV(eps=1e-6, alphas=s.lear_n_alphas, cv=s.lear_cv).fit(Xs, Y[:, h]))
        xf = data.X[t:t + 1]
        xf = scaler.transform(xf) if scaler else xf
        out[n] = [m.predict(xf)[0] for m in models]
    return out


# --------------------------------------------------------------------------- #
# (D)DNN (nn_rolling_pred.py), rebuilt without tfp.layers for robustness
# --------------------------------------------------------------------------- #
PARAM_NAMES = {"Normal": ["loc", "scale"], "JSU": ["loc", "scale", "tailweight", "skewness"], None: []}
ACTIVATIONS = ["sigmoid", "relu", "elu", "tanh", "softplus", "softmax"]


def _tf(threads: int):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass  # already initialised in this process
    import tf_keras as keras
    return tf, keras


def _transform(tf, raw, dist):
    """Raw linear outputs -> distribution parameters (same links as the paper code)."""
    p = {"loc": raw[..., :24], "scale": 1e-3 + 3 * tf.math.softplus(raw[..., 24:48])}
    if dist == "JSU":
        p["tailweight"] = 1 + 3 * tf.math.softplus(raw[..., 48:72])
        p["skewness"] = raw[..., 72:96]
    return p


def build_nn(params: dict, n_inputs: int, dist: str | None, threads: int = 2):
    tf, keras = _tf(threads)
    reg = keras.regularizers.L1
    inputs = keras.Input(shape=(n_inputs,))
    x = keras.layers.BatchNormalization()(inputs)
    if params.get("dropout"):
        x = keras.layers.Dropout(params["dropout_rate"])(x)
    for k in (1, 2):
        act_rate = params[f"h{k}_activation_rate_l1"] if params.get(f"regularize_h{k}_activation") else 0.0
        ker_rate = params[f"h{k}_kernel_rate_l1"] if params.get(f"regularize_h{k}_kernel") else 0.0
        x = keras.layers.Dense(params[f"neurons_{k}"], activation=params[f"activation_{k}"],
                               kernel_regularizer=reg(ker_rate), activity_regularizer=reg(act_rate))(x)
    opt = keras.optimizers.Adam(params["learning_rate"])
    if dist is None:
        model = keras.Model(inputs, keras.layers.Dense(24, activation="linear")(x))
        model.compile(optimizer=opt, loss="mae")
        return model
    heads = []
    for name in PARAM_NAMES[dist]:
        rate = params.get(f"{name}_rate_l1", 0.0) if params.get(f"regularize_{name}") else 0.0
        heads.append(keras.layers.Dense(24, activation="linear", kernel_regularizer=reg(rate))(x))
    model = keras.Model(inputs, keras.layers.Concatenate()(heads))
    import tensorflow_probability as tfp
    tfd = tfp.distributions

    def nll(y, raw):
        p = _transform(tf, raw, dist)
        d = tfd.Normal(p["loc"], p["scale"]) if dist == "Normal" else \
            tfd.JohnsonSU(skewness=p["skewness"], tailweight=p["tailweight"], loc=p["loc"], scale=p["scale"])
        return -d.log_prob(y)

    model.compile(optimizer=opt, loss=nll)
    return model


def nn_forecast(data: DailyData, params: dict, dist: str | None, oos_days: list[int], s: RunSettings,
                seed: int) -> dict[str, np.ndarray]:
    """Rolling (re)training on the last 1,449 days; returns {'point'} or distribution params."""
    tf, keras = _tf(s.tf_threads)
    mask = column_mask(params)
    names = PARAM_NAMES[dist] or ["point"]
    out = {k: np.zeros((len(oos_days), 24)) for k in names}
    model = None
    for n, i in enumerate(oos_days):
        t = N_INSAMPLE + i
        if model is None or i % s.retrain_every == 0:
            keras.backend.clear_session()
            day_seed = seed + t
            tf.keras.utils.set_random_seed(day_seed)
            rng = np.random.default_rng(day_seed)
            rows = np.arange(t - NN_TRAIN_DAYS, t)
            X, Y = data.X[rows][:, mask].astype("float32"), data.Y[rows].astype("float32")
            perm = rng.permutation(len(rows))
            n_tr = int((1 - s.val_fraction) * len(perm))
            tr, va = perm[:n_tr], perm[n_tr:]
            model = build_nn(params, int(mask.sum()), dist, s.tf_threads)
            cb = [keras.callbacks.EarlyStopping(patience=s.patience, restore_best_weights=True)]
            model.fit(X[tr], Y[tr], validation_data=(X[va], Y[va]), epochs=s.epochs,
                      batch_size=s.batch_size, callbacks=cb, verbose=0)
        raw = model(data.X[t:t + 1][:, mask].astype("float32"), training=False)
        if dist is None:
            out["point"][n] = raw.numpy()[0]
        else:
            p = _transform(tf, raw, dist)
            for k in names:
                out[k][n] = p[k].numpy()[0]
    return out


# --------------------------------------------------------------------------- #
# Naive benchmark: similar-day rule + empirical residual quantiles (1,456 days)
# --------------------------------------------------------------------------- #
def naive_point(data: DailyData) -> np.ndarray:
    """Y_{T-7} for Mon/Sat/Sun, Y_{T-1} otherwise (rows < 7 are NaN)."""
    n = len(data.Y)
    out = np.full((n, 24), np.nan)
    d = np.arange(MAX_LAG, n)
    dow = data.dates.dayofweek.to_numpy()[d]
    weekly = np.isin(dow, (0, 5, 6))
    out[d] = np.where(weekly[:, None], data.Y[d - 7], data.Y[d - 1])
    return out


def naive_quantiles(data: DailyData, oos_days: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """q_tau(T,h) = naive(T,h) + empirical tau-quantile of the in-window residuals of hour h."""
    point = naive_point(data)
    resid = data.Y - point
    qs = np.zeros((len(oos_days), 24, len(TAUS)))
    pts = np.zeros((len(oos_days), 24))
    for n, i in enumerate(oos_days):
        t = N_INSAMPLE + i
        win = resid[max(MAX_LAG, t - N_INSAMPLE):t]                 # rolling 1,456-day window
        qs[n] = point[t][:, None] + np.quantile(win, TAUS, axis=0).T
        pts[n] = point[t]
    return pts, qs


# --------------------------------------------------------------------------- #
# QRA / QRM (QRA.py): pooled-hour linear quantile regression on 182 OOS days
# --------------------------------------------------------------------------- #
def quantile_regression(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Exact linear quantile regression via the LP dual (HiGHS).

    Columns are rescaled to unit max-abs for numerical conditioning (the solution
    is mapped back); if a solver variant ends without a proven optimum, the
    next one is tried.
    """
    scale = np.abs(X).max(0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    res = None
    for method in ("highs", "highs-ds", "highs-ipm"):
        res = linprog(-y, A_eq=Xs.T, b_eq=(1 - tau) * Xs.sum(0), bounds=(0, 1), method=method)
        if res.status == 0:
            return -res.eqlin.marginals / scale
    raise RuntimeError(f"Quantile regression failed (tau={tau}): {res.message}")


def qra_forecast(forecasts: np.ndarray, actuals: np.ndarray, test_days: list[int], s: RunSettings,
                 average: bool) -> np.ndarray:
    """forecasts: (736, 24, K) OOS point forecasts; actuals (736, 24).

    QRA: regressors = the K forecasts + intercept; QRM: their mean + intercept.
    Quantiles are sorted per hour to avoid crossing (as in QRA.py).
    test_days are indices 0..553 of the test period (OOS index N_QRA + j).
    """
    F = forecasts.mean(-1, keepdims=True) if average else forecasts
    out = np.zeros((len(test_days), 24, len(TAUS)))
    betas = None
    for n, j in enumerate(test_days):
        i = N_QRA + j
        if betas is None or j % s.qra_refit_every == 0:
            Xc = np.concatenate([F[i - N_QRA:i].reshape(-1, F.shape[-1]), np.ones((N_QRA * 24, 1))], axis=1)
            yc = actuals[i - N_QRA:i].reshape(-1)
            betas = np.column_stack([quantile_regression(Xc, yc, tau) for tau in TAUS])   # (K+1, T)
        Xf = np.concatenate([F[i], np.ones((24, 1))], axis=1)
        out[n] = np.sort(Xf @ betas, axis=1)
    return out
