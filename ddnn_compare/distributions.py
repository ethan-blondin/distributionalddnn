"""Closed-form quantiles, CDFs and means for Normal / Johnson's SU, plus mixtures.

Parameterisation follows tfp.distributions.JohnsonSU (as in the original code):
    X = loc + scale * sinh((Z - skewness) / tailweight),   Z ~ N(0, 1)
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

from .config import TAUS


def quantiles(dist: str, p: dict, taus: np.ndarray = TAUS) -> np.ndarray:
    """p: arrays of shape (...,) -> returns (..., n_taus)."""
    z = ndtri(taus)
    loc, scale = p["loc"][..., None], p["scale"][..., None]
    if dist == "Normal":
        return loc + scale * z
    if dist == "JSU":
        skew, tail = p["skewness"][..., None], p["tailweight"][..., None]
        return loc + scale * np.sinh((z - skew) / tail)
    raise ValueError(dist)


def cdf(dist: str, p: dict, x: np.ndarray) -> np.ndarray:
    """x broadcastable to p[...]."""
    if dist == "Normal":
        return ndtr((x - p["loc"]) / p["scale"])
    if dist == "JSU":
        return ndtr(p["skewness"] + p["tailweight"] * np.arcsinh((x - p["loc"]) / p["scale"]))
    raise ValueError(dist)


def mean(dist: str, p: dict) -> np.ndarray:
    if dist == "Normal":
        return p["loc"]
    if dist == "JSU":
        d, g = p["tailweight"], p["skewness"]
        return p["loc"] - p["scale"] * np.exp(0.5 / d**2) * np.sinh(g / d)
    raise ValueError(dist)


def mixture_quantiles(dist: str, runs: list[dict], taus: np.ndarray = TAUS, n_iter: int = 60) -> np.ndarray:
    """Quantiles of the equal-weight mixture (vertical / probability averaging, pEns).

    Solved by vectorised bisection on the averaged CDF, bracketed by the min/max
    of the component quantiles (the mixture quantile always lies between them).
    """
    comp = np.stack([quantiles(dist, r, taus) for r in runs])   # (R, ..., T)
    lo, hi = comp.min(0), comp.max(0)
    t = np.broadcast_to(taus, lo.shape)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        F = np.mean([cdf(dist, {k: v[..., None] for k, v in r.items()}, mid) for r in runs], axis=0)
        below = F < t
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)
