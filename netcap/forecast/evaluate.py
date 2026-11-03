"""Forecast metrics and significance. Same estimators as the forecasting module (common.dm_test,
common.block_bootstrap_gain), re-implemented without TensorFlow and checked equal in tests.

All errors are over the (sample, flow) cells where the sample mask is True, in the panel's unit.
"""
from math import ceil, erfc, sqrt

import numpy as np

from .. import config

LEVEL = 0.90                     # split-conformal interval level, fixed in advance


def point_metrics(y, pred, m, to_unit):
    e = (pred - y) * to_unit
    ae = np.abs(e)[m]
    den = (np.abs(y) + np.abs(pred))[m]
    num = 2 * np.abs(pred - y)[m]
    smape = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    return {"mae": float(ae.mean()), "rmse": float(np.sqrt((e[m] ** 2).mean())), "smape_pct": float(smape.mean() * 100),
            "cells": int(m.sum())}


def step_loss(y, pred, m, to_unit):
    """Mean absolute error per sample (time step) over its known flows; NaN-free (steps with no flow dropped)."""
    ae = np.where(m, np.abs(pred - y) * to_unit, 0.0)
    cnt = m.sum(1)
    keep = cnt > 0
    return ae[keep].sum(1) / cnt[keep], keep


def dm_test(loss_a, loss_b, lag):
    """Diebold-Mariano with Newey-West (Bartlett) variance; returns (mean d, stat, two-sided p). d < 0: A better."""
    d = np.asarray(loss_a, np.float64) - np.asarray(loss_b, np.float64)
    n = len(d)
    e = d - d.mean()
    var = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    stat = d.mean() / sqrt(max(var, 1e-300) / n)
    return float(d.mean()), float(stat), float(erfc(abs(stat) / sqrt(2)))


def block_bootstrap_gain(err_a, err_b, block, n_boot=2000, seed=config.SEED):
    """MAE change of A vs B in % (positive = A better) with a 95% moving-block bootstrap interval."""
    a, b = np.asarray(err_a, np.float64), np.asarray(err_b, np.float64)
    n = len(a)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    g = (1 - a[idx].sum(1) / b[idx].sum(1)) * 100
    lo, hi = np.percentile(g, [2.5, 97.5])
    return float((1 - a.sum() / b.sum()) * 100), float(lo), float(hi)


def compare(y, pa, pb, m, to_unit, block):
    """A vs B on the same cells: gain %, 95% day-block interval, DM p-value (lag = one day)."""
    la, keep = step_loss(y, pa, m, to_unit)
    lb, _ = step_loss(y, pb, m, to_unit)
    g, lo, hi = block_bootstrap_gain(la, lb, block)
    _, stat, pval = dm_test(la, lb, block)
    return {"gain_pct": g, "ci95": [lo, hi], "dm_stat": stat, "dm_p": pval}


def conformal_margins(y_va, p_va, m_va, level=LEVEL):
    """Per-flow split-conformal margin: the ceil((n+1) level)-th smallest |error| on validation.
    Returns (F,) margins in raw units (inf where a flow has too few validation cells)."""
    F = y_va.shape[1]
    q = np.full(F, np.inf)
    for f in range(F):
        r = np.sort(np.abs(p_va[m_va[:, f], f] - y_va[m_va[:, f], f]))
        n = len(r)
        k = ceil((n + 1) * level)
        if n and k <= n:
            q[f] = r[k - 1]
    return q


def conformal_report(y_va, p_va, m_va, y_te, p_te, m_te, to_unit, level=LEVEL):
    q = conformal_margins(y_va, p_va, m_va, level)
    inside = (np.abs(p_te - y_te) <= q)[m_te]
    upper = (y_te <= p_te + q)[m_te]
    w = np.broadcast_to(2 * q, y_te.shape)[m_te]
    return {"level": level, "coverage_two_sided": float(inside.mean()), "coverage_upper": float(upper.mean()),
            "mean_width": float(np.mean(w[np.isfinite(w)]) * to_unit), "flows_without_margin": int((~np.isfinite(q)).sum())}
