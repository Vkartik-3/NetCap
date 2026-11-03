"""Anomaly detectors and labelled evaluation.

Detectors (all score every (time, series) cell; higher = more anomalous; log1p space):
  threshold  robust z of the value against the series' training median and MAD (no seasonality): the baseline.
  stl        robust z of the STL residual (statsmodels STL, robust=True, period = one day). RETROSPECTIVE:
             STL uses the whole series, so this is offline detection, not an online alarm.
  iforest    scikit-learn IsolationForest pooled over series on causal features (value vs training
             median, change from the previous step, from the same time yesterday, from the median of the
             last 12 steps), each scaled by the series' training MAD.

Labels
  INJECTED   synthetic anomalies (spike, level shift, dropout) written into a copy of the real series at
             random places (seeded). They are labelled injected everywhere they appear.
  weak       Abilene only: cells removed by the cleaning rules R2 (flow outage), R3 (router outage) and
             R4 (impossible value) on the RAW, uncleaned data. These are data faults, not traffic
             anomalies, and detectors also flag real anomalies the rules do not cover, so precision
             against weak labels is a lower bound. R1 (duplicated day) is not a value anomaly and is excluded.
Thresholds are chosen on the validation period (max F1 on injected labels) and applied to test.
"""
import numpy as np

from .. import config

STL_SEASONAL = 13            # see DECISIONS.md 4.1: the default 7 absorbed single-step spikes into the season
SPLIT = (0.6, 0.8)               # train [0, 60%), val [60, 80%), test [80, 100%) of the time rows
INJECT = {"per_period": 300, "spike_factor": (3.0, 8.0), "shift_factor_up": (1.8, 3.0),
          "shift_factor_down": (0.15, 0.5), "shift_steps": (12, 72), "dropout_steps": (6, 36)}
IFOREST = dict(n_estimators=200, max_samples=256, random_state=config.SEED)
IFOREST_TRAIN_CELLS = 200_000


def periods(n):
    a, b = int(SPLIT[0] * n), int(SPLIT[1] * n)
    return {"train": (0, a), "val": (a, b), "test": (b, n)}


def robust_scale(x, known, rows):
    """Median and MAD (x 1.4826) per series over the known cells of the given rows."""
    med = np.zeros(x.shape[1])
    mad = np.ones(x.shape[1])
    for f in range(x.shape[1]):
        v = x[rows][known[rows, f], f]
        if len(v):
            med[f] = np.median(v)
            m = 1.4826 * np.median(np.abs(v - med[f]))
            mad[f] = m if m > 0 else max(v.std(), 1e-6)
    return med, mad


def inject(values, known, active, day, rng, scale=None):
    """Copy of `values` with labelled anomalies; returns (values, label (N, F) bool, events list)."""
    v = values.copy()
    lab = np.zeros(values.shape, bool)
    events = []
    n = len(values)
    flows = np.flatnonzero(active)
    for name, (lo, hi) in periods(n).items():
        for _ in range(INJECT["per_period"]):
            kind = rng.choice(["spike", "shift_up", "shift_down", "dropout"])
            f = int(rng.choice(flows))
            if kind == "spike":
                L = 1
            elif kind == "dropout":
                L = int(rng.integers(*INJECT["dropout_steps"]))
            else:
                L = int(rng.integers(*INJECT["shift_steps"]))
            t = int(rng.integers(max(lo, day), hi - L))
            sl = slice(t, t + L)
            if not known[sl, f].all() or lab[max(0, t - day):t + L + day, f].any():
                continue                                          # keep events apart and on known data
            if kind == "spike":
                v[sl, f] *= rng.uniform(*INJECT["spike_factor"])
            elif kind == "shift_up":
                v[sl, f] *= rng.uniform(*INJECT["shift_factor_up"])
            elif kind == "shift_down":
                v[sl, f] *= rng.uniform(*INJECT["shift_factor_down"])
            else:
                v[sl, f] = 0.0
            lab[sl, f] = True
            events.append({"period": name, "kind": kind, "series": f, "start": t, "steps": L})
    return v, lab, events


def score_threshold(x, known, tr):
    lx = np.log1p(x)
    med, mad = robust_scale(lx, known, tr)
    return np.abs(lx - med) / mad


STL_JOBS = 6                 # series are decomposed in parallel processes (same result, less wall time)


def _stl_one(s, k, period, tr_start, tr_stop):
    from statsmodels.tsa.seasonal import STL
    if k.sum() < 4 * period:
        return np.zeros_like(s)
    s = s.copy()
    first = np.flatnonzero(k)[0]
    s[:first] = s[first]
    for i in np.flatnonzero(~k):
        if i >= first:
            s[i] = s[i - period] if i >= period else s[i - 1]
    r = STL(s, period=period, seasonal=STL_SEASONAL, robust=True).fit().resid
    rows = np.arange(tr_start, tr_stop)
    rv = r[rows][k[rows]]
    med = np.median(rv)
    mad = 1.4826 * np.median(np.abs(rv - med)) or rv.std() or 1.0
    return np.abs(r - med) / mad


def score_stl(x, known, tr, period, n_jobs=STL_JOBS):
    """STL on log1p values; unknown cells are filled with the previous day's value before decomposing
    (and are never scored). One independent decomposition per series, run in parallel."""
    from joblib import Parallel, delayed
    lx = np.log1p(x)
    cols = Parallel(n_jobs=n_jobs)(delayed(_stl_one)(lx[:, f], known[:, f], period, tr.start, tr.stop)
                                   for f in range(lx.shape[1]))
    return np.stack(cols, 1)


def iforest_features(x, known, tr, day):
    lx = np.log1p(x)
    med, mad = robust_scale(lx, known, tr)
    prev = np.vstack([lx[:1], lx[:-1]])
    yday = np.vstack([np.repeat(lx[:1], day, 0), lx[:-day]])
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.vstack([np.repeat(lx[:1], 12, 0), lx])
    roll = np.median(sliding_window_view(pad, 12, axis=0)[:-1], axis=-1)        # median of the 12 previous steps
    feats = [(lx - med) / mad, (lx - prev) / mad, (lx - yday) / mad, (lx - roll) / mad]
    return np.stack(feats, -1)


def score_iforest(x, known, tr, day, seed=config.SEED):
    from sklearn.ensemble import IsolationForest
    X = iforest_features(x, known, tr, day)
    rng = np.random.default_rng(seed)
    ti, fi = np.nonzero(known[tr])
    pick = rng.choice(len(ti), size=min(IFOREST_TRAIN_CELLS, len(ti)), replace=False)
    m = IsolationForest(**IFOREST).fit(X[tr][ti[pick], fi[pick]])
    s = -m.score_samples(X.reshape(-1, X.shape[-1])).reshape(x.shape)
    return s


def evaluate(score, label, known, rows_val, rows_test, events=None, n_thresholds=200):
    """Threshold chosen on validation (max F1), applied to test. Also threshold-free average precision."""
    from sklearn.metrics import average_precision_score
    sv, lv = score[rows_val][known[rows_val]], label[rows_val][known[rows_val]]
    st, lt = score[rows_test][known[rows_test]], label[rows_test][known[rows_test]]
    cand = np.unique(np.quantile(sv, np.linspace(0.90, 0.99995, n_thresholds)))
    best = (-1.0, None)
    for c in cand:
        pr = sv >= c
        tp = (pr & lv).sum()
        f1 = 2 * tp / max(pr.sum() + lv.sum(), 1)
        if f1 > best[0]:
            best = (f1, c)
    thr = best[1]
    pr = st >= thr
    tp, fp, fn = int((pr & lt).sum()), int((pr & ~lt).sum()), int((~pr & lt).sum())
    res = {"threshold_from_val": float(thr), "val_f1": float(best[0]),
           "test": {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                    "f1": 2 * tp / max(2 * tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn,
                    "flag_rate": float(pr.mean()), "positives": int(lt.sum()), "cells": int(len(lt))},
           "test_average_precision": float(average_precision_score(lt, st)) if lt.any() else None,
           "test_base_rate": float(lt.mean())}
    if events is not None:
        by = {}
        for e in events:
            if e["period"] != "test":
                continue
            hit = bool((score[e["start"]:e["start"] + e["steps"], e["series"]] >= thr).any())
            d = by.setdefault(e["kind"], [0, 0])
            d[0] += hit
            d[1] += 1
        res["test_event_recall"] = {k: {"detected": a, "events": b, "recall": a / b} for k, (a, b) in by.items()}
    return res
