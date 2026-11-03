"""Forecasting methods. Every specification here was fixed before any Phase 2 result was seen
(docs/netcap/DECISIONS.md, "Phase 2 frozen specifications"). Choices marked "on validation" are made
on the fold's validation split only; nothing is chosen on test. No TensorFlow anywhere.

All methods forecast y[t] from data up to the origin o = t - h and return raw units clipped at 0.
Notation: last = y[o]; seasonal values are looked up at the TARGET time t (t - lag <= o since h <= lag).
"""
import time

import numpy as np

from .. import config
from .panel import WINDOW, sample_mask

GRID_W = np.linspace(0, 1, 101)                       # blend weights (as forecasting module Phase 8)
ALPHAS = 10.0 ** np.arange(-2, 5)                     # ridge penalties x n (as forecasting module Phase 8)
KS, TIE = (3, 4, 5, 6, 8), 0.05                       # fallback rule thresholds (as forecasting module Phase 10)
TAUS = (1, 2, 3, 5, 8)                                # smooth gate scales (burst attempt 4)
ETS_ALPHA = (0.01, 0.03, 0.1, 0.3, 0.6)
ETS_GAMMA = (0.0, 0.01, 0.05, 0.15)
GBM = dict(loss="absolute_error", max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
           l2_regularization=1.0, early_stopping=False)
GBM_TRAIN_ROWS = 400_000
ARIMA_ORDER = (2, 0, 1)
ARIMA_FOURIER = {"day": 3, "week": 2}                 # harmonics of the daily and weekly cycle
ARIMA_TOP = 10                                        # busiest flows by training mean
ARIMA_ORDERS_SEL = ((2, 0, 1), (1, 1, 1), (0, 1, 1))  # ADDED after results (DECISIONS.md 2.5): order on validation


def mae(y, p, m):
    return float(np.abs(p - y)[m].mean())


def gain(m, base):
    return (1 - m / base) * 100


# --------------------------------------------------------------------------- simple baselines
def lookback(p, t, lag):
    src = t - lag
    ok = src >= 0
    out = np.full((len(t), p.values.shape[1]), np.nan)
    out[ok] = np.where(p.known[src[ok]], p.values[src[ok]], np.nan)
    return out


def seasonal_mean(p, t, n=7):
    stack = np.stack([lookback(p, t, k * p.day) for k in range(1, n + 1)])
    cnt = (~np.isnan(stack)).sum(0)
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, np.nansum(stack, 0) / np.maximum(cnt, 1), np.nan)


def fill(a, last):
    return np.where(np.isnan(a), last, a)


def base_inputs(p, t, h):
    last = p.values[t - h]
    return {"last": last, "seasonal_day": fill(lookback(p, t, p.day), last),
            "seasonal_week": fill(lookback(p, t, p.week), last), "mean7": fill(seasonal_mean(p, t), last)}


def burst_z(p, t, h, tr_t):
    """|log1p(mean7) - log1p(last)| in units of that flow's training spread (as the forecasting module's rule)."""
    def d(tt):
        b = base_inputs(p, tt, h)
        return np.log1p(b["mean7"]) - np.log1p(b["last"])
    sd = d(tr_t).std(0)
    return np.abs(d(t)) / np.where(sd > 0, sd, 1)


def choose_k(y_va, last_va, pred_va, z_va, m_va):
    """Fallback rule threshold on validation: the largest gain over persistence; within TIE of the best
    the smaller (more cautious) k wins (forecasting module Phase 10 rule)."""
    pv = mae(y_va, last_va, m_va)
    sc = {k: gain(mae(y_va, np.where(z_va > k, last_va, pred_va), m_va), pv) for k in KS}
    best = max(sc.values())
    return min(k for k in KS if sc[k] >= best - TIE), sc


# --------------------------------------------------------------------------- ETS(A,N,A)
def ets(p, h_targets, val_targets, period=None):
    """ETS with additive error, no trend, additive season of one day, per flow.

    Parameters (alpha, gamma) per flow and horizon from the ETS_ALPHA x ETS_GAMMA grid, chosen on the
    validation targets. State starts from the first week of the series (always inside training).
    Unknown steps do not update the state. Returns {h: forecast function rows -> (n, F)}, chosen params.
    """
    m = period or p.day
    y, kn = p.values, p.known
    N, F = y.shape
    A = np.array([a for a in ETS_ALPHA for _ in ETS_GAMMA])[:, None]
    G = np.array([g for _ in ETS_ALPHA for g in ETS_GAMMA])[:, None]
    init = slice(0, p.week)
    cnt = kn[init].sum(0)
    lvl0 = np.where(cnt > 0, (y[init] * kn[init]).sum(0) / np.maximum(cnt, 1), 0)
    seas0 = np.zeros((m, F))
    for s in range(m):
        r = np.arange(s, p.week, m)
        c = kn[r].sum(0)
        seas0[s] = np.where(c > 0, (y[r] * kn[r]).sum(0) / np.maximum(c, 1) - lvl0, 0)

    def run(alpha, gamma, want):
        """One pass; want = {h: (rows, out array)}: stores level + season forecast for row t = tau + h."""
        lvl = np.broadcast_to(lvl0, alpha.shape[:-1] + (F,)).copy()
        seas = np.broadcast_to(seas0[:, None, :] if lvl.ndim == 2 else seas0, (m,) + lvl.shape).copy()
        idx = {h: np.full(N, -1) for h in want}
        for h, (rows, _) in want.items():
            idx[h][rows] = np.arange(len(rows))
        for tau in range(N):
            s = tau % m
            k = kn[tau]
            e = np.where(k, y[tau] - (lvl + seas[s]), 0.0)
            lvl += alpha * e
            seas[s] += gamma * e
            for h, (rows, out) in want.items():
                if tau + h < N and idx[h][tau + h] >= 0:
                    out[idx[h][tau + h]] = lvl + seas[(tau + h) % m]
        return want

    chosen, fns = {}, {}
    for h, vt in val_targets.items():
        out = np.zeros((len(vt), len(A), F))
        run(A, G, {h: (vt, out)})
        mv = sample_mask(p, vt, h)
        err = np.where(mv[:, None, :], np.abs(np.clip(out, 0, None) - y[vt][:, None, :]), 0).sum(0)
        g = err.argmin(0)
        chosen[h] = {"alpha": A[g, 0], "gamma": G[g, 0]}
    for h, rows in h_targets.items():
        out = np.zeros((len(rows), F))
        run(chosen[h]["alpha"], chosen[h]["gamma"], {h: (rows, out)})
        fns[h] = (rows, np.clip(out, 0, None))
    return fns, {h: {"alpha_counts": dict(zip(*map(list, np.unique(c["alpha"], return_counts=True)))),
                     "gamma_counts": dict(zip(*map(list, np.unique(c["gamma"], return_counts=True))))}
                 for h, c in chosen.items()}


# --------------------------------------------------------------------------- ridge
def windows(p, t, h, clip_max=None):
    rows = (t - h)[:, None] - np.arange(WINDOW)[::-1]
    X = p.values[rows]                                  # (n, W, F)
    if clip_max is not None:
        X = np.minimum(X, clip_max)
    return X.reshape(len(t), -1)


def ridge_path(X, Y, alphas):
    """Same estimator as common.ridge_path (standardised inputs, centred outputs, penalty alpha * n)."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    lam, V = np.linalg.eigh(Xs.T @ Xs)
    VtXtY = V.T @ (Xs.T @ (Y - ym))
    out = []
    for a in alphas:
        W = (V @ (VtXtY / (lam + a * len(X))[:, None])) / sd[:, None]
        out.append((W, ym - mu @ W))
    return out


def ridge(p, h, tr, va, te, clip=False):
    """Linear model on the last 12 values of every flow (multi-output), penalty chosen on validation.
    clip=True (burst attempt 3): inputs clipped to each flow's training maximum before the linear map."""
    cm = None
    if clip:
        cm = p.values[np.unique((tr - h)[:, None] - np.arange(WINDOW))].max(0)        # (F,) training input maximum
    Xtr, Ytr = windows(p, tr, h, cm), p.values[tr]
    path = ridge_path(Xtr, Ytr, ALPHAS)
    del Xtr
    Xva = windows(p, va, h, cm)
    mv = sample_mask(p, va, h)
    scores = [mae(p.values[va], np.clip(Xva @ W + b, 0, None), mv) for W, b in path]
    i = int(np.argmin(scores))
    W, b = path[i]
    return {"val": np.clip(Xva @ W + b, 0, None), "test": np.clip(windows(p, te, h, cm) @ W + b, 0, None),
            "alpha": float(ALPHAS[i]), "val_mae_by_alpha": dict(zip(ALPHAS.tolist(), scores))}


def ridge_basedef(p, h, tr, va, te):
    """ADDED AFTER SEEING PHASE 2 RESULTS (DECISIONS.md 2.2): the forecasting module's own ridge definition
    (scripts/06_baselines2.py): predict the CHANGE y - last, in two versions (log1p min-max scaled inputs
    and targets with the scaler fitted on training inputs, or raw units); version and penalty on validation."""
    rows_tr = np.unique((tr - h)[:, None] - np.arange(WINDOW))
    lo = np.log1p(p.values[rows_tr]).min(0)
    rg = np.log1p(p.values[rows_tr]).max(0) - lo
    rg = np.where(rg > 0, rg, 1.0)
    sc = lambda a: (np.log1p(a) - lo) / rg
    inv = lambda z: np.expm1(z * rg + lo)
    best = (np.inf, None)
    for space in ("scaled", "raw"):
        f = (lambda X: sc(X)) if space == "scaled" else (lambda X: X)
        def design(t):
            return f(p.values[(t - h)[:, None] - np.arange(WINDOW)[::-1]]).reshape(len(t), -1)
        last = lambda t: f(p.values[t - h])
        Xtr = design(tr)
        path = ridge_path(Xtr, f(p.values[tr]) - last(tr), ALPHAS)
        del Xtr
        Xva = design(va)
        mv = sample_mask(p, va, h)
        for a, (W, b) in zip(ALPHAS, path):
            z = Xva @ W + b + last(va)
            pred = np.clip(inv(z) if space == "scaled" else z, 0, None)
            m_ = mae(p.values[va], pred, mv)
            if m_ < best[0]:
                best = (m_, (space, float(a), W, b))
    space, a, W, b = best[1]
    f = sc if space == "scaled" else (lambda X: X)
    out = {}
    for key, t in (("val", va), ("test", te)):
        z = f(p.values[(t - h)[:, None] - np.arange(WINDOW)[::-1]]).reshape(len(t), -1) @ W + b + f(p.values[t - h])
        out[key] = np.clip(inv(z) if space == "scaled" else z, 0, None)
    out.update(space=space, alpha=a)
    return out


# --------------------------------------------------------------------------- gradient boosting
MAX_CATEGORIES = 255          # scikit-learn's limit for a categorical feature


def _gbm_features(p, t, h, flows, level=None):
    """Rows (sample, flow) for the given flow index array per sample. Log-space, relative to the last value.
    The last feature is the series id (categorical) or, when there are more than MAX_CATEGORIES series,
    `level`: each series' mean log1p value over its training targets (numeric)."""
    b = base_inputs(p, t, h)
    lv = np.log1p(b["last"])
    feats = [lv]
    for k in range(1, WINDOW):
        x = p.values[t - h - k]
        feats.append(np.where(p.known[t - h - k], np.log1p(x), np.nan) - lv)
    for key in ("seasonal_day", "seasonal_week", "mean7"):
        feats.append(np.log1p(b[key]) - lv)
    feats.append(np.log1p(fill(seasonal_mean(p, t - h), b["last"])) - lv)     # seasonal level at the origin
    n = len(t)
    tod = 2 * np.pi * (t % p.day) / p.day
    dow = 2 * np.pi * ((t // p.day) % 7) / 7
    tf = [np.repeat(f[:, None], p.values.shape[1], 1) for f in (np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow))]
    ids = np.arange(p.values.shape[1]) if level is None else level
    fid = np.repeat(ids[None, :], n, 0)
    X = np.stack(feats + tf + [fid], -1).astype(np.float32)          # (n, F, d)
    return X[flows], lv[flows]


def gbm(p, h, tr, va, te, seed=config.SEED):
    """One HistGradientBoostingRegressor pooled over all flows; target log1p(y) - log1p(last); settings
    fixed in advance (GBM), not tuned. Trained on GBM_TRAIN_ROWS random (sample, flow) training rows."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(seed)
    mtr = sample_mask(p, tr, h)
    si, fi = np.nonzero(mtr)
    level = None
    if p.values.shape[1] > MAX_CATEGORIES:                     # DECISIONS.md 2.4
        lv_tr = np.where(mtr, np.log1p(p.values[tr]), 0.0)
        level = lv_tr.sum(0) / np.maximum(mtr.sum(0), 1)
    pick = rng.choice(len(si), size=min(GBM_TRAIN_ROWS, len(si)), replace=False)
    Xall, lvall = _gbm_features(p, tr, h, (slice(None), slice(None)), level)
    X, lv = Xall[si[pick], fi[pick]], lvall[si[pick], fi[pick]]
    del Xall, lvall
    yv = np.log1p(p.values[tr[si[pick]], fi[pick]]) - lv
    d = X.shape[1]
    model = HistGradientBoostingRegressor(**GBM, categorical_features=[d - 1] if level is None else None,
                                          random_state=seed)
    t0 = time.perf_counter()
    model.fit(X, yv)
    fit_s = time.perf_counter() - t0

    def pred(t):
        Xa, la = _gbm_features(p, t, h, (slice(None), slice(None)), level)
        n, F, dd = Xa.shape
        out = la.reshape(-1) + model.predict(Xa.reshape(-1, dd))
        return np.clip(np.expm1(out), 0, None).reshape(n, F)

    t0 = time.perf_counter()
    res = {"val": pred(va), "test": pred(te)}
    res.update(fit_s=fit_s, predict_s=time.perf_counter() - t0, train_rows=int(len(pick)), n_iter=int(model.n_iter_))
    return res


# --------------------------------------------------------------------------- dynamic harmonic regression
def fourier(n_rows, p, start=0):
    t = np.arange(start, start + n_rows)
    cols = [np.ones(n_rows)]
    for per, K in ((p.day, ARIMA_FOURIER["day"]), (p.week, ARIMA_FOURIER["week"])):
        for k in range(1, K + 1):
            cols += [np.sin(2 * np.pi * k * t / per), np.cos(2 * np.pi * k * t / per)]
    return np.stack(cols, 1)


def arima_flow(p, f, train_end, targets_by_h, order=ARIMA_ORDER):
    """ARMA(2,1) errors + daily/weekly Fourier terms (+ constant) for one flow, fitted on rows [0, train_end].
    Exact h-step forecasts from every origin with the fitted parameters held fixed:
    y_hat[t | t-h] = x_t b + Z T^(h-1) a[t-h+1 | t-h]   (Kalman predicted state), checked in tests."""
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = np.where(p.known[:, f], p.values[:, f], np.nan)
    ex = fourier(len(y), p)
    if order[1] > 0:
        ex = ex[:, 1:]                                   # differenced: a constant is not identified
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(y[:train_end + 1], exog=ex[:train_end + 1], order=order, trend="n")
        res = mod.fit(disp=False, maxiter=200)
        full = res.apply(y, exog=ex)
    fit_s = time.perf_counter() - t0
    ssm = full.filter_results
    Z, T = ssm.design[:, :, 0], ssm.transition[:, :, 0]
    a = ssm.predicted_state                              # (k_states, N + 1): a[:, i] = state i given data < i
    beta = full.params[:ex.shape[1]]
    out = {}
    for h, rows in targets_by_h.items():
        Th = np.linalg.matrix_power(T, h - 1)
        out[h] = np.clip(ex[rows] @ beta + (Z @ Th @ a[:, rows - h + 1]).ravel(), 0, None)
    return out, {"fit_s": fit_s, "params": dict(zip(full.model.param_names, map(float, full.params))),
                 "converged": bool(res.mle_retvals.get("converged", False)) if res.mle_retvals else None}, full
