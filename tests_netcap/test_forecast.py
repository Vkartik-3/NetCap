import subprocess
import sys

import numpy as np
import pytest

from netcap import config
from netcap.forecast import evaluate as ev
from netcap.forecast import methods as M
from netcap.forecast.panel import FOLDS, Panel, abilene_targets, abilene_timeline, fold_bounds, sample_mask

needs_abilene = pytest.mark.skipif(not (config.RAW_ABILENE / "X01.gz").exists(), reason="Abilene raw data not present")


def shared_common():
    sys.path.insert(0, str(config.ROOT / "research"))
    import common
    return common


def toy_panel(n=24 * 7 * 6, F=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    vals = 100 + 30 * np.sin(2 * np.pi * t / 24)[:, None] + rng.normal(0, 3, (n, F)) + np.arange(F) * 10
    return Panel("toy", np.clip(vals, 0, None), np.ones((n, F), bool), 24, 168, 1.0, "u", 60,
                 [f"f{i}" for i in range(F)], targets={}, times=np.arange(n))


@needs_abilene
def test_timeline_targets_and_folds_equal_reference():
    common = shared_common()
    from netcap.data import abilene
    weeks = abilene.load_weeks()
    cl = abilene.clean(weeks)
    values, known, to_time = abilene_timeline(weeks, cl)
    bv, bk, bt = common.clean_timeline("abilene")
    assert np.array_equal(values, bv) and np.array_equal(known, bk) and np.array_equal(to_time, bt)
    segs, _ = common.abilene_clean(common.load_abilene_raw())
    for h in (1, 6, 12):
        want = np.concatenate([s + 12 + h - 1 + np.arange(len(seg) - 12 - h + 1) for s, seg in segs if len(seg) >= 12 + h])
        assert np.array_equal(abilene_targets(cl, to_time, h), to_time[want])
        n = len(want)
        X = np.zeros((n, 1))
        tr, _, va, _, te, _ = common.chronological_split(X, X)
        assert [b - a for a, b in fold_bounds(n, "original")] == [len(tr), len(va), len(te)]
        for f, split in FOLDS.items():
            if split:
                assert fold_bounds(n, f) == tuple(zip((0,) + tuple(int(x * n) for x in split[:2]),
                                                      tuple(int(x * n) for x in split)))


def test_dm_bootstrap_ridge_equal_reference():
    common = shared_common()
    rng = np.random.default_rng(1)
    a, b = rng.gamma(2, 1, 600), rng.gamma(2, 1.1, 600)
    assert ev.dm_test(a, b, 24) == pytest.approx(common.dm_test(a, b, 24))
    assert ev.block_bootstrap_gain(a, b, 24) == pytest.approx(common.block_bootstrap_gain(a, b, 24))
    X, Y = rng.normal(size=(200, 8)), rng.normal(size=(200, 3))
    for (w1, b1), (w2, b2) in zip(M.ridge_path(X, Y, [0.1, 10]), common.ridge_path(X, Y, [0.1, 10])):
        assert np.allclose(w1, w2) and np.allclose(b1, b2)


def test_methods_never_look_past_the_origin():
    p = toy_panel()
    h, t = 6, np.arange(24 * 8, 24 * 9)
    before = M.base_inputs(p, t, h)
    p.values[t.min() - h + 1:] *= 50                     # change everything after the first origin
    after = M.base_inputs(p, t[:1], h)
    for k in before:
        assert np.allclose(before[k][:1], after[k])


def test_ets_is_causal_and_learns_a_seasonal_pattern():
    p = toy_panel()
    n = len(p.values)
    rows = np.arange(24 * 8, n)
    va = rows[: len(rows) // 2]
    fc, _ = M.ets(p, {1: rows}, {1: va})
    r, out = fc[1]
    pers = np.abs(p.values[r - 1] - p.values[r]).mean()
    assert np.abs(out - p.values[r]).mean() < 0.5 * pers           # seasonal shape is learned
    cut = va[-1] + 1                                                 # after validation: params cannot change
    p2 = toy_panel()
    p2.values[cut:] += 1000
    fc2, _ = M.ets(p2, {1: rows}, {1: va})
    early = r <= cut                                                 # origin r - 1 < cut
    assert np.allclose(out[early], fc2[1][1][early])


@pytest.mark.parametrize("order", [M.ARIMA_ORDER, (1, 1, 1)])
def test_arima_multistep_formula_equals_statsmodels_forecast(order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    p = toy_panel(n=24 * 7 * 3, F=1)
    p.values[100:103] = 0
    p.known[100:103] = False                                         # a gap: the filter must skip it
    origin, h = 350, 5
    out, meta, full = M.arima_flow(p, 0, 300, {h: np.array([origin + h])}, order=order)
    y = np.where(p.known[:, 0], p.values[:, 0], np.nan)
    ex = M.fourier(len(y), p)[:, (1 if order[1] else 0):]
    ref = SARIMAX(y[:origin + 1], exog=ex[:origin + 1], order=order, trend="n").filter(full.params)
    want = ref.forecast(h, exog=ex[origin + 1:origin + h + 1])[-1]
    assert out[h][0] == pytest.approx(max(want, 0), rel=1e-6)


def test_conformal_margin_is_finite_sample_quantile():
    y = np.zeros((9, 1))
    p = np.arange(1, 10, dtype=float)[:, None]                        # |errors| 1..9
    q = ev.conformal_margins(y, p, np.ones((9, 1), bool), level=0.8)
    assert q[0] == 8                                                  # ceil(10 * 0.8) = 8th smallest
    m = sample_mask(toy_panel(), np.array([30, 40]), 6)
    assert m.all()


def test_forecasting_and_api_never_import_tensorflow():
    code = ("import sys, netcap.forecast.run, netcap.forecast.methods, netcap.forecast.evaluate; "
            "print('tensorflow' in sys.modules or 'keras' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=config.ROOT)
    assert out.stdout.strip() == "False", out.stderr


def test_gbm_features_use_nothing_after_the_origin():
    p = toy_panel(n=24 * 7 * 4)
    p.day, p.week = 24, 168
    rng = np.random.default_rng(3)
    for h in (1, 6):
        for t in rng.integers(24 * 8, len(p.values) - 1, 5):
            t = np.array([t])
            before, lv = M._gbm_features(p, t, h, (slice(None), slice(None)))
            q = toy_panel(n=24 * 7 * 4)
            q.values[t[0] - h + 1:] = rng.uniform(0, 1e6, q.values[t[0] - h + 1:].shape)    # rewrite the future
            after, lv2 = M._gbm_features(q, t, h, (slice(None), slice(None)))
            assert np.array_equal(before, after, equal_nan=True) and np.array_equal(lv, lv2)


@needs_abilene
def test_persistence_equals_project_logged_metrics():
    """Logged persistence metrics, Abilene, 5 min: val 1028174.2265, test 1204681.6653 raw units."""
    from netcap.forecast.panel import load_abilene
    p = load_abilene((1,))
    t = p.targets[1]
    (_, _), (b0, b1), (c0, c1) = fold_bounds(len(t), "original")
    for (lo, hi), want in (((b0, b1), 1028174.2265485929), ((c0, c1), 1204681.6652538313)):
        r = t[lo:hi]
        got = M.mae(p.values[r], p.values[r - 1], np.ones((len(r), 144), bool))
        assert got == pytest.approx(want, rel=1e-6)            # forecasting pipeline uses float32 scaled splits
