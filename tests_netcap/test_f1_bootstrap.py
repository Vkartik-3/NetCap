"""Warning-F1 day-block bootstrap (netcap.scenario.run.f1_bootstrap); synthetic counts, no data needed."""
import numpy as np

from netcap.scenario.run import BLOCK_STEPS, f1_bootstrap


def _counts(n, tp_rows, fp_rows=(), fn_rows=()):
    c = np.zeros((n, 3), int)
    c[list(tp_rows), 0] = 1
    c[list(fp_rows), 1] = 1
    c[list(fn_rows), 2] = 1
    return c


def test_identical_method_has_zero_difference_and_events_are_reported():
    n = BLOCK_STEPS * 10
    rows = np.arange(n)
    ev = range(3 * BLOCK_STEPS, 3 * BLOCK_STEPS + 50)
    c = _counts(n, ev)
    out = f1_bootstrap({"persistence": (rows, c), "same": (rows, c.copy())})
    assert out["same"]["warning_f1_diff_vs_persistence"] == 0
    assert out["same"]["warning_f1_diff_ci95"] == [0.0, 0.0]
    assert out["_events"]["days_with_events"] == 1 and out["_events"]["largest_day_share"] == 1.0


def test_resamples_without_events_are_undefined_not_zero():
    n = BLOCK_STEPS * 10
    rows = np.arange(n)
    ev = list(range(50))                                   # events only on day 0
    out = f1_bootstrap({"persistence": (rows, _counts(n, ev)), "never": (rows, _counts(n, [], fn_rows=ev))})
    r = out["never"]
    assert r["bootstrap_resamples_defined"] < 2000          # resamples that miss day 0 are dropped
    assert r["warning_f1_diff_vs_persistence"] == -1.0
    assert r["share_resamples_worse_than_persistence"] == 1.0


def test_methods_on_other_rows_are_skipped():
    n = BLOCK_STEPS * 4
    c = _counts(n, [1, 2])
    out = f1_bootstrap({"persistence": (np.arange(n), c), "shifted": (np.arange(n) + 1, c)})
    assert "shifted" not in out
