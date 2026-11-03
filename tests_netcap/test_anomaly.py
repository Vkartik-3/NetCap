import numpy as np

from netcap.anomaly import detect as D


def series(n=24 * 60, F=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.exp(5 + 0.5 * np.sin(2 * np.pi * t / 24))[:, None] * rng.lognormal(0, 0.05, (n, F))


def test_injection_labels_exactly_the_changed_cells():
    x = series()
    known = np.ones(x.shape, bool)
    known[100:110, 0] = False
    v, lab, ev = D.inject(x, known, np.ones(4, bool), 24, np.random.default_rng(1))
    assert ev and np.array_equal(lab, ~np.isclose(v, x))       # every label is a changed cell and vice versa
    assert not lab[100:110, 0].any()                            # never on unknown data
    for e in ev:
        assert lab[e["start"]:e["start"] + e["steps"], e["series"]].all()


def test_detectors_find_a_large_spike_and_threshold_comes_from_validation():
    x = series()
    n = len(x)
    known = np.ones(x.shape, bool)
    lab = np.zeros(x.shape, bool)
    P = D.periods(n)
    for t, k in ((P["val"][0] + 50, 10), (P["test"][0] + 50, 40)):    # one val spike sets the threshold
        x[t, 2] *= k
        lab[t, 2] = True
    tr = slice(*P["train"])
    for s in (D.score_threshold(x, known, tr), D.score_stl(x, known, tr, 24), D.score_iforest(x, known, tr, 24)):
        r = D.evaluate(s, lab, known, slice(*P["val"]), slice(*P["test"]))
        assert r["test"]["tp"] == 1 and r["test"]["fn"] == 0
        assert r["threshold_from_val"] <= s[P["val"][0] + 50, 2]
