import numpy as np

from netcap.data import abilene as ab

from .conftest import needs_abilene

STARTS3 = ["2004-03-01", "2004-03-08", "2004-03-15"]       # three back-to-back weeks


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    w = rng.uniform(1e3, 1e5, size=(3, ab.STEPS_PER_WEEK, ab.N_FLOWS))
    f7 = w[:, :, 7]                                                # view
    f7[:] = 0
    f7[rng.random((3, ab.STEPS_PER_WEEK)) < 0.3] = 50.0           # flow 7 is zero ~70% of the time: never R2
    return w


def test_clean_untouched_synthetic_keeps_everything():
    c = ab.clean(synthetic(), STARTS3)
    assert c.keep.all() and c.segments == [(0, 3 * ab.STEPS_PER_WEEK)]
    assert c.removed() == {r: 0 for r in ab.REASONS}


def test_r1_duplicate_day_removed_and_cuts_segment():
    w = synthetic()
    w[1, :288] = w[0, -288:]
    c = ab.clean(w, STARTS3)
    T = ab.STEPS_PER_WEEK
    assert c.duplicate_weeks == [2]
    assert (c.reason[T:T + 288] == "duplicate").all() and c.removed()["duplicate"] == 288
    assert c.segments == [(0, T), (T + 288, 3 * T)]


def test_r2_flow_outage_needs_one_full_day():
    w = synthetic()
    w[0, 100:100 + 288, 3] = 0                    # exactly 1 day: removed
    w[2, 500:500 + 287, 4] = 0                    # one step short: kept
    c = ab.clean(w, STARTS3)
    assert c.flow_outages == [(3, 100, 388)]
    assert c.removed()["flow"] == 288 and c.keep[2 * ab.STEPS_PER_WEEK + 500:].all()


def test_r3_router_outage_one_hour():
    w = synthetic()
    r = 5
    w[1, 40:52, r * 12:(r + 1) * 12] = 0          # all traffic OUT of router 5 for 12 steps
    w[1, 900:911, r::12] = 0                      # all traffic INTO router 5 for 11 steps: kept
    c = ab.clean(w, STARTS3)
    assert [o[:2] for o in c.router_outages] == [(ab.NODES[r], "out of")]
    assert c.removed()["router"] == 12


def test_r4_impossible_value_single_step():
    w = synthetic()
    w[2, 10, 99] = ab.MAX_FLOW_RAW * 1.01
    c = ab.clean(w, STARTS3)
    assert c.impossible_steps == [2 * ab.STEPS_PER_WEEK + 10] and c.removed()["impossible"] == 1


def test_first_rule_wins_and_non_adjacent_weeks_cut():
    w = synthetic()
    w[0, 0:288, 3] = 0                            # flow outage ...
    w[0, 5, 1] = ab.MAX_FLOW_RAW * 2              # ... that also holds an impossible value
    c = ab.clean(w, ["2004-03-01", "2004-03-08", "2004-03-20"])     # week 3 not 7 days after week 2
    assert c.reason[5] == "flow" and c.impossible_steps == [5]
    assert (2 * ab.STEPS_PER_WEEK) in [a for a, _ in c.segments]


def test_step_times_assumed_clock():
    t = ab.step_times(2, STARTS3)
    assert str(t[0]) == "2004-03-01T00:00:00.000000" and str(t[287]) == "2004-03-01T23:55:00.000000"
    assert str(t[ab.STEPS_PER_WEEK]) == "2004-03-08T00:00:00.000000"


@needs_abilene
def test_clean_matches_reference_on_real_data():
    import common                                    # TF-free shared forecasting module
    weeks = ab.load_weeks()
    assert np.array_equal(weeks, common.load_abilene_raw(use_cache=False))
    c = ab.clean(weeks)
    segs, info = common.abilene_clean(weeks)
    assert np.array_equal(c.keep, info["keep"]) and np.array_equal(c.reason, info["reason"])
    assert [(a, b - a) for a, b in c.segments] == [(s, len(g)) for s, g in segs]
    assert c.duplicate_weeks == info["duplicate_weeks"] and c.flow_outages == info["flow_outages"]
    assert c.router_outages == info["router_outages"] and c.impossible_steps == info["impossible_steps"]
    assert c.removed() == {"duplicate": 1152, "flow": 6336, "router": 40, "impossible": 1}


@needs_abilene
def test_topology_and_routing_matrix():
    import common
    t = ab.load_topology()
    assert t.A.shape == (54, 144) and set(np.unique(t.A)) == {0.0, 1.0}
    assert np.array_equal(t.A, common.load_abilene_routing()["A"])
    internal = [l for l in t.links if l[4] == 0]
    assert len(internal) == 30 and all(l[5] in (9_920_000, 2_480_000) and l[6] > 0 for l in internal)
    slow = sorted(l[1] for l in internal if l[5] == 2_480_000)
    assert slow == ["ATLAng,IPLSng", "IPLSng,ATLAng"]
    # every flow enters at its source router and leaves at its destination router
    for k, s, d in ab.flow_pairs():
        used = {t.links[i][1] for i in np.flatnonzero(t.A[:, k]) if t.links[i][4] != 0}
        assert used == {f"*,{s}", f"{d},*"}


def test_flow_pairs_index():
    p = ab.flow_pairs()
    assert p[0] == (0, "ATLA-M5", "ATLA-M5") and p[86] == (86, "LOSAng", "CHINng") and p[143][1:] == ("WASHng",) * 2
