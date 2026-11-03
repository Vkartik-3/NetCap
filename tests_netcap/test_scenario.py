import numpy as np
import pytest

from netcap import config
from netcap.scenario import engine as E
from netcap.scenario.te import min_max_utilisation


def square(weights=(1, 1, 1, 1), cap=100.0):
    """A -> B -> D and A -> C -> D (and reverse), plus access links for every router."""
    nodes = ["A", "B", "C", "D"]
    core = [("A", "B", weights[0]), ("B", "D", weights[1]), ("A", "C", weights[2]), ("C", "D", weights[3])]
    links, lid = [], 1
    for s, d, w in core:
        links += [(lid, s, d, cap, w, "core"), (lid + 1, d, s, cap, w, "core")]
        lid += 2
    for v in nodes:
        links += [(lid, "*", v, None, None, "ingress"), (lid + 1, v, "*", None, None, "egress")]
        lid += 2
    return E.Net(nodes, links)


def flow(net, s, d):
    return net.nodes.index(s) * len(net.nodes) + net.nodes.index(d)


def lid(net, s, d):
    return next(l[0] for l in net.links if l[1] == s and l[2] == d and l[5] == "core")


def test_ecmp_splits_evenly_and_access_links_carry_everything():
    net = square()
    R, unr = E.routing_matrix(net)
    i = net.link_index()
    f = flow(net, "A", "D")
    assert unr == []
    assert R[i[lid(net, "A", "B")], f] == 0.5 and R[i[lid(net, "A", "C")], f] == 0.5
    assert R[i[lid(net, "B", "D")], f] == 0.5 and R[:, f].sum() == pytest.approx(2 + 2 * 0.5 * 2)
    ff = flow(net, "B", "B")
    assert R[:, ff].sum() == 2                                      # ingress + egress only


def test_weights_pick_the_shortest_path_and_failure_reroutes():
    net = square(weights=(1, 1, 5, 5))
    R, _ = E.routing_matrix(net)
    i = net.link_index()
    f = flow(net, "A", "D")
    assert R[i[lid(net, "A", "B")], f] == 1 and R[i[lid(net, "A", "C")], f] == 0
    cut = E.fail_links(net, {lid(net, "A", "B"), lid(net, "B", "A")})
    R2, unr = E.routing_matrix(cut)
    assert unr == [] and R2[i[lid(net, "A", "C")], f] == 1 and R2[i[lid(net, "A", "B")], f] == 0


def test_partition_and_node_failure_are_reported_as_unroutable():
    net = square()
    cut = E.fail_links(net, {lid(net, a, b) for a, b in (("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"))})
    _, unr = E.routing_matrix(cut)
    assert flow(net, "A", "D") in unr and flow(net, "D", "A") in unr and flow(net, "B", "D") not in unr
    _, unr = E.routing_matrix(E.fail_node(net, "B"))
    assert flow(net, "B", "D") in unr and flow(net, "D", "B") in unr and flow(net, "A", "D") not in unr


def test_growth_upgrade_and_lp():
    net = square(weights=(1, 1, 5, 5))
    R, unr = E.routing_matrix(net)
    dem = np.zeros((2, net.n_flows))
    dem[:, flow(net, "A", "D")] = [40.0, 60.0]
    s = E.summarise(net, R, dem, unr, threshold=0.5)
    assert s["max_util"] == pytest.approx(0.6) and s["worst_link"] == lid(net, "A", "B")
    g = E.growth_headroom(net, R, dem, 0.9)
    assert g["factor_to_threshold_max"] == pytest.approx(1.5)
    assert E.months_to_threshold(2.0, 1.0) == pytest.approx(12.0)    # doubling at 100%/year: one year
    up = E.set_capacity(net, {lid(net, "A", "B"), lid(net, "B", "D")}, 200.0)
    assert E.summarise(up, R, dem, unr, 0.5)["max_util"] == pytest.approx(0.3)
    lp = min_max_utilisation(net, dem[1])
    assert lp["status"] == "optimal" and lp["U"] == pytest.approx(0.3)          # 60 split over two paths
    assert lp["link_flow"][lid(net, "A", "B")] + lp["link_flow"][lid(net, "A", "C")] == pytest.approx(60)


def test_lp_respects_unequal_capacity():
    net = E.set_capacity(square(), {1, 2, 3, 4}, 300.0)             # path via B has 3x capacity
    dem = np.zeros(net.n_flows)
    dem[flow(net, "A", "D")] = 400.0
    assert min_max_utilisation(net, dem)["U"] == pytest.approx(1.0)  # 300 + 100 at full use


@pytest.mark.skipif(not (config.RAW_ABILENE / "A").exists(), reason="Abilene raw data not present")
def test_abilene_routing_reproduces_matrix_A():
    from netcap.data import abilene
    topo = abilene.load_topology()
    R, unr = E.routing_matrix(E.from_abilene(topo))
    assert unr == [] and np.array_equal(R, topo.A)


def test_upgrade_search_finds_the_binding_link_and_sizes_it():
    from netcap.scenario.run import upgrade_search
    net = square(cap=100.0)
    net = E.set_capacity(net, {lid(net, "A", "B"), lid(net, "B", "A")}, 10.0)     # one thin link pair
    R0, _ = E.routing_matrix(net)
    dem = np.zeros((3, len(net.nodes) ** 2))
    dem[:, flow(net, "A", "D")] = [20.0, 40.0, 30.0]                              # ECMP: 20 Mbit/s peak on each path
    names = {l[0]: f"{l[1]}->{l[2]}" for l in net.links}
    s = upgrade_search(net, R0, dem, names)
    assert s["worst_case_util_before"] == pytest.approx(4.0)                      # A-C fails: 40 on the 10 Mbit/s A->B
    assert s["best_per_gbps"]["link"] == "A-B"
    assert s["binding_link"]["worst_case_peak_load_mbps"] == pytest.approx(40.0)
    assert s["binding_link"]["min_capacity_mbps_below_0.8"] == pytest.approx(50.0)
    assert s["greedy_plan"]["reached"] and s["greedy_plan"]["worst_case_util_final"] < 0.8
    assert all(c["capacity_mbps_after_ASSUMED"] > c["capacity_mbps_before"] for c in s["ranking"])
