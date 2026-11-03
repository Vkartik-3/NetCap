"""Scenario engine: OSPF routing on a NetworkX topology, link loads, growth / failure / upgrade scenarios.

Routing is OSPF shortest paths on the link weights with equal-cost multipath split evenly at every
router (as OSPF ECMP does: each router divides what it forwards to a destination equally among its
shortest-path next hops). Correctness gate: on the real Abilene topology the computed routing matrix
must equal the matrix A that ships with the data (results/netcap/phase3_routing_gate.json).

Units: demands and loads in Mbit/s, capacities in Mbit/s, utilisation as a fraction of capacity.
Access links (ingress '*,X' and egress 'X,*') carry every flow of their router and have no capacity.
"""
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class Net:
    nodes: list                          # router names; flow id = src_index * n + dst_index
    links: list                          # [(link_id, src, dst, capacity_mbps | None, weight | None, kind)]
    graph: nx.DiGraph = field(default=None)

    def __post_init__(self):
        if self.graph is None:
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(self.nodes)
            for lid, s, d, cap, w, kind in self.links:
                if kind == "core":
                    self.graph.add_edge(s, d, link_id=lid, capacity=cap, weight=w)

    @property
    def n_flows(self):
        return len(self.nodes) ** 2

    def link_index(self):
        return {l[0]: i for i, l in enumerate(self.links)}

    def capacities(self):
        return np.array([np.nan if l[3] is None else l[3] for l in self.links])

    def core_mask(self):
        return np.array([l[5] == "core" for l in self.links])

    def copy(self):
        return Net(list(self.nodes), list(self.links), self.graph.copy())


def from_abilene(topo):
    """netcap.data.abilene.Topology -> Net (capacity kbit/s -> Mbit/s)."""
    kinds = {0: "core", 1: "ingress", 2: "egress"}
    links = [(lid, s, d, None if cap is None else cap / 1000, w, kinds[t]) for lid, _, s, d, t, cap, w in topo.links]
    return Net([n[0] for n in topo.nodes], links)


def routing_matrix(net):
    """(n_links, n_flows) fraction of each flow on each link, OSPF ECMP. Also returns unroutable flows."""
    G, n = net.graph, len(net.nodes)
    idx = net.link_index()
    R = np.zeros((len(net.links), net.n_flows))
    ingress = {l[2]: idx[l[0]] for l in net.links if l[5] == "ingress"}
    egress = {l[1]: idx[l[0]] for l in net.links if l[5] == "egress"}
    unroutable = []
    for di, d in enumerate(net.nodes):
        if d not in G:
            unroutable += [si * n + di for si in range(n)]
            continue
        dist = nx.single_source_dijkstra_path_length(G.reverse(copy=False), d, weight="weight")
        nxt = {u: [v for v in G.successors(u) if v in dist and np.isclose(G[u][v]["weight"] + dist[v], dist[u])]
               for u in dist if u != d}
        order = sorted(dist, key=lambda u: -dist[u])            # farthest first: flow only moves closer
        for si, s in enumerate(net.nodes):
            f = si * n + di
            if s not in G or s not in dist:
                unroutable.append(f)
                continue
            if s in ingress:
                R[ingress[s], f] = 1.0
            if d in egress:
                R[egress[d], f] = 1.0
            share = {s: 1.0}
            for u in order:
                x = share.pop(u, 0.0)
                if x == 0.0 or u == d:
                    continue
                for v in nxt[u]:
                    R[idx[G[u][v]["link_id"]], f] += x / len(nxt[u])
                    share[v] = share.get(v, 0.0) + x / len(nxt[u])
    return R, sorted(unroutable)


def link_loads(R, demand):
    """demand (T, n_flows) or (n_flows,) Mbit/s -> loads (T, n_links)."""
    return np.asarray(demand) @ R.T


def utilisation(net, loads):
    cap = net.capacities()
    with np.errstate(invalid="ignore", divide="ignore"):
        return loads / cap


def fail_links(net, link_ids):
    """Remove the given core links (by id). A cut fibre takes both directions: pass both ids."""
    out = net.copy()
    for lid, s, d, *_ in net.links:
        if lid in link_ids and out.graph.has_edge(s, d) and out.graph[s][d]["link_id"] == lid:
            out.graph.remove_edge(s, d)
    return out


def fail_node(net, node):
    out = net.copy()
    out.graph.remove_node(node)
    return out


def set_capacity(net, link_ids, capacity_mbps):
    out = net.copy()
    links = []
    for lid, s, d, cap, w, kind in out.links:
        if lid in link_ids:
            cap = capacity_mbps
            out.graph[s][d]["capacity"] = cap
        links.append((lid, s, d, cap, w, kind))
    out.links = links
    return out


def bidirectional_pairs(net):
    """[(id_a, id_b, 'A-B')] for core links that come in both directions."""
    core = {(s, d): lid for lid, s, d, _, _, kind in net.links if kind == "core"}
    out = []
    for (s, d), lid in sorted(core.items(), key=lambda kv: kv[1]):
        back = core.get((d, s))
        if back is not None and lid < back:
            out.append((lid, back, f"{s}-{d}"))
    return out


def summarise(net, R, demand, unroutable, threshold, names=None):
    """Per-link max and p95 utilisation over the demand period; the worst core link; dropped traffic."""
    loads = link_loads(R, demand)
    u = utilisation(net, loads)
    core = net.core_mask()
    mx, p95 = np.nanmax(u[:, core], 0), np.nanpercentile(u[:, core], 95, axis=0)
    ids = [l[0] for l in net.links if l[5] == "core"]
    w = int(np.argmax(mx))
    dropped = np.asarray(demand)[:, unroutable].sum(1) if unroutable else np.zeros(len(demand))
    total = np.asarray(demand).sum(1)
    return {"max_util": float(mx[w]), "worst_link": ids[w], "worst_link_name": (names or {}).get(ids[w], str(ids[w])),
            "p95_util_worst": float(np.max(p95)), "links_over_threshold_max": int((mx >= threshold).sum()),
            "links_over_threshold_p95": int((p95 >= threshold).sum()), "threshold": threshold,
            "steps_any_link_over_threshold": int((np.nan_to_num(u[:, core]) >= threshold).any(1).sum()),
            "unroutable_flows": len(unroutable), "dropped_traffic_share_max": float(np.max(dropped / np.where(total > 0, total, 1))),
            "per_link": {int(i): {"max": float(a), "p95": float(b)} for i, a, b in zip(ids, mx, p95)}}


def growth_headroom(net, R, demand, threshold):
    """Largest uniform growth factor g such that every core link stays below threshold at its p95
    (and at its max). Loads are linear in demand, so g = threshold / current utilisation."""
    u = utilisation(net, link_loads(R, demand))[:, net.core_mask()]
    p95 = np.nanpercentile(u, 95, axis=0)
    mx = np.nanmax(u, 0)
    return {"factor_to_threshold_p95": float(threshold / np.max(p95)), "factor_to_threshold_max": float(threshold / np.max(mx))}


def months_to_threshold(factor, annual_growth):
    """Months until demand grown at `annual_growth` (ASSUMED rate, e.g. 0.4 = 40%/year) reaches `factor`."""
    if factor <= 1:
        return 0.0
    return float(12 * np.log(factor) / np.log1p(annual_growth))
