"""Traffic engineering: the minimum achievable maximum link utilisation (a linear program).

min U  subject to  flow conservation for every demand,  sum of flows on link e <= U * capacity_e,  flows >= 0.
Demands are aggregated by source (one commodity per source router): a single-source flow can always be
split back into per-destination paths, so this gives the same optimum as one commodity per OD pair,
with 12 x 30 + 1 variables on Abilene instead of 132 x 30 + 1. Solved with HiGHS (scipy.optimize.linprog).

The optimum is a lower bound for any routing (OSPF weights, MPLS-TE, segment routing) on that topology
and traffic matrix; the gap to OSPF shows how much traffic engineering could at most gain.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


def min_max_utilisation(net, demand):
    """demand (n_flows,) Mbit/s (flow = src * n + dst). Returns dict with U, per-link flow and status."""
    G, nodes = net.graph, list(net.nodes)
    n = len(nodes)
    present = [v for v in nodes if v in G]
    pos = {v: i for i, v in enumerate(present)}
    edges = list(G.edges(data=True))
    E = len(edges)
    D = np.asarray(demand, float).reshape(n, n).copy()
    np.fill_diagonal(D, 0.0)                                     # same-router traffic uses no core link
    srcs = [s for s in range(n) if nodes[s] in G and D[s].sum() > 0]
    nv = len(srcs) * E + 1
    A_eq = lil_matrix((len(srcs) * len(present), nv))
    b_eq = np.zeros(len(srcs) * len(present))
    for k, s in enumerate(srcs):
        for j, (u, v, _) in enumerate(edges):
            A_eq[k * len(present) + pos[u], k * E + j] += 1.0      # out of u
            A_eq[k * len(present) + pos[v], k * E + j] -= 1.0      # into v
        for d in range(n):
            if D[s, d] > 0:
                if nodes[d] not in G:
                    raise ValueError(f"demand to removed router {nodes[d]}")
                b_eq[k * len(present) + pos[nodes[d]]] -= D[s, d]
                b_eq[k * len(present) + pos[nodes[s]]] += D[s, d]
    A_ub = lil_matrix((E, nv))
    for j, (_, _, a) in enumerate(edges):
        for k in range(len(srcs)):
            A_ub[j, k * E + j] = 1.0
        A_ub[j, nv - 1] = -a["capacity"]
    c = np.zeros(nv)
    c[-1] = 1.0
    r = linprog(c, A_ub=A_ub.tocsr(), b_ub=np.zeros(E), A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=(0, None), method="highs")
    if r.status != 0:
        return {"status": r.message, "U": None}
    x = r.x[:-1].reshape(len(srcs), E).sum(0)
    return {"status": "optimal", "U": float(r.x[-1]),
            "link_flow": {int(a["link_id"]): float(f) for (_, _, a), f in zip(edges, x)},
            "variables": nv, "equalities": int(A_eq.shape[0]), "inequalities": E}
