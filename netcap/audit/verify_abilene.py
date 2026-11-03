"""Phase 0 audit: verify Abilene raw files. Read-only. Usage: python -I verify_abilene.py <raw_dir> <out.json>"""
import json, sys
from pathlib import Path
import numpy as np
import networkx as nx

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
NODES = ["ATLA-M5", "ATLAng", "CHINng", "DNVRng", "HSTNng", "IPLSng",
         "KSCYng", "LOSAng", "NYCMng", "SNVAng", "STTLng", "WASHng"]   # common.ABILENE_NODES
R = {}

def rows(name):
    return [l.split() for l in (raw / name).read_text().splitlines() if l.strip() and not l.startswith("#")]

# --- demands: dmd index <-> (s,d) names
dem = {int(i): tuple(n.split(",")) for n, i in rows("demands")}
idx_ok = all(dem[k + 1] == (NODES[k // 12], NODES[k % 12]) for k in range(144))
R["demands"] = {"n": len(dem), "flow_index_is_src*12+dst_with_common_node_order": idx_ok}

# --- links file
links = [(n, int(i), int(t)) for n, i, t in rows("links")]
R["links"] = {"n": len(links), "internal": sum(t == 0 for *_, t in links),
              "inbound": sum(t == 1 for *_, t in links), "outbound": sum(t == 2 for *_, t in links),
              "indices_1_to_54_in_order": [i for _, i, _ in links] == list(range(1, 55))}
link_name = {i: n for n, i, _ in links}

# --- topo: routers, capacities, OSPF weights
topo = (raw / "topo-2003-04-10.txt").read_text().splitlines()
sec, routers, tl = None, [], []
for l in topo:
    s = l.strip()
    if s in ("router", "link"):
        sec = s; continue
    if not s or s.startswith("#"):
        continue
    (routers if sec == "router" else tl).append(s.split())
topo_nodes = [r[0] for r in routers]
topo_links = {(x, y): (float(c), int(w)) for x, y, c, w in tl}
internal = [n for n, _, t in links if t == 0]
R["topo"] = {"routers": topo_nodes, "router_order_matches_common": topo_nodes == NODES,
             "n_links": len(topo_links),
             "internal_links_in_links_file_all_in_topo": all(tuple(n.split(",")) in topo_links for n in internal),
             "topo_links_all_in_links_file": all(f"{x},{y}" in internal for x, y in topo_links),
             "symmetric_capacity_and_weight": all(topo_links.get((y, x)) == v for (x, y), v in topo_links.items()),
             "capacities_kbps": sorted({c for c, _ in topo_links.values()}),
             "links_below_max_capacity": [f"{x},{y}" for (x, y), (c, _) in topo_links.items() if c < 9920000]}

# --- A
A = np.zeros((54, 144)); name_mismatch = 0; n_entries = 0
for lname, dname, li, di, frac in rows("A"):
    li, di = int(li), int(di); n_entries += 1
    A[li - 1, di - 1] = float(frac)
    name_mismatch += (link_name[li] != lname) + (dem[di] != tuple(dname.split(",")))
fr = A[A > 0]
ingress_ok = egress_ok = True
for k in range(144):
    s, d = NODES[k // 12], NODES[k % 12]
    inb = [i for i in range(30, 54) if A[i, k] > 0 and links[i][2] == 1]
    outb = [i for i in range(30, 54) if A[i, k] > 0 and links[i][2] == 2]
    ingress_ok &= len(inb) == 1 and link_name[inb[0] + 1] == f"*,{s}"
    egress_ok &= len(outb) == 1 and link_name[outb[0] + 1] == f"{d},*"
R["A"] = {"entries": n_entries, "shape": [54, 144], "name_index_mismatches": int(name_mismatch),
          "distinct_fractions": sorted(set(fr.tolist())), "all_fractions_1": bool(np.all(fr == 1)),
          "every_flow_one_ingress_link_at_its_source": bool(ingress_ok),
          "every_flow_one_egress_link_at_its_destination": bool(egress_ok),
          "note": "link type 1 '*,X' is entry at X (flows sourced at X use it); type 2 'X,*' is exit at X"}

# --- preview (Phase 3 gate is the real check): OSPF shortest paths vs A on internal links
G = nx.DiGraph()
for (x, y), (c, w) in topo_links.items():
    G.add_edge(x, y, weight=w)
lidx = {tuple(n.split(",")): i for i, (n, _, t) in enumerate(links) if t == 0}
match, ecmp, diffs = 0, 0, []
for k in range(144):
    s, d = NODES[k // 12], NODES[k % 12]
    a_set = {i for i in range(30) if A[i, k] > 0}
    if s == d:
        sp_sets = [set()]
    else:
        sp_sets = [{lidx[(p[j], p[j + 1])] for j in range(len(p) - 1)} for p in nx.all_shortest_paths(G, s, d, weight="weight")]
    ecmp += len(sp_sets) > 1
    if a_set in sp_sets:
        match += 1
    else:
        diffs.append({"flow": k, "pair": f"{s}->{d}", "A_links": sorted(link_name[i + 1] for i in a_set),
                      "ospf_paths": [sorted(link_name[i + 1] for i in p) for p in sp_sets]})
R["routing_preview"] = {"flows_where_A_path_is_an_ospf_shortest_path": match, "of": 144,
                        "flows_with_equal_cost_ties": ecmp, "discrepancies": diffs,
                        "label": "preview only; Phase 3 correctness gate repeats this with tests"}

# --- X files: shape, units, value checks (realOD = column 0)
xs = {}
for w in range(1, 25):
    f = raw / f"X{w:02d}.gz"
    a = np.loadtxt(f)
    v = a.reshape(2016, 144, 5)
    real = v[:, :, 0]
    # all 5 versions share per-router out/in totals if the 144 axis is s*12+d (grouping test)
    tm = v.reshape(2016, 12, 12, 5)
    out_tot, in_tot = tm.sum(2), tm.sum(1)
    xs[f.name] = {"shape": list(a.shape), "bytes": f.stat().st_size,
                  "nan": int(np.isnan(a).sum()), "negative": int((a < 0).sum()),
                  "realOD_max": float(real.max()), "realOD_zero_frac": float((real == 0).mean()),
                  "row_totals_equal_across_5_versions": bool(np.allclose(out_tot, out_tot[..., :1], rtol=1e-3, atol=1)
                                                            and np.allclose(in_tot, in_tot[..., :1], rtol=1e-3, atol=1))}
MBIT = 100 * 8 / 300 / 1e6
allmax = max(x["realOD_max"] for x in xs.values())
second = sorted(x["realOD_max"] for x in xs.values())[-2]
R["X"] = {"files": 24, "all_shape_2016x720": all(x["shape"] == [2016, 720] for x in xs.values()),
          "total_rows": 24 * 2016, "total_values": 24 * 2016 * 720, "total_bytes_gz": sum(x["bytes"] for x in xs.values()),
          "any_nan": any(x["nan"] for x in xs.values()), "any_negative": any(x["negative"] for x in xs.values()),
          "unit_per_readme": "100 bytes per 5 minutes (100 = packet sampling rate)",
          "max_realOD_raw": allmax, "max_realOD_mbit_s": allmax * MBIT,
          "second_largest_weekly_max_mbit_s": second * MBIT,
          "unit_sanity": "with the README unit, the largest weekly maxima other than the R4 outlier are below the 9.92 Gbit/s link speed",
          "per_router_totals_equal_across_5_versions_all_weeks": all(x["row_totals_equal_across_5_versions"] for x in xs.values()),
          "per_file": xs}
out.write_text(json.dumps(R, indent=1))
summary = {k: (v if k != "X" else {kk: vv for kk, vv in v.items() if kk != "per_file"}) for k, v in R.items()}
summary["routing_preview"] = {k: v for k, v in R["routing_preview"].items() if k != "discrepancies"} | {"n_discrepancies": len(diffs), "first": diffs[:6]}
print(json.dumps(summary, indent=1))
