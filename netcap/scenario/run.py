"""Phase 3: routing gate, growth / N-1 failure / upgrade / combined scenarios and TE on Abilene.

  python -m netcap.scenario.run

Demand = the MEASURED Abilene traffic matrix (realOD) of every kept 5-minute step (40,855 steps), in
Mbit/s. Utilisation = 5-minute average load / capacity (bursts shorter than 5 minutes are invisible).
Growth rates, the 2x growth case and the upgrade size are ASSUMPTIONS, labelled as such in every result.
Saves results/netcap/phase3_scenarios.json (+ phase3_routing_gate.json) and the scenarios table.
"""
import json
import time

import numpy as np

from .. import config
from ..data import abilene
from ..provenance import run_record, save_result
from . import engine as E
from .te import min_max_utilisation

THRESHOLDS = (0.5, 0.8)          # 0.5: the forecasting module's congestion threshold; 0.8: a common planning rule
ASSUMED_GROWTH = (0.2, 0.4, 0.6)  # ASSUMED annual growth rates for "months to threshold"
COMBINED_GROWTH = 2.0             # ASSUMED growth factor for the combined scenario
TE_STEPS = 20                     # busiest steps (by OSPF max utilisation) for the TE comparison
TE_EVERY = 12                     # hourly sweep: the LP at every 12th kept step, for a bootstrap interval
BOOT, BLOCK_STEPS = 2000, 288     # day-block bootstrap of the hourly LP/OSPF ratios
COMMON = ("Demand is the measured Abilene realOD matrix (2004) on the 2003-04-10 topology and OSPF weights; "
          "utilisation is a 5-minute average; access links have no capacity and are excluded.")


def demand_matrix():
    weeks = abilene.load_weeks()
    cl = abilene.clean(weeks)
    full = weeks.reshape(-1, abilene.N_FLOWS)[cl.keep] * abilene.TO_MBIT_S
    return full, np.flatnonzero(cl.keep)


def failure_cases(net, R0, dem, names, thr):
    out = []
    for a, b, name in E.bidirectional_pairs(net):
        n2 = E.fail_links(net, {a, b})
        R, unr = E.routing_matrix(n2)
        s = E.summarise(n2, R, dem, unr, thr, names)
        s.pop("per_link")
        out.append({"failure": f"link {name}", "link_ids": [a, b], **s,
                    "flows_rerouted": int((~np.isclose(R, R0)).any(0).sum())})
    for v in net.nodes:
        n2 = E.fail_node(net, v)
        R, unr = E.routing_matrix(n2)
        keep = np.setdiff1d(np.arange(net.n_flows), unr)
        s = E.summarise(n2, R, dem, unr, thr, names)
        s.pop("per_link")
        out.append({"failure": f"router {v}", "node": v, **s,
                    "flows_rerouted": int((~np.isclose(R[:, keep], R0[:, keep])).any(0).sum())})
    return out


def main():
    t0 = time.perf_counter()
    topo = abilene.load_topology()
    net = E.from_abilene(topo)
    names = {l[0]: f"{l[1]}->{l[2]}" for l in net.links}
    R0, unr0 = E.routing_matrix(net)
    gate = {"routing_equals_A": bool(np.array_equal(R0, topo.A)), "flows_matching": int((R0 == topo.A).all(0).sum()),
            "flows": int(topo.A.shape[1]), "max_abs_diff": float(np.abs(R0 - topo.A).max()), "unroutable": unr0,
            "ecmp_flows": int(((R0 > 0) & (R0 < 1)).any(0).sum())}
    save_result("phase3_routing_gate", run_record({"topology": "raw/topo-2003-04-10.txt", "A": "raw/A"}, **gate))
    if not gate["routing_equals_A"]:
        raise SystemExit("GATE FAILED: computed routing differs from A (stop and ask, brief rule a)")
    dem, steps = demand_matrix()
    thr = THRESHOLDS[0]
    out = {"gate": gate, "demand": {"steps": len(dem), "flows": dem.shape[1], "unit": "Mbit/s",
                                    "total_mean_mbps": float(dem.sum(1).mean()), "total_max_mbps": float(dem.sum(1).max())},
           "thresholds": THRESHOLDS, "assumptions": COMMON}

    base = E.summarise(net, R0, dem, unr0, thr, names)
    base["per_link_named"] = {names[i]: v for i, v in base.pop("per_link").items()}
    base["headroom"] = {str(t): E.growth_headroom(net, R0, dem, t) for t in THRESHOLDS}
    base["months_to_threshold_ASSUMED_growth"] = {
        str(t): {f"{int(g * 100)}%/yr": E.months_to_threshold(base["headroom"][str(t)]["factor_to_threshold_p95"], g)
                 for g in ASSUMED_GROWTH} for t in THRESHOLDS}
    out["baseline"] = base

    fails = failure_cases(net, R0, dem, names, thr)
    worst = max(fails, key=lambda f: f["max_util"])
    out["n_minus_1"] = {"cases": fails, "worst": worst["failure"], "worst_max_util": worst["max_util"],
                        "cases_with_unroutable_flows": [f["failure"] for f in fails if f["unroutable_flows"]],
                        "cases_over_threshold": {str(t): [f["failure"] for f in fails if f["max_util"] >= t] for t in THRESHOLDS}}

    # upgrade rule fixed in advance: the core link that is worst most often across the N-1 link cases
    link_cases = [f for f in fails if f["failure"].startswith("link")]
    cnt = {}
    for f in link_cases:
        cnt[f["worst_link"]] = cnt.get(f["worst_link"], 0) + 1
    up_link = max(cnt, key=lambda k: (cnt[k], -k))
    up_pair = next((a, b) for a, b, _ in E.bidirectional_pairs(net) if up_link in (a, b))
    cap0 = net.capacities()[net.link_index()[up_link]]
    new_cap = 9920.0 if cap0 < 9920.0 else 2 * cap0
    net_up = E.set_capacity(net, set(up_pair), new_cap)
    fails_up = failure_cases(net_up, R0, dem, names, thr)
    wu = max(fails_up, key=lambda f: f["max_util"])
    out["upgrade"] = {"rule": "upgrade (both directions of) the core link that is the worst link in the most N-1 link-failure cases",
                      "link": names[up_link], "link_ids": list(up_pair), "capacity_mbps_before": float(cap0),
                      "capacity_mbps_after_ASSUMED": new_cap, "worst_link_counts": {names[k]: v for k, v in cnt.items()},
                      "baseline_max_util_after": E.summarise(net_up, R0, dem, unr0, thr, names)["max_util"],
                      "n_minus_1_worst_after": wu["failure"], "n_minus_1_worst_max_util_after": wu["max_util"],
                      "n_minus_1_worst_max_util_before": worst["max_util"]}

    # ADDED after the first run (DECISIONS.md 3.1): upgrade the overloaded link of the worst N-1 case instead
    alt_link = worst["worst_link"]
    alt_pair = next((a, b) for a, b, _ in E.bidirectional_pairs(net) if alt_link in (a, b))
    alt_cap0 = net.capacities()[net.link_index()[alt_link]]
    alt_new = 9920.0 if alt_cap0 < 9920.0 else 2 * alt_cap0
    net_alt = E.set_capacity(net, set(alt_pair), alt_new)
    fa = failure_cases(net_alt, R0, dem, names, thr)
    wa = max(fa, key=lambda f: f["max_util"])
    out["upgrade_alt"] = {"rule": "upgrade (both directions of) the most loaded link in the worst N-1 case",
                          "label": "added after seeing results", "link": names[alt_link], "link_ids": list(alt_pair),
                          "capacity_mbps_before": float(alt_cap0), "capacity_mbps_after_ASSUMED": alt_new,
                          "n_minus_1_worst_after": wa["failure"], "n_minus_1_worst_max_util_after": wa["max_util"],
                          "n_minus_1_worst_link_after": wa["worst_link_name"],
                          "cases_over_threshold_after": {str(t): [f["failure"] for f in fa if f["max_util"] >= t] for t in THRESHOLDS}}
    out["upgrade_search"] = upgrade_search(net, R0, dem, names)
    comb = {}
    for label, n2 in (("before_upgrade", net), ("after_upgrade", net_up), ("after_upgrade_alt", net_alt)):
        f = worst.get("link_ids")
        n3 = E.fail_links(n2, set(f)) if f else E.fail_node(n2, worst["node"])
        R, unr = E.routing_matrix(n3)
        s = E.summarise(n3, R, dem * COMBINED_GROWTH, unr, thr, names)
        s.pop("per_link")
        comb[label] = s
    out["combined"] = {"growth_factor_ASSUMED": COMBINED_GROWTH, "failure": worst["failure"],
                       "upgrade": names[up_link], **comb}

    # TE: LP optimum vs OSPF on the busiest steps, normal and under the worst N-1 failure
    u0 = E.utilisation(net, E.link_loads(R0, dem))[:, net.core_mask()]
    top = np.argsort(-np.nanmax(u0, 1))[:TE_STEPS]
    te = []
    for i in top:
        lp = min_max_utilisation(net, dem[i])
        te.append({"step": int(steps[i]), "ospf_max_util": float(np.nanmax(u0[i])), "lp_min_max_util": lp["U"]})
    nf = E.fail_links(net, set(worst["link_ids"])) if worst.get("link_ids") else E.fail_node(net, worst["node"])
    Rf, unrf = E.routing_matrix(nf)
    uf = E.utilisation(nf, E.link_loads(Rf, dem))[:, nf.core_mask()]
    topf = np.argsort(-np.nanmax(uf, 1))[:TE_STEPS]
    te_f = []
    for i in topf:
        d = dem[i].copy()
        d[unrf] = 0
        lp = min_max_utilisation(nf, d)
        te_f.append({"step": int(steps[i]), "ospf_max_util": float(np.nanmax(uf[i])), "lp_min_max_util": lp["U"]})
    sweep = {"normal": te_sweep(net, R0, unr0, dem, steps), "worst_failure": te_sweep(nf, Rf, unrf, dem, steps)}
    ratio = lambda rows: [r["lp_min_max_util"] / r["ospf_max_util"] for r in rows]
    out["te"] = {"busiest_steps": te, "under_worst_failure": {"failure": worst["failure"], "steps": te_f},
                 "lp_over_ospf_ratio": {"normal": {"min": min(ratio(te)), "median": float(np.median(ratio(te))), "max": max(ratio(te))},
                                        "worst_failure": {"min": min(ratio(te_f)), "median": float(np.median(ratio(te_f))),
                                                          "max": max(ratio(te_f))}},
                 "lp_size": {k: lp[k] for k in ("variables", "equalities", "inequalities")},
                 "hourly_sweep": sweep}

    out["forecast_driven"] = forecast_driven(net, R0)
    out["wall_s"] = time.perf_counter() - t0
    path = save_result("phase3_scenarios", run_record({"abilene": "raw/X01..X24 realOD, cleaned (R1-R4)",
                                                       "topology": "raw/topo-2003-04-10.txt"}, **out))
    store(out)
    print(path)
    print(json.dumps({"gate": gate["routing_equals_A"], "baseline_max": base["max_util"], "worst_n1": worst["failure"],
                      "worst_n1_max": worst["max_util"], "upgrade": out["upgrade"]["link"],
                      "te_ratio": out["te"]["lp_over_ospf_ratio"]}, indent=1))


UPGRADE_FACTORS = (2, 4)          # ASSUMED candidate sizes: each core link pair at 2x and 4x its current capacity
PEAK_FACTORS = (1.0, 1.2, 1.5, 2.0)  # ASSUMED ratio of sub-5-minute peak to the 5-minute average (not measured)


def upgrade_search(net, R0, dem, names):
    """Every single core-link upgrade (both directions, ASSUMED sizes) against the no-failure case and every N-1 case.

    OSPF routing depends on weights, not capacities, so each case's routing and per-link peak load are computed once;
    a candidate only changes the capacities. Worst-case utilisation = max over cases, core links and steps of
    load / capacity. Ranked by reduction of that worst case per Gbit/s added (per direction)."""
    cases = [("no failure", R0, net)]
    for a, b, name in E.bidirectional_pairs(net):
        n2 = E.fail_links(net, {a, b})
        cases.append((f"link {name}", E.routing_matrix(n2)[0], n2))
    for v in net.nodes:
        n2 = E.fail_node(net, v)
        cases.append((f"router {v}", E.routing_matrix(n2)[0], n2))
    core = net.core_mask()
    ids = np.array([l[0] for l in net.links])[core]
    cap0 = net.capacities()[core]
    peak = np.stack([np.max(E.link_loads(R, dem)[:, core], 0) for _, R, _ in cases])     # (cases, core links) Mbit/s
    labels = [c[0] for c in cases]

    def worst(cap, P=None):
        u = (peak if P is None else P) / cap
        i, j = np.unravel_index(np.argmax(u), u.shape)
        return float(u[i, j]), labels[i], names[int(ids[j])]

    w0, case0, link0 = worst(cap0)
    pos = {int(l): k for k, l in enumerate(ids)}

    # size-free answer for the binding link pair: the smallest capacity at which it stops being the worst case
    # (the next-worst link takes over), and the capacity that keeps it below the planning threshold
    j0 = int(np.argmax(np.max(peak / cap0, 0)))
    pair = next((a, b, n) for a, b, n in E.bidirectional_pairs(net) if int(ids[j0]) in (a, b))
    jj = [pos[pair[0]], pos[pair[1]]]
    c_inf = cap0.copy()
    c_inf[jj] = np.inf
    w_rest, case_rest, link_rest = worst(c_inf)
    pk = float(peak[:, jj].max())
    binding = {"link": pair[2], "capacity_mbps_now": float(cap0[j0]), "worst_case_peak_load_mbps": pk,
               "next_worst_util_if_unbounded": w_rest, "next_worst_case": case_rest, "next_worst_link": link_rest,
               "min_capacity_mbps_to_stop_binding": pk / w_rest,
               f"min_capacity_mbps_below_{THRESHOLDS[1]:g}": pk / THRESHOLDS[1]}
    cands = []
    for a, b, name in E.bidirectional_pairs(net):
        for fct in UPGRADE_FACTORS:
            cap = cap0.copy()
            before = float(cap[pos[a]])
            cap[[pos[a], pos[b]]] *= fct
            w, case, link = worst(cap)
            added = (fct - 1) * before / 1000
            cands.append({"link": name, "link_ids": [a, b], "capacity_mbps_before": before,
                          "capacity_mbps_after_ASSUMED": before * fct, "gbps_added_per_direction": added,
                          "worst_case_util_after": w, "worst_case_after": case, "worst_link_after": link,
                          "reduction_pp": (w0 - w) * 100, "reduction_pp_per_gbps": (w0 - w) * 100 / added})
    cands.sort(key=lambda c: (-c["reduction_pp_per_gbps"], -c["reduction_pp"]))
    useful = [c for c in cands if c["reduction_pp"] > 1e-9]

    # greedy plan: repeatedly apply the single upgrade that most reduces the total overload above the planning
    # threshold (sum over core links of max(0, worst-case utilisation - THRESHOLDS[1])) per Gbit/s added, until no link
    # is at or above it. Total overload, not the max alone, because ties (e.g. two mirror links at the same
    # utilisation) would stall a max-only greedy. Capacities compound (ASSUMED sizes).
    T = THRESHOLDS[1]

    def overload(cap, P):
        return float(np.clip((P / cap).max(0) - (T - 1e-6), 0, None).sum())   # a link exactly at T still counts

    def greedy(P):
        cap, plan, total = cap0.copy(), [], 0.0
        ov = overload(cap, P)
        for _ in range(30):
            if worst(cap, P)[0] < T:
                break
            best = None
            for a, b, name in E.bidirectional_pairs(net):
                for fct in UPGRADE_FACTORS:
                    c2 = cap.copy()
                    c2[[pos[a], pos[b]]] *= fct
                    added = (fct - 1) * float(cap[pos[a]]) / 1000
                    score = (ov - overload(c2, P)) * 100 / added
                    if score > 1e-9 and (best is None or score > best[0]):
                        best = (score, name, fct, c2, added, float(cap[pos[a]]))
            if best is None:
                break
            score, name, fct, cap, added, before = best
            total += added
            ov = overload(cap, P)
            w, case, link = worst(cap, P)
            plan.append({"step": len(plan) + 1, "link": name, "capacity_mbps_before": before,
                         "capacity_mbps_after_ASSUMED": before * fct, "gbps_added_per_direction": added,
                         "worst_case_util_after": w, "worst_case_after": case, "worst_link_after": link,
                         "overload_reduction_pp_per_gbps": score, "total_overload_pp_after": ov * 100,
                         "cumulative_gbps_added": total})
        w = worst(cap, P)[0]
        return {"target_worst_case_util": T, "objective": "total overload above target per Gbit/s",
                "reached": bool(w < T), "worst_case_util_final": w, "gbps_added_per_direction_total": total,
                "upgrades": len(plan), "steps": plan}

    plan0 = greedy(peak)

    # peak-to-average sensitivity: 5-minute averages hide bursts. Loads are linear, so a peak factor f (ASSUMED)
    # scales every case's peak load by f. The chosen upgrade is the greedy plan at f = 1 (ATLA-IPLS to 9.92 Gbit/s).
    cap_plan = cap0.copy()
    for st in plan0["steps"]:
        a, b = next((a, b) for a, b, n in E.bidirectional_pairs(net) if n == st["link"])
        cap_plan[[pos[a], pos[b]]] = st["capacity_mbps_after_ASSUMED"]
    sens = []
    for f in PEAK_FACTORS:
        P = peak * f
        wb, cb, lb = worst(cap0, P)
        wa, ca, la = worst(cap_plan, P)
        ua = P / cap_plan
        g = greedy(P)
        sens.append({"peak_factor_ASSUMED": f, "no_failure_max_util": float((P[0] / cap0).max()),
                     "worst_case_util_before": wb, "worst_case_before": cb,
                     "worst_case_util_after_chosen_upgrade": wa, "worst_case_after_chosen_upgrade": ca,
                     "worst_link_after_chosen_upgrade": la,
                     "cases_at_or_over_threshold_after_chosen_upgrade": int((ua.max(1) >= THRESHOLDS[1]).sum()),
                     "cases_over_100pct_after_chosen_upgrade": int((ua.max(1) > 1).sum()),
                     f"binding_link_min_capacity_mbps_below_{THRESHOLDS[1]:g}": pk * f / THRESHOLDS[1],
                     "greedy_plan": {k: g[k] for k in ("reached", "worst_case_util_final", "gbps_added_per_direction_total", "upgrades")}
                                    | {"upgrades": [f"{st['link']} {st['capacity_mbps_before']:.0f}->{st['capacity_mbps_after_ASSUMED']:.0f}"
                                                    for st in g["steps"]]}})
    return {"rule": "every core link pair x ASSUMED sizes (2x, 4x current) against no failure + every N-1 link and router case; "
                    "ranked by worst-case utilisation reduction (pp) per Gbit/s added per direction",
            "label": "added after seeing results (search replaces the two fixed rules as the planning answer; both rules kept)",
            "cases": len(cases), "candidates": len(cands), "capacities_ASSUMED": True,
            "worst_case_util_before": w0, "worst_case_before": case0, "worst_link_before": link0,
            "candidates_that_reduce_worst_case": len(useful), "binding_link": binding,
            "best_per_gbps": useful[0] if useful else None,
            "best_absolute": max(cands, key=lambda c: (c["reduction_pp"], -c["gbps_added_per_direction"])),
            "greedy_plan": plan0,
            "peak_factor_sensitivity": {"note": "peak factors are ASSUMED (sub-5-minute peak / 5-minute average, not "
                                                "measured on Abilene); chosen upgrade = greedy plan at factor 1",
                                        "chosen_upgrade": [st["link"] + f" to {st['capacity_mbps_after_ASSUMED']:.0f} Mbit/s"
                                                           for st in plan0["steps"]],
                                        "max_peak_factor_chosen_upgrade_stays_below_threshold":
                                            THRESHOLDS[1] / worst(cap_plan)[0],
                                        "caveat": "candidate sizes are coarse (2x / 4x), so plans at different factors can "
                                                  "coincide and overshoot; the per-link minimum capacities are size-free",
                                        "rows": sens},
            "ranking": cands}


def te_sweep(net, R, unr, dem, steps):
    """LP at every TE_EVERY-th kept step; ratio LP / OSPF max utilisation with a day-block bootstrap 95% interval."""
    idx = np.arange(0, len(dem), TE_EVERY)
    u = E.utilisation(net, E.link_loads(R, dem[idx]))[:, net.core_mask()]
    ospf = np.nanmax(u, 1)
    lp = np.empty(len(idx))
    t0 = time.perf_counter()
    for j, i in enumerate(idx):
        d = dem[i].copy()
        d[unr] = 0
        lp[j] = min_max_utilisation(net, d)["U"]
    secs = time.perf_counter() - t0
    r = lp / ospf
    rng = np.random.default_rng(config.SEED)
    block = max(1, BLOCK_STEPS // TE_EVERY)                    # one day of hourly points
    k = int(np.ceil(len(r) / block))
    starts = rng.integers(0, len(r) - block + 1, size=(BOOT, k))
    b = r[(starts[:, :, None] + np.arange(block)).reshape(BOOT, -1)[:, :len(r)]]
    ci = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]
    return {"steps_solved": int(len(idx)), "lp_solve_s_total": secs, "ratio_mean": float(r.mean()),
            "ratio_mean_ci95": ci(b.mean(1)), "ratio_median": float(np.median(r)), "ratio_median_ci95": ci(np.median(b, 1)),
            "ratio_min": float(r.min()), "ratio_max": float(r.max()),
            "ospf_max_util_mean": float(ospf.mean()), "lp_max_util_mean": float(lp.mean()),
            "bootstrap": f"{BOOT} resamples, blocks of one day ({block} hourly points), seed {config.SEED}"}


def forecast_driven(net, R0):
    """Per-link p95 utilisation in the original test period: planned from Phase 2 forecasts vs actual.

    Uses the 60-minute test forecasts of every method saved by Phase 2 (point and conformal upper bound).
    A link's planned p95 is the p95 of the forecast loads; the error is planned - actual (pp)."""
    import polars as pl
    root = config.DATA / "forecasts" / "panel=abilene" / "h=12"
    if not root.exists():
        return {"status": "not run: Phase 2 forecasts missing"}
    res, counts = {}, {}
    core = net.core_mask()
    cap = net.capacities()[core]
    files = sorted(root.glob("*.parquet")) + sorted((config.DATA / "forecasts" / "panel=abilene_neural" / "h=12").glob("*.parquet"))
    for f in files:
        df = pl.read_parquet(f).sort(["row", "flow"])
        rows = df["row"].unique().sort().to_numpy()
        if df.height != len(rows) * net.n_flows:
            continue                                   # needs every flow at every step
        shape = (len(rows), net.n_flows)
        act = df["actual"].to_numpy().reshape(shape) * abilene.TO_MBIT_S
        out = {}
        for col in ("point", "hi"):
            pr = df[col].to_numpy().reshape(shape) * abilene.TO_MBIT_S
            u_p = (pr @ R0.T)[:, core] / cap
            u_a = (act @ R0.T)[:, core] / cap
            p95p, p95a = np.percentile(u_p, 95, 0), np.percentile(u_a, 95, 0)
            ev_a, ev_p = u_a >= THRESHOLDS[0], u_p >= THRESHOLDS[0]          # warning: link-step at/above 50%
            tp, fp, fn = int((ev_a & ev_p).sum()), int((~ev_a & ev_p).sum()), int((ev_a & ~ev_p).sum())
            out[col] = {"mean_abs_error_pp": float(np.abs(p95p - p95a).mean() * 100),
                        "max_abs_error_pp": float(np.abs(p95p - p95a).max() * 100),
                        "links_under_planned": int((p95p < p95a).sum()),
                        "worst_link_planned_p95": float(p95p.max()), "worst_link_actual_p95": float(p95a.max()),
                        "link_load_mae_pp": float(np.abs(u_p - u_a).mean() * 100),
                        "warning_f1": 2 * tp / max(2 * tp + fp + fn, 1), "warning_precision": tp / max(tp + fp, 1),
                        "warning_recall": tp / max(tp + fn, 1), "warning_events_actual": int(ev_a.sum())}
            if col == "point":                         # per-row counts for the F1 bootstrap
                counts[f.stem] = (rows, np.stack([(ev_a & ev_p).sum(1), (~ev_a & ev_p).sum(1), (ev_a & ~ev_p).sum(1)], 1))
        res[f.stem] = out
    boot = f1_bootstrap(counts)
    events = boot.pop("_events", None)
    for m, b in boot.items():
        res[m]["point"].update(b)
    return {"horizon_minutes": 60, "period": "original fold test period", "methods": res, "warning_events": events,
            "warning_f1_bootstrap": f"paired day-block bootstrap over test rows: {BOOT} resamples, blocks of {BLOCK_STEPS} "
                                    f"5-minute rows (one day), seed {config.SEED}; same resampled rows for every method; "
                                    "difference = method - persistence; point forecasts"}


def f1_bootstrap(counts, ref="persistence"):
    """95% day-block bootstrap intervals for each method's warning F1 and its difference to persistence.

    counts[m] = (rows, per-row [tp, fp, fn]). F1 is recomputed from the summed counts of each resample (F1 is not
    an average over rows). Methods whose rows differ from persistence's are skipped."""
    if ref not in counts:
        return {}
    rows0, c0 = counts[ref]
    n = len(rows0)
    rng = np.random.default_rng(config.SEED)
    k = int(np.ceil(n / BLOCK_STEPS))
    starts = rng.integers(0, n - BLOCK_STEPS + 1, size=(BOOT, k))
    idx = (starts[:, :, None] + np.arange(BLOCK_STEPS)).reshape(BOOT, -1)[:, :n]

    def f1(c):                                         # c: (..., 3) summed tp, fp, fn; undefined (NaN) with no events at all
        den = 2 * c[..., 0] + c[..., 1] + c[..., 2]
        return np.where(den > 0, 2 * c[..., 0] / np.where(den > 0, den, 1), np.nan)

    f_ref = f1(c0[idx].sum(1))
    ci = lambda a: [float(x) for x in np.nanpercentile(a, [2.5, 97.5])]
    actual = c0[:, 0] + c0[:, 2]                       # actual warning events per row (tp + fn)
    per_day = np.bincount(np.arange(n) // BLOCK_STEPS, weights=actual).astype(int)
    out = {"_events": {"actual_events": int(actual.sum()), "test_days": int(len(per_day)),
                       "days_with_events": int((per_day > 0).sum()), "events_per_day": per_day.tolist(),
                       "largest_day_share": float(per_day.max() / max(actual.sum(), 1))}}
    for m, (rows, c) in counts.items():
        if len(rows) != n or not np.array_equal(rows, rows0):
            continue
        f_m = f1(c[idx].sum(1))
        d = f_m - f_ref
        dd = d[np.isfinite(d)]                         # shares over defined resamples only (NaN > 0 is False)
        lo, hi = ci(d)
        out[m] = {"warning_f1_ci95": ci(f_m), "warning_f1_diff_vs_persistence": float(f1(c.sum(0)) - f1(c0.sum(0))),
                  "warning_f1_diff_ci95": [lo, hi], "warning_f1_diff_excludes_0": bool(lo > 0 or hi < 0),
                  "bootstrap_resamples_defined": int(np.isfinite(d).sum()),
                  "share_resamples_better_than_persistence": float((dd > 0).mean()) if len(dd) else None,
                  "share_resamples_worse_than_persistence": float((dd < 0).mean()) if len(dd) else None}
    return out


def store(out):
    from ..db import load as db
    rows = []
    rid = "phase3"
    rows.append(("baseline", "growth", {"thresholds": THRESHOLDS, "annual_growth_ASSUMED": ASSUMED_GROWTH},
                 {k: out["baseline"][k] for k in ("max_util", "p95_util_worst", "worst_link_name", "headroom",
                                                  "months_to_threshold_ASSUMED_growth")},
                 COMMON + " Growth rates are ASSUMED (not measured)."))
    for c in out["n_minus_1"]["cases"]:
        rows.append((f"n1:{c['failure']}", "failure", {"failure": c["failure"]}, c, COMMON + " A cut link fails in both directions."))
    rows.append(("upgrade", "upgrade", {"link": out["upgrade"]["link"]}, out["upgrade"],
                 COMMON + " The new capacity is ASSUMED."))
    us = out["upgrade_search"]
    rows.append(("upgrade_search", "upgrade", {"factors_ASSUMED": list(UPGRADE_FACTORS), "cases": us["cases"]},
                 {k: us[k] for k in ("worst_case_util_before", "best_per_gbps", "best_absolute", "binding_link", "greedy_plan", "peak_factor_sensitivity")},
                 COMMON + " Candidate capacities are ASSUMED (2x and 4x current)."))
    rows.append(("combined", "combined", {k: out["combined"][k] for k in ("growth_factor_ASSUMED", "failure", "upgrade")},
                 out["combined"], COMMON + " The growth factor and new capacity are ASSUMED."))
    rows.append(("te", "te", {"steps": TE_STEPS}, out["te"]["lp_over_ospf_ratio"],
                 COMMON + " LP optimum is a lower bound for any routing on this topology and matrix."))
    import polars as pl
    frame = pl.DataFrame({"scenario_id": [f"{rid}:{r[0]}" for r in rows], "kind": [r[1] for r in rows],
                          "params": [json.dumps(r[2], default=str) for r in rows],
                          "result": [json.dumps(r[3], default=str) for r in rows], "assumptions": [r[4] for r in rows]})
    with db.connect() as conn:
        db.apply_schema(conn)
        db.upsert(conn, "scenarios", [frame], ["scenario_id", "kind", "params", "result", "assumptions"])


if __name__ == "__main__":
    main()
