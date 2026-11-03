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
    res = {}
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
        res[f.stem] = out
    return {"horizon_minutes": 60, "period": "original fold test period", "methods": res}


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
