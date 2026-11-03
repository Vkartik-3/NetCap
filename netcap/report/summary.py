"""Phase 7: collect the headline numbers of every phase into results/netcap/summary.json.

  python -m netcap.report.summary

Nothing is computed from data here: every value is copied from a phase result file, and each entry names
its source file and key path, so RESULTS.md and RESUME_FACTS.md can cite them. Missing files are reported
as "not produced".
"""
import json

from .. import config
from ..provenance import run_record, save_result


class Src:
    def __init__(self):
        self.cache = {}

    def load(self, name):
        if name not in self.cache:
            p = config.RESULTS / f"{name}.json"
            self.cache[name] = json.loads(p.read_text()) if p.exists() else None
        return self.cache[name]

    def get(self, name, *path):
        d = self.load(name)
        if d is None:
            return {"value": None, "source": f"{name}.json (not produced)"}
        for k in path:
            if isinstance(d, list):
                d = d[int(k)]
            elif isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return {"value": None, "source": f"{name}.json: {'/'.join(map(str, path))} (missing)"}
        return {"value": d, "source": f"{name}.json: {'/'.join(map(str, path))}"}


def pct(x, d=1):
    return None if x is None else f"{x:+.{d}f}%"


def main():
    S = Src()
    v = lambda *a: S.get(*a)["value"]
    out = {"phase1": {}, "phase2": {}, "phase3": {}, "phase4": {}, "neural": {}, "phase6": {}}

    # Phase 1
    p1 = "phase1_summary"
    out["phase1"] = {
        "abilene_rows_by_table": S.get(p1, "abilene", "rows_by_table"),
        "abilene_measured_od_observations": S.get(p1, "abilene", "measured_od_observations"),
        "cesnet_rows_by_aggregation": S.get(p1, "cesnet_institutions", "rows_by_aggregation"),
        "cesnet_parquet": S.get(p1, "cesnet_institutions", "parquet"),
        "cesnet_postgres": S.get(p1, "cesnet_institutions", "postgres"),
        "abilene_parquet": S.get(p1, "abilene", "parquet"),
        "abilene_postgres_wall_s": S.get(p1, "abilene", "postgres", "wall_s"),
        "query_latency_ms": S.get(p1, "query_latency_ms_median_warm"),
        "mawi_used_attempt": S.get(p1, "mawi", "attempts", "-1"),
        "mawi_idle_control": S.get(p1, "mawi", "idle_swap_control"),
    }
    mw = S.load(p1)["mawi"]["attempts"] if S.load(p1) else []
    out["phase1"]["mawi_used_attempt"] = {"value": mw[-1] if mw else None, "source": f"{p1}.json: mawi/attempts/{len(mw) - 1}"}

    # Phase 2
    for panel in ("abilene", "cesnet_1h"):
        name = f"phase2_forecast_{panel}"
        R = v(name, "results")
        if not R:
            out["phase2"][panel] = {"value": None, "source": f"{name}.json (not produced)"}
            continue
        rows = {}
        for fold, hs in R.items():
            for h, c in hs.items():
                m = c["methods"]
                rows[f"{fold}/{h}"] = {
                    "selected_on_validation": c["selected_on_validation"]["method"],
                    "selected_test_gain_vs_persistence_pct": c["selected_on_validation"]["test_gain_vs_persistence_pct"],
                    "selected_ci95": c["selected_on_validation"]["test_vs_persistence"]["ci95"],
                    "selected_dm_p": c["selected_on_validation"]["test_vs_persistence"]["dm_p"],
                    "persistence_test_mae": m["persistence"]["test"]["mae"],
                    "gbm_test_gain_pct": m["gbm"]["test_gain_vs_persistence_pct"],
                    "gbm_vs_best_simple": {"baseline": c["best_simple_baseline_on_validation"], **c["gbm_vs_best_simple"]},
                    "best_simple_test_gain_pct": m[c["best_simple_baseline_on_validation"]]["test_gain_vs_persistence_pct"],
                    "ets_test_gain_pct": m["ets"]["test_gain_vs_persistence_pct"],
                    "gbm_conformal_coverage": m["gbm"]["conformal"]["coverage_two_sided"],
                    "persistence_conformal_coverage": m["persistence"]["conformal"]["coverage_two_sided"],
                }
                if "test_flow86" in m["persistence"]:
                    rows[f"{fold}/{h}"]["flow86_mae"] = {k: m[k]["test_flow86"]["mae"] for k in
                                                        ("persistence", "blend", "blend_rule", "blend_gate", "gbm", "gbm_rule",
                                                         "ets", "ets_rule", "ridge", "ridge_clip", "ridge_basedef") if k in m}
                if "top_flows_subset" in c:
                    t = c["top_flows_subset"]["test"]
                    rows[f"{fold}/{h}"]["top10_subset_mae"] = {k: t[k]["mae"] for k in t if isinstance(t[k], dict) and "mae" in t[k]
                                                               and not k.endswith("_val")}
        out["phase2"][panel] = {"value": rows, "source": f"{name}.json: results/<fold>/<h>",
                                "unit": v(name, "unit"), "wall_s": v(name, "wall_s"), "timings_s": v(name, "timings_s")}

    # Phase 3
    s3 = "phase3_scenarios"
    out["phase3"] = {k: S.get(s3, *path) for k, path in {
        "routing_gate": ("gate",), "demand": ("demand",), "baseline_max_util": ("baseline", "max_util"),
        "baseline_worst_link": ("baseline", "worst_link_name"), "baseline_p95_worst": ("baseline", "p95_util_worst"),
        "headroom": ("baseline", "headroom"), "months_to_threshold_ASSUMED_growth": ("baseline", "months_to_threshold_ASSUMED_growth"),
        "n1_worst": ("n_minus_1", "worst"), "n1_worst_max_util": ("n_minus_1", "worst_max_util"),
        "n1_cases_over_threshold": ("n_minus_1", "cases_over_threshold"),
        "n1_unroutable": ("n_minus_1", "cases_with_unroutable_flows"), "upgrade": ("upgrade",), "combined": ("combined",),
        "te_ratio": ("te", "lp_over_ospf_ratio"), "forecast_driven": ("forecast_driven",)}.items()}

    # Phase 4
    a4 = v("phase4_anomaly", "results")
    if a4:
        out["phase4"] = {"value": {case: {d: {k: r.get(k) for k in ("test_average_precision", "average_precision_all_rows",
                                                                     "recall_top1pct", "precision_top1pct_vs_weak", "test_event_recall")}
                                          | ({"test_f1": r["test"]["f1"], "test_precision": r["test"]["precision"],
                                              "test_recall": r["test"]["recall"]} if "test" in r else {})
                                          for d, r in c["detectors"].items()} for case, c in a4.items()},
                         "source": "phase4_anomaly.json: results/<case>/detectors/<detector>"}

    # Neural
    out["neural"] = {"timing": S.get("phase_neural_timing"), "retrain": S.get("phase_neural_retrain", "results")}
    # Phase 6
    q = S.load("phase6_quality") or {}
    out["phase6"] = {"tests": {"value": {k: q.get("tests", {}).get(k) for k in ("passed", "skipped", "failed")},
                               "source": "phase6_quality.json: tests"},
                     "coverage_total_percent": S.get("phase6_quality", "coverage_total_percent"),
                     "coverage_by_package": S.get("phase6_quality", "coverage_by_package"),
                     "container_stack_status": S.get("phase6_quality", "container_stack_status"),
                     "idle_benchmarks": {n: {"timing_label": v("" + n + "_process", "timing_label") if S.load(n + "_process") else v(n, "cpu_at_save", "timing_label"),
                                             "source": f"{n}.json"} for n in
                                         ("phase6_idle_ingest_parquet_cesnet", "phase6_idle_ingest_postgres_cesnet",
                                          "phase6_idle_query_latency", "phase6_idle_forecast_throughput")},
                     "mawi_idle": S.get("phase6_idle_mawi_process"),
                     "forecast_throughput": S.get("phase6_idle_forecast_throughput", "results")}

    out["kpis"] = kpis(out)
    out["warnings_finding_html"] = warnings_html(out)
    print(save_result("summary", run_record({"sources": "results/netcap/*.json"}, **out)))


def kpis(o):
    k = []
    p1 = o["phase1"]
    rows = (p1["cesnet_rows_by_aggregation"]["value"] or {}).get("10_minutes")
    if rows:
        k.append({"label": "CESNET rows ingested (10-min)", "value": f"{rows:,}", "source": p1["cesnet_rows_by_aggregation"]["source"]})
    m = p1["mawi_used_attempt"]["value"]
    if m:
        k.append({"label": "MAWI packets reconciled exactly", "value": f"{m['reconciliation']['flow_record_packets']:,}",
                  "source": p1["mawi_used_attempt"]["source"]})
    a = o["phase2"].get("abilene", {}).get("value")
    if a:
        r = a["original/h12"]
        k.append({"label": f"60-min forecast vs persistence ({r['selected_on_validation']}, chosen on validation)",
                  "value": pct(r["selected_test_gain_vs_persistence_pct"]),
                  "source": "phase2_forecast_abilene.json: results/original/h12"})
    g = o["phase3"]["routing_gate"]["value"]
    if g:
        k.append({"label": "OSPF routing reproduces matrix A", "value": f"{g['flows_matching']}/{g['flows']} flows",
                  "source": o["phase3"]["routing_gate"]["source"]})
    te = o["phase3"]["te_ratio"]["value"]
    if te:
        k.append({"label": "TE LP max-util / OSPF max-util (median, busiest steps)", "value": f"{te['normal']['median']:.2f}",
                  "source": o["phase3"]["te_ratio"]["source"]})
    return k


def warnings_html(o):
    fd = o["phase3"]["forecast_driven"]["value"]
    if not fd or "methods" not in fd:
        return "Phase 3 forecast-driven check not produced yet."
    ci = lambda a: f"[{a[0]:+.3f}, {a[1]:+.3f}]" if a else "-"
    rows = []
    for m, r in sorted(fd["methods"].items()):
        p = r["point"]
        rows.append(f"<tr><td>{m}</td><td>{p['link_load_mae_pp']:.3f}</td><td>{p['warning_f1']:.3f}</td>"
                    f"<td>{p.get('warning_f1_diff_vs_persistence', float('nan')):+.3f} {ci(p.get('warning_f1_diff_ci95'))}</td>"
                    f"<td>{r['hi']['warning_f1']:.3f}</td></tr>")
    ev = fd.get("warning_events") or {}
    sig = sorted(m for m, r in fd["methods"].items() if r["point"].get("warning_f1_diff_excludes_0"))
    note = (f" The test period has {ev['actual_events']:,} warning events on only {ev['days_with_events']} of {ev['test_days']} days "
            f"({ev['largest_day_share']:.0%} on one day), so the intervals are wide: "
            + (f"{len(sig)} method(s) differ from persistence ({', '.join(sig)})." if sig else
               "no method's F1 difference from persistence excludes 0.")) if ev else ""
    return ("<p>Link-load error and congestion-warning F1 (link at or above 50% utilisation) in the original test period, "
            "60-minute forecasts. F1 difference vs persistence with a 95% paired day-block bootstrap interval." + note +
            " Source: phase3_scenarios.json, forecast_driven.</p><table><tr><th>method</th>"
            "<th>link-load MAE (pp)</th><th>warning F1 (point)</th><th>F1 − persistence [95% CI]</th>"
            "<th>warning F1 (90% upper)</th></tr>" + "".join(rows) + "</table>")

if __name__ == "__main__":
    main()
