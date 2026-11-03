"""Collect the Phase 1 headline numbers from results/netcap/phase1_*.json into phase1_summary.json.

  python -m netcap.pipeline.phase1_summary
Nothing is recomputed: every value is copied from a result file, and its source file is named.
"""
import json

from .. import config
from ..provenance import run_record, save_result


ABILENE_KIND = {"nodes": "reference (topology)", "links": "reference (topology)", "flows": "reference (OD pairs)",
                "routing": "reference (non-zero entries of the 54 x 144 routing matrix A)",
                "time_steps": "reference (one row per 5-minute step with cleaning label)",
                "traffic_matrix": "MEASURED: OD-flow traffic per 5-minute step (realOD)",
                "link_load": "DERIVED: traffic_matrix x routing matrix A^T, not a measurement"}


def load(name):
    return json.loads((config.RESULTS / f"{name}.json").read_text())


MAWI_ATTEMPTS = ("chunked", "chunked_lowmem")     # bounded-memory re-runs (Phase 1 follow-up 1), in order


def mawi_section():
    """Only the bounded-memory re-runs are used for memory and throughput (the first run is superseded)."""
    tid = "202510011400"
    out = {"trace": tid, "superseded": {"file": f"phase1_mawi_{tid}.json", "why": "whole trace in memory; "
           "peak footprint 16.8 GB and exit long after the work finished; numbers not used"},
           "idle_swap_control": {k: v for k, v in load("phase1_swap_control_idle_440s").items()
                                 if k in ("process_wall_s", "vm_stat_delta_system_wide")},
           "attempts": []}
    for suffix in MAWI_ATTEMPTS:
        f = config.RESULTS / f"phase1_mawi_{tid}_{suffix}.json"
        if not f.exists():
            continue
        mw, pr = load(f"phase1_mawi_{tid}_{suffix}"), load(f"phase1_mawi_{tid}_{suffix}_process")
        out["attempts"].append({
            "source": f"phase1_mawi_{tid}_{suffix}.json, phase1_mawi_{tid}_{suffix}_process.json",
            "settings": {"chunk_records": mw["decode"]["chunk_records"], "duckdb_memory_limit": mw["chunked"]["duckdb_memory_limit"]},
            "label": mw["label"], "reconciliation": mw["reconciliation"], "flow_records": mw["totals"]["flows"],
            "softflowd_wall_s": mw["softflowd"]["wall_s"], "softflowd_packets_per_s": mw["softflowd"]["packets_per_s"],
            "decode_and_aggregate_s": mw["decode"]["decode_and_chunk_aggregate_s"], "pipeline_wall_s_in_process": mw["wall_s"],
            "process_wall_s_to_exit": pr["process_wall_s"], "teardown_s": pr["teardown_s"],
            "peak_rss_bytes_pipeline": mw["peak_rss_bytes"], "peak_rss_bytes_children_max": mw["peak_rss_bytes_children"],
            "peak_memory_footprint_bytes": pr["peak_memory_footprint_bytes"],
            "swapouts_system_wide": pr["vm_stat_delta_system_wide"]["swapouts"],
            "swap_used_before_after": [pr["swapusage_before"], pr["swapusage_after"]],
            "aggregates_vs_phase1": {k: v for k, v in mw["stores"]["parquet"].items()},
            "throughput_mbps_10s": mw["throughput_mbps_10s"],
            "protocol_byte_share": {x["proto"]: x["byte_share"] for x in mw["protocol_mix"][:6]}})
    return out


def main():
    pa, pc = load("phase1_ingest_parquet_abilene"), load("phase1_ingest_parquet_cesnet")
    ga, gc = load("phase1_ingest_postgres_abilene"), load("phase1_ingest_postgres_cesnet")
    ql = load("phase1_query_latency")
    out = {
        "abilene": {
            "source": "phase1_ingest_parquet_abilene.json / phase1_ingest_postgres_abilene.json",
            "rows": pa["runs"][0]["rows_total"],
            "rows_by_table": {t: {"rows": v["rows"], "kind": ABILENE_KIND[t]} for t, v in pa["runs"][0]["tables"].items()},
            "measured_od_observations": {
                "all_steps": pa["runs"][0]["tables"]["traffic_matrix"]["rows"],
                "kept_steps_after_cleaning": pa["runs"][0]["kept_steps"] * 144,
                "note": "traffic_matrix rows = 48,384 five-minute steps x 144 OD flows (realOD, column 0). "
                        "Only these are traffic observations; link_load is computed from them and the routing "
                        "matrix, and the other tables are reference data."},
            "parquet": {"wall_s": pa["runs"][0]["wall_s"], "rows_per_s": pa["runs"][0]["rows_per_s"],
                        "peak_rss_bytes": pa["runs"][0]["peak_rss_bytes"], "rerun_changed_nothing": pa["rerun_changed_nothing"]},
            "postgres": {"wall_s": ga["runs"][0]["wall_s"], "rows_changed_run1": ga["runs"][0]["rows_changed"],
                         "rows_changed_run2": ga["runs"][1]["rows_changed"],
                         "rerun_changed_nothing": ga["rerun_changed_nothing"],
                         "table_bytes": {t: v["total_bytes"] for t, v in ga["runs"][-1]["table_sizes"].items()}},
            "cleaning_matches_reference": "tests_netcap/test_abilene.py::test_clean_matches_reference_on_real_data",
        },
        "cesnet_institutions": {
            "source": "phase1_ingest_parquet_cesnet.json / phase1_ingest_postgres_cesnet.json",
            "rows_by_aggregation": {k: v["rows"] for k, v in pc["runs"][0]["per_aggregation"].items()},
            "rows_total": pc["runs"][0]["rows_total"], "source_csv_bytes": pc["runs"][0]["source_csv_bytes"],
            "parquet": {"bytes": pc["runs"][0]["table"]["bytes"], "files": pc["runs"][0]["table"]["files"],
                        "wall_s": pc["runs"][0]["wall_s"], "rows_per_s": pc["runs"][0]["rows_per_s"],
                        "peak_rss_bytes": pc["runs"][0]["peak_rss_bytes"], "rerun_changed_nothing": pc["rerun_changed_nothing"]},
            "postgres": {"wall_s": gc["runs"][0]["wall_s"], "rows_per_s": gc["runs"][0]["rows_per_s"],
                         "rows_changed_run2": gc["runs"][1]["rows_changed"], "rerun_changed_nothing": gc["rerun_changed_nothing"],
                         "table_bytes": gc["runs"][-1]["table_sizes"]["cesnet_series"]},
            "scope": "per-institution aggregate series; not link-level; not used for routing scenarios",
        },
        "query_latency_ms_median_warm": {
            "source": "phase1_query_latency.json",
            "rows": ql["postgres"]["sizes"]["bench_heap"]["rows"],
            "postgres": {v: {q: r["median_ms"] for q, r in d.items()} for v, d in ql["postgres"]["latency"].items()},
            "parquet_duckdb": {v: {q: r["median_ms"] for q, r in d.items()} for v, d in ql["parquet_duckdb"]["latency"].items()},
            "identical_results_across_variants": ql["identical_results_across_variants"],
        },
        "mawi": mawi_section(),
        "scale_statement": ("Largest single table: CESNET institutions, 10,446,225 rows at 10-minute aggregation "
                            "(12,405,074 across all three aggregations); MAWI trace: 207,302,561 IP packets -> "
                            "56,523,852 flow records. Use these numbers rather than the words 'large-scale'."),
    }
    print(save_result("phase1_summary", run_record({"sources": "results/netcap/phase1_*.json"}, **out)))


if __name__ == "__main__":
    main()
