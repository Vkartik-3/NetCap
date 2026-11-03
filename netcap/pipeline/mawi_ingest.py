"""MAWI trace -> flow records (softflowd, IPFIX over TCP) -> aggregates in Parquet and Postgres.

  python -m netcap.pipeline.mawi_ingest 202510011400               # convert + aggregate, keep raw
  python -m netcap.pipeline.mawi_ingest 202510011400               # again: rebuilds from the IPFIX stream
  python -m netcap.pipeline.mawi_ingest 202510011400 --delete-raw  # same, then delete pcap and IPFIX stream

Only derived aggregates are kept long term. Every run reconciles three independent packet counts
(pcap record walk, softflowd's counters, decoded flow records). Memory is bounded: the IPFIX stream is
decoded in chunks (netcap.data.ipfix.iter_decode) and aggregated chunk by chunk (mawi.ChunkedAggregates).
Results: results/netcap/phase1_mawi_<id>_chunked.json (conversion run) and ..._chunked_rerun.json (runs
from the IPFIX stream). The first Phase 1 run (phase1_mawi_<id>.json, whole trace in memory) is kept as a
record but superseded: its memory and timing numbers are not used.
"""
import argparse
import json
import subprocess
import sys
import time

import polars as pl

from .. import config
from ..data import ipfix, mawi
from ..db import load as db
from ..provenance import peak_rss_bytes, run_record, save_result, sha256_file
from ..store.parquet import TableWriter

URL = "https://mawi.wide.ad.jp/mawi/samplepoint-F/{y}/{id}.pcap.gz"
COLS = {
    "mawi_traces": ["trace_id", "source_url", "start_ts", "end_ts", "flows", "packets", "bytes", "pcap_packets",
                    "flow_tool"],
    "mawi_protocol_mix": ["trace_id", "proto", "flows", "packets", "bytes", "byte_share", "packet_share"],
    "mawi_top_talkers": ["trace_id", "kind", "rank", "key", "flows", "packets", "bytes", "byte_share"],
    "mawi_throughput": ["trace_id", "bin_start", "bin_s", "bytes", "mbps", "flows_active"],
}
TOOL = ("softflowd 1.1.1 (0260261) -> IPFIX/TCP; timeouts " + ",".join(f"{k}={v}" for k, v in mawi.TIMEOUTS.items())
        + "; unidirectional records")


def write_stores(tables, tid):
    out = {"parquet": {}}
    for name, frame in tables.items():
        root = config.DATA / "mawi" / name.removeprefix("mawi_")
        w = TableWriter(root)
        for other in sorted(root.glob("trace_id=*")):          # keep other traces' files
            if other.name != f"trace_id={tid}":
                for f in other.glob("*.parquet"):
                    w.files[str(f.relative_to(root))] = {"sha256": sha256_file(f), "bytes": f.stat().st_size,
                                                         "rows": pl.read_parquet(f).height, "status": "unchanged"}
        w.write(frame.select(COLS[name]), f"trace_id={tid}/part-000.parquet")
        out["parquet"][name] = w.finish()
    with db.connect() as conn:
        db.apply_schema(conn)
        out["postgres"] = {name: db.timed_upsert(conn, name, [frame.select(COLS[name])], COLS[name])
                           for name, frame in tables.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_id")
    ap.add_argument("--delete-raw", action="store_true", help="delete pcap and IPFIX stream after success")
    ap.add_argument("--chunk-records", type=int, default=4_000_000, help="flow records decoded per chunk")
    ap.add_argument("--duckdb-memory", default="3GB", help="DuckDB memory limit for the top-talker group-bys")
    ap.add_argument("--result-suffix", default="chunked", help="results/netcap/phase1_mawi_<id>_<suffix>.json")
    a = ap.parse_args(argv)
    tid = a.trace_id
    pcap = config.RAW_MAWI / f"{tid}.pcap.gz"
    stream = config.RAW_MAWI / "ipfix" / f"{tid}.ipfix"
    html = config.RAW_MAWI / f"{tid}.html"
    info = mawi.parse_trace_info(html.read_text()) if html.exists() else {}
    rec = {"trace_id": tid, "source_url": URL.format(y=tid[:4], id=tid), "archive_summary": info,
           "label": "REAL flow data from MAWI (WIDE samplepoint-F), not Abilene; addresses anonymised by MAWI"}
    t_all = time.perf_counter()
    converted = False
    if not stream.exists():
        if not pcap.exists():
            raise SystemExit(f"neither {pcap} nor {stream} exists")
        rec["pcap_bytes"] = pcap.stat().st_size
        rec["pcap_sha256"] = sha256_file(pcap)
        counter = subprocess.Popen(f"gzcat '{pcap}' | '{sys.executable}' -m netcap.data.pcapcount", shell=True,
                                   stdout=subprocess.PIPE, text=True, cwd=config.ROOT)
        rec["softflowd"] = mawi.pcap_to_ipfix(pcap, stream)
        t0 = time.perf_counter()
        rec["pcap_count"] = json.loads(counter.communicate()[0])
        rec["pcap_count"]["extra_wait_s"] = time.perf_counter() - t0
        sf = rec["softflowd"]
        if info.get("packets"):
            sf["packets_per_s"] = info["packets"] / sf["wall_s"]
        converted = True
    else:
        rec["softflowd"] = "skipped: aggregates rebuilt from the existing IPFIX stream"
    t0 = time.perf_counter()
    st = ipfix._Stats()
    agg = mawi.ChunkedAggregates(config.DATA / "mawi_spill" / tid, memory_limit=a.duckdb_memory)
    t_conv = 0.0
    for recs in ipfix.iter_decode(stream, chunk_records=a.chunk_records, stats=st):
        t1 = time.perf_counter()
        df = mawi.records_to_frame(recs)
        del recs
        t_conv += time.perf_counter() - t1
        agg.add(df)
        del df
    rec["decode"] = {**st.as_dict(), "decode_and_chunk_aggregate_s": time.perf_counter() - t0,
                     "records_to_frame_s": t_conv, "chunk_records": a.chunk_records,
                     "flow_records": sum(st.records[t] for t, d in st.templates.items() if "packetDeltaCount" in d.names),
                     "ipfix_sha256": sha256_file(stream), "peak_rss_bytes_after_decode": peak_rss_bytes()}
    t0 = time.perf_counter()
    (tot, proto, talkers, tp), ainfo = agg.finish(tid)
    rec["aggregate_finish_s"] = time.perf_counter() - t0
    rec["chunked"] = {"chunks": ainfo["chunks"], "spill_bytes": ainfo["spill_bytes"],
                      "duckdb_memory_limit": a.duckdb_memory, "spill_deleted": True}
    rec["totals"] = tot
    rec["by_ip_version"] = ainfo["by_ip_version"]
    if converted:
        c, sf = rec["pcap_count"], rec["softflowd"]
        rec["reconciliation"] = {
            "pcap_ip_records (independent walk)": c["ethertype_ipv4"] + c["ethertype_ipv6"],
            "softflowd_packets_processed": sf["packets_processed"],
            "flow_record_packets": tot["packets"],
            "archive_summary_packets (all, incl. non-IP)": rec["archive_summary"].get("packets"),
            "pcap_records (all)": c["records"],
            "flow_bytes": tot["bytes"], "pcap_wire_bytes_minus_14_per_record": c["wire_bytes"] - 14 * c["records"],
            "packets_match_exactly": c["ethertype_ipv4"] + c["ethertype_ipv6"] == tot["packets"]
                                     == sf["packets_processed"],
            "export_failures": sf["export_failures"]}
    trace = pl.DataFrame([{"trace_id": tid, "source_url": rec["source_url"], "start_ts": tot["t_first"],
                           "end_ts": tot["t_last"], "flows": tot["flows"], "packets": tot["packets"],
                           "bytes": tot["bytes"], "pcap_packets": info.get("packets"), "flow_tool": TOOL}]
                         ).with_columns(pl.from_epoch(pl.col("start_ts") * 1e6, time_unit="us").dt.replace_time_zone("UTC"),
                                        pl.from_epoch(pl.col("end_ts") * 1e6, time_unit="us").dt.replace_time_zone("UTC"))
    t0 = time.perf_counter()
    rec["stores"] = write_stores({"mawi_traces": trace, "mawi_protocol_mix": proto, "mawi_top_talkers": talkers,
                                  "mawi_throughput": tp}, tid)
    rec["stores_s"] = time.perf_counter() - t0
    rec["wall_s"] = time.perf_counter() - t_all
    rec["peak_rss_bytes"] = peak_rss_bytes()
    import resource
    ch = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    rec["peak_rss_bytes_children"] = int(ch if sys.platform == "darwin" else ch * 1024)   # softflowd, gzcat, counter
    rec["protocol_mix"] = proto.to_dicts()
    rec["top5"] = {k: talkers.filter(pl.col("kind") == k).head(5).to_dicts() for k in ("src", "dst", "pair", "dport")}
    q = tp["mbps"]
    rec["throughput_mbps_10s"] = {"bins": tp.height, "mean": q.mean(), "p50": q.median(), "p95": q.quantile(0.95),
                                  "max": q.max(), "approximation": "flow bytes spread uniformly over flow duration"}
    if a.delete_raw:
        for p in (pcap, stream):
            if p.exists():
                p.unlink()
        rec["raw_deleted"] = [str(p.name) for p in (pcap, stream)]
    name = f"phase1_mawi_{tid}_{a.result_suffix}" + ("" if converted else "_rerun")
    path = save_result(name, run_record({"dataset": "MAWI samplepoint-F", "trace_id": tid,
                                         "pcap_sha256": rec.get("pcap_sha256", "see phase1_mawi_" + tid)}, **rec))
    print("saved", path)
    print(json.dumps({k: rec[k] for k in ("reconciliation", "totals", "throughput_mbps_10s") if k in rec},
                     indent=1, default=str))
    print("NETCAP_END", time.time(), flush=True)          # netcap.ops.measure: teardown time after this mark


if __name__ == "__main__":
    main()
