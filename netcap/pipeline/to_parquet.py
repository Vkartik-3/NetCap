"""Raw files -> partitioned Parquet store (data_netcap/). Idempotent; see netcap.store.parquet."""
import time

import numpy as np
import polars as pl

from .. import config
from ..data import abilene as ab
from ..data import cesnet as ce
from ..provenance import peak_rss_bytes
from ..store.parquet import TableWriter


def abilene_frames(weeks=None, raw=None):
    """All Abilene tables as polars frames. Removed steps are kept and flagged, never deleted."""
    weeks = ab.load_weeks(raw) if weeks is None else weeks
    c = ab.clean(weeks)
    topo = ab.load_topology(raw)
    n = weeks.shape[0] * ab.STEPS_PER_WEEK
    full = weeks.reshape(n, ab.N_FLOWS)
    seg = np.full(n, -1, np.int32)
    for i, (a, b) in enumerate(c.segments):
        seg[a:b] = i
    steps = pl.DataFrame({
        "step": np.arange(n, dtype=np.int32),
        "week": (np.arange(n) // ab.STEPS_PER_WEEK + 1).astype(np.int16),
        "step_in_week": (np.arange(n) % ab.STEPS_PER_WEEK).astype(np.int16),
        "ts_assumed": ab.step_times(weeks.shape[0]),
        "keep": c.keep, "reason": c.reason, "segment_id": seg})
    tm = pl.DataFrame({
        "step": np.repeat(np.arange(n, dtype=np.int32), ab.N_FLOWS),
        "flow_id": np.tile(np.arange(ab.N_FLOWS, dtype=np.int16), n),
        "value_raw": full.ravel()}).with_columns(week=(pl.col("step") // ab.STEPS_PER_WEEK + 1).cast(pl.Int16))
    load = full @ topo.A.T                                      # (n, 54) raw units
    cap = np.array([l[5] if l[5] is not None else np.nan for l in topo.links])   # kbit/s, internal links only
    mbps = load * ab.TO_MBIT_S
    ll = pl.DataFrame({
        "step": np.repeat(np.arange(n, dtype=np.int32), len(topo.links)),
        "link_id": np.tile(np.array([l[0] for l in topo.links], np.int16), n),
        "load_raw": load.ravel(), "load_mbps": mbps.ravel(),
        "utilization": (mbps / (cap / 1000)).ravel()}).with_columns(
        week=(pl.col("step") // ab.STEPS_PER_WEEK + 1).cast(pl.Int16),
        utilization=pl.col("utilization").fill_nan(None))
    nodes = pl.DataFrame(topo.nodes, schema=["name", "city", "lat", "lon"], orient="row").with_row_index("node_id")
    links = pl.DataFrame(topo.links, schema=["link_id", "name", "src", "dst", "link_type", "capacity_kbps",
                                             "igp_weight"], orient="row")
    flows = pl.DataFrame(ab.flow_pairs(), schema=["flow_id", "src", "dst"], orient="row")
    li, fi = np.nonzero(topo.A)
    routing = pl.DataFrame({"link_id": (li + 1).astype(np.int16), "flow_id": fi.astype(np.int16),
                            "fraction": topo.A[li, fi]})
    return {"nodes": nodes, "links": links, "flows": flows, "routing": routing,
            "time_steps": steps, "traffic_matrix": tm, "link_load": ll}, c


def abilene_to_parquet(root=None, raw=None):
    root = (root or config.DATA) / "abilene"
    t0 = time.perf_counter()
    weeks = ab.load_weeks(raw)
    t_parse = time.perf_counter() - t0
    frames, c = abilene_frames(weeks, raw)
    out = {}
    for name in ("nodes", "links", "flows", "routing", "time_steps"):
        w = TableWriter(root / name)
        w.write(frames[name], "part-000.parquet")
        out[name] = w.finish()
    for name in ("traffic_matrix", "link_load"):
        w = TableWriter(root / name)
        for (wk,), part in frames[name].partition_by("week", as_dict=True, maintain_order=True).items():
            w.write(part.drop("week"), f"week={wk:02d}/part-000.parquet")
        out[name] = w.finish()
    wall = time.perf_counter() - t0
    rows = sum(v["rows"] for v in out.values())
    return {"tables": out, "rows_total": rows, "wall_s": wall, "parse_s": t_parse,
            "rows_per_s": rows / wall, "peak_rss_bytes": peak_rss_bytes(),
            "removed_steps": c.removed(), "kept_steps": int(c.keep.sum()), "segments": len(c.segments)}


def cesnet_to_parquet(root=None, raw=None, aggs=tuple(ce.AGGREGATIONS), batch_size=32):
    """Chunked: batch_size institution files at a time; one file per (aggregation, month, batch)."""
    root = (root or config.DATA) / "cesnet" / "series"
    t0 = time.perf_counter()
    w = TableWriter(root)
    per_agg = {}
    for agg in aggs:
        ta, rows = time.perf_counter(), 0
        for b, df in ce.iter_batches(agg, batch_size, raw):
            df = df.with_columns(month=pl.col("ts").dt.strftime("%Y-%m"))
            for (month,), part in df.partition_by("month", as_dict=True, maintain_order=True).items():
                w.write(part.drop("month"), f"aggregation={agg}/month={month}/part-{b:03d}.parquet")
            rows += df.height
        per_agg[agg] = {"rows": rows, "wall_s": time.perf_counter() - ta}
    stats = w.finish()
    wall = time.perf_counter() - t0
    return {"table": stats, "per_aggregation": per_agg, "rows_total": stats["rows"], "wall_s": wall,
            "rows_per_s": stats["rows"] / wall, "peak_rss_bytes": peak_rss_bytes(),
            "source_csv_bytes": ce.source_bytes(raw, aggs), "batch_size": batch_size}
