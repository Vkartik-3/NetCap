"""Query latency before/after indexing and partitioning, CESNET 10-minute series (Postgres and Parquet).

  python -m netcap.bench.query_latency [--runs 15]

Postgres variants (built from cesnet_series, same rows, same insertion order):
  heap         no index at all
  indexed      + btree (institution_id, ts) and btree (ts)
  partitioned  range-partitioned by month on ts, same indexes on every partition
Parquet variants (DuckDB): one flat file vs the store's Hive month partitions.
Warm cache: 1 warm-up run then --runs timed runs; client wall time; median and p95 reported.
Every variant must return identical results (checked), otherwise the benchmark fails.
"""
import argparse
import hashlib
import statistics
import time

import duckdb

from .. import config
from ..db import load as db
from ..provenance import peak_rss_bytes, run_record, save_result

INST, WEEK, MONTH, DAY = 42, ("2024-03-04", "2024-03-11"), ("2024-03-01", "2024-04-01"), ("2024-03-05", "2024-03-06")
QUERIES = {
    "q1_one_institution_one_week":
        "SELECT ts, n_bytes FROM {t} WHERE institution_id = {inst} AND ts >= '{w0} 00:00:00+00'::timestamptz AND ts < '{w1} 00:00:00+00'::timestamptz "
        "ORDER BY ts",
    "q2_network_hourly_total_one_month":
        "SELECT date_trunc('hour', ts) AS h, sum(n_bytes) AS b FROM {t} WHERE ts >= '{m0} 00:00:00+00'::timestamptz AND ts < '{m1} 00:00:00+00'::timestamptz "
        "GROUP BY 1 ORDER BY 1",
    "q3_top10_institutions_one_day":
        "SELECT institution_id, sum(n_bytes) AS b FROM {t} WHERE ts >= '{d0} 00:00:00+00'::timestamptz AND ts < '{d1} 00:00:00+00'::timestamptz "
        "GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 10",
}
FMT = dict(inst=INST, w0=WEEK[0], w1=WEEK[1], m0=MONTH[0], m1=MONTH[1], d0=DAY[0], d1=DAY[1])


def _digest(rows):
    return hashlib.sha256(repr([tuple(str(v) for v in r) for r in rows]).encode()).hexdigest()


def _time(fn, runs):
    fn()                                  # warm-up
    ts, out = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"median_ms": statistics.median(ts), "p95_ms": ts[min(len(ts) - 1, int(round(0.95 * (len(ts) - 1))))],
            "min_ms": ts[0], "runs": runs, "result_rows": len(out), "result_sha256": _digest(out)}


def build_postgres(conn):
    src = "SELECT * FROM cesnet_series WHERE aggregation = '10_minutes' ORDER BY institution_id, id_time"
    t = {}
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS bench_heap, bench_indexed, bench_partitioned CASCADE")
        t0 = time.perf_counter()
        cur.execute(f"CREATE TABLE bench_heap AS {src}")
        t["heap_build_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cur.execute(f"CREATE TABLE bench_indexed AS {src}")
        cur.execute("CREATE INDEX ON bench_indexed (institution_id, ts)")
        cur.execute("CREATE INDEX ON bench_indexed (ts)")
        t["indexed_build_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cur.execute("CREATE TABLE bench_partitioned (LIKE cesnet_series) PARTITION BY RANGE (ts)")
        cur.execute("SELECT date_trunc('month', min(ts)), date_trunc('month', max(ts)) FROM bench_heap")
        lo, hi = cur.fetchone()
        cur.execute("SELECT generate_series(%s::timestamptz, %s::timestamptz, interval '1 month')", (lo, hi))
        months = [r[0] for r in cur.fetchall()]
        for m in months:
            cur.execute(f"CREATE TABLE bench_p_{m:%Y_%m} PARTITION OF bench_partitioned FOR VALUES FROM "
                        f"('{m.isoformat()}') TO ('{m.isoformat()}'::timestamptz + interval '1 month')")
        cur.execute(f"INSERT INTO bench_partitioned {src}")
        cur.execute("CREATE INDEX ON bench_partitioned (institution_id, ts)")
        cur.execute("CREATE INDEX ON bench_partitioned (ts)")
        t["partitioned_build_s"] = time.perf_counter() - t0
        t["partitions"] = len(months)
        for name in ("bench_heap", "bench_indexed", "bench_partitioned"):
            cur.execute(f"ANALYZE {name}")
        sizes = {}
        for name in ("bench_heap", "bench_indexed", "bench_partitioned"):
            cur.execute("SELECT count(*) FROM " + name)
            n = cur.fetchone()[0]
            cur.execute("SELECT coalesce(sum(pg_total_relation_size(c.oid)), pg_total_relation_size(%s::regclass)) "
                        "FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid WHERE i.inhparent = %s::regclass",
                        (name, name))
            sizes[name] = {"rows": n, "total_bytes": int(cur.fetchone()[0])}
    conn.commit()
    return t, sizes


def bench_postgres(conn, runs):
    out = {}
    with conn.cursor() as cur:
        for variant in ("bench_heap", "bench_indexed", "bench_partitioned"):
            out[variant] = {}
            for q, sql in QUERIES.items():
                s = sql.format(t=variant, **FMT)
                def run(s=s):
                    cur.execute(s)
                    return cur.fetchall()
                r = _time(run, runs)
                cur.execute("EXPLAIN (FORMAT JSON) " + s)
                plan = cur.fetchone()[0][0]["Plan"]
                r["plan_root"] = plan["Node Type"]
                r["plan_scans"] = sorted({n for n in _walk(plan)})
                out[variant][q] = r
    return out


def _walk(p):
    if "Scan" in p["Node Type"]:
        yield f"{p['Node Type']}:{p.get('Relation Name', '')}"
    for c in p.get("Plans", []):
        yield from _walk(c)


def bench_parquet(runs):
    part = config.DATA / "cesnet" / "series" / "aggregation=10_minutes"
    flat = config.DATA / "cesnet" / "bench_flat_10_minutes.parquet"
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    t0 = time.perf_counter()
    con.execute(f"COPY (SELECT * FROM read_parquet('{part}/**/*.parquet') ORDER BY institution_id, id_time) "
                f"TO '{flat}' (FORMAT parquet, COMPRESSION zstd)")
    flat_build = time.perf_counter() - t0
    sources = {"flat_file": f"read_parquet('{flat}')",
               "hive_month_partitions": f"read_parquet('{part}/**/*.parquet', hive_partitioning = true)"}
    out = {}
    for variant, src in sources.items():
        out[variant] = {}
        for q, sql in QUERIES.items():
            s = sql.format(t=src, **FMT)
            out[variant][q] = _time(lambda s=s: con.execute(s).fetchall(), runs)
    files = list(part.rglob("*.parquet"))
    return out, {"flat_build_s": flat_build, "flat_bytes": flat.stat().st_size,
                 "partitioned_files": len(files), "partitioned_bytes": sum(f.stat().st_size for f in files)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--result-name", default="phase1_query_latency")
    a = ap.parse_args(argv)
    with db.connect() as conn:
        conn.execute("SET TimeZone = 'UTC'")          # same hour boundaries and output as DuckDB
        build, sizes = build_postgres(conn)
        pg = bench_postgres(conn, a.runs)
        with conn.cursor() as cur:
            cur.execute("SHOW server_version")
            server = cur.fetchone()[0]
            cur.execute("SHOW shared_buffers")
            sb = cur.fetchone()[0]
    pq, pq_meta = bench_parquet(a.runs)
    digests = {q: {(e, v): r[q]["result_sha256"] for e, d in (("pg", pg), ("pq", pq)) for v, r in d.items()}
               for q in QUERIES}
    same = {q: len(set(d.values())) == 1 for q, d in digests.items()}
    payload = run_record({"dataset": "CESNET-TimeSeries24 institutions, aggregation 10_minutes"},
                         queries={q: s.format(t="<table>", **FMT) for q, s in QUERIES.items()},
                         postgres={"server_version": server, "shared_buffers": sb, "build": build, "sizes": sizes,
                                   "latency": pg},
                         parquet_duckdb={"duckdb_version": duckdb.__version__, **pq_meta, "latency": pq},
                         identical_results_across_variants=same, cache="warm (1 warm-up run per query)",
                         peak_rss_bytes=peak_rss_bytes())
    path = save_result(a.result_name, payload)
    if not all(same.values()):
        raise SystemExit(f"result mismatch across variants: {same} (saved {path})")
    print("saved", path)
    for eng, d in (("postgres", pg), ("parquet", pq)):
        for v, r in d.items():
            print(f"{eng:9s} {v:22s} " + "  ".join(f"{q[:2]} {x['median_ms']:8.2f} ms" for q, x in r.items()))


if __name__ == "__main__":
    main()
