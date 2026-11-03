"""Ingest CLI. Each invocation runs the ingest `--runs` times (default 2) and records whether the
second run changed anything, with timings, throughput and peak memory, in
results/netcap/phase1_ingest_<target>_<dataset>.json.

  python -m netcap.pipeline.ingest parquet abilene
  python -m netcap.pipeline.ingest parquet cesnet
  python -m netcap.pipeline.ingest postgres abilene
  python -m netcap.pipeline.ingest postgres cesnet
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone

from .. import config
from ..provenance import run_record, save_result, sha256_file

TABLES = {"abilene": ["nodes", "links", "flows", "routing", "time_steps", "traffic_matrix", "link_load"],
          "cesnet": ["cesnet_series"]}


def data_version(dataset):
    if dataset == "abilene":
        files = sorted(config.RAW_ABILENE.glob("X*.gz")) + [config.RAW_ABILENE / n for n in ("A", "links")]
        files += sorted(config.RAW_ABILENE.glob("topo-*.txt"))
        h = hashlib.sha256("".join(sha256_file(f) for f in files).encode()).hexdigest()
        return {"dataset": "Abilene TM (Zhang), 24 weeks", "files": len(files), "combined_sha256": h}
    rec = json.loads((config.RAW_CESNET.parent / "record.json").read_text())
    md5 = {f["key"]: f["checksum"] for f in rec["files"] if f["key"] in ("institutions.tar.gz", "times.tar.gz")}
    return {"dataset": "CESNET-TimeSeries24 institutions", "zenodo_record": rec["id"], "doi": rec.get("doi"),
            "archive_md5": md5}


def store_fingerprint(dataset):
    root = config.DATA / ("abilene" if dataset == "abilene" else "cesnet")
    files = sorted(p for p in root.rglob("*.parquet"))
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "content_sha256": hashlib.sha256("".join(sha256_file(p) for p in files).encode()).hexdigest(),
            "mtime_ns_sha256": hashlib.sha256("".join(str(p.stat().st_mtime_ns) for p in files).encode()).hexdigest()}


def run_parquet(dataset):
    from .to_parquet import abilene_to_parquet, cesnet_to_parquet
    return abilene_to_parquet() if dataset == "abilene" else cesnet_to_parquet()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["parquet", "postgres"])
    ap.add_argument("dataset", choices=["abilene", "cesnet"])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--result-name", default=None, help="default phase1_ingest_<target>_<dataset>")
    a = ap.parse_args(argv)
    runs, fps = [], []
    if a.target == "parquet":
        fps.append(store_fingerprint(a.dataset))
        for _ in range(a.runs):
            runs.append(run_parquet(a.dataset))
            fps.append(store_fingerprint(a.dataset))
    else:
        from ..db import load as db
        from .to_postgres import abilene_to_postgres, cesnet_to_postgres
        with db.connect() as conn:
            db.apply_schema(conn)
            fp = lambda: {t: db.table_fingerprint(conn, t, "dataset = 'abilene'" if a.dataset == "abilene" else "TRUE")
                          for t in TABLES[a.dataset]}
            fps.append(fp())
            for _ in range(a.runs):
                started = datetime.now(timezone.utc)
                r = abilene_to_postgres(conn) if a.dataset == "abilene" else cesnet_to_postgres(conn)
                db.log_run(conn, a.dataset, "postgres", started, r["rows_seen"], r["rows_changed"], r["wall_s"], r)
                runs.append(r)
                fps.append(fp())
            with conn.cursor() as cur:
                sizes = {}
                for t in TABLES[a.dataset]:
                    cur.execute("SELECT pg_total_relation_size(%s), pg_relation_size(%s), pg_indexes_size(%s)",
                                (t, t, t))
                    sizes[t] = dict(zip(("total_bytes", "heap_bytes", "index_bytes"), cur.fetchone()))
                cur.execute("SHOW server_version")
                server = cur.fetchone()[0]
        runs[-1]["table_sizes"] = sizes
        runs[-1]["postgres_server_version"] = server
    idempotent = all(fps[i] == fps[1] for i in range(2, len(fps))) if len(fps) > 2 else None
    if a.target == "postgres" and len(runs) > 1:
        idempotent = idempotent and all(r["rows_changed"] == 0 for r in runs[1:])
    payload = run_record(data_version(a.dataset), target=a.target, runs=runs, fingerprints=fps,
                         rerun_changed_nothing=idempotent)
    path = save_result(a.result_name or f"phase1_ingest_{a.target}_{a.dataset}", payload)
    print(json.dumps({"saved": str(path), "rerun_changed_nothing": idempotent,
                      "run1": {k: v for k, v in runs[0].items() if k not in ("tables", "table", "table_sizes")}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
