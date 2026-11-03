"""Free disk space by deleting only things that can be regenerated, and record what was freed.

  python -m netcap.ops.cleanup            # dry run: list targets and sizes
  python -m netcap.ops.cleanup --apply    # delete, then save results/netcap/phase1_cleanup.json

Every target names the command that rebuilds it. Raw archives (tarballs) and the Parquet store are kept.
"""
import argparse
import os
import shutil
from pathlib import Path

from .. import config
from ..db import load as db
from ..provenance import run_record, save_result

SCRATCH = Path(os.environ.get("NETCAP_SCRATCH", "/nonexistent"))
TARGETS = [
    (config.ROOT / "raw_cesnet" / "extracted", "CSV files extracted from the kept tarballs",
     "mkdir -p raw_cesnet/extracted && tar -xzf raw_cesnet/institutions.tar.gz -C raw_cesnet/extracted && tar -xzf raw_cesnet/times.tar.gz -C raw_cesnet/extracted"),
    (config.ROOT / "raw_geant" / "totem", "GÉANT XML extracted from the kept archive (GÉANT not used)",
     "mkdir -p raw_geant/totem && tar -xjf raw_geant/dl/traffic-matrices-anonymized-v2.tar.bz2 -C raw_geant/totem"),
    (SCRATCH / "repro", "Phase 0 re-run of the forecasting module scripts (results kept in results/netcap/audit*)",
     "python -m netcap.audit.build_audit"),
    (SCRATCH / "build", "softflowd and nfdump source trees (installed under ~/.local)", "see docs/netcap/NOTES.md 5"),
    (config.ROOT / ".coverage", "coverage data file", "pytest --cov"),
]
BENCH_TABLES = "bench_heap, bench_indexed, bench_partitioned"


def size(p):
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def free_bytes():
    return shutil.disk_usage(config.ROOT).free


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    before = free_bytes()
    rows = []
    for path, what, rebuild in TARGETS:
        n = size(path)
        rows.append({"path": str(path), "what": what, "bytes": n, "rebuild": rebuild, "existed": path.exists()})
        if a.apply and path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT coalesce(sum(pg_total_relation_size(c.oid)), 0) FROM pg_class c "
                    "WHERE c.relname LIKE 'bench\\_%' AND c.relkind IN ('r', 'p')")
        n = int(cur.fetchone()[0])
        rows.append({"path": f"postgres tables {BENCH_TABLES}", "what": "query-latency benchmark copies",
                     "bytes": n, "rebuild": "python -m netcap.bench.query_latency", "existed": n > 0})
        if a.apply:
            cur.execute(f"DROP TABLE IF EXISTS {BENCH_TABLES} CASCADE")
            conn.commit()
    for r in rows:
        print(f"{r['bytes'] / 2**30:7.2f} GiB  {r['path']}")
    if a.apply:
        after = free_bytes()
        print(save_result("phase1_cleanup", run_record(
            "n/a", targets=rows, free_bytes_before=before, free_bytes_after=after,
            freed_bytes_measured=after - before, freed_bytes_listed=sum(r["bytes"] for r in rows))))
        print(f"free: {before / 2**30:.1f} GiB -> {after / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
