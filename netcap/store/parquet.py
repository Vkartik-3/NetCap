"""Idempotent Parquet writer.

A file is written to a temporary path, hashed, and only moved into place when its content
differs from what is already there, so re-running an ingest with the same inputs leaves every
file (and its mtime) untouched. Files under a table directory that the current run did not
produce are removed, so the store always equals the latest run. Partitions use Hive-style
directories (key=value) so DuckDB/Polars can prune them.
"""
import hashlib
import json
import os
from pathlib import Path


COMPRESSION = "zstd"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def write_file(df, path):
    """Write df to path unless identical content is already there. Returns (status, sha256, bytes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    df.write_parquet(tmp, compression=COMPRESSION, statistics=True)
    sha = _sha(tmp)
    if path.exists() and _sha(path) == sha:
        tmp.unlink()
        return "unchanged", sha, path.stat().st_size
    status = "replaced" if path.exists() else "created"
    os.replace(tmp, path)
    return status, sha, path.stat().st_size


class TableWriter:
    """Collects the files of one table during a run; finish() removes stale files and writes a manifest."""

    def __init__(self, root):
        self.root = Path(root)
        self.files = {}

    def write(self, df, relpath):
        status, sha, size = write_file(df, self.root / relpath)
        self.files[str(relpath)] = {"sha256": sha, "bytes": size, "rows": df.height, "status": status}
        return status

    def finish(self):
        removed = []
        for p in self.root.rglob("*.parquet"):
            rel = str(p.relative_to(self.root))
            if rel not in self.files:
                p.unlink()
                removed.append(rel)
        manifest = {"files": dict(sorted(self.files.items())),
                    "rows": sum(f["rows"] for f in self.files.values()),
                    "bytes": sum(f["bytes"] for f in self.files.values())}
        mpath = self.root / "_manifest.json"
        # the manifest records content only (no run status), so it is unchanged when the files are
        stable = json.dumps({**manifest, "files": {k: {kk: vv for kk, vv in v.items() if kk != "status"}
                                                   for k, v in manifest["files"].items()}}, indent=1, sort_keys=True)
        if not mpath.exists() or mpath.read_text() != stable:
            mpath.write_text(stable)
        statuses = [f["status"] for f in self.files.values()]
        return {"rows": manifest["rows"], "bytes": manifest["bytes"], "files": len(self.files),
                "created": statuses.count("created"), "replaced": statuses.count("replaced"),
                "unchanged": statuses.count("unchanged"), "removed_stale": removed}
