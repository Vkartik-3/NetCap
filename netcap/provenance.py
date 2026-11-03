"""Every saved result carries how it was produced: command, code version, seed, date, data version."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def code_version():
    """Git commit of the NetCap repo if there is one, plus a hash of the netcap/ sources.

    The source hash identifies the exact code even while the repo has no commits yet.
    """
    import os
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True, text=True)
        commit = r.stdout.strip() if r.returncode == 0 else None
    except FileNotFoundError:                       # no git (e.g. inside the container image)
        commit = os.environ.get("NETCAP_COMMIT") or "unknown (git not available)"
        return {"netcap_commit": commit, "dirty": None, "netcap_source_sha256": _source_hash(),
                "project_release": config.PROJECT_RELEASE}
    dirty = None
    if commit:
        s = subprocess.run(["git", "status", "--porcelain", "netcap"], cwd=config.ROOT, capture_output=True, text=True)
        dirty = bool(s.stdout.strip())
    return {"netcap_commit": commit or "uncommitted (repo has no commits yet)", "dirty": dirty,
            "netcap_source_sha256": _source_hash(), "project_release": config.PROJECT_RELEASE}


def _source_hash():
    h = hashlib.sha256()
    for p in sorted((config.ROOT / "netcap").rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sql"):
            h.update(str(p.relative_to(config.ROOT)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


CONTENTION_LOAD = 4.0


def cpu_load():
    """1-minute load average now, and the label NetCap uses for timings (DECISIONS.md, user condition 2)."""
    import os
    la = os.getloadavg()[0]
    return {"load_average_1min": la, "cpu_count": os.cpu_count(),
            "timing_label": ("upper bound, measured under CPU contention" if la > CONTENTION_LOAD
                             else "measured on an otherwise idle machine")}


def run_record(data_version, seed=config.SEED, **extra):
    return {"cpu_at_save": cpu_load(), "command": " ".join([Path(sys.executable).name, *sys.argv]),
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": seed, "python": platform.python_version(), "platform": platform.platform(),
            "code": code_version(), "data_version": data_version, **extra}


def save_result(name, payload):
    """Write results/netcap/<name>.json and return its path."""
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path


def peak_rss_bytes():
    """Peak resident memory of this process so far (ru_maxrss is bytes on macOS, KiB on Linux)."""
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)
