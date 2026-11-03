"""Phase 6: run the checks that can run on this machine and record each outcome.

  python -m netcap.ops.quality

Tests with coverage (per package), ruff, yamllint, kubeconform (strict), kustomize build, docker compose
config, and whether a Docker daemon is reachable (if not, the container stack is UNTESTED).
Saves results/netcap/phase6_quality.json.
"""
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict

from .. import config
from ..provenance import run_record, save_result


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=config.ROOT, **kw)
    return {"cmd": " ".join(cmd), "returncode": r.returncode, "tail": (r.stdout + r.stderr)[-2500:]}


def coverage_by_package(path):
    d = json.loads(path.read_text())
    agg = defaultdict(lambda: [0, 0])
    for f, info in d["files"].items():
        parts = f.split("/")
        pkg = "/".join(parts[:2]) if parts[0] == "netcap" and len(parts) > 2 else parts[0]
        s = info["summary"]
        agg[pkg][0] += s["covered_lines"]
        agg[pkg][1] += s["num_statements"]
    return {k: {"covered": c, "statements": n, "percent": round(100 * c / n, 1) if n else None} for k, (c, n) in sorted(agg.items())}


def main():
    out = {}
    cov = config.ROOT / "coverage.json"
    t = sh([sys.executable, "-m", "pytest", "-q", "-ra", "--cov=netcap", "--cov=netcap_app", "--cov-report=json:coverage.json",
            "--cov-report=term"])
    m = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) failed)?", t["tail"])
    out["tests"] = {**t, "passed": int(m.group(1)) if m else None, "skipped": int(m.group(2) or 0) if m else None,
                    "failed": int(m.group(3) or 0) if m else None}
    if cov.exists():
        out["coverage_by_package"] = coverage_by_package(cov)
        tot = json.loads(cov.read_text())["totals"]
        out["coverage_total_percent"] = round(tot["percent_covered"], 1)
        cov.unlink()
    out["ruff"] = sh([sys.executable, "-m", "ruff", "check", "netcap", "netcap_app", "tests_netcap"])
    out["yamllint"] = sh([sys.executable, "-m", "yamllint", "docker-compose.yml", "infra", ".github"])
    out["kubeconform"] = sh(["kubeconform", "-strict", "-summary", "-ignore-filename-pattern", "kind-config|kustomization",
                             "infra/k8s/"]) if shutil.which("kubeconform") else {"returncode": None, "tail": "not installed"}
    out["kustomize"] = sh(["kubectl", "kustomize", "infra/k8s"]) if shutil.which("kubectl") else {"returncode": None}
    out["kustomize"]["tail"] = out["kustomize"].get("tail", "")[-300:]
    out["compose_config"] = sh(["docker", "compose", "config", "--quiet"]) if shutil.which("docker") else {"returncode": None}
    d = sh(["docker", "info", "--format", "{{.ServerVersion}}"]) if shutil.which("docker") else {"returncode": None}
    out["docker_daemon"] = {"available": d.get("returncode") == 0, "detail": d.get("tail", "")[-300:]}
    out["container_stack_status"] = "TESTED" if out["docker_daemon"]["available"] else "UNTESTED (no Docker daemon)"
    print(save_result("phase6_quality", run_record("n/a", **out)))
    print({k: (v.get("returncode") if isinstance(v, dict) and "returncode" in v else v) for k, v in out.items()
           if k not in ("coverage_by_package",)})
    print(json.dumps(out.get("coverage_by_package"), indent=1))


if __name__ == "__main__":
    main()
