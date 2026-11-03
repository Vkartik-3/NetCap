"""Record the repository's GitHub Actions runs (public API, no token).

  python -m netcap.ops.ci_status [--limit 20]

Saves results/netcap/phase6_ci_runs.json: per run the commit, workflow, status, conclusion and time.
"""
import argparse
import json
import urllib.request

from ..provenance import run_record, save_result

REPO = "Vkartik-3/NetCap"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    a = ap.parse_args(argv)
    url = f"https://api.github.com/repos/{REPO}/actions/runs?per_page={a.limit}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "netcap-ci-status"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    runs = [{"commit": x["head_sha"][:7], "workflow": x["name"], "event": x["event"], "branch": x["head_branch"],
             "status": x["status"], "conclusion": x["conclusion"], "created_at": x["created_at"]} for x in d["workflow_runs"]]
    ci = [x for x in runs if x["workflow"] == "ci"]
    print(save_result("phase6_ci_runs", run_record(
        {"source": url}, repository=REPO, total_runs_reported=d["total_count"], runs=runs,
        ci_runs=len(ci), ci_success=sum(x["conclusion"] == "success" for x in ci),
        latest_ci=ci[0] if ci else None)))
    print({"ci_runs": len(ci), "ci_success": sum(x["conclusion"] == "success" for x in ci), "latest": ci[0] if ci else None})


if __name__ == "__main__":
    main()
