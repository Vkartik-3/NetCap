"""Check a running NetCap stack over HTTP and record the outcome.

  python -m netcap.ops.stack_check compose --api http://localhost:8000 --prometheus http://localhost:9090
  python -m netcap.ops.stack_check kind --api http://localhost:8080

Saves results/netcap/phase6_stack_<name>.json.
"""
import argparse
import json
import time
import urllib.request

from ..provenance import run_record, save_result

GETS = ["/health", "/metrics", "/api/topology", "/api/link/1/utilisation", "/api/forecast/summary",
        "/api/forecast?flow=86&horizon=12&method=gbm&limit=10", "/api/results/summary", "/api/limitations", "/"]


def call(url, data=None):
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"content-type": "application/json"}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            return {"status": r.status, "ms": (time.perf_counter() - t0) * 1000, "bytes": len(body), "body": body}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "ms": (time.perf_counter() - t0) * 1000, "bytes": 0, "body": b""}
    except Exception as e:
        return {"status": None, "error": repr(e), "ms": None, "bytes": 0, "body": b""}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--api", required=True)
    ap.add_argument("--prometheus")
    a = ap.parse_args(argv)
    out = {"api": a.api, "endpoints": {}}
    for g in GETS:
        r = call(a.api + g)
        out["endpoints"][g] = {k: v for k, v in r.items() if k != "body"}
        if g == "/health" and r["status"] == 200:
            out["health"] = json.loads(r["body"])
    for name, body in (("POST /api/scenario failure", {"fail_links": [16, 19]}), ("POST /api/scenario growth", {"growth_factor": 2})):
        r = call(a.api + "/api/scenario", body)
        out["endpoints"][name] = {k: v for k, v in r.items() if k != "body"}
        if r["status"] == 200:
            out["endpoints"][name]["max_util"] = json.loads(r["body"])["max_util"]
    r = call(a.api + "/api/te", {})
    out["endpoints"]["POST /api/te"] = {k: v for k, v in r.items() if k != "body"}
    if a.prometheus:
        r = call(a.prometheus + "/api/v1/targets")
        t = json.loads(r["body"])["data"]["activeTargets"] if r["status"] == 200 else []
        out["prometheus_targets"] = [{"job": x["labels"]["job"], "health": x["health"], "lastError": x["lastError"]} for x in t]
        r = call(a.prometheus + "/api/v1/query?query=sum(netcap_http_requests_total)")
        out["prometheus_requests_total"] = json.loads(r["body"])["data"]["result"] if r["status"] == 200 else None
    out["all_ok"] = all(e.get("status") == 200 for e in out["endpoints"].values()) and out.get("health", {}).get("database", False)
    print(save_result(f"phase6_stack_{a.name}", run_record("live stack", **out)))
    print(json.dumps({k: (v["status"], round(v["ms"] or -1)) for k, v in out["endpoints"].items()}, indent=0))
    print(out.get("health"), out.get("prometheus_targets"), out["all_ok"])


if __name__ == "__main__":
    main()
