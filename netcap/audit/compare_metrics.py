"""Compare measured metrics with a reference CSV. Usage: python -I compare_metrics.py <reference.csv> <measured.csv> <out.json>"""
import csv, json, sys
reference = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
repro = list(csv.DictReader(open(sys.argv[2], encoding="utf-8")))
key = lambda r: (r["name"], r["dataset"], r["horizon"], r["notes"])
rows, worst = [], 0.0
for r in repro:
    cands = [o for o in reference if key(o) == key(r)]
    if not cands:
        rows.append({"key": key(r), "reference": None}); continue
    diffs = {}
    for o in cands:   # the reference may contain repeated rows; report the closest and how many exist
        d = {m: abs(float(r[m]) - float(o[m])) / max(abs(float(o[m])), 1e-12) for m in ("mae", "rmse", "mape", "mape_coverage")}
        if not diffs or max(d.values()) < max(diffs.values()):
            diffs, best = d, o
    worst = max(worst, max(diffs.values()))
    rows.append({"key": key(r), "reference_rows_with_key": len(cands), "reference_timestamp": best["timestamp"],
                 "mae_repro": float(r["mae"]), "mae_reference": float(best["mae"]), "max_rel_diff": max(diffs.values())})
res = {"repro_rows": len(repro), "matched": sum(x.get("reference") is not None or "mae_reference" in x for x in rows),
       "unmatched": [x["key"] for x in rows if "mae_reference" not in x], "max_rel_diff_any_metric": worst, "rows": rows}
json.dump(res, open(sys.argv[3], "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
for x in rows:
    if "mae_reference" in x:
        print(f"{x['key'][0]:18s} {x['key'][1]:8s} h{x['key'][2]:3s} rel diff {x['max_rel_diff']:.2e}  ({x['reference_rows_with_key']} reference rows)  {x['key'][3][:50]}")
