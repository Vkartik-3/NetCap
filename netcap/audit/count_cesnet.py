"""Count rows/bytes in extracted CESNET-TimeSeries24 institutions CSVs. Usage: python -I count_cesnet.py <extracted_dir> <out.json>"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
t0 = time.time()
agg = defaultdict(lambda: {"files": 0, "rows": 0, "bytes": 0})
header = {}
for f in sorted(root.rglob("*.csv")):
    key = str(f.parent.relative_to(root))
    with open(f, "rb") as fh:
        first = fh.readline()
        n = sum(1 for _ in fh)
    header.setdefault(key, first.decode(errors="replace").strip())
    a = agg[key]; a["files"] += 1; a["rows"] += n; a["bytes"] += f.stat().st_size
res = {"per_directory": dict(agg), "headers": header,
       "total_rows": sum(a["rows"] for a in agg.values()), "total_bytes": sum(a["bytes"] for a in agg.values()),
       "total_files": sum(a["files"] for a in agg.values()), "count_seconds": round(time.time() - t0, 1)}
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
