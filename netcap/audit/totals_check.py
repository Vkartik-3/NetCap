import sys, json
from pathlib import Path
import numpy as np
raw = Path(sys.argv[1]); res = {}
for w in range(1, 25):
    v = np.loadtxt(raw / f"X{w:02d}.gz").reshape(2016, 12, 12, 5)
    o, i = v.sum(2), v.sum(1)            # (2016,12,5) out / in totals
    def rel(t):
        d = np.abs(t[..., 1:] - t[..., :1]); return d, d / np.maximum(t[..., :1], 1)
    do, ro = rel(o); di, ri = rel(i)
    bad = (ro > 1e-3) & (do > 1)
    badi = (ri > 1e-3) & (di > 1)
    res[w] = {"steps_out_mismatch": int(bad.any((1, 2)).sum()), "steps_in_mismatch": int(badi.any((1, 2)).sum()),
              "max_rel_out": float(ro.max()), "max_rel_in": float(ri.max()),
              "mismatch_by_version_out": bad.any((0, 1)).tolist()}
bad_w = {w: r for w, r in res.items() if r["steps_out_mismatch"] or r["steps_in_mismatch"]}
print(json.dumps(bad_w, indent=1)); print("weeks with any mismatch:", list(bad_w))
