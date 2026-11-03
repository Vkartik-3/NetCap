"""Phase 0 audit: TF-free partial reproduction of scripts/06_baselines2.py.

Runs cells header..3 of the ORIGINAL script source unchanged, except the line
`import tensorflow as tf` is removed (TF is only used in cell 4, which loads saved
neural models; models/ is empty in this checkout). Then prints the cell-5 table and
cell-6 significance for the baselines only (no NN rows), with the script's own code paths.
Builds the h6/h12 and GEANT splits first with the same calls 03d/03e use.
Usage: TRAFFIC_PROJECT=<repro> python run06_baselines.py <repro>/scripts/06_baselines2.py <out.json>
"""
import json, sys
from pathlib import Path

src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(src_path.parent.parent))
import common
for h in (6, 12):
    d = common.PROCESSED / f"h{h}"
    if not (d / "scaler.npz").exists():
        common.prepare_abilene(horizon=h, directory=d)
for h in (1, 4):
    d = common.PROCESSED / "geant" / f"h{h}"
    if not (d / "scaler.npz").exists():
        common.prepare_geant(horizon=h, directory=d)

src = src_path.read_text()
head = src[:src.index("# %% 4.")]
assert head.count("import tensorflow as tf\n") == 1
head = head.replace("import tensorflow as tf\n", "")
ns = {"__file__": str(src_path), "__name__": "__main__"}
exec(compile(head, str(src_path), "exec"), ns)

# cell 5 / cell 6 logic restricted to baselines (copied expressions from the script)
CASES, C, R, BASELINES, LABEL = ns["CASES"], ns["C"], ns["R"], ns["BASELINES"], ns["LABEL"]
mae, gain, UNIT, case_label = ns["mae"], ns["gain"], ns["UNIT"], ns["case_label"]
np = ns["np"]
out = {"cases": {}, "arima_top5": {}}
for ds, h in CASES:
    case = (ds, h); c = C[case]; u = UNIT[ds]
    pv, pt = mae(c["y"][1], R[case]["persistence"]["val"]), mae(c["y"][2], R[case]["persistence"]["test"])
    best = min((b for b in BASELINES if b != "persistence"), key=lambda b: mae(c["y"][1], R[case][b]["val"]))
    row = {"persistence_mae_mbit": {"val": pv * u, "test": pt * u}, "best_baseline_by_val": best,
           "blend_cfg": list(map(str, c["blend_cfg"])), "ridge_cfg": list(map(str, c["ridge_cfg"])),
           "gain_vs_persistence_pct": {}}
    print(f"{case_label(ds, h)}   persistence MAE: val {pv * u:.3f}, test {pt * u:.3f} Mbit/s  best={best}")
    for name in BASELINES[1:]:
        mv, mt = mae(c["y"][1], R[case][name]["val"]), mae(c["y"][2], R[case][name]["test"])
        row["gain_vs_persistence_pct"][name] = {"val": gain(mv, pv), "test": gain(mt, pt)}
        print(f"  {LABEL[name]:30s} {gain(mv, pv):+7.1f}% {gain(mt, pt):+7.1f}%")
    day = ns["STEPS_PER_DAY"][ds]
    ea = np.abs(R[case][best]["test"] - c["y"][2]).mean(1)
    eb = np.abs(R[case]["persistence"]["test"] - c["y"][2]).mean(1)
    _, stat, p = ns["dm_test"](ea, eb, day)
    g, lo, hi = ns["block_bootstrap_gain"](ea, eb, day)
    row["best_vs_persistence_test"] = {"gain_pct": g, "ci95": [lo, hi], "dm_p": p}
    print(f"  best vs persistence (test): {g:+.1f}% [{lo:+.1f}, {hi:+.1f}] DM p={p:.3f}")
    out["cases"][f"{ds}_h{h}"] = row
for h, (ma, mp, chosen, _) in ns["ARIMA_RES"].items():
    out["arima_top5"][f"h{h}"] = {"arima_mae_mbit": ma * common.MBIT, "persistence_mae_mbit": mp * common.MBIT,
                                  "gain_pct": gain(ma, mp), "orders": [list(o) for o in chosen]}
out_path.write_text(json.dumps(out, indent=1, default=str))
print("saved", out_path)
