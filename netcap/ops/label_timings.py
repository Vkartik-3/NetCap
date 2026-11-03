"""Add a timing label to result files whose timings were measured while the CPU was shared.

  python -m netcap.ops.label_timings phase2_forecast_abilene phase2_forecast_cesnet_1h phase4_anomaly --evidence "..."

Only the key `timing_label` (and `timing_label_evidence`) is added; no other field is changed. Used for runs that
started before load averages were recorded automatically (DECISIONS.md, Phase 6/7 notes).
"""
import argparse
import json

from .. import config

LABEL = "upper bound, measured under CPU contention"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--evidence", required=True)
    a = ap.parse_args(argv)
    for n in a.names:
        p = config.RESULTS / f"{n}.json"
        d = json.loads(p.read_text())
        d["timing_label"] = LABEL
        d["timing_label_evidence"] = a.evidence
        p.write_text(json.dumps(d, indent=1, default=str))
        print("labelled", p.name)


if __name__ == "__main__":
    main()
