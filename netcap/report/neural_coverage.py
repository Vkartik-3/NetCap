"""Interval coverage of the GPU-retrained CNN-LSTM forecasts (no TensorFlow needed).

  python -m netcap.report.neural_coverage

Reads data_netcap/forecasts/panel=abilene_neural/h=12/*.parquet (3-seed mean test forecasts with split-conformal 90%
intervals, exported by netcap.neural.retrain) and records, per method: two-sided and upper coverage, mean width and
cells, next to the GBM's coverage on the same test period from phase2_forecast_abilene.json.
Saves results/netcap/phase_neural_coverage.json.
"""
import json

import polars as pl

from .. import config
from ..data import abilene
from ..provenance import run_record, save_result

ROOT = config.DATA / "forecasts" / "panel=abilene_neural" / "h=12"


def main():
    if not ROOT.exists():
        raise SystemExit(f"no neural forecasts in {ROOT} (run netcap.neural.retrain on a GPU first)")
    out = {}
    for f in sorted(ROOT.glob("*.parquet")):
        df = pl.read_parquet(f)
        a, lo, hi = df["actual"], df["lo"], df["hi"]
        out[f.stem] = {"coverage_two_sided": float(((a >= lo) & (a <= hi)).mean()),
                       "coverage_upper": float((a <= hi).mean()),
                       "mean_width_mbps": float(((hi - lo) * abilene.TO_MBIT_S).mean()),
                       "cells": df.height}
    p2 = json.loads((config.RESULTS / "phase2_forecast_abilene.json").read_text())["results"]["original"]["h12"]["methods"]
    ref = {m: p2[m]["conformal"]["coverage_two_sided"] for m in ("persistence", "gbm")}
    print(save_result("phase_neural_coverage", run_record(
        {"forecasts": "data_netcap/forecasts/panel=abilene_neural/h=12 (from phase_neural_retrain.json)"},
        level=0.9, horizon_minutes=60, fold="original", methods=out,
        reference_coverage_two_sided_phase2=ref,
        note="split-conformal 90% intervals, per-flow margins from validation errors of the 3-seed mean forecast")))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
