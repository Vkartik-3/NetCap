"""Forecasting throughput, measured alone on an otherwise idle machine (Phase 2/4 timings ran concurrently
with other jobs and are only upper bounds).

  python -m netcap.bench.forecast_throughput

For each panel (Abilene 144 flows, CESNET 283 institutions), original fold, one horizon: wall time to fit
and forecast the validation + test targets with ETS, the GBM and the simple baselines, and forecasts
(series x targets) per second. Saves results/netcap/phase6_forecast_throughput.json.
"""
import time

import numpy as np

from ..provenance import peak_rss_bytes, run_record, save_result
from ..forecast import methods as M
from ..forecast.panel import fold_bounds, load_abilene, load_cesnet, sample_mask


def bench(p, h):
    t_all = p.targets[h]
    (a0, a1), (b0, b1), (c0, c1) = fold_bounds(len(t_all), "original")
    tr, va, te = t_all[a0:a1], t_all[b0:b1], t_all[c0:c1]
    vt = np.concatenate([va, te])
    n_fc = int(sample_mask(p, vt, h).sum())
    out = {"series": int(p.values.shape[1]), "train_targets": len(tr), "forecast_cells": n_fc}
    t0 = time.perf_counter()
    b = M.base_inputs(p, vt, h)
    _ = b["last"] + 0.3 * (b["mean7"] - b["last"])
    out["simple_baselines_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    M.ets(p, {h: vt}, {h: va})
    out["ets_select_and_forecast_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = M.gbm(p, h, tr, va, te)
    out["gbm_total_s"] = time.perf_counter() - t0
    out["gbm_fit_s"], out["gbm_predict_s"] = g["fit_s"], g["predict_s"]
    for k in ("simple_baselines_s", "ets_select_and_forecast_s", "gbm_total_s", "gbm_predict_s"):
        out[k.replace("_s", "_forecasts_per_s")] = n_fc / out[k]
    return out


def main():
    import os
    load = os.getloadavg()
    res = {"abilene_h12": bench(load_abilene((12,)), 12), "cesnet_h1": bench(load_cesnet((1,)), 1)}
    print(save_result("phase6_idle_forecast_throughput", run_record(
        {"abilene": "raw realOD", "cesnet": "data_netcap cesnet 1_hour"}, results=res, load_average_at_start=load,
        cpu_count=os.cpu_count(), peak_rss_bytes=peak_rss_bytes())))
    print(res)


if __name__ == "__main__":
    main()
