"""Limited neural retrain using NetCap's forecasting implementation (the only code that uses TensorFlow).

Run in the separate TensorFlow environment, from the repository root:
  .venv-tf/bin/python -m netcap.neural.retrain --time-one-epoch     # timing gate (DECISIONS.md)
  .venv-tf/bin/python -m netcap.neural.retrain                      # 6 models, only if the gate passed

Abilene, 60-minute horizon (12 steps), original fold, seeds 42-44: the base CNN-LSTM and the seasonal
CNN-LSTM (+ the fallback rule, threshold chosen on validation exactly as scripts/08_peaks.py). Uses
common.py's functions and scripts/07_features.py's settings. Exports the 3-seed mean forecasts to
data_netcap/forecasts/panel=abilene_neural/h=12/cnn_lstm_*.parquet (the API reads these; it never imports TF).
Saves results/netcap/phase_neural_timing.json and results/netcap/phase_neural_retrain.json.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402

import common  # noqa: E402
from netcap.provenance import run_record, save_result  # noqa: E402

H, SEEDS = 12, (42, 43, 44)
EPOCHS, PATIENCE, LR = 200, 15, 1e-3            # scripts/07_features.py
KS, TIE = (3, 4, 5, 6, 8), 0.05                 # scripts/08_peaks.py
BUDGET_S = 3 * 3600
PREFIX = "netcap_retrain"
REFERENCE_RESULTS = {"base": 1.1, "seasonal": 2.9, "rule": 10.8, "source": "docs/NETCAP_RESEARCH_LOG.md section 21.3 (GPU, 3-seed mean)"}


def data():
    d = common.data_dir("abilene", H)
    if not (d / "target_steps.npy").exists():
        common.prepare_abilene(horizon=H, directory=d)
    base = common.context_dataset("abilene", H)
    seas = common.context_dataset("abilene", H, seasonal="seasonal_mean7")
    return base, seas


def builder(seasonal):
    return lambda: common.build_cnn_lstm(window=common.WINDOW, n_flows=common.N_FLOWS, residual=True, seasonal=seasonal)


def time_one_epoch():
    import tensorflow as tf
    base, _ = data()
    loss = common.make_loss("mae_raw", base["scaler"], float(base["y_raw"][0].mean()))
    common.set_seeds(42)
    model = builder(False)()
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=loss)
    x, y = common.model_inputs(base, 0), base["y"][0]
    xv, yv = common.model_inputs(base, 1), base["y"][1]
    t0 = time.perf_counter()
    model.fit(x, y, validation_data=(xv, yv), epochs=1, batch_size=128, shuffle=True, verbose=0)
    per_epoch = time.perf_counter() - t0
    proj = per_epoch * EPOCHS * 2 * len(SEEDS)
    rec = {"seconds_one_epoch_incl_compile": per_epoch, "train_samples": int(len(y)), "max_epochs": EPOCHS,
           "models": 2 * len(SEEDS), "projected_upper_bound_s": proj, "budget_s": BUDGET_S,
           "gate_passed": proj <= BUDGET_S, "device": [d.device_type for d in tf.config.list_physical_devices()],
           "tensorflow": tf.__version__}
    print(save_result("phase_neural_timing", run_record({"abilene": "processed/h12 (common.prepare_abilene)"}, **rec)))
    print(rec)
    return rec


def run():
    import tensorflow as tf
    base, seas = data()
    sc = base["scaler"]
    loss = common.make_loss("mae_raw", sc, float(base["y_raw"][0].mean()))
    out, preds, fits = {}, {}, {}
    for key, d, seasonal in (("base", base, False), ("seasonal", seas, True)):
        preds[key] = {1: [], 2: []}
        for seed in SEEDS:
            name = f"{PREFIX}_abilene_h{H}_{key}_s{seed}"
            t0 = time.perf_counter()
            model, hist, _ = common.fit_or_load(builder(seasonal), name, common.model_inputs(d, 0), d["y"][0],
                                                common.model_inputs(d, 1), d["y"][1], retrain=False, seed=seed,
                                                epochs=EPOCHS, patience=PATIENCE, lr=LR, loss=loss)
            fits[name] = {"epochs": len(hist["loss"]), "best_val_loss": float(min(hist["val_loss"])),
                          "wall_s": time.perf_counter() - t0}
            for i in (1, 2):
                preds[key][i].append(sc.inverse_transform(common.predict(model, common.model_inputs(d, i), scaler=sc))
                                     .astype(np.float64))
            tf.keras.backend.clear_session()
    y = {i: base["y_raw"][i] for i in (1, 2)}
    last = {i: base["last_raw"][i] for i in (1, 2)}
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    gain = lambda m, b: (1 - m / b) * 100
    pers = {i: mae(y[i], last[i]) for i in (1, 2)}
    sd = seas["S"][0][:, 0].std(0)
    z = {i: np.abs(seas["S"][i][:, 0]) / np.where(sd > 0, sd, 1) for i in (1, 2)}
    mean = {k: {i: np.mean(preds[k][i], 0) for i in (1, 2)} for k in preds}
    sc_k = {k: gain(mae(y[1], np.where(z[1] > k, last[1], mean["seasonal"][1])), pers[1]) for k in KS}
    k_rule = min(k for k in KS if sc_k[k] >= max(sc_k.values()) - TIE)
    rule = {i: np.where(z[i] > k_rule, last[i], mean["seasonal"][i]) for i in (1, 2)}
    per_seed = {k: [gain(mae(y[2], p), pers[2]) for p in preds[k][2]] for k in preds}
    per_seed["rule"] = [gain(mae(y[2], np.where(z[2] > k_rule, last[2], p)), pers[2]) for p in preds["seasonal"][2]]
    res = {}
    for k, fc in (("base", mean["base"]), ("seasonal", mean["seasonal"]), ("rule", rule)):
        g = gain(mae(y[2], fc[2]), pers[2])
        s = per_seed[k]
        lo, hi = np.mean(s) - 2 * np.std(s, ddof=1), np.mean(s) + 2 * np.std(s, ddof=1)
        res[k] = {"test_gain_3seed_mean_forecast_pct": g, "val_gain_pct": gain(mae(y[1], fc[1]), pers[1]),
                  "test_gain_per_seed_pct": s, "seed_band_mean_pm_2sd": [lo, hi], "reference_pct": REFERENCE_RESULTS[k],
                  "reproduced": bool(lo <= REFERENCE_RESULTS[k] <= hi)}
    out = {"results": res, "rule_k": k_rule, "rule_val_gain_by_k": sc_k, "rule_share_test": float((z[2] > k_rule).mean()),
           "persistence_test_mae_raw": pers[2], "fits": fits, "reference_results": REFERENCE_RESULTS,
           "devices": [d.device_type for d in tf.config.list_physical_devices()],
           "gpu_names": [tf.config.experimental.get_device_details(d).get("device_name", "?")
                         for d in tf.config.list_physical_devices("GPU")],
           "tensorflow": tf.__version__,
           "note": ("GPU retrain" if tf.config.list_physical_devices("GPU") else "CPU retrain")
                   + "; the reference numbers use GPU models (section 20.1: GPU and CPU differ slightly)"}
    print(save_result("phase_neural_retrain", run_record({"abilene": "processed/h12 (common.prepare_abilene)"}, **out)))
    export(base, mean, rule)          # after saving: an export failure must not lose the metrics
    print({k: (round(v["test_gain_3seed_mean_forecast_pct"], 2), v["reproduced"]) for k, v in res.items()})


def export(base, mean, rule):
    """3-seed mean test forecasts to the same Parquet layout as Phase 2 (row = real-time row, step = Abilene step)."""
    import polars as pl
    from netcap.forecast import evaluate as ev
    from netcap.forecast.panel import load_abilene
    from netcap.store.parquet import TableWriter
    p = load_abilene((H,))
    t_all = p.targets[H]
    n = len(t_all)
    a = int(n * 0.6)
    b = a + int(n * 0.2)
    va, te = t_all[a:b], t_all[b:]
    # atol: true zeros come back from common's scaler round trip as ~5.5e-7 raw units (16 cells, flow 45, GPU run 2026-11-03)
    if len(te) != len(base["y_raw"][2]) or not np.allclose(p.values[te], base["y_raw"][2], rtol=1e-5, atol=1e-6):
        raise RuntimeError("test targets do not line up with netcap.forecast.panel")
    from netcap import config
    w = TableWriter(config.DATA / "forecasts" / "panel=abilene_neural")     # own directory: Phase 2 files untouched
    m = {"val": np.ones((len(va), 144), bool), "test": np.ones((len(te), 144), bool)}
    for name, fc in (("cnn_lstm_base", mean["base"]), ("cnn_lstm_seasonal", mean["seasonal"]), ("cnn_lstm_seasonal_rule", rule)):
        q = ev.conformal_margins(p.values[va], fc[1], m["val"])
        si, fi = np.nonzero(m["test"])
        point = fc[2][si, fi]
        w.write(pl.DataFrame({"row": te[si].astype(np.int64), "step": p.row_to_step[te[si]].astype(np.int64),
                              "flow": fi.astype(np.int32), "point": point.astype(np.float32),
                              "lo": np.clip(point - q[fi], 0, None).astype(np.float32), "hi": (point + q[fi]).astype(np.float32),
                              "actual": p.values[te][si, fi].astype(np.float32)}), f"h={H}/{name}.parquet")
    w.finish()


GPU_COMMAND = ("python -m venv .venv-tf && .venv-tf/bin/pip install -r research/requirements-gpu.txt polars==2.1.0 pyarrow==26.0.0 "
               "&& .venv-tf/bin/python -m netcap.neural.retrain --time-one-epoch && .venv-tf/bin/python -m netcap.neural.retrain")


def record_not_run():
    """User decision (2026-11-02): skip the retrain and the timing gate on this machine."""
    rec = {"status": "not run: needs GPU (no NVIDIA GPU on this machine)",
           "decision": "user chose to skip the CPU retrain and its timing gate to save time",
           "planned": {"dataset": "abilene", "horizon_steps": H, "fold": "original", "seeds": SEEDS,
                       "models": ["cnn_lstm_base", "cnn_lstm_seasonal (+ fallback rule)"],
                       "settings": {"epochs": EPOCHS, "patience": PATIENCE, "lr": LR}},
           "gpu_command": GPU_COMMAND, "claims": "none: the recorded neural reference results are not claimed as rerun"}
    print(save_result("phase_neural_retrain", run_record("n/a (not run)", **rec)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-one-epoch", action="store_true")
    ap.add_argument("--record-not-run", action="store_true")
    a = ap.parse_args()
    record_not_run() if a.record_not_run else time_one_epoch() if a.time_one_epoch else run()
