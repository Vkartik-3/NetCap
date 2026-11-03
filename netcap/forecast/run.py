"""Phase 2: every forecasting method on identical rolling-origin folds.

  python -m netcap.forecast.run abilene            # 4 folds x horizons 1, 6, 12 (5, 30, 60 min)
  python -m netcap.forecast.run cesnet             # CESNET 1-hour institutions, 4 folds x horizons 1, 6
  python -m netcap.forecast.run abilene --folds original --horizons 12 --no-arima   (quick look)

Saves results/netcap/phase2_forecast_<panel>.json and the test forecasts of the original fold to
data_netcap/forecasts/panel=<p>/h=<h>/<method>.parquet (row, flow, point, lo, hi).
Nothing is selected on test: blend weight, ridge penalty, ETS parameters, rule thresholds and the
"best baseline" are chosen on each fold's validation split.
"""
import argparse
import time

import numpy as np

from .. import config
from ..provenance import peak_rss_bytes, run_record, save_result
from . import evaluate as ev
from . import methods as M
from ..store.parquet import TableWriter
from .panel import FOLDS, fold_bounds, load_abilene, load_cesnet, sample_mask

FLOW86 = 86          # LOSA -> CHIN, the burst flow of the forecasting module (sections 15.2, 18.4, 21)
SIMPLE = ("persistence", "seasonal_day", "seasonal_week", "mean7", "blend")


def run_case(p, fold, h, ets_fc, do_ridge, do_arima, arima_cache, timings):
    t_all = p.targets[h]
    (a0, a1), (b0, b1), (c0, c1) = fold_bounds(len(t_all), fold)
    tr, va, te = t_all[a0:a1], t_all[b0:b1], t_all[c0:c1]
    m = {"val": sample_mask(p, va, h), "test": sample_mask(p, te, h)}
    y = {"val": p.values[va], "test": p.values[te]}
    B = {"val": M.base_inputs(p, va, h), "test": M.base_inputs(p, te, h)}
    P = {k: {s: B[s]["last" if k == "persistence" else k] for s in ("val", "test")}
         for k in ("persistence", "seasonal_day", "seasonal_week", "mean7")}
    info = {"n_samples": {"train": len(tr), "val": len(va), "test": len(te)},
            "periods": {s: [str(p.times[r[0]]), str(p.times[r[-1]])] for s, r in (("train", tr), ("val", va), ("test", te))}}

    # blend: last + w (mean7 - last), w on validation
    sc = [M.mae(y["val"], B["val"]["last"] + w * (B["val"]["mean7"] - B["val"]["last"]), m["val"]) for w in M.GRID_W]
    w = float(M.GRID_W[int(np.argmin(sc))])
    P["blend"] = {s: B[s]["last"] + w * (B[s]["mean7"] - B[s]["last"]) for s in ("val", "test")}
    info["blend_w"] = w

    rows_vt = np.concatenate([va, te])
    rows, fc = ets_fc[h]
    pos = {r: i for i, r in enumerate(rows)}
    sel = np.array([pos[r] for r in rows_vt])
    P["ets"] = {"val": fc[sel[:len(va)]], "test": fc[sel[len(va):]]}

    if do_ridge:
        for key, clip in (("ridge", False), ("ridge_clip", True)):
            t0 = time.perf_counter()
            r = M.ridge(p, h, tr, va, te, clip=clip)
            timings[f"{key}_h{h}_{fold}"] = time.perf_counter() - t0
            P[key] = {"val": r["val"], "test": r["test"]}
            info[f"{key}_alpha"] = r["alpha"]
        t0 = time.perf_counter()
        r = M.ridge_basedef(p, h, tr, va, te)                      # post hoc, labelled (DECISIONS.md 2.2)
        timings[f"ridge_basedef_h{h}_{fold}"] = time.perf_counter() - t0
        P["ridge_basedef"] = {"val": r["val"], "test": r["test"]}
        info["ridge_basedef"] = {"space": r["space"], "alpha": r["alpha"], "label": "added after seeing results"}
    t0 = time.perf_counter()
    g = M.gbm(p, h, tr, va, te)
    timings[f"gbm_h{h}_{fold}"] = time.perf_counter() - t0
    P["gbm"] = {"val": g["val"], "test": g["test"]}
    info["gbm"] = {k: g[k] for k in ("fit_s", "predict_s", "train_rows", "n_iter")}

    # burst attempts: fallback rule on blend and GBM, smooth gate on blend
    z = {"val": M.burst_z(p, va, h, tr), "test": M.burst_z(p, te, h, tr)}
    for base in ("blend", "gbm", "ets"):
        k, scores = M.choose_k(y["val"], B["val"]["last"], P[base]["val"], z["val"], m["val"])
        P[f"{base}_rule"] = {s: np.where(z[s] > k, B[s]["last"], P[base][s]) for s in ("val", "test")}
        info[f"{base}_rule"] = {"k": k, "val_gain_by_k": scores, "share_test_forecasts_switched":
                                float((z["test"] > k)[m["test"]].mean())}
    best = (np.inf, None)
    for tau in M.TAUS:
        for ww in M.GRID_W[::5]:
            g_va = B["val"]["last"] + ww / (1 + (z["val"] / tau) ** 2) * (B["val"]["mean7"] - B["val"]["last"])
            s_ = M.mae(y["val"], g_va, m["val"])
            if s_ < best[0]:
                best = (s_, (tau, float(ww)))
    tau, ww = best[1]
    P["blend_gate"] = {s: B[s]["last"] + ww / (1 + (z[s] / tau) ** 2) * (B[s]["mean7"] - B[s]["last"]) for s in ("val", "test")}
    info["blend_gate"] = {"tau": tau, "w": ww}

    # metrics
    res = {"info": info, "methods": {}}
    pers = P["persistence"]
    for name, pr in P.items():
        r = {s: ev.point_metrics(y[s], pr[s], m[s], p.to_unit) for s in ("val", "test")}
        r["test_gain_vs_persistence_pct"] = M.gain(r["test"]["mae"], ev.point_metrics(y["test"], pers["test"], m["test"], p.to_unit)["mae"])
        r["val_gain_vs_persistence_pct"] = M.gain(r["val"]["mae"], ev.point_metrics(y["val"], pers["val"], m["val"], p.to_unit)["mae"])
        if name != "persistence":
            r["test_vs_persistence"] = ev.compare(y["test"], pr["test"], pers["test"], m["test"], p.to_unit, p.day)
        r["conformal"] = ev.conformal_report(y["val"], pr["val"], m["val"], y["test"], pr["test"], m["test"], p.to_unit)
        if p.name == "abilene":
            mf = np.zeros_like(m["test"])
            mf[:, FLOW86] = m["test"][:, FLOW86]
            r["test_flow86"] = ev.point_metrics(y["test"], pr["test"], mf, p.to_unit)
            r["test_without_flow86"] = ev.point_metrics(y["test"], pr["test"], m["test"] & ~mf, p.to_unit)
        res["methods"][name] = r
    cands = [k for k in P if k != "persistence"]
    chosen = min(cands, key=lambda k: res["methods"][k]["val"]["mae"])
    res["selected_on_validation"] = {"method": chosen, "val_mae": res["methods"][chosen]["val"]["mae"],
                                     "test_mae": res["methods"][chosen]["test"]["mae"],
                                     "test_gain_vs_persistence_pct": res["methods"][chosen]["test_gain_vs_persistence_pct"],
                                     "test_vs_persistence": res["methods"][chosen]["test_vs_persistence"]}
    simple = [k for k in SIMPLE if k != "persistence"]
    sb = min(simple, key=lambda k: res["methods"][k]["val"]["mae"])
    res["best_simple_baseline_on_validation"] = sb
    res["gbm_vs_best_simple"] = ev.compare(y["test"], P["gbm"]["test"], P[sb]["test"], m["test"], p.to_unit, p.day)

    if do_arima:
        top = arima_cache["flows"]
        fa = {s: arima_cache["fc"][h][s] for s in ("val", "test")}
        mt = {s: np.zeros_like(m[s]) for s in m}
        for s in mt:
            mt[s][:, top] = m[s][:, top]
        sub = {}
        for name in ("persistence", "mean7", "blend", "blend_rule", "ets", "gbm", "gbm_rule") + (("ridge",) if do_ridge else ()):
            sub[name] = ev.point_metrics(y["test"], P[name]["test"], mt["test"], p.to_unit)
        pa = {s: P["persistence"][s].copy() for s in ("val", "test")}
        for s in pa:
            pa[s][:, top] = fa[s]
        sub["arima_dhr"] = ev.point_metrics(y["test"], pa["test"], mt["test"], p.to_unit)
        sub["arima_dhr_val"] = ev.point_metrics(y["val"], pa["val"], mt["val"], p.to_unit)
        sub["arima_dhr_vs_persistence"] = ev.compare(y["test"], pa["test"], P["persistence"]["test"], mt["test"], p.to_unit, p.day)
        sub["arima_dhr_conformal"] = ev.conformal_report(y["val"], pa["val"], mt["val"], y["test"], pa["test"], mt["test"], p.to_unit)
        ps = {s: P["persistence"][s].copy() for s in ("val", "test")}
        for s in ps:
            ps[s][:, top] = arima_cache["sel"][h][s]
        sub["arima_dhr_sel"] = ev.point_metrics(y["test"], ps["test"], mt["test"], p.to_unit)
        sub["arima_dhr_sel_val"] = ev.point_metrics(y["val"], ps["val"], mt["val"], p.to_unit)
        sub["arima_dhr_sel_vs_persistence"] = ev.compare(y["test"], ps["test"], P["persistence"]["test"], mt["test"], p.to_unit, p.day)
        sub["arima_dhr_sel_orders"] = arima_cache["chosen"][h]
        sub["arima_dhr_sel_label"] = "added after seeing results (DECISIONS.md 2.5)"
        P["arima_dhr_sel"] = ps
        res["top_flows_subset"] = {"flows": [int(f) for f in top], "names": [p.flow_names[f] for f in top],
                                   "test": sub}
        P["arima_dhr"] = pa
    return res, P, (va, te), m


def save_forecasts(writer, p, h, P, va, te, m):
    """Test forecasts with per-flow split-conformal 90% bounds, one Parquet file per method."""
    import polars as pl
    y_va, y_te = p.values[va], p.values[te]
    for name, pr in P.items():
        q = ev.conformal_margins(y_va, pr["val"], m["val"])
        si, fi = np.nonzero(m["test"])
        point = pr["test"][si, fi]
        df = pl.DataFrame({"row": te[si].astype(np.int64), "step": p.row_to_step[te[si]].astype(np.int64),
                           "flow": p.entity_ids[fi].astype(np.int32), "point": point.astype(np.float32),
                           "lo": np.clip(point - q[fi], 0, None).astype(np.float32),
                           "hi": (point + q[fi]).astype(np.float32), "actual": y_te[si, fi].astype(np.float32)})
        writer.write(df, f"h={h}/{name}.parquet")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", choices=["abilene", "cesnet"])
    ap.add_argument("--folds", nargs="*", default=list(FOLDS))
    ap.add_argument("--horizons", nargs="*", type=int)
    ap.add_argument("--no-arima", action="store_true")
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    hs = a.horizons or ([1, 6, 12] if a.panel == "abilene" else [1, 6])
    p = load_abilene(hs) if a.panel == "abilene" else load_cesnet(hs)
    do_ridge = a.panel == "abilene"
    timings, out, writer = {}, {}, None
    for fold in a.folds:
        out[fold] = {}
        t0 = time.perf_counter()
        bounds = {h: fold_bounds(len(p.targets[h]), fold) for h in hs}
        val_t = {h: p.targets[h][b[1][0]:b[1][1]] for h, b in bounds.items()}
        vt = {h: p.targets[h][b[1][0]:b[2][1]] for h, b in bounds.items()}
        ets_fc, ets_params = M.ets(p, vt, val_t)
        timings[f"ets_{fold}"] = time.perf_counter() - t0
        arima_cache = None
        if not a.no_arima:
            t0 = time.perf_counter()
            h0 = hs[0]
            tr0 = p.targets[h0][:bounds[h0][0][1]]
            mu = np.where(p.known[: tr0[-1] + 1], p.values[: tr0[-1] + 1], 0).sum(0) / np.maximum(p.known[: tr0[-1] + 1].sum(0), 1)
            top = np.argsort(-mu)[:M.ARIMA_TOP]
            fc = {h: {"val": np.zeros((len(val_t[h]), M.ARIMA_TOP)), "test": None} for h in hs}
            tests = {h: p.targets[h][bounds[h][2][0]:bounds[h][2][1]] for h in hs}
            fits = []
            sel = {h: {"val": np.zeros((len(val_t[h]), M.ARIMA_TOP)), "test": np.zeros((len(tests[h]), M.ARIMA_TOP))} for h in hs}
            chosen = {h: [] for h in hs}
            train_end = int(min(p.targets[h][bounds[h][0][1] - 1] for h in hs))
            for j, f in enumerate(top):
                cand = {}
                for order in M.ARIMA_ORDERS_SEL:
                    o, meta, _ = M.arima_flow(p, int(f), train_end, {h: np.concatenate([val_t[h], tests[h]]) for h in hs},
                                              order=order)
                    fits.append({"flow": int(f), "order": list(order), **meta})
                    cand[order] = o
                    if order == M.ARIMA_ORDER:                       # the frozen specification
                        for h in hs:
                            if fc[h]["test"] is None:
                                fc[h]["test"] = np.zeros((len(tests[h]), M.ARIMA_TOP))
                            fc[h]["val"][:, j] = o[h][:len(val_t[h])]
                            fc[h]["test"][:, j] = o[h][len(val_t[h]):]
                for h in hs:                                         # post hoc: order on validation per flow
                    yv = p.values[val_t[h], f]
                    mv = sample_mask(p, val_t[h], h)[:, f]
                    best = min(cand, key=lambda od: np.abs(cand[od][h][:len(val_t[h])] - yv)[mv].mean())
                    sel[h]["val"][:, j] = cand[best][h][:len(val_t[h])]
                    sel[h]["test"][:, j] = cand[best][h][len(val_t[h]):]
                    chosen[h].append({"flow": int(f), "order": list(best)})
            arima_cache = {"flows": top, "fc": fc, "fits": fits, "sel": sel, "chosen": chosen}
            timings[f"arima_{fold}"] = time.perf_counter() - t0
            print(f"  {fold}: ARIMA fitted on {len(top)} flows in {timings[f'arima_{fold}']:.0f} s", flush=True)
        for h in hs:
            t0 = time.perf_counter()
            res, P, (va, te), m = run_case(p, fold, h, ets_fc, do_ridge, arima_cache is not None, arima_cache, timings)
            res["ets_params"] = ets_params[h]
            if arima_cache:
                res["arima_fits"] = arima_cache["fits"]
            out[fold][f"h{h}"] = res
            timings[f"case_h{h}_{fold}"] = time.perf_counter() - t0
            s = res["methods"]
            print(f"{p.name} {fold} {p.horizon_minutes(h)} min: persistence {s['persistence']['test']['mae']:.3f}; " +
                  ", ".join(f"{k} {v['test_gain_vs_persistence_pct']:+.1f}%" for k, v in s.items() if k != "persistence") +
                  f"; selected on val: {res['selected_on_validation']['method']}", flush=True)
            if fold == "original":
                if writer is None:
                    writer = TableWriter(config.DATA / "forecasts" / f"panel={p.name}")
                save_forecasts(writer, p, h, P, va, te, m)
            del P
    files = writer.finish() if writer is not None else None
    rec = run_record({"panel": p.name, "values_shape": list(p.values.shape), "known_cells": int(p.known.sum()),
                      "source": "data_netcap (Phase 1 Parquet)"},
                     panel=p.name, unit=p.unit, step_minutes=p.step_minutes,
                     horizons_minutes={str(h): p.horizon_minutes(h) for h in hs}, folds={f: FOLDS[f] for f in a.folds},
                     frozen_spec={"blend_grid": [0, 1, 101], "ridge_alphas": M.ALPHAS.tolist(), "rule_ks": M.KS,
                                  "rule_tie_pct": M.TIE, "gate_taus": M.TAUS, "ets_alpha": M.ETS_ALPHA,
                                  "ets_gamma": M.ETS_GAMMA, "gbm": M.GBM, "gbm_train_rows": M.GBM_TRAIN_ROWS,
                                  "arima_order": M.ARIMA_ORDER, "arima_fourier": M.ARIMA_FOURIER,
                                  "arima_top_flows": M.ARIMA_TOP, "conformal_level": ev.LEVEL},
                     results=out, timings_s=timings, forecast_files=files, wall_s=time.perf_counter() - t_start,
                     peak_rss_bytes=peak_rss_bytes())
    print(save_result(f"phase2_forecast_{p.name}", rec))


if __name__ == "__main__":
    main()
