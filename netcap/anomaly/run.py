"""Phase 4: anomaly detection on Abilene (injected + weak labels) and CESNET (injected labels).

  python -m netcap.anomaly.run

Saves results/netcap/phase4_anomaly.json and the flagged test cells in the anomalies table
(injected = true for the runs on series with injected anomalies).
"""
import time

import numpy as np

from .. import config
from ..data import abilene
from ..provenance import peak_rss_bytes, run_record, save_result
from . import detect as D

DETECTORS = ("threshold", "stl", "iforest")


def scores(x, known, tr, day, timings, tag):
    out = {}
    for name in DETECTORS:
        t0 = time.perf_counter()
        if name == "threshold":
            out[name] = D.score_threshold(x, known, tr)
        elif name == "stl":
            out[name] = D.score_stl(x, known, tr, day)
        else:
            out[name] = D.score_iforest(x, known, tr, day)
        timings[f"{tag}_{name}"] = time.perf_counter() - t0
        print(f"  {tag} {name}: {timings[f'{tag}_{name}']:.0f} s", flush=True)
    return out


def injected_case(name, values, known, day, timings, rng):
    n = len(values)
    P = D.periods(n)
    tr = slice(*P["train"])
    med = np.array([np.median(values[tr][known[tr, f], f]) if known[tr, f].any() else 0 for f in range(values.shape[1])])
    active = med > 0
    x, lab, events = D.inject(values, known, active, day, rng)
    sc = scores(x, known, tr, day, timings, f"{name}_injected")
    rv, rt = slice(*P["val"]), slice(*P["test"])
    res = {"series": int(values.shape[1]), "active_series": int(active.sum()), "rows": n,
           "events_by_period": {p: int(sum(e["period"] == p for e in events)) for p in P},
           "events_by_kind_test": {k: int(sum(e["period"] == "test" and e["kind"] == k for e in events))
                                   for k in ("spike", "shift_up", "shift_down", "dropout")},
           "label": "INJECTED synthetic anomalies in a copy of the real series", "detectors": {}}
    for d in DETECTORS:
        res["detectors"][d] = D.evaluate(sc[d], lab, known & active[None, :], rv, rt, events)
    return res, sc, lab, x


def weak_case(timings):
    """Raw (uncleaned) Abilene on the week-step axis; labels from rules R2-R4."""
    weeks = abilene.load_weeks()
    cl = abilene.clean(weeks)
    x = weeks.reshape(-1, abilene.N_FLOWS).astype(np.float64)
    n = len(x)
    known = np.ones(x.shape, bool)
    lab = np.zeros(x.shape, bool)
    kinds = np.full(x.shape, "", dtype="<U10")
    for f, a, b in cl.flow_outages:
        lab[a:b, f] = True
        kinds[a:b, f] = "R2 flow"
    nn = abilene.N_NODES
    for node, direction, a, b in cl.router_outages:
        r = abilene.NODES.index(node)
        fl = [r * nn + j for j in range(nn)] if direction == "out of" else [i * nn + r for i in range(nn)]
        lab[a:b, fl] = True
        kinds[a:b, fl] = np.where(kinds[a:b, fl] == "", "R3 router", kinds[a:b, fl])
    for s in cl.impossible_steps:
        c = x[s] > abilene.MAX_FLOW_RAW
        lab[s, c] = True
        kinds[s, c] = "R4 impossible"
    P = D.periods(n)
    tr = slice(*P["train"])
    sc = scores(x, known, tr, abilene.STEPS_PER_DAY, timings, "abilene_weak")
    res = {"axis": "week file x 2016 + step (raw, uncleaned; 24 weeks)", "rows": n,
           "label": "WEAK ground truth: cells removed by cleaning rules R2-R4 (R1 duplicate days excluded)",
           "labelled_cells": {k: int((kinds == k).sum()) for k in ("R2 flow", "R3 router", "R4 impossible")},
           "labelled_cells_by_period": {p: int(lab[slice(*b)].sum()) for p, b in P.items()}, "detectors": {}}
    from sklearn.metrics import average_precision_score
    for d in DETECTORS:
        s = sc[d]
        r = {"average_precision_all_rows": float(average_precision_score(lab.ravel(), s.ravel())),
             "base_rate": float(lab.mean())}
        # recall of each rule's cells when flagging the top 1% of cells (a fixed budget, set in advance)
        thr = np.quantile(s, 0.99)
        flag = s >= thr
        r["top1pct_threshold"] = float(thr)
        r["recall_top1pct"] = {k: float(flag[kinds == k].mean()) if (kinds == k).any() else None
                               for k in ("R2 flow", "R3 router", "R4 impossible")}
        r["precision_top1pct_vs_weak"] = float(lab[flag].mean())
        res["detectors"][d] = r
    return res


def store(name, sc, lab, known, P, injected):
    """Flagged test cells (score above the validation threshold) into the anomalies table."""
    import polars as pl
    from ..db import load as db
    frames = []
    rt = slice(*P["test"])
    for d, s in sc.items():
        thr = D.evaluate(s, lab, known, slice(*P["val"]), rt)["threshold_from_val"]
        ti, fi = np.nonzero((s[rt] >= thr) & known[rt])
        frames.append(pl.DataFrame({"dataset": name, "detector": d, "entity_kind": "flow" if name == "abilene" else "institution",
                                    "entity_id": fi.astype(np.int32), "target_step": (ti + rt.start).astype(np.int64),
                                    "score": s[rt][ti, fi].astype(np.float64), "is_anomaly": True,
                                    "injected": injected, "run_id": "phase4"}))
    cols = ["dataset", "detector", "entity_kind", "entity_id", "target_step", "score", "is_anomaly", "injected", "run_id"]
    with db.connect() as conn:
        db.apply_schema(conn)
        return db.upsert(conn, "anomalies", frames, cols)


def main():
    t0 = time.perf_counter()
    timings, out = {}, {}
    rng = np.random.default_rng(config.SEED)
    from ..forecast.panel import load_abilene, load_cesnet
    pa = load_abilene(horizons=())
    res, sc, lab, _ = injected_case("abilene", pa.values, pa.known, pa.day, timings, rng)
    out["abilene_injected"] = res
    out["abilene_injected"]["axis"] = "real time, cleaned steps only (removed steps unknown, never scored)"
    P = D.periods(len(pa.values))
    out["abilene_injected"]["stored"] = store("abilene", sc, lab, pa.known, P, True)
    del sc
    out["abilene_weak"] = weak_case(timings)
    pc = load_cesnet(horizons=())
    res, sc, lab, _ = injected_case("cesnet", pc.values, pc.known, pc.day, timings, rng)
    out["cesnet_injected"] = res
    out["cesnet_injected"]["axis"] = "1-hour institution series (n_bytes); missing institution-hours never scored"
    out["cesnet_injected"]["stored"] = store("cesnet_1h", sc, lab, pc.known, D.periods(len(pc.values)), True)
    rec = run_record({"abilene": "raw X01..X24 realOD", "cesnet": "data_netcap cesnet 1_hour n_bytes"},
                     spec={"split": D.SPLIT, "inject": D.INJECT, "iforest": D.IFOREST,
                           "iforest_train_cells": D.IFOREST_TRAIN_CELLS, "detectors": DETECTORS},
                     results=out, timings_s=timings, wall_s=time.perf_counter() - t0, peak_rss_bytes=peak_rss_bytes())
    print(save_result("phase4_anomaly", rec))
    for k, v in out.items():
        for d, r in v["detectors"].items():
            print(k, d, {kk: r[kk] for kk in r if kk in ("test_average_precision", "average_precision_all_rows")},
                  r.get("test", {}).get("f1"), r.get("recall_top1pct"))


if __name__ == "__main__":
    main()
