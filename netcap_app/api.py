"""NetCap API (FastAPI). Never imports TensorFlow: forecasts are read from Parquet files written by the
forecasting step (statistical models, and neural models exported by a separate training step).

  uvicorn netcap_app.api:app --port 8000          # then open http://localhost:8000/

Data: data_netcap/ (Parquet store), results/netcap/*.json (every number shown comes from these files),
Postgres (optional; /health reports whether it is reachable).
"""
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from netcap import config
from netcap.logs import event, get_logger
from netcap.scenario import engine as E
from netcap.scenario import store
from netcap.scenario.te import min_max_utilisation

STATIC = Path(__file__).parent / "static"
LOG = get_logger("netcap.api")
REQS = Counter("netcap_http_requests_total", "HTTP requests", ["method", "path", "status"])
LAT = Histogram("netcap_http_request_seconds", "HTTP request latency", ["path"],
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
SCEN = Counter("netcap_scenarios_total", "Scenario evaluations", ["kind"])

app = FastAPI(title="NetCap", version="1.0",
              description="Backbone capacity planning on real Abilene data: forecasts, scenarios, TE, anomalies.")


@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        resp = await call_next(request)
        status = resp.status_code
        return resp
    finally:
        dt = time.perf_counter() - t0
        route = request.scope.get("route")
        path = route.path if route is not None else "unmatched"
        REQS.labels(request.method, path, str(status)).inc()
        LAT.labels(path).observe(dt)
        event(LOG, "request", method=request.method, path=request.url.path, route=path, status=status,
              duration_ms=round(dt * 1000, 2))


# --------------------------------------------------------------------------- cached data
@lru_cache(maxsize=1)
def net():
    return store.load_net()


@lru_cache(maxsize=1)
def routing():
    R, unr = E.routing_matrix(net())
    return R, unr


@lru_cache(maxsize=1)
def demand():
    D, steps, ts = store.load_demand()
    return D, steps, ts["ts_assumed"].to_numpy()


def result(name):
    p = config.RESULTS / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, f"{name}.json has not been produced yet")
    return json.loads(p.read_text())


def names():
    return {l[0]: f"{l[1]}->{l[2]}" for l in net().links}


# --------------------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    db = False
    try:
        import psycopg
        with psycopg.connect(config.DATABASE_URL, connect_timeout=2) as c:
            c.execute("SELECT 1")
        db = True
    except Exception:
        pass
    return {"status": "ok", "database": db, "parquet_store": (config.DATA / "abilene" / "links").exists()}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/topology")
def topology():
    n = net()
    R, _ = routing()
    D, _, _ = demand()
    u = E.utilisation(n, E.link_loads(R, D))
    links = []
    for i, (lid, s, d, cap, w, kind) in enumerate(n.links):
        if kind != "core":
            continue
        links.append({"link_id": lid, "src": s, "dst": d, "capacity_mbps": cap, "ospf_weight": w,
                      "max_util": float(np.nanmax(u[:, i])), "p95_util": float(np.nanpercentile(u[:, i], 95))})
    return {"nodes": [{"name": k, "lat": v[0], "lon": v[1], "city": v[2]} for k, v in n.coords.items()],
            "links": links, "demand_steps": int(len(D)), "source": "measured Abilene realOD, kept steps; topology 2003-04-10"}


@app.get("/api/link/{link_id}/utilisation")
def link_series(link_id: int, every: int = Query(12, ge=1, le=288)):
    n = net()
    idx = n.link_index()
    if link_id not in idx or n.links[idx[link_id]][5] != "core":
        raise HTTPException(404, "unknown core link")
    R, _ = routing()
    D, steps, ts = demand()
    u = E.utilisation(n, E.link_loads(R, D[::every]))[:, idx[link_id]]
    return {"link": names()[link_id], "ts_assumed": [str(t) for t in ts[::every]], "utilisation": u.tolist(),
            "note": "timestamps are ASSUMED (file clock); 5-minute averages, one point every `every` steps"}


@app.get("/api/forecast/summary")
def forecast_summary(panel: str = "abilene"):
    r = result(f"phase2_forecast_{panel}")
    rows = []
    for fold, hs in r["results"].items():
        for h, c in hs.items():
            for m, v in c["methods"].items():
                rows.append({"fold": fold, "horizon": h, "method": m, "test_mae": v["test"]["mae"],
                             "test_gain_vs_persistence_pct": v["test_gain_vs_persistence_pct"],
                             "coverage90": v["conformal"]["coverage_two_sided"]})
    return {"unit": r["unit"], "rows": rows, "selected_on_validation":
            {f"{f}/{h}": c["selected_on_validation"]["method"] for f, hs in r["results"].items() for h, c in hs.items()}}


@app.get("/api/forecast")
def forecast(flow: int = Query(86, ge=0), horizon: int = Query(12), method: str = "persistence",
             panel: str = "abilene", limit: int = Query(2016, ge=1, le=20000)):
    f = config.DATA / "forecasts" / f"panel={panel}" / f"h={horizon}" / f"{method}.parquet"
    if not f.exists() or "/" in method or ".." in method:
        raise HTTPException(404, "no forecasts for that panel / horizon / method")
    df = pl.scan_parquet(f).filter(pl.col("flow") == flow).sort("row").head(limit).collect()
    to = 1e-6 * 8 / 300 * 100 if panel == "abilene" else 8 / 3600 / 1e6
    return {"flow": flow, "method": method, "horizon_steps": horizon, "unit": "Mbit/s",
            "interval": "split-conformal 90% (per-flow margin from validation errors)",
            "step": df["step"].to_list(), **{c: (df[c] * to).to_list() for c in ("point", "lo", "hi", "actual")}}


class ScenarioIn(BaseModel):
    growth_factor: float = Field(1.0, gt=0, le=100, description="ASSUMED uniform demand multiplier")
    fail_links: list[int] = Field(default_factory=list, description="core link ids to remove (give both directions for a cut)")
    fail_node: str | None = None
    upgrades: dict[int, float] = Field(default_factory=dict, description="link id -> new capacity in Mbit/s (ASSUMED)")
    threshold: float = Field(0.5, gt=0, le=2)


@app.post("/api/scenario")
def scenario(s: ScenarioIn):
    n = net()
    for lid, cap in s.upgrades.items():
        n = E.set_capacity(n, {lid}, cap)
    if s.fail_links:
        n = E.fail_links(n, set(s.fail_links))
    if s.fail_node:
        if s.fail_node not in n.nodes:
            raise HTTPException(400, "unknown router")
        n = E.fail_node(n, s.fail_node)
    R, unr = E.routing_matrix(n)
    D, _, _ = demand()
    out = E.summarise(n, R, D * s.growth_factor, unr, s.threshold, names())
    out["per_link"] = {names()[k]: v for k, v in out["per_link"].items()}
    kind = "combined" if sum([s.growth_factor != 1, bool(s.fail_links or s.fail_node), bool(s.upgrades)]) > 1 else \
        ("growth" if s.growth_factor != 1 else "failure" if (s.fail_links or s.fail_node) else "upgrade" if s.upgrades else "baseline")
    SCEN.labels(kind).inc()
    out["kind"] = kind
    out["assumptions"] = ("Measured Abilene demand (kept 5-minute steps) on the 2003-04-10 topology; OSPF ECMP rerouting; "
                          "growth factor and upgraded capacities are ASSUMED inputs; utilisation is a 5-minute average.")
    return out


class TEIn(BaseModel):
    step: int | None = Field(None, description="Abilene step; default: the busiest step under this routing")
    fail_links: list[int] = Field(default_factory=list, description="core link ids to remove first")


@app.post("/api/te")
def te(body: TEIn | None = None):
    body = body or TEIn()
    step = body.step
    n = net()
    if body.fail_links:
        n = E.fail_links(n, set(body.fail_links))
    R, unr = E.routing_matrix(n)
    D, steps, _ = demand()
    u = E.utilisation(n, E.link_loads(R, D))[:, n.core_mask()]
    if step is None:
        i = int(np.nanargmax(np.nanmax(u, 1)))                 # the busiest step under this routing
    else:
        i = int(np.searchsorted(steps, step))
        if i >= len(steps) or steps[i] != step:
            raise HTTPException(404, "step not among the kept steps")
    d = D[i].copy()
    d[unr] = 0
    lp = min_max_utilisation(n, d)
    return {"step": int(steps[i]), "ospf_max_util": float(np.nanmax(u[i])), "lp_min_max_util": lp["U"], "status": lp["status"],
            "note": "LP optimum = lower bound on max utilisation for any routing of this matrix on this topology"}


@app.get("/api/anomalies")
def anomalies(dataset: str = "abilene", detector: str = "stl", limit: int = Query(200, ge=1, le=5000)):
    try:
        import psycopg
        with psycopg.connect(config.DATABASE_URL, connect_timeout=2) as c:
            rows = c.execute("SELECT entity_id, target_step, score, injected FROM anomalies WHERE dataset = %s AND "
                             "detector = %s ORDER BY score DESC LIMIT %s", (dataset, detector, limit)).fetchall()
    except Exception as e:
        raise HTTPException(503, f"database unavailable: {type(e).__name__}")
    return {"dataset": dataset, "detector": detector, "label": "evaluation runs contain INJECTED anomalies",
            "rows": [{"entity_id": r[0], "step": r[1], "score": r[2], "injected_run": r[3]} for r in rows]}


@app.get("/api/results/{name}")
def results(name: str):
    if not name.replace("_", "").isalnum():
        raise HTTPException(400, "bad name")
    return result(name)


@app.get("/api/limitations")
def limitations():
    p = config.ROOT / "docs" / "netcap" / "LIMITATIONS.md"
    return PlainTextResponse(p.read_text() if p.exists() else "LIMITATIONS.md not written yet", media_type="text/markdown")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
