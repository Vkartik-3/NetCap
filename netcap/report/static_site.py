"""Export the dashboard as a static site (for GitHub Pages): the same page, reading precomputed JSON files.

  python -m netcap.report.static_site [--out site]

Every file is produced by calling the API in-process (FastAPI TestClient), so the static numbers are exactly
what the live API returns. Scenarios are precomputed for no failure, every link pair and every router, at growth
factors 1, 1.5 and 2 (threshold 0.5); forecasts for the 10 busiest flows (and flow 86) at 5, 30 and 60 minutes
for the main methods (and the GPU-retrained CNN-LSTMs at 60 minutes), first 1,008 test steps (3.5 days). Anything else needs the live API.
"""
import argparse
import json
import shutil
from pathlib import Path

from .. import config

GROWTH = (1, 1.5, 2)
METHODS = ("persistence", "gbm", "gbm_rule", "blend_rule", "blend_gate", "ets", "arima_dhr_sel")
NEURAL_METHODS = ("cnn_lstm_base", "cnn_lstm_seasonal", "cnn_lstm_seasonal_rule")   # panel=abilene_neural, h=12
HORIZONS = (1, 6, 12)
FORECAST_POINTS = 1008


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.ROOT / "site"))
    a = ap.parse_args(argv)
    from fastapi.testclient import TestClient
    from netcap_app.api import app
    c = TestClient(app)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)

    def save(name, obj):
        (out / "data" / name).write_text(json.dumps(obj, separators=(",", ":")))

    def get(url):
        r = c.get(url)
        r.raise_for_status()
        return r.json()

    topo = get("/api/topology")
    save("topology.json", topo)
    for l in topo["links"]:
        save(f"link_{l['link_id']}.json", get(f"/api/link/{l['link_id']}/utilisation?every=12"))
    fs = get("/api/forecast/summary")
    save("forecast_summary.json", fs)
    for name in ("summary", "phase4_anomaly"):
        save(f"results_{name}.json", get(f"/api/results/{name}"))
    res = json.loads((config.RESULTS / "phase2_forecast_abilene.json").read_text())
    flows = sorted(set(res["results"]["original"]["h12"]["top_flows_subset"]["flows"]) | {86})
    n_fc = 0
    for f in flows:
        for h in HORIZONS:
            for m in METHODS:
                r = c.get(f"/api/forecast?flow={f}&horizon={h}&method={m}&limit={FORECAST_POINTS}")
                if r.status_code == 200:
                    save(f"forecast_{f}_{h}_{m}.json", r.json())
                    n_fc += 1
        for m in NEURAL_METHODS:                      # GPU retrain: 60 minutes only
            r = c.get(f"/api/forecast?flow={f}&horizon=12&method={m}&panel=abilene_neural&limit={FORECAST_POINTS}")
            if r.status_code == 200:
                save(f"forecast_{f}_12_{m}.json", r.json())
                n_fc += 1
    pairs = [[l["link_id"], next(m["link_id"] for m in topo["links"] if m["src"] == l["dst"] and m["dst"] == l["src"])]
             for l in topo["links"] if l["src"] < l["dst"]]
    cases = [([], None)] + [(p, None) for p in pairs] + [([], n["name"]) for n in topo["nodes"]]
    n_sc = 0
    for fl, node in cases:
        for g in GROWTH:
            r = c.post("/api/scenario", json={"fail_links": fl, "fail_node": node, "growth_factor": g, "threshold": 0.5})
            r.raise_for_status()
            save(f"scenario_{'-'.join(map(str, fl)) or 'none'}_{node or 'none'}_{g:g}.json", r.json())
            n_sc += 1
    for ds in ("abilene", "cesnet_1h"):
        for det in ("stl", "iforest", "threshold"):
            r = c.get(f"/api/anomalies?dataset={ds}&detector={det}&limit=50")
            if r.status_code == 200:
                save(f"anomalies_{ds}_{det}.json", r.json())
    (out / "data" / "limitations.md").write_text(c.get("/api/limitations").text)
    html = (Path(__file__).resolve().parents[2] / "netcap_app" / "static" / "index.html").read_text()
    html = html.replace("<head>", "<head>\n<script>window.NETCAP_STATIC = " + json.dumps(
        {"flows": flows, "methods": METHODS + NEURAL_METHODS, "growth": GROWTH}) + ";</script>", 1)
    (out / "index.html").write_text(html)
    (out / ".nojekyll").write_text("")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"static site: {out} ({n_fc} forecast files, {n_sc} scenarios, {size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
