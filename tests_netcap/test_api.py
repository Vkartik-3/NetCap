import pytest

from netcap import config

pytestmark = pytest.mark.skipif(not (config.DATA / "abilene" / "links").exists(), reason="Parquet store not built")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from netcap_app.api import app
    return TestClient(app)


def test_health_metrics_and_structured_logs(client, caplog):
    import json
    import logging
    from netcap.logs import JsonFormatter
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["parquet_store"]
    logging.getLogger("netcap.api").propagate = True              # let caplog see the records
    with caplog.at_level(logging.INFO, logger="netcap.api"):
        client.get("/api/topology")
    logging.getLogger("netcap.api").propagate = False
    rec = json.loads(JsonFormatter().format(caplog.records[-1]))
    assert rec["route"] == "/api/topology" and rec["status"] == 200 and rec["duration_ms"] >= 0
    m = client.get("/metrics").text
    assert 'netcap_http_requests_total{method="GET",path="/api/topology",status="200"}' in m


def test_topology_matches_scenario_baseline(client):
    t = client.get("/api/topology").json()
    assert len(t["nodes"]) == 12 and len(t["links"]) == 30 and t["demand_steps"] == 40855
    base = client.post("/api/scenario", json={}).json()
    assert base["kind"] == "baseline" and base["unroutable_flows"] == 0
    assert base["max_util"] == pytest.approx(max(l["max_util"] for l in t["links"]))


def test_failure_growth_and_te(client):
    pair = [l["link_id"] for l in client.get("/api/topology").json()["links"] if {l["src"], l["dst"]} == {"IPLSng", "CHINng"}]
    base = client.post("/api/scenario", json={}).json()["max_util"]
    grown = client.post("/api/scenario", json={"growth_factor": 2.0}).json()
    assert grown["kind"] == "growth" and grown["max_util"] == pytest.approx(2 * base)
    f = client.post("/api/scenario", json={"fail_links": pair}).json()
    assert f["kind"] == "failure" and f["unroutable_flows"] == 0
    n = client.post("/api/scenario", json={"fail_node": "ATLAng"}).json()
    assert n["unroutable_flows"] > 0                               # ATLA-M5 hangs off ATLAng only
    te = client.post("/api/te").json()
    assert client.post("/api/te", json={}).json()["status"] == "optimal"
    tf = client.post("/api/te", json={"fail_links": pair}).json()
    assert tf["status"] == "optimal" and tf["lp_min_max_util"] <= tf["ospf_max_util"] + 1e-9
    assert te["status"] == "optimal" and te["lp_min_max_util"] <= te["ospf_max_util"] + 1e-9
    assert client.post("/api/scenario", json={"fail_node": "NOPE"}).status_code == 400


def test_bad_inputs_are_rejected(client):
    assert client.get("/api/forecast?method=../../etc").status_code == 404
    assert client.get("/api/results/..%2Fsecret").status_code in (400, 404)
    assert client.post("/api/scenario", json={"growth_factor": -1}).status_code == 422
    assert client.get("/api/link/999/utilisation").status_code == 404


def test_api_never_imports_tensorflow():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", "import sys, netcap_app.api; print('tensorflow' in sys.modules)"],
                         capture_output=True, text=True, cwd=config.ROOT)
    assert out.stdout.strip() == "False", out.stderr
