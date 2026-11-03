# Phase 5 report: API and dashboard
Built: `netcap_app/api.py` (FastAPI) and `netcap_app/static/index.html` (Plotly, no build step). Endpoints: /health,
/metrics (Prometheus), /api/topology, /api/link/{id}/utilisation, /api/forecast/summary, /api/forecast, POST
/api/scenario, POST /api/te, /api/anomalies, /api/results/{name}, /api/limitations, / (dashboard).
Measured: 5 API tests pass (`tests_netcap/test_api.py`: health, metrics, structured logs, scenario baseline = topology
peak, growth linearity, failures, LP <= OSPF, bad-input rejection, no TensorFlow import); every dashboard endpoint
returns 200 on the real result files. Shows the warnings finding (from `phase3_scenarios.json`) and LIMITATIONS.md.
Open issues: no authentication; data is read from the local Parquet store; not deployed anywhere.
