# NetCap limitations

What NetCap's results do and do not show. Numbers are in docs/netcap/RESULTS.md and results/netcap/*.json.

## Data
- **One link-level network, and it is old.** Scenarios and TE use Abilene: 12 routers, 15 bidirectional core
  links, traffic from March to September 2004 on the topology and OSPF weights of 10 April 2003. The routing
  matrix computed from those weights equals the one shipped with the data for all 144 flows, but a 2004 network
  with 2003 weights is still an assumption.
- **Timestamps are assumed.** The Abilene files give only a start date per week; times of day are inferred
  (file clock, 5-minute steps). Daily seasonality is real; the absolute clock and time zone are not known.
- **5-minute averages.** Utilisation is a 5-minute average. Bursts shorter than 5 minutes, which is what
  actually fills buffers, are invisible; real planning would add a peak-to-average factor.
- **Abilene utilisation is low.** The busiest core link stays well below common planning thresholds, so most
  "time to threshold" numbers depend on an ASSUMED growth rate, not on measured growth.
- **CESNET is per institution, not per link.** It is used for forecasting across 283 series, throughput and anomaly
  detection, never for routing scenarios.
- **MAWI is one 15-minute trace.** It shows the flow pipeline end to end at 207M packets; aggregates only.
  Throughput per 10 s spreads each flow's bytes evenly over its lifetime (an approximation).
- **GÉANT was not used** (optional per the user; the download link has moved, see NOTES.md).

## Forecasting
- Folds overlap in time (the forecasting module's rolling-origin design), so the four test periods are not independent samples.
- The SARIMA-type model is ARMA(2,1) errors with Fourier seasonality (a seasonal ARIMA with a 288-step period is not
  practical), fitted on the 10 busiest flows only.
- Burst handling (the fallback rule) was designed by the forecasting module after looking at its test period; NetCap applies
  the same rule with thresholds chosen on validation only, but the idea itself is not test-blind.
- One added baseline (`ridge_basedef`) was added after the first results; it is labelled as such.

## Scenarios and TE
- Failures are single (N-1): one bidirectional link or one router. No shared-risk groups, no N-2.
- Growth is uniform across all flows; real growth is uneven.
- The TE LP gives the lowest possible maximum utilisation for a traffic matrix (a bound); it is not a deployable
  routing (it can split flows arbitrarily) and ignores latency.
- Access links (router to outside) have no published capacity and are excluded from utilisation.

## Anomaly detection
- **Injected anomalies are synthetic** (spikes, level shifts, dropouts written into copies of real series); detection
  rates on them say how well a detector finds those shapes, not how well it finds real incidents.
- **Weak labels are data faults** (cleaning rules R2-R4), not traffic anomalies. Precision against them is a lower
  bound because detectors also flag real anomalies the rules do not cover.
- The STL detector is retrospective (it decomposes the whole series); it is not an online alarm.

## Forecasts and warnings
- **Better forecasts did not give better congestion warnings** in the forecasting module (its section 22). NetCap checks the
  same question on its own forecasts in Phase 3 (`phase3_scenarios.json`, `forecast_driven`).

## Engineering
- Docker Compose and Kubernetes were run on one laptop (kind, single node); there is no hosted or multi-node deployment.
- Neural models: see RESULTS.md for whether the limited CPU retrain ran and whether it reproduced the forecasting module.
