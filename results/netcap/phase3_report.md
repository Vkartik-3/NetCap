# Phase 3 report: scenario engine and traffic engineering

Source: `phase3_scenarios.json`, `phase3_routing_gate.json` (`phase3_scenarios_run1_single_upgrade_rule.json`
is the first run, before the second upgrade rule; DECISIONS 3.1). Demand: the measured Abilene matrix at every
kept 5-minute step (40,855 steps × 144 flows), on the 2003-04-10 topology. ASSUMED inputs are labelled.

## What was built
`netcap/scenario/`: NetworkX topology, OSPF shortest paths with even ECMP splitting per router, link loads and
utilisation, failures (link pair or router), capacity upgrades, growth, the min-max-utilisation LP (HiGHS),
toy-topology tests (ECMP split, weights, reroute, partition, node failure, growth, upgrade, LP optimum).

## Correctness gate
Computed routing equals matrix A for **144 / 144 flows** (max difference 0, no ECMP ties, 0 unroutable).

## Baseline
Busiest core link: KSCY→IPLS, peak **75.4%** (5-minute average); the worst link's p95 is 9.3%. 572 steps have some
link at or above 50%. Peaks come from short bursts: uniform growth of 5.38× would bring the worst p95 to 50%,
but the peak already exceeds 50%. Months to a 50% p95 under ASSUMED growth: 111 (20%/yr), 60 (40%/yr), 43 (60%/yr).

## N-1 (15 link pairs, 12 routers)
| Worst cases | Peak utilisation | Overloaded link |
|---|---|---|
| link IPLS–KSCY fails | **297%** | ATLA→IPLS (2.48 Gbit/s) |
| router KSCY fails | 296% | ATLA→IPLS |
| link CHIN–IPLS fails | 77% | WASH→NYCM |
Only the IPLS–KSCY link and the KSCY router push a link above 80%. Every router failure strands its own traffic
(e.g. KSCY: 23 flows, up to 18.9% of traffic); ATLA-M5 is single-homed to ATLAng.

## Upgrades (capacities ASSUMED)
- Rule fixed in advance (KSCY–IPLS, the link worst in the most N-1 cases, 9.92 → 19.84 Gbit/s): **no effect** on the
  worst case (still 297%). A real negative result.
- Rule added after seeing that (ATLA–IPLS 2.48 → 9.92 Gbit/s): worst N-1 case drops to **77%** (CHIN–IPLS failure);
  no case above 80%.
- Combined: 2× ASSUMED growth + IPLS–KSCY failure: 595% before, 152% after the ATLA–IPLS upgrade.

## Traffic engineering (LP lower bound vs OSPF)
| | LP / OSPF max utilisation | 95% interval |
|---|---|---|
| hourly, normal (3,405 LPs) | mean 0.704, median 0.710 | mean [0.696, 0.712], median [0.702, 0.717] |
| hourly, IPLS–KSCY failed | mean 0.493, median 0.478 | mean [0.483, 0.504], median [0.469, 0.490] |
| 20 busiest steps, normal | 0.509-0.513 | – |
| 20 busiest steps, failed | 0.256-0.266 | – |
Day-block bootstrap, 2,000 resamples. Any routing could at best cut OSPF's peak utilisation by about 30% on a typical
hour and about half on the busiest steps; under the worst failure, by about half to three quarters.
3,405 LPs solve in 10.5 s (load average 4.9 at start, 3.9 when saved: borderline, so treat as an upper bound; not a headline claim).

## Forecast-driven check: do better forecasts give better warnings? (60 min, original test period)
| Method | Link-load MAE (pp) | Warning F1 (link ≥ 50%) | Recall |
|---|---|---|---|
| gbm | **0.426** | 0.890 | 0.846 |
| persistence | 0.494 | 0.896 | 0.896 |
| blend + smooth gate | 0.519 | **0.906** | 0.893 |
| blend + rule | 0.532 | 0.903 | 0.866 |
| arima (order on validation) | 0.502 | 0.898 | 0.896 |
| ets | 0.570 | 0.781 | 0.718 |
1,091 actual link-steps at or above 50%. The most accurate forecaster of link load (GBM, 14% lower error than
persistence) has a slightly lower warning F1 than persistence, and the good methods span only 0.890-0.906; no
interval was computed for F1, so none of these F1 differences is shown to be significant. Accuracy on average
does not translate into better congestion warnings here: congestion is a few long bursts on one flow, which
persistence already tracks (consistent with the forecasting module's section 22, measured independently here).

## Open issues
- Uniform growth and single failures only; no SRLGs; LP is a bound, not a deployable routing.
