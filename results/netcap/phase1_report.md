# Phase 1 report: data layer (with the Phase 1 follow-ups)

Every number below is copied from `results/netcap/phase1_summary.json` (which copies the per-run files
named in it). Machine: Apple M-series laptop, 16 GB RAM, macOS; local PostgreSQL 17.4 on port 5433.

## What was built
- Parquet store (idempotent writer: temp file, hash compare, atomic replace; manifest; stale-file removal)
  and Postgres schema with idempotent upserts (COPY to staging, `INSERT … ON CONFLICT … WHERE IS DISTINCT FROM`).
- Ingestion for Abilene (24 weeks), CESNET-TimeSeries24 institutions (3 aggregations) and one MAWI trace.
- Query-latency benchmark (Postgres heap / B-tree / monthly partitions; DuckDB flat vs Hive partitions).
- MAWI pcap → flow records (softflowd → IPFIX over TCP → own decoder) → aggregates, now with bounded memory.

## Rows, by what they are
| Abilene table | Rows | Kind |
|---|---|---|
| traffic_matrix | 6,967,296 | MEASURED OD traffic (48,384 steps × 144 flows); 5,883,120 in kept steps |
| link_load | 2,612,736 | DERIVED (traffic × routing matrix), not a measurement |
| time_steps / routing / flows / links / nodes | 48,384 / 644 / 144 / 54 / 12 | reference |
| **total** | **9,629,270** | |

CESNET institutions: 10,446,225 rows at 10 minutes, 1,880,182 at 1 hour, 78,667 at 1 day (12,405,074 total)
from 1,244,197,731 bytes of CSV, stored as 270 Parquet files (491,818,321 bytes).

## Throughput and idempotency
| Step | Wall time | Rate | Re-run changed |
|---|---|---|---|
| Abilene → Parquet | 2.6 s | 3.68M rows/s | nothing (all files unchanged) |
| Abilene → Postgres | 25.2 s | | 0 rows |
| CESNET → Parquet | 6.0 s | 2.06M rows/s (peak RSS 1.23 GB) | nothing |
| CESNET → Postgres | 76.1 s | 163k rows/s; table 3.33 GB | 0 rows |

## Query latency (CESNET 10-minute table, 10,446,225 rows, warm cache, median of repeated runs, identical results in every variant)
| Query | Postgres heap (no index) | B-tree (institution_id, ts) + (ts) | monthly range partitions + same indexes |
|---|---|---|---|
| Q1 one institution, one week | 186.4 ms | 0.37 ms | 0.38 ms |
| Q2 network hourly totals, one month | 229.5 ms | 235.4 ms (index not used) | 71.6 ms |
| Q3 top-10 institutions, one day | 192.8 ms | 3.69 ms | 3.86 ms |

DuckDB on Parquet (same queries): flat file 4.4 / 46.2 / 31.0 ms; Hive month partitions 6.7 / 25.5 / 8.0 ms.
Reading: the B-tree index makes the selective queries (Q1, Q3) about 500× and 52× faster than a full scan;
for the month-wide aggregate (Q2) the index does not help and partition pruning gives 3.2×.

## MAWI 2025-10-01 14:00 (real WIDE samplepoint-F traffic, not Abilene)
Reconciliation (identical in all three runs): IP packets in the pcap (independent walk) = softflowd packets
processed = packets in the flow records = **207,302,561**; 0 export failures; 56,523,852 flow records;
flow bytes 181,053,517,547 vs 181,065,327,685 wire bytes minus Ethernet headers (−0.0065%).

Memory follow-up (bounded-memory rewrite; only these re-measured numbers are used):

| Run | Chunk / DuckDB limit | Process wall to exit | Teardown | Peak footprint | Pipeline peak RSS | softflowd etc. peak RSS | System-wide swap-outs |
|---|---|---|---|---|---|---|---|
| Phase 1 original (superseded) | whole trace in memory | 1,465 s* | ≈ 900 s* | 16.8 GB* | 7.86 GB | – | not measured |
| attempt 1 | 4M records / 3 GB | 440.7 s | 0.10 s | 3.77 GB | 2.77 GB | 1.55 GB | **92,796** |
| attempt 2 (used) | 1M records / 1 GB | **415.8 s** | **0.09 s** | **1.85 GB** | 1.41 GB | 1.56 GB | **0** |
| idle control (`sleep 440`) | – | 440.2 s | – | – | – | – | 0 |

\* Not recorded in any result file (how they were measured is not documented); `phase1_mawi_202510011400.json` records the pipeline's
peak RSS (7,861,927,936 bytes) and 566.0 s in-process wall time. Comparisons use peak RSS: 7.86 GB → 1.41 GB.

Throughput of the used run: softflowd 193.0 s for the trace (1.08M pcap records/s), decode + aggregation 206.2 s
(57 chunks, 412 MB of temporary spill files, deleted). Aggregates equal Phase 1's: three tables byte-identical;
the throughput table differs only by floating-point summation order (largest change 1.5e-11 Mbit/s).
Average trace throughput 1,591.7 Mbit/s (10-second bins, p95 1,875.1, max 2,100.2; bytes spread uniformly over each flow).

## Disk
Cleanup deleted only regenerable data (`phase1_cleanup.json`): free space 43.7 → 53.3 GiB (9.6 GiB freed,
mostly the 6.7 GB benchmark tables). After the MAWI re-downloads and the TensorFlow environment: 47 GiB free.

## Failures and attempts (all recorded)
- nfdump `nfpcapd` rejected (kept 25.6% of packets on 96-byte snaplen traces); softflowd → UDP → nfcapd rejected (~25% export loss).
- Bounded-memory attempt 1 still coincided with 92,796 system-wide swap-outs while an idle control of the same length had 0;
  attempt 2 lowered the chunk size and DuckDB limit and had 0.
- Each re-measurement needed the 5.36 GB pcap again (the IPFIX stream had been deleted); sha256 identical each time.

## Open issues
- Swap-outs are only measurable system-wide on macOS; 0 is conclusive, a non-zero value is not attributable.
- Docker still not running; the private Postgres on 5433 is used.
