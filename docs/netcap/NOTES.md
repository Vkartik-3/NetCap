# NetCap working notes

Findings about the data and the forecasting module that are not results in themselves but matter for anyone
reusing them. Each item names the command that shows it.

## 1. Abilene: only per-source totals agree across the five column versions

**Claim in the forecasting module** (`research/RESEARCH_LOG.md`, section 10.3): "all five versions have exactly the
same per-router totals (as the estimation models require), so only one of them can be the measurement".

**What the data shows** (Phase 0 audit):

| Totals | Agreement across realOD and the 4 estimates |
|---|---|
| Per source router (row sums of the 12 x 12 matrix) | Agree in every week; largest relative difference 0.65% (X05) |
| Per destination router (column sums) | Differ in almost every step of every week; largest relative difference 842% (X23) |

So the statement holds only for per-source (ingress) totals. It does not change the forecasting module's choice
of `realOD` (column 0): that choice rests on `raw/readme.txt`, which names `realOD` as the measured
traffic. The flow index (source x 12 + destination) is confirmed independently by `raw/demands`.

Reproduce: `.venv/bin/python -I netcap/audit/totals_check.py raw`
(output saved in `results/netcap/audit_detail/router_totals_check.txt`).

## 2. GÉANT download link has moved

The TOTEM dataset page links to `totem.run.montefiore.ulg.ac.be`, which no longer resolves. The
archive is served (HTTP 301 from the `uliege.be` host) at
`https://totem.run.montefiore.uliege.be/files/data/traffic-matrices-anonymized-v2.tar.bz2`
(17,883,919 bytes, sha256 in `results/netcap/audit_detail/geant_sha256.txt`). GÉANT is not used in
NetCap for now.

## 3. Abilene timestamps are assumed

`raw/readme.txt` gives only the first date of each week file, with no time or time zone. NetCap stores
`ts_assumed` = README date 00:00 + 5 minutes x step, without time zone, as the forecasting module does
(`step_label`). Because four weeks start with a duplicated copy of the previous day (rule R1), two
steps can share a `ts_assumed`; the key is `step` (week index x 2016 + step in week), never the time.

## 4. CESNET timestamps

`times_10_minutes.csv` carries local offsets (+02:00 / +01:00 across the October 2023 DST change) and
irregular seconds (e.g. `02:03:49`); hourly and daily times are UTC. NetCap converts all to UTC.
The 10-minute series has 10,446,225 rows, fewer than 283 institutions x 40,298 slots
(11,404,334), so some institution-slots are absent in the source; they are not filled.

## 5. MAWI: choosing a flow meter for 96-byte truncated traces

MAWI samplepoint-F traces keep only the first 96 bytes of each packet. Three routes were tried on trace
202510011400 (evidence: `results/netcap/phase1_mawi_flowmeter_attempts.json`):

| Route | Outcome |
|---|---|
| nfdump 1.7.10 `nfpcapd` (built with `--enable-nfpcapd`; Homebrew's build omits it) | **Rejected.** Skips every packet whose IP length exceeds the captured bytes, so it kept 53,075,430 of 207,553,081 packets (25.6%) and 2.69 GB. |
| softflowd 1.1.1 → NetFlow v9 over UDP → `nfcapd` | **Rejected.** Metering correct, but about 25% of export datagrams lost on a 4.2M-packet sample (11,499 sequence errors), and flow times took the wall clock without `-a`. |
| softflowd 1.1.1 → IPFIX over **TCP** → `netcap.data.ipfix` collector | **Adopted.** On the sample, flow packets = 4,232,077 = IP packets counted independently in the pcap; 0 export failures; times match the capture. |

softflowd counts bytes as the pcap record's original length minus the link header, so byte counts are
the real IP lengths even though only 96 bytes were captured.

Building the tools (not in the repository; installed under `~/.local`):

```
git clone --depth 1 --branch softflowd-v1.1.1 https://github.com/irino/softflowd.git
cd softflowd && autoreconf -if && ./configure --prefix=$HOME/.local/softflowd && make && make install
```

## 6. Local Postgres during development

Docker was not running during Phase 1, so a private PostgreSQL 17.4 cluster was started from the
locally installed binaries (`data_netcap/pgdata`, localhost:5433, user `netcap`). The Docker Compose
`db` service (Phase 6) replaces it; the URL is set with `NETCAP_DATABASE_URL`.
