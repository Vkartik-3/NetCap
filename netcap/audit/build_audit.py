"""Phase 0: assemble results/netcap/audit.json from the audit detail files.

Usage (from repo root): .venv/bin/python -I netcap/audit/build_audit.py
Values marked "observed" were read from HTTP responses / command output during the
audit session (2026-11-02) and are recorded verbatim; everything else is loaded
from results/netcap/audit_detail/*.json produced by the other netcap/audit scripts.
"""
import json, platform, subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results" / "netcap" / "audit_detail"
load = lambda n: json.loads((D / n).read_text())
git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

ab = load("abilene_verify.json")
mc = load("metrics_compare.json")
r06 = load("06_baselines_repro.json")
ce = load("cesnet_counts.json")
X = {k: v for k, v in ab["X"].items() if k != "per_file"}

audit = {
    "meta": {
        "phase": 0, "date": date.today().isoformat(),
        "project_commit": git("rev-parse", "HEAD"), "repository": "https://github.com/Vkartik-3/NetCap",
        "working_tree_clean_of_tracked_changes": git("status", "--porcelain", "--untracked-files=no") == "",
        "seed": 42, "python": platform.python_version(), "platform": platform.platform(),
        "env": "Python 3.13 .venv with requirements.txt pins minus tensorflow/keras/ipykernel, plus networkx 3.7",
        "command": "python -I netcap/audit/build_audit.py",
        "note": "Reproductions ran in an isolated workspace (TRAFFIC_PROJECT=<scratch>/repro with raw/ and "
                "raw_geant/ symlinked) so the primary results and figures were not modified.",
    },
    "project_log_summary": {
        "done": [
            "Phases 0-12 per docs/NETCAP_RESEARCH_LOG.md: data layer with cleaning rules R1-R4 (Abilene) and G1-G4 (GEANT); "
            "persistence/ARIMA/seasonal/blend/ridge baselines; LSTM/CNN/CNN-LSTM (residual, original-units MAE loss); "
            "horizons 5/30/60 min; link loads and congestion warnings via A; GEANT generalisation; error analysis; "
            "DM and day-block bootstrap tests; seasonal inputs; burst fallback rule (k sd) and learned gate; "
            "conformal q90 warning margins; rolling-origin folds A/B/C; 3-seed ablations.",
        ],
        "key_findings_from_log": [
            "Architecture alone: +0.3-2.5% vs persistence on Abilene (sections 14-15, 19, 23).",
            "Seasonal inputs + fallback rule: +8.7-13.1% vs persistence at 60 min across 4 Abilene folds (section 23.1).",
            "Better forecasts do not give better congestion warnings (sections 16, 22).",
            "GEANT results not robust across folds (section 23.1).",
            "80% utilisation threshold gives zero Abilene events; 50% used (section 16.2).",
        ],
        "open_per_log": [
            "Written report not started (status table row 7); presentation files do exist in docs/presentation/, so that row is stale.",
            "Not tried: peak-weighted loss, ARIMA + network combination (section 21.4).",
            "Seasonal ARIMA with 288-step season not done (section 11.3).",
            "Rule threshold k is not well determined by validation (sections 21.4, 23.1).",
            "Section 9 'Next steps' and section 5 script table are stale (describe Phase 4 as upcoming).",
        ],
    },
    "abilene": {
        "source": "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/", "downloaded": "2026-11-02",
        "files": 29, "sha256_file": "results/netcap/audit_detail/abilene_sha256.txt",
        "command": "python -I netcap/audit/verify_abilene.py raw results/netcap/audit_detail/abilene_verify.json",
        "X": X, "demands": ab["demands"], "links": ab["links"], "topo": ab["topo"], "A": ab["A"],
        "routing_preview": {k: v for k, v in ab["routing_preview"].items()},
        "checks_passed": {
            "units_100_bytes_per_5_min": "stated in raw/readme.txt; consistent: all weekly maxima except the R4 outlier "
                                          f"are below link speed (2nd largest weekly max {X['second_largest_weekly_max_mbit_s']:.1f} Mbit/s)",
            "flow_index_src*12+dst": ab["demands"]["flow_index_is_src*12+dst_with_common_node_order"],
            "A_54x144_all_fractions_1": ab["A"]["all_fractions_1"] and ab["A"]["name_index_mismatches"] == 0,
            "node_numbering_consistent_X_A_links_topo": ab["topo"]["router_order_matches_common"]
                                                        and ab["topo"]["internal_links_in_links_file_all_in_topo"]
                                                        and ab["topo"]["topo_links_all_in_links_file"],
        },
        "discrepancies_with_docs": [{
            "claim": "PROJECT_LOG 10.3: 'all five versions have exactly the same per-router totals'",
            "finding": "True for per-SOURCE totals only (max relative difference 0.65%, week 5). Per-DESTINATION totals differ "
                       "across versions in almost every step (max relative difference 8.42 in X23).",
            "impact": "None on results: realOD choice rests on raw/readme.txt, and the flow index is confirmed by raw/demands.",
            "command": "python -I netcap/audit/totals_check.py raw",
            "output": "results/netcap/audit_detail/router_totals_check.txt",
        }],
    },
    "reproduction": {
        "commands": [
            "TRAFFIC_PROJECT=<repro> BROWSER=true MPLBACKEND=Agg .venv/bin/python scripts/01_data.py",
            "TRAFFIC_PROJECT=<repro> BROWSER=true MPLBACKEND=Agg .venv/bin/python scripts/02_baselines.py",
            "TRAFFIC_PROJECT=<repro> .venv/bin/python netcap/audit/run06_baselines.py <repro>/scripts/06_baselines2.py out.json",
            "python -I netcap/audit/compare_metrics.py results/metrics.csv <repro>/results/metrics.csv out.json",
        ],
        "01_data": {"status": "reproduced, identical to PROJECT_LOG 10.5-10.7",
                    "removed_steps": {"duplicate": 1152, "flow": 6336, "router": 40, "impossible": 1},
                    "kept_pct": 84.4, "segments": 9, "samples": {"train": 24448, "val": 8149, "test": 8150},
                    "peak_trough": 1.53, "wall_s": 11.97, "peak_rss_bytes": 2064613376, "log": "audit_detail/01.log"},
        "02_baselines": {"status": "reproduced, identical to PROJECT_LOG 11.1-11.2",
                         "persistence_mae_mbit": {"val": 2.742, "test": 3.212},
                         "arima_top5_vs_persistence_test_mae_mbit": [21.871, 21.281], "arima_change_pct": -2.8,
                         "wall_s": 26.49, "log": "audit_detail/02.log"},
        "06_baselines2": {"status": "PARTIAL: cells 1-3 (seasonal, blend, yday_change, ridge, ARIMA) + baseline-only tables "
                                    "and best-baseline-vs-persistence significance. Cell 4+ neural rows NOT reproduced "
                                    "(needs TensorFlow and saved models; models/ is empty).",
                          "matches_log_section_19": True, "results": r06, "wall_s": 134.85,
                          "peak_rss_bytes": 3705847808, "log": "audit_detail/06.log"},
        "metrics_csv_comparison": {k: v for k, v in mc.items() if k != "rows"},
        "differences": "None beyond float noise: max relative difference over all compared metrics "
                       f"{mc['max_rel_diff_any_metric']:.2e} (ARIMA rows); persistence rows identical.",
    },
    "datasets": {
        "geant_totem": {
            "available": True, "license": "no license stated; TOTEM page asks publications to cite Uhlig et al., "
                                          "SIGCOMM CCR 36(1), 2006",
            "url_in_readme": "https://totem.info.ucl.ac.be/dataset.html",
            "archive_url_working": "https://totem.run.montefiore.uliege.be/files/data/traffic-matrices-anonymized-v2.tar.bz2",
            "issue": "link on the dataset page (totem.run.montefiore.ulg.ac.be) no longer resolves; host moved to uliege.be (HTTP 301)",
            "archive_bytes": 17883919, "sha256_file": "results/netcap/audit_detail/geant_sha256.txt",
            "matrices": 10772, "extracted_bytes_approx": "206M (du -sh)",
            "rows_note": "10,772 matrices x 529 OD pairs (sparse: files list only some pairs)",
        },
        "cesnet_timeseries24": {
            "available": True, "license": "CC-BY-4.0 (Zenodo record 13382427)", "doi": "10.5281/zenodo.13382427",
            "downloaded": ["institutions.tar.gz", "times.tar.gz", "weekends_and_holidays.csv"],
            "md5_verified_against_zenodo": True,
            "institutions_tar_gz_bytes": 479428489,
            "counts": ce,
            "time_range_10min": "2023-10-09 02:03:49+02:00 .. 2024-07-14 23:50:52+02:00 (40,298 slots)",
            "not_downloaded_observed_sizes_bytes": {"ip_addresses_full.tar.gz": 40048881181,
                                                    "institution_subnets.tar.gz": 774004934,
                                                    "ip_addresses_sample.tar.gz": 170864480},
            "note": "Per-institution aggregate time series (flows, packets, bytes, ...), not a topology or traffic matrix. "
                    "Full IP-level set (40.0 GB compressed) not feasible on this machine (66 GiB free disk).",
        },
        "mawi_wide": {
            "available": True, "samplepoint": "F", "daily_traces_2025": 365,
            "example": {"id": "202510011400", "duration_s": 899.95, "packets": 207553081,
                        "pcap_gz_bytes": 5357837608, "avg_rate_mbps": 1635.64, "caplen_bytes": 96},
            "license": "NOT FOUND on the archive index page; must be checked before redistribution of derived data",
            "tools": "nfdump 1.7.10 available via Homebrew (includes nfpcapd for pcap->flow); softflowd not in Homebrew",
            "note": "Not downloaded yet (5.36 GB per 15-min trace). Real flow data from WIDE, unrelated to Abilene.",
        },
    },
    "models": {
        "local_models_dir": "empty (only .gitkeep)", "elsewhere_on_disk": "none found (*.keras search of home)",
        "github_releases": 0, "github_license": None,
        "plan": "Serve TF-free statistical forecasters. Neural (CNN-LSTM seasonal+rule) forecasts can only be imported "
                "as precomputed artifacts after retraining with TF 2.21 (available on PyPI for py3.13); otherwise skipped "
                "and stated as such.",
    },
    "environment_constraints": {
        "ram_bytes": 17179869184, "cpus": 8, "free_disk": "66Gi",
        "docker_daemon": "not running at audit time", "kind": "v0.32.0", "postgres_client": "psql 17.4",
        "local_dns_resolver_flaky": "mawi.wide.ad.jp and the old TOTEM host intermittently failed to resolve; worked on retry",
    },
    "gate": {
        "status": "AWAITING USER APPROVAL",
        "blocking_issues": [],
        "decisions_needed": [
            "Neural forecasts: retrain CNN-LSTM seasonal+rule on CPU to produce artifacts, or skip neural challengers.",
            "CESNET scope: institutions (10.4M rows at 10-min) is downloaded; the 40 GB IP-level set is not feasible here.",
            "MAWI: how many 15-min traces to download (5.36 GB each); license must be confirmed.",
            "Repository release workflow and CI activation.",
        ],
    },
}
out = ROOT / "results" / "netcap" / "audit.json"
out.write_text(json.dumps(audit, indent=1, default=str))
print("wrote", out, f"({out.stat().st_size} bytes)")
