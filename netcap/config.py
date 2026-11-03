"""Paths and settings. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("NETCAP_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAW_ABILENE = Path(os.environ.get("NETCAP_RAW_ABILENE", ROOT / "raw"))
RAW_CESNET = Path(os.environ.get("NETCAP_RAW_CESNET", ROOT / "raw_cesnet" / "extracted"))
RAW_MAWI = Path(os.environ.get("NETCAP_RAW_MAWI", ROOT / "raw_mawi"))
DATA = Path(os.environ.get("NETCAP_DATA", ROOT / "data_netcap"))      # Parquet store (gitignored)
RESULTS = ROOT / "results" / "netcap"

DATABASE_URL = os.environ.get("NETCAP_DATABASE_URL", "postgresql://netcap:netcap@localhost:5433/netcap")
SEED = 42
PROJECT_RELEASE = "1.0"
