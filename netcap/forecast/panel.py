"""Forecasting data: a panel of series on a real-time axis, the samples and the rolling-origin folds.

Abilene samples are exactly the forecasting module's (common._prepare_segments: window 12, a sample's window and
target never cross a segment edge), and the folds are the forecasting module's four chronological splits:
  original  train [0, 60%), val [60%, 80%), test [80%, 100%)   (common.chronological_split)
  A, B, C   train [0, a), val [a, b), test [b, c) with (a, b, c) from scripts/10_robustness.py FOLDS.
tests_netcap/test_forecast.py checks the targets and fold sizes against common.py.
Fractions are of the time-ordered samples of that horizon, so fold boundaries differ slightly by horizon,
exactly as in the forecasting module.

CESNET (1-hour institution series) uses the same fold fractions; a sample needs its target, its last value
and all 12 window values known for that institution (missing institution-hours are never filled).
"""
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .. import config
from ..data import abilene

WINDOW = 12
FOLDS = {"original": None, "A": (0.30, 0.40, 0.55), "B": (0.45, 0.55, 0.70), "C": (0.60, 0.70, 0.80)}
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)


@dataclass
class Panel:
    name: str
    values: np.ndarray          # (N, F) float64, raw units; 0 where unknown
    known: np.ndarray           # (N, F) bool
    day: int                    # steps per day
    week: int                   # steps per week
    to_unit: float              # raw value -> reported unit
    unit: str
    step_minutes: int
    flow_names: list
    targets: dict = field(default_factory=dict)      # horizon -> target rows (real-time), sample order
    times: np.ndarray = None    # (N,) datetime64 of each row (assumed for Abilene)
    row_to_step: np.ndarray = None   # (N,) store key of each row (Abilene step, CESNET id_time); -1 = none

    def horizon_minutes(self, h):
        return h * self.step_minutes


def fold_bounds(n, fold):
    """(train, val, test) as (lo, hi) sample index pairs for n time-ordered samples."""
    if FOLDS[fold] is None:
        a = int(n * SPLIT_FRACTIONS[0])
        b = a + int(n * SPLIT_FRACTIONS[1])
        return (0, a), (a, b), (b, n)
    a, b, c = (int(f * n) for f in FOLDS[fold])
    return (0, a), (a, b), (b, c)


def abilene_timeline(weeks, cl):
    """Cleaned traffic on a real-time axis (as common.clean_timeline). Returns values, known, to_time."""
    full, keep = weeks.reshape(-1, weeks.shape[2]).astype(np.float64), cl.keep
    d0 = datetime.fromisoformat(abilene.WEEK_STARTS[0])
    offs = np.array([(datetime.fromisoformat(d) - d0).days * abilene.STEPS_PER_DAY for d in abilene.WEEK_STARTS])
    i = np.arange(len(full))
    to_time = offs[i // abilene.STEPS_PER_WEEK] + i % abilene.STEPS_PER_WEEK
    rows = to_time[keep]
    if len(np.unique(rows)) != len(rows):
        raise ValueError("two kept steps fall on the same real time")
    values = np.zeros((to_time.max() + 1, full.shape[1]))
    known = np.zeros(to_time.max() + 1, bool)
    values[rows], known[rows] = full[keep], True
    return values, known, to_time


def abilene_targets(cl, to_time, h, window=WINDOW):
    """Real-time rows of the targets, in the forecasting module's sample order (per segment, in time)."""
    idx = [a + window + h - 1 + np.arange(b - a - window - h + 1) for a, b in cl.segments if b - a >= window + h]
    return to_time[np.concatenate(idx)]


def load_abilene(horizons=(1, 6, 12), raw=None):
    weeks = abilene.load_weeks(raw)
    cl = abilene.clean(weeks)
    values, known, to_time = abilene_timeline(weeks, cl)
    d0 = np.datetime64(abilene.WEEK_STARTS[0])
    p = Panel("abilene", values, np.repeat(known[:, None], values.shape[1], 1), abilene.STEPS_PER_DAY,
              abilene.STEPS_PER_WEEK, abilene.TO_MBIT_S, "Mbit/s", 5,
              [f"{s}->{d}" for _, s, d in abilene.flow_pairs()],
              times=d0 + np.arange(len(values)) * np.timedelta64(300, "s"))
    p.row_to_step = np.full(len(values), -1)
    kept = np.flatnonzero(cl.keep)
    p.row_to_step[to_time[kept]] = kept
    for h in horizons:
        p.targets[h] = abilene_targets(cl, to_time, h)
    p.entity_ids = np.arange(values.shape[1])
    return p


def load_cesnet(horizons=(1, 6), metric="n_bytes", root=None):
    """CESNET-TimeSeries24 institutions, 1-hour aggregation: (6,717 hours x 283 institutions)."""
    import polars as pl
    root = root or config.DATA / "cesnet" / "series" / "aggregation=1_hour"
    df = pl.read_parquet(root / "*" / "*.parquet", columns=["institution_id", "id_time", "ts", metric])
    inst = np.sort(df["institution_id"].unique().to_numpy())
    n = int(df["id_time"].max()) + 1
    col = np.searchsorted(inst, df["institution_id"].to_numpy())
    t = df["id_time"].to_numpy()
    values = np.zeros((n, len(inst)))
    known = np.zeros((n, len(inst)), bool)
    values[t, col] = df[metric].to_numpy().astype(np.float64)
    known[t, col] = True
    obs = df.select("id_time", pl.col("ts").dt.convert_time_zone("UTC").dt.replace_time_zone(None)).unique("id_time")
    t0 = obs["ts"].to_numpy()[0] - obs["id_time"][0] * np.timedelta64(3600, "s")
    ts = t0 + np.arange(n) * np.timedelta64(3600, "s")
    if not (ts[obs["id_time"].to_numpy()] == obs["ts"].to_numpy()).all():
        raise ValueError("CESNET 1-hour id_time is not a regular hourly grid")
    p = Panel("cesnet_1h", values, known, 24, 168, 8 / 3600 / 1e6, "Mbit/s", 60,
              [f"institution {i}" for i in inst], times=ts, row_to_step=np.arange(n))
    p.entity_ids = inst
    for h in horizons:
        p.targets[h] = cesnet_targets(known, h)
    return p


def cesnet_targets(known, h, window=WINDOW):
    """Rows t where at least one institution has its whole window and target known (per-flow mask applies)."""
    n = len(known)
    t = np.arange(window + h - 1, n)
    return t[known[t].any(1)]


def sample_mask(p, t, h, window=WINDOW):
    """(n, F) bool: target, last value and the whole window known for that flow."""
    m = p.known[t].copy()
    for k in range(window):
        m &= p.known[t - h - k]
    return m
