"""Abilene loaders, TF-free. Re-implements the forecasting module's cleaning rules R1-R4 (common.abilene_clean).

Facts verified in Phase 0 (results/netcap/audit.json):
  - X01..X24.gz: 2016 rows x 720 values; value 5k+c of a row is flow k, version c; c = 0 is realOD (measured)
  - unit: 100 bytes per 5 minutes; flow k = src * 12 + dst in NODES order (raw/demands)
  - A: 54 links x 144 flows, all fractions 1 (single path per flow)
Week files are not all back to back. Timestamps assume each file starts at 00:00 on the README date;
the README gives dates only and no time zone, so `ts` is an ASSUMED file-clock timestamp.
Two weeks that overlap in time (R1 duplicates) mean `ts` is not unique; the key is `step`.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config

NODES = ["ATLA-M5", "ATLAng", "CHINng", "DNVRng", "HSTNng", "IPLSng",
         "KSCYng", "LOSAng", "NYCMng", "SNVAng", "STTLng", "WASHng"]
N_NODES, N_FLOWS, STEPS_PER_WEEK, STEPS_PER_DAY = 12, 144, 2016, 288
WEEK_STARTS = ["2004-03-01", "2004-03-08", "2004-04-02", "2004-04-09", "2004-04-22", "2004-05-01",
               "2004-05-08", "2004-05-15", "2004-05-22", "2004-05-29", "2004-06-05", "2004-06-12",
               "2004-06-19", "2004-06-26", "2004-07-03", "2004-07-10", "2004-07-17", "2004-07-24",
               "2004-07-31", "2004-08-07", "2004-08-13", "2004-08-21", "2004-08-28", "2004-09-04"]
UNIT_BYTES = 100
STEP_SECONDS = 300
TO_MBIT_S = UNIT_BYTES * 8 / STEP_SECONDS / 1e6        # raw value -> Mbit/s averaged over the step

# cleaning thresholds, identical to the forecasting module
OUTAGE_FLOW_STEPS = 288          # R2: active flow at zero >= 1 day
OUTAGE_ROUTER_STEPS = 12         # R3: all traffic in/out of a router at zero >= 1 hour
ACTIVE_FLOW_MAX_ZERO = 0.5       # R2 applies only to flows zero < 50% of the time
MAX_LINK_KBPS = 9_920_000        # R4: fastest link (topo file)
MAX_FLOW_RAW = MAX_LINK_KBPS * 1e3 / 8 * STEP_SECONDS / UNIT_BYTES
REASONS = ("duplicate", "flow", "router", "impossible")


def load_week(week, raw=None, column=0):
    """(2016, 144) float64 of one week file, raw units."""
    raw = Path(raw or config.RAW_ABILENE)
    a = np.loadtxt(raw / f"X{week:02d}.gz")
    if a.shape != (STEPS_PER_WEEK, N_FLOWS * 5):
        raise ValueError(f"X{week:02d}.gz has shape {a.shape}")
    return a.reshape(STEPS_PER_WEEK, N_FLOWS, 5)[:, :, column]


def load_weeks(raw=None, column=0):
    """(24, 2016, 144) float64 realOD, raw units."""
    return np.stack([load_week(w, raw, column) for w in range(1, 25)])


def _runs(b):
    """(start, end) of runs of True in a 1-D bool array, end exclusive."""
    d = np.diff(np.concatenate([[0], np.asarray(b, np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


@dataclass
class Cleaned:
    keep: np.ndarray                      # (n_steps,) bool
    reason: np.ndarray                    # (n_steps,) str, "" when kept
    segments: list                        # [(start_step, end_step)] runs of consecutive kept steps
    duplicate_weeks: list = field(default_factory=list)
    flow_outages: list = field(default_factory=list)      # (flow, start, end)
    router_outages: list = field(default_factory=list)    # (node, "into"/"out of", start, end)
    impossible_steps: list = field(default_factory=list)

    def removed(self):
        return {r: int((self.reason == r).sum()) for r in REASONS}


def clean(weeks, week_starts=WEEK_STARTS):
    """Apply rules R1-R4 to (n_weeks, 2016, 144) raw weeks. Same semantics as common.abilene_clean.

    R1 duplicate: a week whose first day exactly copies the previous week's last day -> those steps.
    R2 flow outage: a flow zero < 50% of the time reads zero >= 1 day -> those steps (all flows).
    R3 router outage: all traffic into, or out of, a router is zero >= 1 hour -> those steps.
    R4 impossible: any flow above the fastest link's capacity -> that step.
    The first rule that fires names the reason. Segments are cut at removed steps and at week joins
    that are not proven back to back (dates not 7 days apart, or R1 fired at that join).
    """
    n_w, T, F = weeks.shape
    full = weeks.reshape(-1, F)
    reason = np.full(len(full), "", dtype="<U10")

    def mark(a, b, r):
        sl = reason[a:b]
        sl[sl == ""] = r

    dup = []
    for w in range(1, n_w):
        if np.array_equal(weeks[w, :STEPS_PER_DAY], weeks[w - 1, -STEPS_PER_DAY:]):
            mark(w * T, w * T + STEPS_PER_DAY, "duplicate")
            dup.append(w + 1)

    zero = full == 0
    flow_out = []
    for k in np.flatnonzero(zero.mean(0) < ACTIVE_FLOW_MAX_ZERO):
        for a, b in _runs(zero[:, k]):
            if b - a >= OUTAGE_FLOW_STEPS:
                mark(a, b, "flow")
                flow_out.append((int(k), a, b))

    tm = full.reshape(-1, N_NODES, N_NODES)
    router_out = []
    for r in range(N_NODES):
        for direction, tot in (("into", tm[:, :, r].sum(1)), ("out of", tm[:, r, :].sum(1))):
            for a, b in _runs(tot == 0):
                if b - a >= OUTAGE_ROUTER_STEPS:
                    mark(a, b, "router")
                    router_out.append((NODES[r], direction, a, b))

    imp = np.flatnonzero((full > MAX_FLOW_RAW).any(1))
    sel = reason[imp] == ""
    reason[imp[sel]] = "impossible"

    keep = reason == ""
    starts = [datetime.fromisoformat(d) for d in week_starts]
    cut = np.zeros(len(full), bool)
    for w in range(1, n_w):
        if not ((starts[w] - starts[w - 1]).days == 7 and (w + 1) not in dup):
            cut[w * T] = True
    segments = []
    for a, b in _runs(keep):
        edges = [a, *(np.flatnonzero(cut[a + 1:b]) + a + 1).tolist(), b]
        segments += list(zip(edges[:-1], edges[1:]))
    return Cleaned(keep, reason, segments, dup, flow_out, router_out, imp.tolist())


def step_times(n_weeks=24, week_starts=WEEK_STARTS):
    """ASSUMED timestamp of every step (file clock, week file starts 00:00 on its README date)."""
    base = np.array([np.datetime64(d) for d in week_starts[:n_weeks]], dtype="datetime64[s]")
    off = np.arange(STEPS_PER_WEEK) * np.timedelta64(STEP_SECONDS, "s")
    return (base[:, None] + off[None, :]).ravel().astype("datetime64[us]")


def _rows(path):
    return [l.split() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]


@dataclass
class Topology:
    nodes: list          # [(name, city, lat, lon)]
    links: list          # [(link_id 1..54, name, src, dst, link_type, capacity_kbps|None, igp_weight|None)]
    A: np.ndarray        # (54, 144); link load = A @ flows


def load_topology(raw=None):
    """Nodes, links (with capacities and OSPF weights from topo-2003-04-10.txt) and the routing matrix A."""
    raw = Path(raw or config.RAW_ABILENE)
    topo = next(raw.glob("topo-*.txt")).read_text().splitlines()
    sec, nodes, caps = None, [], {}
    for line in topo:
        s = line.strip()
        if s in ("router", "link"):
            sec = s
            continue
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if sec == "router":
            nodes.append((p[0], p[1], float(p[2]), float(p[3])))
        else:
            caps[(p[0], p[1])] = (float(p[2]), int(p[3]))
    if [n[0] for n in nodes] != NODES:
        raise ValueError("node order in topo file differs from NODES")
    links = []
    for name, idx, t in _rows(raw / "links"):
        src, dst = name.split(",")
        cap, w = caps.get((src, dst), (None, None))
        links.append((int(idx), name, src, dst, int(t), cap, w))
    A = np.zeros((len(links), N_FLOWS))
    for _, _, li, di, frac in _rows(raw / "A"):
        A[int(li) - 1, int(di) - 1] = float(frac)
    return Topology(nodes, links, A)


def flow_pairs():
    """[(flow_id, src, dst)] with flow_id = src_index * 12 + dst_index."""
    return [(k, NODES[k // N_NODES], NODES[k % N_NODES]) for k in range(N_FLOWS)]
