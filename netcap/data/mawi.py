"""MAWI Working Group Traffic Archive (WIDE Project), samplepoint-F: pcap -> NetFlow-style records -> aggregates.

REAL flow data from the WIDE transit/IX links in Japan; unrelated to Abilene, no topology.
Source: https://mawi.wide.ad.jp/mawi/ . Raw traces are never stored in the repository: the pcap is
deleted after conversion and only derived aggregates are kept.

Flow meter: softflowd 1.1.1 (built from source, commit 0260261) reading the pcap from stdin,
exporting IPFIX over TCP to a local collector (netcap.data.ipfix):
  gzcat trace.pcap.gz | softflowd -d -a -r - -n 127.0.0.1:<port> -P tcp -v 10 -6 -A milli -m 4000000
                         -t maxlife=60 -t general=15 -t tcp=15 -t tcp.rst=15 -t tcp.fin=15 -t udp=15 -t icmp=15
Records are unidirectional. Bytes are the original IP length (softflowd counts pcap wire length minus
the link header), so they are correct although MAWI truncates every packet to 96 bytes.

Rejected (recorded in results/netcap/phase1_mawi_*.json and docs/netcap/NOTES.md):
  - nfdump 1.7.10 nfpcapd: drops every packet whose IP length exceeds the captured bytes, i.e. nearly
    all packets of a 96-byte-snaplen trace (kept 53.1M of 207.6M packets, 2.69 GB).
  - softflowd -> UDP NetFlow v9 -> nfcapd: ~25% of export packets lost (11,499 sequence errors on a
    4.2M-packet sample) and flow times stamped with the wall clock without -a.
Per-interval throughput spreads each flow's bytes uniformly over [first, last] seen: an approximation
inside a flow. Link capacity is not published, so no utilisation %. Addresses are anonymised by MAWI.
"""
import ipaddress
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import ipfix

SOFTFLOWD = Path(os.environ.get("NETCAP_SOFTFLOWD", Path.home() / ".local" / "softflowd" / "sbin" / "softflowd"))
TIMEOUTS = {"maxlife": 60, "general": 15, "tcp": 15, "tcp.rst": 15, "tcp.fin": 15, "udp": 15, "icmp": 15}
MAX_FLOWS = 4_000_000
BIN_S = 10
PROTO = {1: "ICMP", 6: "TCP", 17: "UDP", 41: "IPv6-in-IPv4", 47: "GRE", 50: "ESP", 51: "AH", 58: "ICMPv6",
         132: "SCTP", 4: "IPIP"}


def softflowd_args(port, ctl_dir):
    args = [str(SOFTFLOWD), "-d", "-a", "-r", "-", "-n", f"127.0.0.1:{port}", "-P", "tcp", "-v", "10", "-6",
            "-A", "milli", "-m", str(MAX_FLOWS), "-p", str(ctl_dir / "softflowd.pid"), "-c", str(ctl_dir / "sf.ctl")]
    for k, v in TIMEOUTS.items():
        args += ["-t", f"{k}={v}"]
    return args


def pcap_to_ipfix(pcap_gz, out_ipfix, port=4739):
    """Stream the gzipped pcap through softflowd; collect the IPFIX stream losslessly over TCP."""
    out_ipfix = Path(out_ipfix)
    out_ipfix.parent.mkdir(parents=True, exist_ok=True)
    ready, result = threading.Event(), {}
    th = threading.Thread(target=lambda: result.update(n=ipfix.collect(port, out_ipfix, ready=ready)))
    th.start()
    ready.wait(10)
    t0 = time.perf_counter()
    gz = subprocess.Popen(["gzcat", str(pcap_gz)], stdout=subprocess.PIPE)
    sf = subprocess.run(softflowd_args(port, out_ipfix.parent), stdin=gz.stdout, capture_output=True, text=True)
    gz.stdout.close()
    gz.wait()
    th.join(600)
    wall = time.perf_counter() - t0
    log = sf.stdout + sf.stderr
    if sf.returncode != 0 or gz.returncode != 0:
        raise RuntimeError(f"softflowd {sf.returncode} / gzcat {gz.returncode}: {log[-2000:]}")
    num = lambda pat: int(re.search(pat, log).group(1)) if re.search(pat, log) else None
    return {"wall_s": wall, "ipfix_bytes": result.get("n"), "command": " ".join(softflowd_args(port, Path("."))),
            "packets_processed": num(r"Packets processed: (\d+)"), "ignored_non_ip": num(r"\((\d+) non-IP"),
            "flows_exported": num(r"Flows exported: (\d+)"), "records_exported": num(r"\((\d+) records\)"),
            "export_failures": num(r"\((\d+) failures\)"), "softflowd_log_tail": log[-2500:]}


def _addr(a, version):
    """Raw address bytes -> (hi, lo) uint64 columns (IPv4 in lo)."""
    raw = np.ascontiguousarray(a).view(np.uint8).reshape(len(a), -1)
    if version == 4:
        return np.zeros(len(a), np.uint64), raw.view(">u4").ravel().astype(np.uint64)
    v = raw.view(">u8")
    return v[:, 0].astype(np.uint64), v[:, 1].astype(np.uint64)


def records_to_frame(recs):
    """Decoded IPFIX records (dict template id -> structured array) -> one flow frame."""
    parts = []
    for tid, a in recs.items():
        names = a.dtype.names
        if "packetDeltaCount" not in names:
            continue                                     # options records (exporter metadata)
        v = 4 if "sourceIPv4Address" in names else 6
        sh, sl = _addr(a[f"sourceIPv{v}Address"], v)
        dh, dl = _addr(a[f"destinationIPv{v}Address"], v)
        icmp = f"icmpTypeCodeIPv{v}" if v == 6 else "icmpTypeCodeIPv4"
        if "sourceTransportPort" in names:
            sport, dport = a["sourceTransportPort"].astype(np.uint16), a["destinationTransportPort"].astype(np.uint16)
        else:                                            # ICMP: dport carries type * 256 + code
            sport, dport = np.zeros(len(a), np.uint16), a[icmp].astype(np.uint16)
        parts.append(pl.DataFrame({
            "ip_version": np.full(len(a), v, np.uint8), "src_hi": sh, "src_lo": sl, "dst_hi": dh, "dst_lo": dl,
            "proto": a["protocolIdentifier"].astype(np.uint8), "sport": sport, "dport": dport,
            "icmp": np.full(len(a), "sourceTransportPort" not in names),
            "t_first": a["flowStartMilliseconds"].astype(np.float64) / 1e3,
            "t_last": a["flowEndMilliseconds"].astype(np.float64) / 1e3,
            "packets": a["packetDeltaCount"].astype(np.uint64), "bytes": a["octetDeltaCount"].astype(np.uint64)}))
    return pl.concat(parts)


def _fmt_addr(version, hi, lo):
    if version == 4:
        return str(ipaddress.IPv4Address(int(lo)))
    return str(ipaddress.IPv6Address((int(hi) << 64) | int(lo)))


def proto_name(p):
    return PROTO.get(int(p), str(int(p)))


def throughput(df, bin_s=BIN_S):
    """Bytes per bin with each flow spread uniformly over its duration. Returns (bin_start_epoch, bytes, flows_active)."""
    a, b, by = df["t_first"].to_numpy(), df["t_last"].to_numpy(), df["bytes"].to_numpy().astype(np.float64)
    t0 = np.floor(a.min() / bin_s) * bin_s
    first, last = ((a - t0) // bin_s).astype(np.int64), ((b - t0) // bin_s).astype(np.int64)
    n_bins = int(last.max()) + 1
    dur = b - a
    out = np.zeros(n_bins)
    active = np.zeros(n_bins, np.int64)
    for k in range(int((last - first).max()) + 1):
        idx = first + k
        ok = idx <= last
        lo = np.maximum(a, t0 + idx * bin_s)
        hi = np.minimum(b, t0 + (idx + 1) * bin_s)
        frac = np.where(dur > 0, np.clip(hi - lo, 0, None) / np.where(dur > 0, dur, 1), (k == 0) * 1.0)
        out += np.bincount(idx[ok], weights=(by * frac)[ok], minlength=n_bins)
        active += np.bincount(idx[ok], minlength=n_bins)
    assert abs(out.sum() - by.sum()) <= 1e-6 * max(by.sum(), 1)          # every byte assigned exactly once
    return t0 + np.arange(n_bins) * bin_s, out, active


GROUPS = {"src": ["ip_version", "src_hi", "src_lo"], "dst": ["ip_version", "dst_hi", "dst_lo"],
          "pair": ["ip_version", "src_hi", "src_lo", "dst_hi", "dst_lo"], "dport": ["proto", "icmp", "dport"]}
SPILL_COLS = ["ip_version", "src_hi", "src_lo", "dst_hi", "dst_lo", "proto", "icmp", "dport", "packets", "bytes"]


def _talker_key(kind, r):
    if kind == "src":
        return _fmt_addr(r["ip_version"], r["src_hi"], r["src_lo"])
    if kind == "dst":
        return _fmt_addr(r["ip_version"], r["dst_hi"], r["dst_lo"])
    if kind == "pair":
        return (_fmt_addr(r["ip_version"], r["src_hi"], r["src_lo"]) + "->" +
                _fmt_addr(r["ip_version"], r["dst_hi"], r["dst_lo"]))
    return f"{proto_name(r['proto'])}/" + (f"{r['dport'] >> 8}.{r['dport'] & 255}" if r["icmp"] else str(r["dport"]))


def _finish(tot, proto, tops, tp_parts, trace_id):
    """Shared output formatting for the in-memory and the chunked path.

    proto: frame (proto int, flows, packets, bytes); tops: {kind: top-n frame with key columns + sums}.
    """
    proto = (proto.with_columns(pl.col("flows").cast(pl.UInt32)).sort(["bytes", "proto"], descending=[True, False])
             .with_columns(pl.col("proto").map_elements(proto_name, return_dtype=pl.String),
                           byte_share=pl.col("bytes") / tot["bytes"], packet_share=pl.col("packets") / tot["packets"],
                           trace_id=pl.lit(trace_id)))
    talkers = []
    for kind, g in tops.items():
        rows = g.to_dicts()
        talkers.append(pl.DataFrame({"trace_id": trace_id, "kind": kind,
                                     "rank": np.arange(1, len(rows) + 1, dtype=np.int32),
                                     "key": [_talker_key(kind, r) for r in rows],
                                     "flows": g["flows"].cast(pl.UInt32), "packets": g["packets"],
                                     "bytes": g["bytes"]})
                       .with_columns(byte_share=pl.col("bytes") / tot["bytes"]))
    starts, by, act = tp_parts
    tp = pl.DataFrame({"trace_id": trace_id, "bin_start": (starts * 1e6).astype("datetime64[us]"),
                       "bin_s": BIN_S, "bytes": by, "mbps": by * 8 / BIN_S / 1e6, "flows_active": act}
                      ).with_columns(pl.col("bin_start").dt.replace_time_zone("UTC"))
    return tot, proto, pl.concat(talkers), tp


def aggregates(df, trace_id, top_n=20):
    """Derived aggregates kept in the store (the flow records themselves are not). In memory: for tests and
    small inputs; the pipeline uses ChunkedAggregates, which gives the same result with bounded memory."""
    tot = df.select(pl.len().alias("flows"), pl.col("packets").sum().cast(pl.Int64), pl.col("bytes").sum().cast(pl.Int64),
                    pl.col("t_first").min(), pl.col("t_last").max()).row(0, named=True)
    sums = [pl.len().cast(pl.Int64).alias("flows"), pl.col("packets").sum().cast(pl.Int64),
            pl.col("bytes").sum().cast(pl.Int64)]
    proto = df.group_by("proto").agg(*sums).with_columns(pl.col("proto").cast(pl.Int64))
    tops = {kind: df.group_by(keys).agg(*sums).sort(["bytes", *keys], descending=[True] + [False] * len(keys)).head(top_n)
            for kind, keys in GROUPS.items()}
    return _finish(tot, proto, tops, throughput(df), trace_id)


class ThroughputAccumulator:
    """Adds per-chunk `throughput` results on a common grid of bins aligned to multiples of bin_s.

    Bins are aligned to absolute epoch multiples of bin_s, so a chunk's bins line up with the global ones
    exactly; the only difference from one global pass is the order of floating-point additions.
    """

    def __init__(self, bin_s=BIN_S):
        self.bin_s, self.base, self.by, self.active = bin_s, None, np.zeros(0), np.zeros(0, np.int64)

    def add(self, df):
        starts, by, act = throughput(df, self.bin_s)
        first = int(round(starts[0] / self.bin_s))
        if self.base is None:
            self.base = first
        if first < self.base:                              # grow to the left
            pad = self.base - first
            self.by, self.active = np.concatenate([np.zeros(pad), self.by]), np.concatenate([np.zeros(pad, np.int64), self.active])
            self.base = first
        lo = first - self.base
        hi = lo + len(by)
        if hi > len(self.by):                              # grow to the right
            self.by = np.concatenate([self.by, np.zeros(hi - len(self.by))])
            self.active = np.concatenate([self.active, np.zeros(hi - len(self.active), np.int64)])
        self.by[lo:hi] += by
        self.active[lo:hi] += act

    def result(self):
        return (self.base + np.arange(len(self.by))) * float(self.bin_s), self.by, self.active


class ChunkedAggregates:
    """Exact aggregates over flow chunks with bounded memory.

    Totals, protocol mix and throughput are additive and kept in memory per chunk. Top talkers need a
    group-by over all flows, so each chunk's key columns are spilled to Parquet under `spill_dir` and
    grouped by DuckDB under a fixed memory limit (DuckDB spills to disk when it needs more). The spill
    files are deleted by `finish`.
    """

    def __init__(self, spill_dir, memory_limit="3GB", threads=4):
        self.spill_dir = Path(spill_dir)
        if self.spill_dir.exists():
            for f in self.spill_dir.glob("*"):
                f.unlink()
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        self.memory_limit, self.threads = memory_limit, threads
        self.n = 0
        self.tp = ThroughputAccumulator()
        self.flows = self.packets = self.bytes = 0
        self.t_first, self.t_last = np.inf, -np.inf
        self.proto, self.ipv = {}, {}

    @staticmethod
    def _add(acc, frame, key):
        for r in frame.iter_rows(named=True):
            a = acc.setdefault(r[key], [0, 0, 0])
            a[0] += r["flows"]
            a[1] += r["packets"]
            a[2] += r["bytes"]

    def add(self, df):
        sums = [pl.len().cast(pl.Int64).alias("flows"), pl.col("packets").sum().cast(pl.Int64),
                pl.col("bytes").sum().cast(pl.Int64)]
        self._add(self.proto, df.group_by("proto").agg(*sums), "proto")
        self._add(self.ipv, df.group_by("ip_version").agg(*sums), "ip_version")
        self.flows += df.height
        self.packets += int(df["packets"].sum())
        self.bytes += int(df["bytes"].sum())
        self.t_first, self.t_last = min(self.t_first, df["t_first"].min()), max(self.t_last, df["t_last"].max())
        self.tp.add(df)
        df.select(SPILL_COLS).write_parquet(self.spill_dir / f"part-{self.n:05d}.parquet", compression="zstd")
        self.n += 1

    def spill_bytes(self):
        return sum(f.stat().st_size for f in self.spill_dir.glob("*.parquet"))

    def finish(self, trace_id, top_n=20):
        import duckdb
        tot = {"flows": self.flows, "packets": self.packets, "bytes": self.bytes,
               "t_first": float(self.t_first), "t_last": float(self.t_last)}
        frame = lambda acc, key: pl.DataFrame([{key: int(k), "flows": v[0], "packets": v[1], "bytes": v[2]}
                                               for k, v in acc.items()])
        tmp = self.spill_dir / "duckdb_tmp"
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{self.memory_limit}'; SET threads={self.threads}; "
                    f"SET temp_directory='{tmp}'; SET preserve_insertion_order=false")
        src = f"read_parquet('{self.spill_dir}/part-*.parquet')"
        tops = {}
        for kind, keys in GROUPS.items():
            k = ", ".join(keys)
            order = ", ".join(f"{c} ASC" for c in keys)
            q = (f"SELECT {k}, count(*)::BIGINT AS flows, sum(packets)::BIGINT AS packets, sum(bytes)::BIGINT AS bytes "
                 f"FROM {src} GROUP BY {k} ORDER BY bytes DESC, {order} LIMIT {top_n}")
            tops[kind] = con.sql(q).pl().with_columns(
                [pl.col(c).cast(t) for c, t in (("ip_version", pl.UInt8), ("src_hi", pl.UInt64), ("src_lo", pl.UInt64),
                                                ("dst_hi", pl.UInt64), ("dst_lo", pl.UInt64), ("proto", pl.UInt8),
                                                ("dport", pl.UInt16)) if c in keys])
        con.close()
        spill = self.spill_bytes()
        for f in self.spill_dir.glob("*.parquet"):
            f.unlink()
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        self.spill_dir.rmdir()
        by_ipv = frame(self.ipv, "ip_version").sort("ip_version").to_dicts()
        out = _finish(tot, frame(self.proto, "proto"), tops, self.tp.result(), trace_id)
        return out, {"chunks": self.n, "spill_bytes": spill, "by_ip_version": by_ipv}


def parse_trace_info(html):
    """Numbers from the MAWI trace page (the archive's own summary of the pcap)."""
    txt = re.sub(r"<[^>]+>", "", html)
    g = lambda pat, f=float: f(re.search(pat, txt).group(1)) if re.search(pat, txt) else None
    return {"packets": g(r"# of packets:\s*(\d+)", int), "duration_s": g(r"TotalTime:\s*([\d.]+)"),
            "avg_rate_mbps": g(r"AvgRate:\s*([\d.]+)Mbps"), "caplen_bytes": g(r"CapLen:\s*(\d+)", int),
            "start": (re.search(r"StartTime:\s*(.+)", txt).group(1).strip() if "StartTime" in txt else None),
            "ip_packets": g(r"\n\s*ip\s+(\d+)", int), "ip6_packets": g(r"\n\s*ip6\s+(\d+)", int)}
