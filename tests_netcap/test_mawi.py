import struct

import numpy as np
import polars as pl

from netcap.data import ipfix, mawi


def flows(rows):
    """rows: (t_first, t_last, proto, src_v4, dst_v4, sport, dport, packets, bytes)"""
    cols = list(zip(*rows))
    return pl.DataFrame({"ip_version": np.full(len(rows), 4, np.uint8), "src_hi": np.zeros(len(rows), np.uint64),
                         "src_lo": np.array(cols[3], np.uint64), "dst_hi": np.zeros(len(rows), np.uint64),
                         "dst_lo": np.array(cols[4], np.uint64), "proto": np.array(cols[2], np.uint8),
                         "sport": np.array(cols[5], np.uint16), "dport": np.array(cols[6], np.uint16),
                         "icmp": np.array([p == 1 for p in cols[2]]), "t_first": np.array(cols[0], float),
                         "t_last": np.array(cols[1], float), "packets": np.array(cols[7], np.uint64),
                         "bytes": np.array(cols[8], np.uint64)})


def test_throughput_spreads_bytes_over_duration():
    df = flows([(100.0, 100.0, 17, 1, 2, 1, 53, 1, 50),       # instant flow: all in its bin
                (100.0, 120.0, 6, 1, 3, 2, 443, 10, 2000),    # 20 s over two 10 s bins
                (105.0, 110.0, 6, 4, 3, 3, 443, 2, 100)])     # half in bin 0, ends on the boundary
    starts, by, active = mawi.throughput(df, bin_s=10)
    assert starts[0] == 100.0
    assert np.allclose(by, [50 + 1000 + 100, 1000, 0])
    assert by.sum() == df["bytes"].sum()
    assert active.tolist() == [3, 2, 1]


def test_aggregates_shares_ranks_and_keys():
    df = flows([(0.0, 1.0, 6, 0x0A000001, 0x0A000002, 1, 443, 10, 900),
                (0.0, 1.0, 1, 0x0A000003, 0x0A000002, 0, 3 * 256 + 1, 1, 100)])   # ICMP type 3 code 1
    tot, proto, talkers, tp = mawi.aggregates(df, "t")
    assert (tot["flows"], tot["bytes"]) == (2, 1000)
    assert proto["proto"].to_list() == ["TCP", "ICMP"] and proto["byte_share"].to_list() == [0.9, 0.1]
    assert talkers.filter(pl.col("kind") == "dst")["key"].to_list() == ["10.0.0.2"]
    assert talkers.filter(pl.col("kind") == "dport")["key"].to_list() == ["TCP/443", "ICMP/3.1"]
    assert talkers.filter(pl.col("kind") == "pair")["key"][0] == "10.0.0.1->10.0.0.2"
    assert tp["bytes"].sum() == 1000


def test_ipfix_decode_round_trip(tmp_path):
    """One template (src/dst v4, start/end ms, octets, packets, proto, ports) and two data records."""
    fields = [(8, 4), (12, 4), (152, 8), (153, 8), (1, 4), (2, 4), (4, 1), (7, 2), (11, 2)]
    tmpl = struct.pack(">HH", 256, len(fields)) + b"".join(struct.pack(">HH", i, l) for i, l in fields)
    tset = struct.pack(">HH", 2, 4 + len(tmpl)) + tmpl
    rec = lambda s, d, t, o, p, pr, sp, dp: struct.pack(">IIQQIIBHH", s, d, t, t + 5, o, p, pr, sp, dp)
    data = rec(0x0A000001, 0x0A000002, 1_759_294_800_000, 1500, 3, 6, 1234, 443) + \
        rec(0xC0A80001, 0x08080808, 1_759_294_801_000, 80, 1, 17, 5353, 53)
    dset = struct.pack(">HH", 256, 4 + len(data)) + data
    body = tset + dset
    msg = struct.pack(">HHIII", 10, 16 + len(body), 0, 1, 0) + body
    (tmp_path / "s.ipfix").write_bytes(msg)
    recs, st = ipfix.decode(tmp_path / "s.ipfix")
    df = mawi.records_to_frame(recs)
    assert st["records"] == {256: 2} and st["trailing_bytes"] == 0
    assert df["bytes"].to_list() == [1500, 80] and df["packets"].to_list() == [3, 1]
    assert df["dport"].to_list() == [443, 53] and df["proto"].to_list() == [6, 17]
    assert df["t_first"][0] == 1_759_294_800.0 and df["t_last"][0] == 1_759_294_800.005
    assert mawi._fmt_addr(4, df["src_hi"][1], df["src_lo"][1]) == "192.168.0.1"


def test_parse_trace_info():
    html = ("<pre>DumpFile:  x.pcap\nStartTime: Wed Oct  1 14:00:00 2025\nTotalTime: 899.95 seconds\n"
            "TotalCapSize: 10847.62MB  CapLen: 96 bytes\n# of packets: 207553081 (175448.49MB)\n"
            "AvgRate: 1635.64Mbps  stddev:296.44M\n total  207553081 (100.00%)\n ip  176328212 ( 84.96%)\n"
            " ip6  30974349 ( 14.92%)\n</pre>")
    i = mawi.parse_trace_info(html)
    assert i["packets"] == 207553081 and i["duration_s"] == 899.95 and i["caplen_bytes"] == 96
    assert i["ip_packets"] == 176328212 and i["ip6_packets"] == 30974349 and i["avg_rate_mbps"] == 1635.64


def random_flows(n, seed=0):
    rng = np.random.default_rng(seed)
    t = 1_759_294_800 + rng.uniform(0, 300, n)
    return flows(list(zip(t, t + rng.exponential(5, n) * (rng.random(n) < 0.7), rng.choice([6, 17, 1, 50], n),
                          rng.integers(1, 40, n), rng.integers(1, 40, n), rng.integers(0, 5, n),
                          rng.choice([53, 80, 443, 3 * 256 + 1], n), rng.integers(1, 100, n),
                          rng.integers(40, 150_000, n))))


def test_chunked_aggregates_equal_in_memory(tmp_path):
    df = random_flows(5000)
    want = mawi.aggregates(df, "t", top_n=7)
    ca = mawi.ChunkedAggregates(tmp_path / "spill", memory_limit="256MB", threads=1)
    for part in (df[2000:3500], df[:2000], df[3500:]):      # out of time order, as softflowd exports
        ca.add(part)
    (tot, proto, talkers, tp), info = ca.finish("t", top_n=7)
    assert tot == want[0]
    assert proto.equals(want[1]) and talkers.equals(want[2])
    assert tp["bin_start"].equals(want[3]["bin_start"]) and tp["flows_active"].equals(want[3]["flows_active"])
    assert np.allclose(tp["bytes"].to_numpy(), want[3]["bytes"].to_numpy(), rtol=1e-12)
    assert info["chunks"] == 3 and not (tmp_path / "spill").exists()          # spill files removed


def test_iter_decode_chunks_match_whole_decode(tmp_path):
    fields = [(8, 4), (12, 4), (152, 8), (153, 8), (1, 4), (2, 4), (4, 1), (7, 2), (11, 2)]
    tmpl = struct.pack(">HH", 256, len(fields)) + b"".join(struct.pack(">HH", i, l) for i, l in fields)
    msgs = [struct.pack(">HHIII", 10, 16 + 4 + len(tmpl), 0, 0, 0) + struct.pack(">HH", 2, 4 + len(tmpl)) + tmpl]
    for m in range(50):                                      # 50 messages of 3 records each
        data = b"".join(struct.pack(">IIQQIIBHH", m, r, 1000 * m, 1000 * m + r, 100 + r, 1, 6, r, 443) for r in range(3))
        msgs.append(struct.pack(">HHIII", 10, 16 + 4 + len(data), 0, m + 1, 0) + struct.pack(">HH", 256, 4 + len(data)) + data)
    (tmp_path / "s.ipfix").write_bytes(b"".join(msgs))
    whole, st = ipfix.decode(tmp_path / "s.ipfix")
    chunks = list(ipfix.iter_decode(tmp_path / "s.ipfix", chunk_records=7, block_bytes=100))   # blocks split messages
    assert len(chunks) == 17                                 # a chunk closes on whole messages: 9 records each
    got = np.concatenate([c[256] for c in chunks])
    assert np.array_equal(got, whole[256]) and len(got) == 150 and st["trailing_bytes"] == 0
    assert (st["first_seq"], st["last_seq"], st["messages"]) == (0, 50, 51)
