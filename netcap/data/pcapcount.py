"""Independent pcap record counter used to check flow-meter output (no packet decoding beyond Ethernet).

  gzcat trace.pcap.gz | python -m netcap.data.pcapcount
Prints JSON: records, sum of original (wire) lengths, first/last timestamp, link type, snaplen,
and how many records are IPv4 / IPv6 / other by EtherType (with up to two 802.1Q tags skipped).
"""
import json
import struct
import sys

ETH_V4, ETH_V6, ETH_VLAN = 0x0800, 0x86DD, (0x8100, 0x88A8)


def count(stream):
    gh = stream.read(24)
    magic = struct.unpack("<I", gh[:4])[0]
    endian = {0xA1B2C3D4: "<", 0xA1B23C4D: "<", 0xD4C3B2A1: ">", 0x4D3CB2A1: ">"}[magic]
    nano = magic in (0xA1B23C4D, 0x4D3CB2A1)
    snaplen, linktype = struct.unpack(endian + "II", gh[16:24])
    rec = struct.Struct(endian + "IIII")
    n = wire = v4 = v6 = other = 0
    first = last = None
    buf = b""
    while True:
        chunk = stream.read(1 << 24)
        if not chunk:
            break
        buf += chunk
        off = 0
        while off + 16 <= len(buf):
            ts, frac, incl, orig = rec.unpack_from(buf, off)
            if off + 16 + incl > len(buf):
                break
            if linktype == 1:
                p = off + 16 + 12
                et = int.from_bytes(buf[p:p + 2], "big") if incl >= 14 else 0
                for _ in range(2):
                    if et in ETH_VLAN and incl >= p - off - 16 + 6:
                        p += 4
                        et = int.from_bytes(buf[p:p + 2], "big")
                v4 += et == ETH_V4
                v6 += et == ETH_V6
                other += et not in (ETH_V4, ETH_V6)
            t = ts + frac / (1e9 if nano else 1e6)
            first = t if first is None else first
            last = t
            n += 1
            wire += orig
            off += 16 + incl
        buf = buf[off:]
    return {"records": n, "wire_bytes": wire, "first_ts": first, "last_ts": last, "linktype": linktype,
            "snaplen": snaplen, "ethertype_ipv4": v4, "ethertype_ipv6": v6, "ethertype_other": other,
            "trailing_partial_bytes": len(buf)}


if __name__ == "__main__":
    print(json.dumps(count(sys.stdin.buffer)))
