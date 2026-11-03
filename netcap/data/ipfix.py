"""Lossless flow export: softflowd -> IPFIX over TCP -> raw stream file -> decoded records.

UDP export to a collector dropped ~25% of records in testing (sequence errors), so NetCap uses TCP:
backpressure slows the exporter instead of losing records. `collect` saves the raw IPFIX byte stream;
`iter_decode` parses it in bounded-memory chunks (RFC 7011: message header, template sets id 2, data sets id >= 256) with numpy
structured dtypes, so it handles any fixed-length template softflowd sends.
"""
import socket
import struct
from collections import defaultdict

import numpy as np

# IPFIX information elements used by softflowd (RFC 7012 / IANA registry)
IE = {1: "octetDeltaCount", 2: "packetDeltaCount", 4: "protocolIdentifier", 5: "ipClassOfService",
      6: "tcpControlBits", 7: "sourceTransportPort", 8: "sourceIPv4Address", 10: "ingressInterface",
      11: "destinationTransportPort", 12: "destinationIPv4Address", 14: "egressInterface",
      27: "sourceIPv6Address", 28: "destinationIPv6Address", 32: "icmpTypeCodeIPv4", 58: "vlanId",
      60: "ipVersion", 136: "flowEndReason", 139: "icmpTypeCodeIPv6", 150: "flowStartSeconds",
      151: "flowEndSeconds", 152: "flowStartMilliseconds", 153: "flowEndMilliseconds",
      154: "flowStartMicroseconds", 155: "flowEndMicroseconds", 156: "flowStartNanoseconds",
      157: "flowEndNanoseconds", 22: "flowStartSysUpTime", 21: "flowEndSysUpTime",
      160: "systemInitTimeMilliseconds"}


def collect(port, out_path, host="127.0.0.1", ready=None):
    """Accept one TCP connection and write everything received to out_path. Returns bytes received."""
    srv = socket.create_server((host, port))
    if ready is not None:
        ready.set()
    conn, _ = srv.accept()
    n = 0
    with open(out_path, "wb") as f:
        while chunk := conn.recv(1 << 20):
            f.write(chunk)
            n += len(chunk)
    conn.close()
    srv.close()
    return n


def _dtype(fields):
    """numpy dtype for a template of (ie_id, length) fields, big-endian."""
    out = []
    for i, (ie, ln) in enumerate(fields):
        name = IE.get(ie, f"ie{ie}")
        if ln in (1, 2, 4, 8) and ie not in (8, 12, 27, 28):
            out.append((name, f">u{ln}"))
        else:
            out.append((name, f"V{ln}"))      # addresses and odd lengths: raw bytes
    return np.dtype(out)


class _Stats:
    def __init__(self):
        self.messages = self.bytes = self.trailing = 0
        self.first_seq = self.last_seq = None
        self.templates, self.records = {}, defaultdict(int)

    def as_dict(self):
        return {"messages": self.messages, "bytes": self.bytes, "trailing_bytes": self.trailing,
                "templates": {tid: list(dt.names) for tid, dt in self.templates.items()},
                "first_seq": self.first_seq, "last_seq": self.last_seq, "records": dict(self.records)}


def _parse_templates(b, q, send, options, templates):
    while q + 4 <= send:
        tid, cnt = struct.unpack_from(">HH", b, q)
        q += 6 if options else 4                           # options templates add a scope field count
        if cnt == 0:                                       # padding or template withdrawal
            break
        fields = []
        for _ in range(cnt):
            ie, ln = struct.unpack_from(">HH", b, q)
            q += 4
            if ie & 0x8000:                                # enterprise-specific: skip enterprise number
                q += 4
                ie &= 0x7FFF
            fields.append((ie, ln))
        templates[tid] = _dtype(fields)


def iter_decode(path, chunk_records=4_000_000, block_bytes=64 << 20, stats=None):
    """Decode an IPFIX stream with bounded memory.

    Reads the file in blocks of about `block_bytes` (whole messages only; a partial message is carried
    into the next block) and yields {template id: structured array} with about `chunk_records` records
    per yield. Memory is bounded by one block plus one chunk. `stats` (a _Stats) is filled as it goes.
    """
    st = stats if stats is not None else _Stats()
    pending, n_pending = defaultdict(list), 0
    with open(path, "rb") as f:
        carry = b""
        while True:
            data = f.read(block_bytes)
            b = carry + data
            off = 0
            while off + 16 <= len(b):
                ver, length, _export_t, seq, _domain = struct.unpack_from(">HHIII", b, off)
                if ver != 10 or length < 16:
                    raise ValueError(f"bad IPFIX message at byte {st.bytes + off}: version {ver}, length {length}")
                if off + length > len(b):
                    break                                  # message continues in the next block
                if st.first_seq is None:
                    st.first_seq = seq
                st.last_seq = seq
                p, end = off + 16, off + length
                while p + 4 <= end:
                    sid, slen = struct.unpack_from(">HH", b, p)
                    if sid in (2, 3):                      # template set / options template set
                        _parse_templates(b, p + 4, p + slen, sid == 3, st.templates)
                    elif sid >= 256:
                        dt = st.templates[sid]
                        n = (slen - 4) // dt.itemsize
                        pending[sid].append(np.frombuffer(b, dtype=dt, count=n, offset=p + 4).copy())
                        st.records[sid] += n
                        n_pending += n
                    p += slen
                off = end
                st.messages += 1
                if n_pending >= chunk_records:
                    yield {tid: np.concatenate(c) for tid, c in pending.items()}
                    pending, n_pending = defaultdict(list), 0
            st.bytes += off
            carry = b[off:]
            if not data:
                break
    st.trailing = len(carry)
    if n_pending:
        yield {tid: np.concatenate(c) for tid, c in pending.items()}


def decode(path):
    """Decode a whole IPFIX stream into memory. Returns (records_by_template {tid: structured array}, stats).

    For small streams and tests; the pipeline uses iter_decode.
    """
    st = _Stats()
    parts = defaultdict(list)
    for chunk in iter_decode(path, stats=st):
        for tid, a in chunk.items():
            parts[tid].append(a)
    recs = {tid: np.concatenate(c) for tid, c in parts.items()}
    return recs, st.as_dict()
