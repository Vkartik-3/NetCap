import json

import numpy as np
import polars as pl
import pytest

from netcap.data import cesnet as ce
from netcap.db import load as db
from netcap.store.parquet import TableWriter

from .conftest import needs_abilene


def frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"k": np.arange(n), "v": rng.normal(size=n)})


def write_all(root, frames):
    w = TableWriter(root)
    for rel, df in frames.items():
        w.write(df, rel)
    return w.finish()


def test_parquet_rerun_changes_nothing(tmp_path):
    frames = {"p=1/part-000.parquet": frame(seed=1), "p=2/part-000.parquet": frame(seed=2)}
    s1 = write_all(tmp_path, frames)
    mt = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    s2 = write_all(tmp_path, frames)
    assert s1["created"] == 2 and s2["unchanged"] == 2 and s2["created"] == s2["replaced"] == 0
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()} == mt   # no file rewritten


def test_parquet_changed_and_stale_files(tmp_path):
    write_all(tmp_path, {"p=1/part-000.parquet": frame(seed=1), "p=2/part-000.parquet": frame(seed=2)})
    s = write_all(tmp_path, {"p=1/part-000.parquet": frame(seed=9)})
    assert s["replaced"] == 1 and s["removed_stale"] == ["p=2/part-000.parquet"]
    assert pl.read_parquet(tmp_path / "p=1/part-000.parquet").equals(frame(seed=9))
    assert json.loads((tmp_path / "_manifest.json").read_text())["rows"] == 1000


def make_cesnet(root):
    (root / "times").mkdir(parents=True)
    (root / "institutions" / "agg_10_minutes").mkdir(parents=True)
    (root / "times" / "times_10_minutes.csv").write_text(
        "id_time,time\n0,2023-10-29 02:50:00+02:00\n1,2023-10-29 02:00:00+01:00\n2,2023-10-31 23:59:00+01:00\n")
    hdr = "id_time," + ",".join(ce.METRICS) + "\n"
    row = lambda t, b: f"{t}," + ",".join(str(b) if m == "n_bytes" else "1" for m in ce.METRICS) + "\n"
    (root / "institutions" / "agg_10_minutes" / "3.csv").write_text(hdr + row(0, 10) + row(2, 30))
    (root / "institutions" / "agg_10_minutes" / "12.csv").write_text(hdr + row(1, 20))


def test_cesnet_reader_utc_and_order(tmp_path):
    make_cesnet(tmp_path)
    tmap = ce.times("10_minutes", tmp_path)
    assert [str(x) for x in tmap["ts"]] == ["2023-10-29 00:50:00+00:00", "2023-10-29 01:00:00+00:00",
                                            "2023-10-31 22:59:00+00:00"]           # DST change handled
    df = ce.read_batch(ce.institution_files("10_minutes", tmp_path), "10_minutes", tmap)
    assert df["institution_id"].to_list() == [3, 3, 12] and df["n_bytes"].to_list() == [10, 30, 20]
    assert df.columns == ce.COLUMNS


def test_cesnet_unknown_time_id_fails(tmp_path):
    make_cesnet(tmp_path)
    f = tmp_path / "institutions" / "agg_10_minutes" / "12.csv"
    f.write_text(f.read_text() + "99," + ",".join("1" for _ in ce.METRICS) + "\n")
    with pytest.raises(ValueError, match="missing from times"):
        ce.read_batch(ce.institution_files("10_minutes", tmp_path), "10_minutes", ce.times("10_minutes", tmp_path))


def test_cesnet_to_parquet_partitions_by_month(tmp_path):
    from netcap.pipeline.to_parquet import cesnet_to_parquet
    make_cesnet(tmp_path / "raw")
    r1 = cesnet_to_parquet(tmp_path / "store", tmp_path / "raw", aggs=("10_minutes",))
    r2 = cesnet_to_parquet(tmp_path / "store", tmp_path / "raw", aggs=("10_minutes",))
    files = sorted(str(p.relative_to(tmp_path / "store")) for p in (tmp_path / "store").rglob("*.parquet"))
    assert files == ["cesnet/series/aggregation=10_minutes/month=2023-10/part-000.parquet"]
    assert r1["rows_total"] == r2["rows_total"] == 3 and r2["table"]["unchanged"] == 1


def tm_frame(values):
    return pl.DataFrame({"dataset": ["t"] * len(values), "step": list(range(len(values))),
                         "flow_id": [1] * len(values), "value_raw": values},
                        schema={"dataset": pl.String, "step": pl.Int32, "flow_id": pl.Int16, "value_raw": pl.Float64})


def test_postgres_upsert_idempotent(pg):
    cols = ["dataset", "step", "flow_id", "value_raw"]
    seen, changed = db.upsert(pg, "traffic_matrix", [tm_frame([1.0, 2.0, 3.0])], cols)
    fp = db.table_fingerprint(pg, "traffic_matrix")
    assert (seen, changed) == (3, 3)
    assert db.upsert(pg, "traffic_matrix", [tm_frame([1.0, 2.0, 3.0])], cols) == (3, 0)
    assert db.table_fingerprint(pg, "traffic_matrix") == fp
    assert db.upsert(pg, "traffic_matrix", [tm_frame([1.0, 2.5, 3.0])], cols) == (3, 1)     # one corrected row
    assert db.table_fingerprint(pg, "traffic_matrix") != fp


def test_postgres_nulls_and_timestamps_round_trip(pg):
    from datetime import datetime, timezone
    cols = ["dataset", "step", "week", "step_in_week", "ts_assumed", "keep", "reason", "segment_id"]
    df = pl.DataFrame({"dataset": ["abilene"] * 2, "step": [0, 1], "week": [1, 1], "step_in_week": [0, 1],
                       "ts_assumed": [datetime(2004, 3, 1, 0, 0), datetime(2004, 3, 1, 0, 5)],
                       "keep": [True, False], "reason": [None, "router"], "segment_id": [0, -1]})
    db.upsert(pg, "time_steps", [df], cols)
    rows = pg.execute("SELECT step, ts_assumed, keep, reason FROM time_steps ORDER BY step").fetchall()
    assert rows == [(0, datetime(2004, 3, 1, 0, 0), True, None), (1, datetime(2004, 3, 1, 0, 5), False, "router")]
    ce_cols = ["aggregation", "institution_id", "id_time", "ts", *ce.METRICS]
    c = pl.DataFrame({"aggregation": ["10_minutes"], "institution_id": [1], "id_time": [0],
                      "ts": [datetime(2023, 10, 29, 0, 50, tzinfo=timezone.utc)],
                      **{m: [1] for m in ce.METRICS}}).with_columns(pl.col("ts").dt.convert_time_zone("UTC"))
    db.upsert(pg, "cesnet_series", [c], ce_cols)
    assert pg.execute("SELECT ts FROM cesnet_series").fetchone()[0] == datetime(2023, 10, 29, 0, 50, tzinfo=timezone.utc)


@needs_abilene
def test_loaded_postgres_matches_raw(live_db):
    from netcap.data import abilene as ab
    n = live_db.execute("SELECT count(*) FROM traffic_matrix WHERE dataset = 'abilene'").fetchone()[0]
    if n == 0:
        pytest.skip("Abilene not loaded; run python -m netcap.pipeline.ingest postgres abilene")
    w = ab.load_weeks().reshape(-1, ab.N_FLOWS)
    assert n == w.size
    rows = live_db.execute("SELECT step, flow_id, value_raw FROM traffic_matrix WHERE dataset = 'abilene' "
                           "AND step IN (0, 20000, 48383) ORDER BY step, flow_id").fetchall()
    for s, f, v in rows:
        assert v == w[s, f]
    kept = live_db.execute("SELECT count(*) FROM time_steps WHERE dataset='abilene' AND keep").fetchone()[0]
    assert kept == 40855
    t = ab.load_topology()
    ll = live_db.execute("SELECT load_raw FROM link_load WHERE dataset='abilene' AND step = 1234 "
                         "ORDER BY link_id").fetchall()
    assert np.allclose([r[0] for r in ll], t.A @ w[1234], rtol=1e-12)


def test_every_table_in_the_schema_has_upsert_keys():
    import re
    from netcap.db import load as db
    tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", db.SCHEMA_SQL.read_text()))
    assert tables - {"ingest_runs"} <= set(db.PKEYS)          # ingest_runs is append-only
