"""Build the scenario engine's inputs from the Parquet store (no raw files needed: used by the API)."""
import numpy as np
import polars as pl

from .. import config
from ..data.abilene import TO_MBIT_S
from .engine import Net

KINDS = {0: "core", 1: "ingress", 2: "egress"}


def load_net(root=None):
    root = root or config.DATA / "abilene"
    nodes = pl.read_parquet(root / "nodes" / "*.parquet").sort("node_id")
    links = pl.read_parquet(root / "links" / "*.parquet").sort("link_id")
    ls = [(r["link_id"], r["src"], r["dst"], None if r["capacity_kbps"] is None else r["capacity_kbps"] / 1000,
           r["igp_weight"], KINDS[r["link_type"]]) for r in links.iter_rows(named=True)]
    net = Net(nodes["name"].to_list(), ls)
    net.coords = {r["name"]: (r["lat"], r["lon"], r["city"]) for r in nodes.iter_rows(named=True)}
    return net


def load_A(net, root=None):
    root = root or config.DATA / "abilene"
    r = pl.read_parquet(root / "routing" / "*.parquet")
    A = np.zeros((len(net.links), net.n_flows))
    idx = net.link_index()
    A[[idx[int(i)] for i in r["link_id"].to_list()], r["flow_id"].to_numpy()] = r["fraction"].to_numpy()
    return A


def load_demand(root=None):
    """Measured demand (kept steps x 144) in Mbit/s and the step ids."""
    root = root or config.DATA / "abilene"
    ts = pl.read_parquet(root / "time_steps" / "*.parquet").filter(pl.col("keep")).sort("step")
    tm = (pl.scan_parquet(root / "traffic_matrix" / "*" / "*.parquet")
          .join(ts.lazy().select("step"), on="step").sort(["step", "flow_id"]).collect())
    steps = ts["step"].to_numpy()
    D = tm["value_raw"].to_numpy().reshape(len(steps), -1) * TO_MBIT_S
    return D, steps, ts
