-- NetCap schema. Safe to apply repeatedly (IF NOT EXISTS everywhere).
-- Units: Abilene value_raw / load_raw are in 100 bytes per 5 minutes (raw/readme.txt).
-- Abilene ts_assumed is a file-clock timestamp: each week file is ASSUMED to start at 00:00 on its
-- README date; no time zone is documented, so it is stored WITHOUT time zone and is not unique.

CREATE TABLE IF NOT EXISTS datasets (
    dataset      text PRIMARY KEY,
    description  text NOT NULL,
    source_url   text NOT NULL,
    license      text NOT NULL,
    unit         text NOT NULL,
    provenance   text NOT NULL            -- 'real' | 'simulated' | 'injected' (rule 4)
);

CREATE TABLE IF NOT EXISTS nodes (
    dataset  text REFERENCES datasets,
    node_id  integer NOT NULL,
    name     text NOT NULL,
    city     text,
    lat      double precision,
    lon      double precision,
    PRIMARY KEY (dataset, name)
);

CREATE TABLE IF NOT EXISTS links (
    dataset        text REFERENCES datasets,
    link_id        integer NOT NULL,
    name           text NOT NULL,
    src            text NOT NULL,
    dst            text NOT NULL,
    link_type      smallint NOT NULL,     -- 0 internal, 1 inbound (entry at dst), 2 outbound (exit at src)
    capacity_kbps  double precision,      -- internal links only
    igp_weight     integer,               -- OSPF weight, internal links only
    PRIMARY KEY (dataset, link_id)
);

CREATE TABLE IF NOT EXISTS flows (
    dataset  text REFERENCES datasets,
    flow_id  integer NOT NULL,            -- src_index * 12 + dst_index
    src      text NOT NULL,
    dst      text NOT NULL,
    PRIMARY KEY (dataset, flow_id)
);

CREATE TABLE IF NOT EXISTS routing (
    dataset   text REFERENCES datasets,
    link_id   integer NOT NULL,
    flow_id   integer NOT NULL,
    fraction  double precision NOT NULL,
    PRIMARY KEY (dataset, link_id, flow_id)
);

CREATE TABLE IF NOT EXISTS time_steps (
    dataset       text REFERENCES datasets,
    step          integer NOT NULL,       -- week_index * 2016 + step_in_week
    week          smallint NOT NULL,
    step_in_week  smallint NOT NULL,
    ts_assumed    timestamp NOT NULL,
    keep          boolean NOT NULL,       -- false = removed by cleaning rule `reason`
    reason        text,                   -- duplicate | flow | router | impossible | NULL
    segment_id    integer NOT NULL,       -- -1 when removed
    PRIMARY KEY (dataset, step)
);

CREATE TABLE IF NOT EXISTS traffic_matrix (
    dataset    text NOT NULL,
    step       integer NOT NULL,
    flow_id    smallint NOT NULL,
    value_raw  double precision NOT NULL,
    PRIMARY KEY (dataset, step, flow_id)
);

CREATE TABLE IF NOT EXISTS link_load (
    dataset      text NOT NULL,
    step         integer NOT NULL,
    link_id      smallint NOT NULL,
    load_raw     double precision NOT NULL,
    load_mbps    double precision NOT NULL,
    utilization  double precision,        -- load / capacity, internal links only
    PRIMARY KEY (dataset, step, link_id)
);

CREATE TABLE IF NOT EXISTS forecasts (
    dataset        text NOT NULL,
    model          text NOT NULL,
    horizon_steps  integer NOT NULL,
    fold           text NOT NULL,
    entity_kind    text NOT NULL,         -- flow | link | institution
    entity_id      integer NOT NULL,
    target_step    bigint NOT NULL,       -- Abilene step, or CESNET id_time
    point          double precision NOT NULL,
    lo             double precision,
    hi             double precision,
    run_id         text NOT NULL,
    PRIMARY KEY (dataset, model, horizon_steps, fold, entity_kind, entity_id, target_step)
);

CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id  text PRIMARY KEY,
    kind         text NOT NULL,           -- growth | failure | upgrade | combined | te
    params       jsonb NOT NULL,
    result       jsonb,
    assumptions  text NOT NULL,           -- every scenario states what is assumed (rule 4)
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS anomalies (
    dataset      text NOT NULL,
    detector     text NOT NULL,
    entity_kind  text NOT NULL,
    entity_id    integer NOT NULL,
    target_step  bigint NOT NULL,
    score        double precision NOT NULL,
    is_anomaly   boolean NOT NULL,
    injected     boolean NOT NULL,        -- true = synthetic anomaly injected for evaluation (rule 4)
    run_id       text NOT NULL,
    PRIMARY KEY (dataset, detector, entity_kind, entity_id, target_step, injected)
);

CREATE TABLE IF NOT EXISTS cesnet_series (
    aggregation           text NOT NULL,  -- 10_minutes | 1_hour | 1_day
    institution_id        integer NOT NULL,
    id_time               integer NOT NULL,
    ts                    timestamptz NOT NULL,
    n_flows               bigint, n_packets bigint, n_bytes bigint,
    sum_n_dest_asn        bigint, average_n_dest_asn double precision, std_n_dest_asn double precision,
    sum_n_dest_ports      bigint, average_n_dest_ports double precision, std_n_dest_ports double precision,
    sum_n_dest_ip         bigint, average_n_dest_ip double precision, std_n_dest_ip double precision,
    tcp_udp_ratio_packets double precision, tcp_udp_ratio_bytes double precision,
    dir_ratio_packets     double precision, dir_ratio_bytes double precision,
    avg_duration          double precision, avg_ttl double precision,
    PRIMARY KEY (aggregation, institution_id, id_time)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id        bigserial PRIMARY KEY,
    dataset       text NOT NULL,
    target        text NOT NULL,          -- postgres | parquet
    started_at    timestamptz NOT NULL,
    finished_at   timestamptz NOT NULL,
    rows_seen     bigint NOT NULL,
    rows_changed  bigint NOT NULL,        -- inserted + updated; 0 on an identical re-run
    wall_s        double precision NOT NULL,
    details       jsonb NOT NULL
);

-- MAWI (WIDE, samplepoint-F): REAL flow data, unrelated to Abilene. Only derived aggregates are stored;
-- the pcap is deleted after conversion. Addresses are anonymised by the archive.
CREATE TABLE IF NOT EXISTS mawi_traces (
    trace_id        text PRIMARY KEY,         -- MAWI id, e.g. 202510011400
    source_url      text NOT NULL,
    start_ts        timestamptz NOT NULL,     -- first flow seen
    end_ts          timestamptz NOT NULL,     -- last flow seen
    flows           bigint NOT NULL,
    packets         bigint NOT NULL,          -- as counted in flow records
    bytes           bigint NOT NULL,          -- IP bytes as counted by nfpcapd
    pcap_packets    bigint,                   -- MAWI's own trace summary
    flow_tool       text NOT NULL             -- tool and settings that made the flow records
);

CREATE TABLE IF NOT EXISTS mawi_protocol_mix (
    trace_id      text NOT NULL REFERENCES mawi_traces,
    proto         text NOT NULL,
    flows         bigint NOT NULL,
    packets       bigint NOT NULL,
    bytes         bigint NOT NULL,
    byte_share    double precision NOT NULL,
    packet_share  double precision NOT NULL,
    PRIMARY KEY (trace_id, proto)
);

CREATE TABLE IF NOT EXISTS mawi_top_talkers (
    trace_id    text NOT NULL REFERENCES mawi_traces,
    kind        text NOT NULL,                -- src | dst | pair | dport (key = proto/port)
    rank        integer NOT NULL,
    key         text NOT NULL,
    flows       bigint NOT NULL,
    packets     bigint NOT NULL,
    bytes       bigint NOT NULL,
    byte_share  double precision NOT NULL,
    PRIMARY KEY (trace_id, kind, rank)
);

CREATE TABLE IF NOT EXISTS mawi_throughput (
    trace_id      text NOT NULL REFERENCES mawi_traces,
    bin_start     timestamptz NOT NULL,
    bin_s         integer NOT NULL,
    bytes         double precision NOT NULL,  -- flow bytes spread uniformly over each flow's duration
    mbps          double precision NOT NULL,
    flows_active  bigint NOT NULL,
    PRIMARY KEY (trace_id, bin_start)
);
