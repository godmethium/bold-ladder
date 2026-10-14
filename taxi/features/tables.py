import numpy as np
import pandas as pd

import config

QUEUE = ["QUE_TO_APT_WIN", "QUE_ARR_WIN", "QUE_AHEAD_APT", "QUE_PUSH_20", "QUE_TO_RWY_WIN", "QUE_AHEAD_RWY", "QUE_PREV_RWY",
         "QUE_NEXT_RWY", "QUE_RWY_RATE", "QUE_FREE"]
GROUPS = [
    ("delay", ["DLY_EOBT_W15", "DLY_EOBT_W30", "DLY_SCHED_W60", "DLY_N_W15", "DLY_OWN_DIFF"]),
    ("delay_runway", ["DLY_RWY_EOBT_W15", "DLY_RWY_EOBT_W30", "DLY_RWY_N_W15", "DLY_RWY_DIFF_W15"]),
    ("arrivals_weather", ["ARRV_APT_15", "ARRV_RWY_15", "ARRV_RWY_SHARE_60", "WX_FZ_0", "WX_FG_0", "WX_SN_0", "WX_SN_3H",
                          "WX_TS_0", "WX_VIS", "WX_WIND", "WX_GUST"]),
    ("forward_windows", ["DLYC_EOBT_15", "DLYF_EOBT_15", "NMT_C15", "NMT_F15", "DEP_APT_AFTER_10", "DEP_RWY_AFTER_10",
                         "ARR_AFTER_10", "DLYC_RWY_15", "NMT_RWY_C15"]),
    ("stand_arrivals", ["IN_DELAY", "IN_TURN", "IN_BUFFER", "NX_GAP", "NX_T"]),
]
GROUP_DIR = config.WORK / "groups"


def table(name):
    return pd.read_parquet(GROUP_DIR / f"{name}.parquet")


def base(split):
    return pd.read_parquet(config.WORK / f"base_{split}.parquet")


def clean_mask(df):
    t = df["TAXITIME_SEC_mvt"]
    return ~((t <= 0) | (df["ADEP_mvt"].eq("LIRF") & (t < 100)))


def add_groups(*frames):
    q = table("queue")[["MVT_ID_mvt"] + QUEUE]
    q[QUEUE] = q[QUEUE].astype(np.float32)
    g = table(GROUPS[0][0])[["MVT_ID_mvt"] + GROUPS[0][1]]
    for name, cols in GROUPS[1:]:
        g = g.merge(table(name)[["MVT_ID_mvt"] + cols], on="MVT_ID_mvt", how="left", validate="one_to_one")
    out = []
    for d in frames:
        d = d.merge(q, on="MVT_ID_mvt", how="left", validate="one_to_one")
        out.append(d.merge(g, on="MVT_ID_mvt", how="left", validate="one_to_one"))
    return out


def _upper(s):
    return s.fillna("nan").astype(str).str.strip().str.upper()


def _te_keys(df):
    apt = df["ADEP_mvt"].astype(str)
    stand, runway = _upper(df["STAND_mvt"]), _upper(df["RUNWAY_mvt"])
    return {
        "STAND_RWY": apt + "_" + stand + "_" + runway,
        "OP_APT": apt + "_" + _upper(df["AIRCRAFT_OPERATOR_flt"]),
        "ROUTE": apt + "_" + _upper(df["ADES_mvt"]),
        "STAND_APT": apt + "_" + stand,
        "RWY_APT": apt + "_" + runway,
        "ACTYPE_APT": apt + "_" + _upper(df["AIRCRAFT_TYPE_mvt"]),
        "PIER_RWY": apt + "_" + stand.str.slice(0, 2) + "_" + runway,
        "HOUR_APT": apt + "_" + df["LOCAL_HOUR"].astype(str),
        "RWY_STAND_RAW": (apt + "_" + df["RUNWAY_mvt"].fillna("nan").astype(str) + "_"
                          + df["STAND_mvt"].fillna("nan").astype(str)),
    }


TE = [
    ("STAND_RWY_TE", "STAND_RWY", 20.0, "mean"), ("OP_APT_TE", "OP_APT", 20.0, "mean"),
    ("ROUTE_TE", "ROUTE", 20.0, "mean"), ("STAND_APT_TE", "STAND_APT", 25.0, "mean"),
    ("RWY_APT_TE", "RWY_APT", 30.0, "mean"), ("ACTYPE_APT_TE", "ACTYPE_APT", 20.0, "mean"),
    ("PIER_RWY_TE", "PIER_RWY", 30.0, "mean"), ("HOUR_APT_TE", "HOUR_APT", 50.0, "mean"),
    ("STAND_RWY_STD_TE", "STAND_RWY", 20.0, "std"), ("STAND_RWY_TE_OFFSET", "RWY_STAND_RAW", 15.0, "mean"),
]
TE_COLUMNS = [t[0] for t in TE]


def _text(k):
    s = pd.Series(k).astype(object)
    return s.where(s.notna(), "nan").astype(str).to_numpy(dtype=object)


def _smooth(n, s, ss, pn, ps, pss, m, stat):
    prior = ps / pn
    if stat == "mean":
        return (s + m * prior) / (n + m)
    prior_std = np.sqrt(np.maximum(pss - ps * ps / pn, 0.0) / (pn - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.maximum(ss - s * s / n, 0.0) / (n - 1)
    std = np.where(n >= 2, np.sqrt(var), prior_std)
    return (n * std + m * prior_std) / (n + m)


def group_oof_te(k_tr, y, groups, k_ap, m, stat):
    code, unique = pd.factorize(_text(k_tr))
    gk, _ = pd.factorize(np.asarray(groups), use_na_sentinel=False)
    y = np.asarray(y, float)
    nk, ng = len(unique), int(gk.max()) + 1
    b = code.astype(np.int64) * ng + gk
    c_kg = np.bincount(b, minlength=nk * ng).reshape(nk, ng).astype(float)
    s_kg = np.bincount(b, weights=y, minlength=nk * ng).reshape(nk, ng)
    q_kg = np.bincount(b, weights=y * y, minlength=nk * ng).reshape(nk, ng)
    c_k, s_k, q_k = c_kg.sum(1), s_kg.sum(1), q_kg.sum(1)
    c_g, s_g, q_g = c_kg.sum(0), s_kg.sum(0), q_kg.sum(0)
    N, S, Q = c_k.sum(), s_k.sum(), q_k.sum()
    tr = _smooth(c_k[code] - c_kg[code, gk], s_k[code] - s_kg[code, gk], q_k[code] - q_kg[code, gk],
                 N - c_g[gk], S - s_g[gk], Q - q_g[gk], m, stat)
    i = pd.Index(unique).get_indexer(_text(k_ap))
    seen = i >= 0
    j = np.maximum(i, 0)
    ap = _smooth(np.where(seen, c_k[j], 0.0), np.where(seen, s_k[j], 0.0), np.where(seen, q_k[j], 0.0), N, S, Q, m, stat)
    return tr.astype(np.float32), ap.astype(np.float32)


def add_te(train, test):
    y, month = train[config.TARGET].to_numpy(float), train["MVT_MONTH"].to_numpy()
    ke, kt = _te_keys(train), _te_keys(test)
    for col, key, m, stat in TE:
        train[col], test[col] = group_oof_te(ke[key], y, month, kt[key], m, stat)


def model_a_frames():
    e = base("train")
    e = e[e[config.TARGET].notna()]
    e = e[clean_mask(e).to_numpy()].reset_index(drop=True)
    t = base("test")
    t = t[t["NM_TAXI_DIFF_SEC"].notna() & t["MVT_MONTH"].isin(config.TEST_MONTHS)].reset_index(drop=True)
    e, t = add_groups(e, t)
    add_te(e, t)
    return e, t
