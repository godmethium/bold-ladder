import warnings

import numpy as np
import pandas as pd

import config
from taxi import data

COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
           "RUNWAY_mvt", "STAND_mvt", "EOBT_1_flt", "AOBT_3_flt"]
DELAY_MIN, DELAY_MAX = -3600.0, 14400.0
METAR_FILES = [config.EXTERNAL / "metar_hourly.parquet", config.EXTERNAL / "metar_hourly_extra.parquet"]


def ep(s):
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(float)


def _count_open(sorted_, low, high):
    n = np.searchsorted(sorted_, high, "left") - np.searchsorted(sorted_, low, "right")
    return np.where(np.isfinite(low) & np.isfinite(high), np.maximum(n, 0), np.nan)


def _airport_queue(dep, arr_t):
    T, O = dep["T"].to_numpy(float), dep["O"].to_numpy(float)
    out = {"QUE_TO_APT_WIN": _count_open(np.sort(T), O, T), "QUE_ARR_WIN": _count_open(np.sort(arr_t), O, T)}
    nm = np.isfinite(O)
    Os, Tn = np.sort(O[nm]), np.sort(T[nm])
    out["QUE_AHEAD_APT"] = np.where(nm, np.maximum(np.searchsorted(Os, O, "left") - np.searchsorted(Tn, O, "right"), 0), np.nan)
    out["QUE_PUSH_20"] = np.where(nm, np.searchsorted(Os, O + 600, "right") - np.searchsorted(Os, O - 600, "left") - 1, np.nan)
    for c in ("QUE_TO_RWY_WIN", "QUE_AHEAD_RWY", "QUE_PREV_RWY", "QUE_NEXT_RWY"):
        out[c] = np.full(len(dep), np.nan)
    for _, idx in dep.groupby("RUNWAY_mvt", dropna=False).indices.items():
        t, o = T[idx], O[idx]
        ts = np.sort(t)
        out["QUE_TO_RWY_WIN"][idx] = _count_open(ts, o, t)
        m = np.isfinite(o)
        os_, tn = np.sort(o[m]), np.sort(t[m])
        out["QUE_AHEAD_RWY"][idx] = np.where(m, np.maximum(np.searchsorted(os_, o, "left") - np.searchsorted(tn, o, "right"), 0),
                                             np.nan)
        i, j = np.searchsorted(ts, t, "left"), np.searchsorted(ts, t, "right")
        out["QUE_PREV_RWY"][idx] = t - np.where(i > 0, ts[np.clip(i - 1, 0, None)], np.nan)
        out["QUE_NEXT_RWY"][idx] = np.where(j < len(ts), ts[np.clip(j, None, len(ts) - 1)], np.nan) - t
    nm_min = (T - O) / 60.0
    out["QUE_RWY_RATE"] = np.where(nm_min > 0, out["QUE_TO_RWY_WIN"] / np.maximum(nm_min, 1.0), np.nan)
    out["QUE_FREE"] = (T - O) - 90.0 * out["QUE_TO_RWY_WIN"]
    return pd.DataFrame(out, index=dep.index)


def queue(df):
    df = df.copy()
    df["T"], df["O"] = ep(df["MVT_TIME_UTC_mvt"]), ep(df["AOBT_3_flt"])
    dep, arr = df[df["PHASE_mvt"].eq("DEP")], df[df["PHASE_mvt"].eq("ARR")]
    parts = [pd.concat([d[["MVT_ID_mvt"]], _airport_queue(d, arr.loc[arr["ADES_mvt"].eq(a), "T"].to_numpy(float))], axis=1)
             for a, d in dep.groupby("ADEP_mvt")]
    return pd.concat(parts, ignore_index=True)


def _window_median(T, v, W):
    lo, hi = np.searchsorted(T, T - W, "left"), np.searchsorted(T, T, "left")
    med, n = np.full(len(T), np.nan), np.zeros(len(T))
    for i in range(len(T)):
        a = v[lo[i]:hi[i]]
        a = a[np.isfinite(a)]
        if len(a):
            med[i], n[i] = np.median(a), len(a)
    return med, n


def delay(df):
    df = df[df["PHASE_mvt"].eq("DEP")].copy()
    df["T"] = ep(df["MVT_TIME_UTC_mvt"])
    df["d_eobt"] = np.clip(df["T"] - ep(df["EOBT_1_flt"]), DELAY_MIN, DELAY_MAX)
    df["d_sched"] = np.clip(df["T"] - ep(df["SCHED_TIME_UTC_mvt"]), DELAY_MIN, DELAY_MAX)
    parts = []
    for _, d in df.groupby("ADEP_mvt"):
        d = d.sort_values("T", kind="stable")
        T, e, s = d["T"].to_numpy(float), d["d_eobt"].to_numpy(float), d["d_sched"].to_numpy(float)
        g15, n15 = _window_median(T, e, 900.0)
        g30, _ = _window_median(T, e, 1800.0)
        s60, _ = _window_median(T, s, 3600.0)
        parts.append(pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy(), "DLY_EOBT_W15": g15, "DLY_EOBT_W30": g30,
                                   "DLY_SCHED_W60": s60, "DLY_N_W15": n15, "DLY_OWN_DIFF": e - g30}))
    return pd.concat(parts, ignore_index=True)


def delay_runway(df, dly):
    df = df[df["PHASE_mvt"].eq("DEP")].copy()
    df["T"] = ep(df["MVT_TIME_UTC_mvt"])
    df["d_eobt"] = np.clip(df["T"] - ep(df["EOBT_1_flt"]), DELAY_MIN, DELAY_MAX)
    df["runway"] = df["RUNWAY_mvt"].astype(str)
    parts = []
    for _, d in df.groupby(["ADEP_mvt", "runway"]):
        d = d.sort_values("T", kind="stable")
        T, e = d["T"].to_numpy(float), d["d_eobt"].to_numpy(float)
        g15, n15 = _window_median(T, e, 900.0)
        g30, _ = _window_median(T, e, 1800.0)
        parts.append(pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy(), "DLY_RWY_EOBT_W15": g15,
                                   "DLY_RWY_EOBT_W30": g30, "DLY_RWY_N_W15": n15}))
    out = pd.concat(parts, ignore_index=True).merge(dly[["MVT_ID_mvt", "DLY_EOBT_W15"]], on="MVT_ID_mvt", how="left")
    out["DLY_RWY_DIFF_W15"] = out["DLY_RWY_EOBT_W15"] - out["DLY_EOBT_W15"]
    return out.drop(columns="DLY_EOBT_W15")


def _count_half(s, a, b):
    return np.searchsorted(s, b, "left") - np.searchsorted(s, a, "left")


def metar():
    m = pd.concat([pd.read_parquet(f) for f in METAR_FILES if f.exists()], ignore_index=True)
    m["hour"] = pd.to_datetime(m["hour"])
    m["hour"] = m["hour"].dt.tz_localize("UTC") if m["hour"].dt.tz is None else m["hour"]
    m = m.drop_duplicates(["ICAO", "hour"], keep="last")
    for c in [c for c in m.columns if c.startswith("WX_")]:
        m[c] = m[c].astype("float32")
    return m.set_index(["ICAO", "hour"])


def arrivals_weather(df, m):
    df = df.copy()
    df["T"] = ep(df["MVT_TIME_UTC_mvt"])
    df["runway"] = df["RUNWAY_mvt"].astype(str)
    dep, arr = df[df["PHASE_mvt"].eq("DEP")], df[df["PHASE_mvt"].eq("ARR")]
    parts = []
    for a, d in dep.groupby("ADEP_mvt"):
        d = d.reset_index(drop=True)
        arr_a = arr[arr["ADES_mvt"].eq(a)]
        T, R, at = d["T"].to_numpy(float), d["runway"].to_numpy(), np.sort(arr_a["T"].to_numpy(float))
        o = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy(), "ARRV_APT_15": _count_half(at, T - 900, T).astype(float),
                          "ARRV_RWY_15": np.nan, "ARRV_RWY_SHARE_60": np.nan})
        a60 = _count_half(at, T - 3600, T)
        for r in np.unique(R):
            i = np.where(R == r)[0]
            ar = np.sort(arr_a.loc[arr_a["runway"].eq(r), "T"].to_numpy(float))
            o.loc[i, "ARRV_RWY_15"] = _count_half(ar, T[i] - 900, T[i])
            o.loc[i, "ARRV_RWY_SHARE_60"] = np.where(a60[i] > 0, _count_half(ar, T[i] - 3600, T[i]) / np.maximum(a60[i], 1), np.nan)
        parts.append(o)
    out = pd.concat(parts, ignore_index=True)
    dp = dep.set_index("MVT_ID_mvt").loc[out["MVT_ID_mvt"]]
    h = dp["MVT_TIME_UTC_mvt"].dt.floor("h")

    def get(column, shift):
        idx = pd.MultiIndex.from_arrays([dp["ADEP_mvt"].astype(str).to_numpy(), (h + pd.Timedelta(hours=shift)).to_numpy()])
        return m[column].reindex(idx).to_numpy(float)

    for name, column in (("TS", "WX_TS"), ("FZ", "WX_FZ"), ("SN", "WX_SN"), ("FG", "WX_FG")):
        out[f"WX_{name}_0"] = get(column, 0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["WX_SN_3H"] = np.nanmax(np.column_stack([get("WX_SN", -k) for k in range(4)]), axis=1)
    for name, column in (("VIS", "VIS_M"), ("WIND", "WIND_KT"), ("GUST", "GUST_KT")):
        out[f"WX_{name}"] = get(column, 0)
    return out[["MVT_ID_mvt", "ARRV_APT_15", "ARRV_RWY_15", "ARRV_RWY_SHARE_60", "WX_FZ_0", "WX_FG_0", "WX_SN_0", "WX_SN_3H",
                "WX_TS_0", "WX_VIS", "WX_WIND", "WX_GUST"]]


def _window_median_between(T, v, a_ofs, b_ofs):
    lo = np.searchsorted(T, T + a_ofs, "left" if a_ofs <= 0 else "right")
    hi = np.searchsorted(T, T + b_ofs, "right")
    out = np.full(len(T), np.nan)
    for i in range(len(T)):
        lo_i, hi_i = lo[i], hi[i]
        if hi_i <= lo_i:
            continue
        a = v[lo_i:hi_i]
        if lo_i <= i < hi_i:
            a = np.delete(a, i - lo_i)
        a = a[np.isfinite(a)]
        if len(a):
            out[i] = np.median(a)
    return out


def _count_ahead(s, a, b):
    return np.searchsorted(s, b, "right") - np.searchsorted(s, a, "right")


def forward_windows(df):
    df = df.copy()
    df["T"] = ep(df["MVT_TIME_UTC_mvt"])
    df["runway"] = df["RUNWAY_mvt"].astype(str)
    df["de"] = np.clip(df["T"] - ep(df["EOBT_1_flt"]), -3600, 14400)
    df["nt"] = np.clip(df["T"] - ep(df["AOBT_3_flt"]), 0, 7200)
    dep, arr = df[df["PHASE_mvt"].eq("DEP")], df[df["PHASE_mvt"].eq("ARR")]
    parts = []
    for a, d in dep.groupby("ADEP_mvt"):
        d = d.sort_values("T", kind="stable").reset_index(drop=True)
        T, de, nt, R = d["T"].to_numpy(float), d["de"].to_numpy(float), d["nt"].to_numpy(float), d["runway"].to_numpy()
        o = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()})
        o["DLYC_EOBT_15"], o["DLYF_EOBT_15"] = _window_median_between(T, de, -900, 900), _window_median_between(T, de, 0, 900)
        o["NMT_C15"], o["NMT_F15"] = _window_median_between(T, nt, -900, 900), _window_median_between(T, nt, 0, 900)
        o["DEP_APT_AFTER_10"] = _count_ahead(T, T, T + 600).astype(float)
        o["DEP_RWY_AFTER_10"], o["DLYC_RWY_15"], o["NMT_RWY_C15"] = np.nan, np.nan, np.nan
        for r in np.unique(R):
            i = np.where(R == r)[0]
            o.loc[i, "DEP_RWY_AFTER_10"] = _count_ahead(T[i], T[i], T[i] + 600)
            o.loc[i, "DLYC_RWY_15"] = _window_median_between(T[i], de[i], -900, 900)
            o.loc[i, "NMT_RWY_C15"] = _window_median_between(T[i], nt[i], -900, 900)
        o["ARR_AFTER_10"] = _count_ahead(np.sort(arr.loc[arr["ADES_mvt"].eq(a), "T"].to_numpy(float)), T, T + 600).astype(float)
        parts.append(o)
    return pd.concat(parts, ignore_index=True)


def stand_arrivals(df, window=6 * 3600.0):
    dep, arr = df[df["PHASE_mvt"].eq("DEP")].copy(), df[df["PHASE_mvt"].eq("ARR")].copy()
    arr["apt"], dep["apt"] = arr["ADES_mvt"], dep["ADEP_mvt"]
    arr["B"] = ep(arr["BLOCK_TIME_UTC_mvt"])
    arr["g"] = np.clip(ep(arr["MVT_TIME_UTC_mvt"]) - ep(arr["SCHED_TIME_UTC_mvt"]), -3600, 14400)
    dep["O"], dep["T"], dep["E"] = ep(dep["AOBT_3_flt"]), ep(dep["MVT_TIME_UTC_mvt"]), ep(dep["EOBT_1_flt"])
    dep["ref"] = np.where(np.isfinite(dep["O"]), dep["O"], dep["T"] - 900.0)
    arr = arr[np.isfinite(arr["B"]) & arr["STAND_mvt"].notna()].sort_values("B")
    every = dep[["MVT_ID_mvt"]]
    dep = dep[dep["STAND_mvt"].notna()].sort_values("ref")
    prev = pd.merge_asof(dep[["MVT_ID_mvt", "apt", "STAND_mvt", "ref", "O", "E"]], arr[["apt", "STAND_mvt", "B", "g"]],
                         left_on="ref", right_on="B", by=["apt", "STAND_mvt"], direction="backward", tolerance=window)
    nxt = pd.merge_asof(dep[["MVT_ID_mvt", "apt", "STAND_mvt", "ref", "T"]], arr[["apt", "STAND_mvt", "B"]],
                        left_on="ref", right_on="B", by=["apt", "STAND_mvt"], direction="forward", allow_exact_matches=False,
                        tolerance=window)
    out = pd.DataFrame({"MVT_ID_mvt": prev["MVT_ID_mvt"].to_numpy(), "IN_DELAY": prev["g"].to_numpy(float),
                        "IN_TURN": (prev["O"] - prev["B"]).to_numpy(float), "IN_BUFFER": (prev["E"] - prev["B"]).to_numpy(float)})
    nxt = nxt.assign(NX_GAP=nxt["B"] - nxt["ref"], NX_T=nxt["B"] - nxt["T"])[["MVT_ID_mvt", "NX_GAP", "NX_T"]]
    out = out.merge(nxt, on="MVT_ID_mvt", how="left", validate="one_to_one")
    return every.merge(out, on="MVT_ID_mvt", how="left", validate="one_to_one")


def compute(split="test", log=print):
    df = data.movements(split, COLUMNS).drop_duplicates("MVT_ID_mvt")
    dly = delay(df)
    out = {"queue": queue(df), "delay": dly, "delay_runway": delay_runway(df, dly),
           "arrivals_weather": arrivals_weather(df, metar()), "forward_windows": forward_windows(df),
           "stand_arrivals": stand_arrivals(df)}
    for name, o in out.items():
        assert o["MVT_ID_mvt"].is_unique, name
        for c in o.columns[1:]:
            o[c] = o[c].astype("float32")
        log(f"[groups] {split} {name}: {len(o):,} departures")
    return out


def build(log=print):
    from taxi.features.tables import GROUP_DIR
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    tr, te = compute("train", log), compute("test", log)
    for name in tr:
        d = pd.concat([tr[name], te[name]], ignore_index=True).drop_duplicates("MVT_ID_mvt", keep="last")
        d.to_parquet(GROUP_DIR / f"{name}.parquet", index=False)
        log(f"[groups] {name}: {len(d):,} rows → {(GROUP_DIR / f'{name}.parquet').relative_to(config.ROOT)}")
