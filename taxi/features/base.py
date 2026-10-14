import json
import warnings

import numpy as np
import pandas as pd

import config
from taxi import data

TABLE = {"train": config.WORK / "base_train.parquet", "test": config.WORK / "base_test.parquet"}
WEATHER = config.EXTERNAL / "weather_hourly.parquet"
REF = config.ARTIFACTS / "reference"
DISTANCES = REF / "stand_runway_distances.csv"
RUNWAYS = REF / "runways.csv"
SCHEMA = REF / "base_schema.json"

TIME_COLUMNS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt",
                "AOBT_3_flt", "ARVT_3_flt"]
COLUMNS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
           "TAXITIME_SEC_mvt", "CALLSIGN_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
           "ADES_FILED_flt", "ADES_flt"] + TIME_COLUMNS
TIME_ZONES = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
              "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome",
              "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}
DOMESTIC_PREFIXES = {"EDDF": ("ED", "ET"), "EDDM": ("ED", "ET"), "EGLL": ("EG",), "EHAM": ("EH",), "LEBL": ("LE", "GC"),
                     "LEMD": ("LE", "GC"), "LFPG": ("LF",), "LIRF": ("LI",), "LSZH": ("LS",), "LTFM": ("LT",)}
PEAK_HOURS = (6, 7, 8, 9, 16, 17, 18, 19)
GROUP_MIN = 20
LIRF_NOISY_STANDS = {"102", "105", "824", "826", "828", "829", "907", "HNG"}
WEATHER_COLUMNS = ["TEMPERATURE", "DEW_POINT", "HUMIDITY", "PRECIPITATION", "SNOWFALL", "SNOW_DEPTH", "WIND_SPEED",
                   "WIND_DIR", "CLOUD_COVER", "FROST_RISK", "IS_FREEZING", "SNOWFALL_3H", "SNOWFALL_6H", "PRECIP_3H",
                   "WIND_U", "WIND_V", "TEMP_DROP_3H"]
WEATHER_KEYS = ["ADEP_mvt", "MVT_YEAR", "MVT_MONTH", "MVT_DAY", "MVT_HOUR"]
FLOAT64 = {"MVT_ID_mvt", "NM_OFFSET_SEC", "NM_TAXI_DIFF_SEC", "TAXITIME_SEC_mvt"}


def all_movements():
    h = pd.concat([data.movements("train", COLUMNS).assign(_test=False),
                   data.movements("test", COLUMNS).assign(_test=True)], ignore_index=True)
    h = h.drop_duplicates("MVT_ID_mvt", keep="last").reset_index(drop=True)
    t = h["MVT_TIME_UTC_mvt"]
    key = np.where(h["_test"].to_numpy(), (t.dt.year * 100 + t.dt.month).to_numpy(), 0)
    h = h.iloc[np.argsort(key, kind="stable")].reset_index(drop=True)
    for c in TIME_COLUMNS:
        h[c] = h[c].dt.tz_localize(None).astype("datetime64[ns]")
    return h


def weather_hourly():
    if not WEATHER.exists():
        return None
    return pd.read_parquet(WEATHER, columns=WEATHER_KEYS + WEATHER_COLUMNS)


def runway_dimensions():
    return pd.read_csv(RUNWAYS, dtype={"ADEP_mvt": str, "RUNWAY_mvt": str})


def population_test_ids():
    d = pd.read_parquet(config.POPULATION_TEST_FILE, columns=["MVT_ID_mvt", "PHASE_mvt"])
    return d.loc[d["PHASE_mvt"] == "DEP", "MVT_ID_mvt"].to_numpy()


def _count_before(events, queries, minutes):
    e = np.sort(events[~np.isnat(events)])
    n = np.searchsorted(e, queries, "left") - np.searchsorted(e, queries - np.timedelta64(minutes, "m"), "left")
    return np.where(np.isnat(queries), 0.0, n.astype(float))


def _window_stat(events, values, queries, minutes, stat):
    m = ~np.isnat(events)
    times = pd.DatetimeIndex(np.concatenate([events[m], queries]))
    c = pd.DataFrame({"v": np.concatenate([np.asarray(values, float)[m], np.full(len(queries), np.nan)]),
                      "q": np.r_[np.zeros(m.sum(), bool), np.ones(len(queries), bool)]}, index=times).sort_index()
    r = c["v"].rolling(f"{minutes}min", closed="left")
    r = r.median() if stat == "median" else r.max()
    return r[c["q"].to_numpy()].to_numpy()


def _concurrent(start, end, queries, include_self):
    m = ~(np.isnat(start) | np.isnat(end))
    b, s = start[m], end[m]
    ok = b <= s
    b, s = b[ok], s[ok]
    if len(b) == 0:
        return np.zeros(len(queries))
    t = np.concatenate([b, s])
    order = np.argsort(t, kind="stable")
    t, total = t[order], np.cumsum(np.r_[np.ones(len(b)), -np.ones(len(s))][order])
    i = np.searchsorted(t, queries, "right" if include_self else "left") - 1
    return np.clip(np.where(i >= 0, total[np.clip(i, 0, None)], 0.0), 0, None)


def _stand_turnover(h):
    stand = h["ADEP_mvt"].where(h["PHASE_mvt"].eq("DEP"), h["ADES_mvt"]) + "_" + h["STAND_mvt"]
    t = h["MVT_TIME_UTC_mvt"]
    v = pd.DataFrame({"stand": stand, "_t": t, "PREV_T": t, "PREV_STAND_ARR_TAXI": h["TAXITIME_SEC_mvt"].astype(float)})
    v = v[h["PHASE_mvt"].eq("ARR").to_numpy()].dropna(subset=["stand", "_t"]).sort_values("_t")
    k = pd.DataFrame({"stand": stand, "_t": t})[h["PHASE_mvt"].eq("DEP").to_numpy()].dropna().sort_values("_t")
    m = pd.merge_asof(k.reset_index(), v, on="_t", by="stand", direction="backward",
                      allow_exact_matches=False).set_index("index")
    m["SEC_SINCE_STAND_ARR"] = ((m["_t"] - m["PREV_T"]).dt.total_seconds()).clip(upper=86400)
    return m[["PREV_STAND_ARR_TAXI", "SEC_SINCE_STAND_ARR"]]


def _airport_traffic(dep, arr):
    T, O = dep["MVT_TIME_UTC_mvt"].to_numpy(), dep["AOBT_3_flt"].to_numpy()
    Ta = arr["MVT_TIME_UTC_mvt"].to_numpy()
    taxi_in = arr["TAXITIME_SEC_mvt"].to_numpy(float)
    nm = dep["NM_TAXI_DIFF_SEC"].to_numpy(float)
    runways = dep.groupby("RUNWAY_mvt").indices
    o = {}
    for w in (30, 60):
        o[f"Q_PUSHBACK_{w}"] = _count_before(O, O, w)
    for w in (15, 30, 60):
        o[f"Q_RWY_DEP_{w}"] = np.full(len(dep), np.nan)
        for idx in runways.values():
            o[f"Q_RWY_DEP_{w}"][idx] = _count_before(T[idx], T[idx], w)
    for w in (15, 30, 60):
        o[f"Q_ARR_{w}"] = _count_before(Ta, T, w)
    o["ARR_TAXI_MED_30"] = _window_stat(Ta, taxi_in, T, 30, "median")
    o["ARR_TAXI_MED_60"] = _window_stat(Ta, taxi_in, T, 60, "median")
    o["ARR_TAXI_MAX_30"] = _window_stat(Ta, taxi_in, T, 30, "max")
    o["DEP_TAXI_MED_15"] = _window_stat(T, nm, T, 15, "median")
    o["DEP_TAXI_MED_30"] = _window_stat(T, nm, T, 30, "median")
    o["DEP_QUEUE_N"] = _concurrent(O, T, T, include_self=True)
    o["RWY_QUEUE_N"] = np.full(len(dep), np.nan)
    for idx in runways.values():
        o["RWY_QUEUE_N"][idx] = _concurrent(O[idx], T[idx], T[idx], include_self=True)
    arr_end = Ta + (np.nan_to_num(taxi_in) * 1e9).astype("timedelta64[ns]")
    o["ARR_TAXIING_N"] = _concurrent(Ta, arr_end, T, include_self=False)
    o["SURFACE_N"] = o["DEP_QUEUE_N"] + o["ARR_TAXIING_N"]
    o["DEP_15"], o["ARR_15"] = _count_before(T, T, 15), _count_before(Ta, T, 15)
    o["DEP_ARR_RATIO"] = o["DEP_15"] / (o["ARR_15"] + 1)
    o["MOVEMENTS_15"] = o["DEP_15"] + o["ARR_15"]
    s = dep[["MVT_TIME_UTC_mvt", "RUNWAY_mvt", "WK_TBL_CAT_flt"]].sort_values("MVT_TIME_UTC_mvt")
    previous = s.groupby("RUNWAY_mvt")["WK_TBL_CAT_flt"].shift(1)
    o["WAKE_TURBULENCE_BEHIND_HEAVY"] = previous.isin(["H", "J"]).astype(float).reindex(dep.index).to_numpy()
    o["ACTIVE_RUNWAY_COUNT_30"] = sum((_count_before(T[idx], T, 30) > 0).astype(float) for idx in runways.values())
    return pd.DataFrame(o, index=dep.index)


def _keys(x):
    apt = x["ADEP_mvt"]
    return {"apt": apt, "stand_runway": apt + "_" + x["STAND_mvt"] + "_" + x["RUNWAY_mvt"], "actype": x["AIRCRAFT_TYPE_mvt"],
            "operator": x["AIRCRAFT_OPERATOR_flt"] + "_" + apt}


def _group(ref_nm, ref_key, key, p10=False):
    g = ref_nm.groupby(ref_key)
    t = pd.DataFrame({"med": g.median(), "n": g.count()})
    if p10:
        t["p10"] = g.agg(lambda x: x.quantile(0.10))
    return t.reindex(key.to_numpy())


def _group_stats(d, ref):
    kr, kd, rn = _keys(ref), _keys(d), ref["NM_TAXI_DIFF_SEC"]
    a = _group(rn, kr["apt"], kd["apt"], p10=True)
    apt_med, apt_p10 = a["med"].to_numpy(), a["p10"].to_numpy()
    o = pd.DataFrame(index=d.index)
    g = _group(rn, kr["stand_runway"], kd["stand_runway"], p10=True)
    enough = (g["n"] >= GROUP_MIN).to_numpy()
    o["BASE_TAXI_TIME"] = np.where(enough, g["med"].to_numpy(), apt_med)
    o["STAND_RWY_COUNT"] = g["n"].to_numpy()
    o["UNIMPEDED_TAXI"] = np.where(enough, g["p10"].to_numpy(), apt_p10)
    o["QUEUE_DELAY_EST"] = d["NM_TAXI_DIFF_SEC"] - o["UNIMPEDED_TAXI"]
    for name, key in (("AIRCRAFT_TYPE_MED_TAXI", "actype"), ("OPERATOR_APT_MED_TAXI", "operator")):
        g = _group(rn, kr[key], kd[key])
        o[name] = np.where((g["n"] >= GROUP_MIN).to_numpy(), g["med"].to_numpy(), apt_med)
    return o


def _local_time(t, apt):
    out = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    for a, tz in TIME_ZONES.items():
        m = (apt == a).to_numpy()
        if m.any():
            out[m] = t[m].dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None).astype("datetime64[ns]")
    return out


def departure_features(h, weather=None, log=print):
    turnover = _stand_turnover(h)
    d = h[h["PHASE_mvt"].eq("DEP")].copy()
    pop = (~d["_test"] | d["MVT_ID_mvt"].isin(population_test_ids())).to_numpy()
    log(f"[base] population: {pop.sum():,} departures; test departures outside the population {(~pop).sum():,}")
    mvt = d["MVT_TIME_UTC_mvt"]
    d["NM_TAXI_DIFF_SEC"] = (mvt - d["AOBT_3_flt"]).dt.total_seconds()
    ref = d[pop]
    o = pd.DataFrame(index=d.index)
    for c in ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "CALLSIGN_flt", "MARKET_SEGMENT_flt",
              "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt"]:
        o[c] = d[c]
    o["FLIGHT_CODE"] = d["FLIGHT_mvt"]
    o["AIRLINE_LETTERS"] = d["FLIGHT_mvt"].str.extract(r"([A-Za-z]+)")[0]
    o["FLIGHT_NUMBER"] = pd.to_numeric(d["FLIGHT_mvt"].str.extract(r"(\d+)")[0], errors="coerce")
    o["ADES_FILED_CHANGED"] = (d["ADES_FILED_flt"] != d["ADES_flt"]).astype(float)
    o["ADES_FILED"] = d["ADES_FILED_flt"]
    for name, c in (("SCHED", "SCHED_TIME_UTC_mvt"), ("LOBT", "LOBT_flt"), ("IOBT", "IOBT_flt"), ("EOBT", "EOBT_1_flt"),
                    ("AOBT", "AOBT_3_flt"), ("ARVT3", "ARVT_3_flt")):
        o[f"{name}_DIFF_SEC"] = (d[c] - mvt).dt.total_seconds()
    o["AOBT_MINUS_EOBT"] = o["AOBT_DIFF_SEC"] - o["EOBT_DIFF_SEC"]
    o["NM_TAXI_DIFF_SEC"] = d["NM_TAXI_DIFF_SEC"]
    o["MVT_MONTH"], o["MVT_DAY"], o["MVT_DAYOFWEEK"] = mvt.dt.month, mvt.dt.day, mvt.dt.dayofweek
    o["MVT_HOUR_UTC"], o["MVT_MINOFDAY_UTC"] = mvt.dt.hour, mvt.dt.hour * 60 + mvt.dt.minute
    local = _local_time(mvt, d["ADEP_mvt"])
    o["LOCAL_HOUR"] = local.dt.hour
    o["LOCAL_MINOFDAY"] = local.dt.hour * 60 + local.dt.minute
    o["HOUR_SIN"] = np.sin(2 * np.pi * o["LOCAL_MINOFDAY"] / 1440)
    o["HOUR_COS"] = np.cos(2 * np.pi * o["LOCAL_MINOFDAY"] / 1440)
    o["IS_PEAK_HOUR"] = o["LOCAL_HOUR"].isin(PEAK_HOURS).astype(float)

    arr = h[h["PHASE_mvt"].eq("ARR")]
    tr = [_airport_traffic(d[d["ADEP_mvt"].eq(a)], arr[arr["ADES_mvt"].eq(a)]) for a in sorted(d["ADEP_mvt"].dropna().unique())]
    tr = pd.concat(tr).reindex(d.index)
    o = pd.concat([o, tr], axis=1)
    log(f"[base] traffic: {len(d):,} departures")

    key = pd.DataFrame({"ADEP_mvt": d["ADEP_mvt"].to_numpy(), "MVT_YEAR": mvt.dt.year.to_numpy(),
                        "MVT_MONTH": mvt.dt.month.to_numpy(), "MVT_DAY": mvt.dt.day.to_numpy(),
                        "MVT_HOUR": mvt.dt.hour.to_numpy()})
    if weather is None:
        log(f"[base] WARNING: no weather table ({WEATHER.relative_to(config.ROOT)}); weather columns are NaN")
        w = pd.DataFrame(np.nan, index=d.index, columns=WEATHER_COLUMNS)
    else:
        w = key.merge(weather, on=WEATHER_KEYS, how="left", validate="many_to_one")[WEATHER_COLUMNS]
        w.index = d.index
        log(f"[base] weather: unmatched test departures {w.loc[d['_test'].to_numpy(), 'TEMPERATURE'].isna().mean():.2%}")
    o = pd.concat([o, w], axis=1)
    heading = pd.to_numeric(d["RUNWAY_mvt"].str.extract(r"^(\d{2})")[0], errors="coerce") * 10
    r = np.deg2rad(heading)
    o["RUNWAY_HEADING_DEG"] = heading
    o["HEADWIND"] = o["WIND_U"] * np.sin(r) + o["WIND_V"] * np.cos(r)
    o["CROSSWIND"] = np.abs(o["WIND_U"] * np.cos(r) - o["WIND_V"] * np.sin(r))
    o["TEMP_DEWPOINT_SPREAD"] = o["TEMPERATURE"] - o["DEW_POINT"]

    o = pd.concat([o, _group_stats(d, ref)], axis=1)
    o["NM_MISSING"] = d["AOBT_3_flt"].isna().astype(float)
    late = (mvt - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    o["TAKEOFF_VS_SCHED_SEC"], o["TAKEOFF_VS_SCHED_SEC_POS"], o["TAKEOFF_VS_SCHED_SEC_NEG"] = late, late.clip(lower=0), late.clip(upper=0)
    o = pd.concat([o, turnover.reindex(d.index)], axis=1)
    o["OPERATIONAL_DIFFICULTY"] = o["SURFACE_N"] * o["PRECIPITATION"].fillna(0)
    o["DELAY_QUEUE_PRESSURE"] = o["TAKEOFF_VS_SCHED_SEC_POS"] * o["DEP_QUEUE_N"]
    o["STAND_TYPE"] = d["STAND_mvt"].str[0]
    o["ROUTE_PAIR"] = d["ADEP_mvt"] + "_" + d["ADES_mvt"]
    prefix = d["ADES_mvt"].str[:2]
    o["IS_DOMESTIC"] = np.array([b in DOMESTIC_PREFIXES.get(a, ()) for a, b in zip(d["ADEP_mvt"], prefix)], dtype=float)
    day = d["ADEP_mvt"] + "_" + mvt.dt.date.astype(str)
    o["DEP_RANK_TODAY"] = (d.assign(_g=day).sort_values("MVT_TIME_UTC_mvt").groupby("_g").cumcount() + 1).reindex(d.index)
    o["IS_FIRST_DEP_TODAY"] = (o["DEP_RANK_TODAY"] == 1).astype(float)
    mean = o[pop].groupby(["ADEP_mvt", "LOCAL_HOUR"])["DEP_QUEUE_N"].mean()
    mean = mean.reindex(pd.MultiIndex.from_arrays([d["ADEP_mvt"], o["LOCAL_HOUR"]])).to_numpy()
    o["QUEUE_RATIO_TO_HOUR_MEAN"] = o["DEP_QUEUE_N"] / (mean + 1)
    o["SCHED_DIFF_SEC_HOUR_PCT"] = o.groupby([day, o["LOCAL_HOUR"]])["SCHED_DIFF_SEC"].rank(pct=True)
    o["MVT_ID_mvt"] = d["MVT_ID_mvt"]
    o["TAXITIME_SEC_mvt"] = d["TAXITIME_SEC_mvt"].astype(float)
    o["NM_OFFSET_SEC"] = (d["BLOCK_TIME_UTC_mvt"] - d["AOBT_3_flt"]).dt.total_seconds()
    o["_test"], o["_pop"] = d["_test"], pop
    return o


def _runways_and_cdm(t, runways):
    k = pd.DataFrame({"ADEP_mvt": t["ADEP_mvt"].to_numpy(), "RUNWAY_mvt": t["RUNWAY_mvt"].str.strip().to_numpy()})
    p = k.merge(runways, on=["ADEP_mvt", "RUNWAY_mvt"], how="left", validate="many_to_one")
    for c in ("RWY_PHYSICAL_LENGTH_FT", "RWY_HEADING_DEG_T", "RWY_DISPLACED_THRESH_FT"):
        t[c] = p[c].to_numpy(np.float32)
    z = {c: t[f"{c}_DIFF_SEC"].fillna(0).astype(np.float32) for c in ("AOBT", "EOBT", "LOBT", "IOBT", "ARVT3", "SCHED")}
    for a, b in (("AOBT", "EOBT"), ("AOBT", "LOBT"), ("AOBT", "IOBT"), ("EOBT", "IOBT"), ("EOBT", "LOBT"), ("ARVT3", "AOBT"),
                 ("SCHED", "AOBT")):
        t[f"{a}_MINUS_{b}_CDM"] = z[a] - z[b]


def _distances(t, ref):
    m = pd.read_csv(DISTANCES)
    m["stand"] = m["STAND_ID"].astype(str).str.strip().str.upper()
    m["runway"] = m["RUNWAY"].astype(str).str.strip().str.upper().str.lstrip("0")
    closest = m.groupby(["ADEP", "stand"])["DIST_M"].min().rename("STAND_MIN_DIST_M").reset_index()
    to_runway = (m.drop_duplicates(["ADEP", "stand", "runway"])[["ADEP", "stand", "runway", "DIST_M"]]
                 .rename(columns={"DIST_M": "STAND_TO_RWY_DIST_M"}))

    def match(x):
        k = pd.DataFrame({"ADEP": x["ADEP_mvt"].to_numpy(),
                          "stand": x["STAND_mvt"].fillna("nan").str.strip().str.upper().to_numpy(),
                          "runway": x["RUNWAY_mvt"].fillna("nan").str.strip().str.upper().str.lstrip("0").to_numpy()})
        return pd.DataFrame({"ADEP_mvt": x["ADEP_mvt"].to_numpy(),
                             "STAND_TO_RWY_DIST_M": k.merge(to_runway, on=["ADEP", "stand", "runway"], how="left")["STAND_TO_RWY_DIST_M"].to_numpy(),
                             "STAND_MIN_DIST_M": k.merge(closest, on=["ADEP", "stand"], how="left")["STAND_MIN_DIST_M"].to_numpy()})
    e = match(t)
    ref = e[ref]
    for c, v in (("STAND_TO_RWY_DIST_M", 2200.0), ("STAND_MIN_DIST_M", 1000.0)):
        t[c] = e[c].fillna(t["ADEP_mvt"].map(ref.groupby("ADEP_mvt")[c].median()).fillna(v)).to_numpy(np.float32)
    t["STAND_DIST_RATIO"] = (t["STAND_TO_RWY_DIST_M"] / (t["STAND_MIN_DIST_M"] + 50.0)).clip(1.0, 10.0).astype(np.float32)


def _cdm(t):
    z = lambda c: t[c].fillna(0).astype(np.float32)  # noqa: E731
    ae, q30, q15, qd = z("AOBT_MINUS_EOBT"), z("Q_RWY_DEP_30"), z("Q_RWY_DEP_15"), z("QUEUE_DELAY_EST")
    ad, ed, ld, idf, pk = z("AOBT_DIFF_SEC"), z("EOBT_DIFF_SEC"), z("LOBT_DIFF_SEC"), z("IOBT_DIFF_SEC"), z("IS_PEAK_HOUR")
    t["AOBT_EOBT_QUEUE_PROD"] = ae * q30
    t["AOBT_EOBT_QUEUE15_PROD"] = ae * q15
    t["QUEUE_DELAY_AOBT_SUM"] = qd + ae.clip(0, None)
    t["CDM_REVISION_MAGNITUDE"] = (ld - ed).abs()
    t["LOBT_IOBT_DIFF"] = ld - idf
    t["EOBT_AOBT_DIFF"] = ed - ad
    t["PEAK_AOBT_EOBT"] = pk * ae
    t["AOBT_EOBT_ABS"] = ae.abs()
    t["QUEUE_CDM_RATIO"] = (q30 / (ae.abs() + 1.0)).clip(0, 100)


def _rolling_counts(t):
    ts = t["_mvt"].dt.floor("min").to_numpy("datetime64[ns]")
    for keys, windows, prefix in ((["ADEP_mvt"], (10, 30, 60), "DEP_COUNT_"),
                                  (["ADEP_mvt", "RUNWAY_mvt"], (10, 30), "DEP_RWY_COUNT_")):
        groups = t.groupby(keys, sort=False).indices
        for w in windows:
            out = np.zeros(len(t), dtype=np.float32)
            for idx in groups.values():
                s = np.sort(ts[idx])
                out[idx] = np.searchsorted(s, ts[idx], "left") - np.searchsorted(s, ts[idx] - np.timedelta64(w, "m"), "left")
            t[f"{prefix}{w}"] = out


def block(o, h, split):
    m = (o["_test"] == (split == "test")).to_numpy()
    t = o[m].drop(columns=["_test"]).copy()
    t["_mvt"] = h.loc[t.index, "MVT_TIME_UTC_mvt"]
    t = t.reset_index(drop=True)
    ref = t.pop("_pop").to_numpy() if split == "test" else np.ones(len(t), bool)
    t = t.drop(columns=["_pop"], errors="ignore")
    for c in t.columns:
        if t[c].dtype == np.float64 and c not in FLOAT64:
            t[c] = t[c].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        _runways_and_cdm(t, runway_dimensions())
        _distances(t, ref)
        _cdm(t)
        lirf, noisy = t["ADEP_mvt"].eq("LIRF"), t["STAND_mvt"].isin(LIRF_NOISY_STANDS)
        t["LIRF_NOISY_STAND"] = (lirf & noisy).astype(np.float32)
        t["LIRF_NOISY_16R"] = (lirf & noisy & t["RUNWAY_mvt"].eq("16R")).astype(np.float32)
        _rolling_counts(t)
    out = {}
    for name, kind in json.loads(SCHEMA.read_text(encoding="utf-8")):
        if name not in t:
            out[name] = pd.Series(np.nan, index=t.index, dtype="object" if kind.endswith("string") else "float32")
            continue
        x = t[name]
        if kind in ("int32", "int64") and x.notna().all():
            x = x.astype(kind)
        elif kind == "float":
            x = x.astype(np.float32)
        elif kind == "double":
            x = x.astype(np.float64)
        out[name] = x
    return pd.DataFrame(out)


def compute(weather="file", log=print):
    h = all_movements()
    log(f"[base] movements: {len(h):,} (test {int(h['_test'].sum()):,})")
    o = departure_features(h, weather_hourly() if isinstance(weather, str) else weather, log)
    out = {k: block(o, h, k) for k in ("train", "test")}
    for k, t in out.items():
        log(f"[base] {k}: {len(t):,} departures, {t.shape[1]} columns")
    return out


def build(log=print):
    config.WORK.mkdir(exist_ok=True)
    for k, t in compute(log=log).items():
        t.to_parquet(TABLE[k], index=False)
        log(f"[base] → {TABLE[k].relative_to(config.ROOT)}")
