import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from taxi import data
from taxi.correctors import common as C
from taxi.features import tables as T

DIR = config.MODELS / "adsb"
TRACK = ["ADSB_FIRST", "ADSB_MOVE", "ADSB_TO_DIFF", "ADSB_N", "ADSB_GAP", "ADSB_GAP_5MIN", "ADSB_FIRST_GS", "ADSB_PARKED",
         "ADSB_FIRST_STOP", "ADSB_FIRST_DIST", "ADSB_PATH", "ADSB_STOP", "ADSB_ROLL", "ADSB_MATCH"]
TRACK_SPLIT = ["ADSB_TAXI_START", "ADSB_STAND_LAST", "ADSB_STAND_NEXT", "ADSB_STAND_GAP", "ADSB_STAND_FIRST"]
OPDI = ["OPDI_FIRST_GROUND", "OPDI_STAND_EXIT", "OPDI_STAND_DUR", "OPDI_APRON_EXIT", "OPDI_TAXI_ENTRY", "OPDI_TAXI_TOTAL",
        "OPDI_TAXI_MAX", "OPDI_TWY_N", "OPDI_RWY_ENTRY", "OPDI_RWY_DUR", "OPDI_N"]
PARAMS = dict(objective="huber", alpha=600.0, learning_rate=0.03, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8)
ROUNDS, SEEDS, COPIES, ECHO_BETA, LOG_M = 800, (42, 2026, 7), ("s30", "s10"), 0.25, 30.0
EXTRA_COLUMNS = ["AIRLINE_LETTERS", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt"]


def _frame(d, folder):
    d = d.merge(C.adsb_features(folder), on="MVT_ID_mvt", how="left", validate="one_to_one")
    d = d.merge(C.mvt_epoch(d["MVT_ID_mvt"]), on="MVT_ID_mvt", how="left", validate="one_to_one")
    d = d.merge(T.table("queue"), on="MVT_ID_mvt", how="left", validate="one_to_one")
    return d.merge(C.base_columns(d["MVT_ID_mvt"], EXTRA_COLUMNS), on="MVT_ID_mvt", how="left")


def training_data(folder="adsb_tracks"):
    d = C.oof_base()
    days = {(int(g[5:7]), int(g[8:10])) for g in C.adsb_days(folder, 2025)}
    d = d[[(m, g) in days for m, g in zip(d["MVT_MONTH"], d["MVT_DAY"])]].reset_index(drop=True)
    return _frame(d, folder)


def _log_ref(d):
    good = ((d["ADSB_MATCH"].fillna(0) > 0) & (d["ADSB_FIRST_GS"].fillna(99) < 3) & d["ADSB_MOVE"].notna()
            & (d["y_taxi"] < 7200))
    good_split = ((d["ADSB_MATCH"].fillna(0) > 0) & d["ADSB_STAND_NEXT"].notna() & (d["ADSB_STAND_GAP"] <= 120)
             & (d["y_taxi"] < 7200))
    return [("LOG", d[good].assign(u=d.loc[good, "y_taxi"] - d.loc[good, "ADSB_MOVE"])),
            ("LOG_SPLIT", d[good_split].assign(u=d.loc[good_split, "y_taxi"] - d.loc[good_split, "ADSB_STAND_NEXT"]))]


def _key(df, col):
    return df["ADEP_mvt"].astype(str) + "|" + df[col].astype(str)


def logging_oof(d):
    for prefix, ref in _log_ref(d):
        for name, col in (("AIRLINE", "AIRLINE_LETTERS"), ("STAND", "STAND_mvt")):
            kd, kr, v = _key(d, col), _key(ref, col), np.full(len(d), np.nan)
            for month in np.unique(d["MVT_MONTH"]):
                rr = ref[ref["MVT_MONTH"] != month]
                prior, g = rr.groupby("ADEP_mvt")["u"].mean(), rr.groupby(kr.loc[rr.index])["u"].agg(["sum", "size"])
                i = (d["MVT_MONTH"] == month).to_numpy()
                o = d.loc[i, "ADEP_mvt"].map(prior).to_numpy(float)
                s_, n_ = kd[i].map(g["sum"]).fillna(0).to_numpy(float), kd[i].map(g["size"]).fillna(0).to_numpy(float)
                v[i] = (s_ + LOG_M * o) / (n_ + LOG_M)
            d[f"{prefix}_{name}"] = v
    return d


def logging_tables(d):
    out = []
    for prefix, ref in _log_ref(d):
        prior = ref.groupby("ADEP_mvt")["u"].mean().rename("prior").reset_index().assign(prefix=prefix)
        for name, col in (("AIRLINE", "AIRLINE_LETTERS"), ("STAND", "STAND_mvt")):
            g = ref.groupby(_key(ref, col))["u"].agg(["sum", "size"]).reset_index(names="key")
            out.append(g.assign(prefix=prefix, name=name, kind="key"))
        out.append(prior.rename(columns={"ADEP_mvt": "key"}).assign(name="", kind="prior"))
    return pd.concat(out, ignore_index=True)


def apply_logging(t, table):
    for prefix in ("LOG", "LOG_SPLIT"):
        prior = table[(table["prefix"] == prefix) & (table["kind"] == "prior")].set_index("key")["prior"]
        o = t["ADEP_mvt"].map(prior).to_numpy(float)
        for name, col in (("AIRLINE", "AIRLINE_LETTERS"), ("STAND", "STAND_mvt")):
            g = table[(table["prefix"] == prefix) & (table["name"] == name) & (table["kind"] == "key")].set_index("key")
            k = _key(t, col)
            s_, n_ = k.map(g["sum"]).fillna(0).to_numpy(float), k.map(g["size"]).fillna(0).to_numpy(float)
            t[f"{prefix}_{name}"] = (s_ + LOG_M * o) / (n_ + LOG_M)
    return t


def _centered(diff, d):
    good = (d["ADSB_FIRST_GS"].fillna(99) < 3) & diff.notna()
    group = d["ADEP_mvt"].astype(str) + "_" + d["MVT_MONTH"].astype(str)
    med = diff[good].groupby(group[good]).agg(["median", "size"])
    overall = float(diff[good].median()) if good.any() else 0.0
    return diff - group.map(med["median"].where(med["size"] >= 30)).fillna(overall)


def _local(d, d_move_c, window=1800.0):
    out, count = np.full(len(d), np.nan), np.zeros(len(d))
    ok = ((d["ADSB_FIRST_GS"].fillna(99) < 3) & d_move_c.notna()).to_numpy()
    v, c, t = np.where(ok, d_move_c.clip(-600, 600).fillna(0).to_numpy(), 0.0), ok.astype(float), d["mvt"].to_numpy(float)
    for _, idx in d.groupby("ADEP_mvt").indices.items():
        o = idx[np.argsort(t[idx])]
        ts, vv, cc = t[o], v[o], c[o]
        sv, sc = np.r_[0.0, np.cumsum(vv)], np.r_[0.0, np.cumsum(cc)]
        lo, hi = np.searchsorted(ts, ts - window, "left"), np.searchsorted(ts, ts + window, "right")
        total, n = sv[hi] - sv[lo] - vv, sc[hi] - sc[lo] - cc
        out[o], count[o] = np.where(n > 0, total / np.maximum(n, 1), np.nan), n
    return out, count


def stand_table(min_n=3):
    f = C.adsb_features("adsb_tracks", columns=["MVT_ID_mvt", "ADSB_FIRST_GS", "ADSB_FIRST_LAT", "ADSB_FIRST_LON"])
    f = f[(f["ADSB_FIRST_GS"] < 3) & f["ADSB_FIRST_LAT"].notna()]
    f = f.merge(C.base_columns(f["MVT_ID_mvt"], ["ADEP_mvt", "STAND_mvt"]), on="MVT_ID_mvt", how="inner")
    f = f[f["STAND_mvt"].notna()]
    g = f.groupby(["ADEP_mvt", "STAND_mvt"]).agg(stand_lat=("ADSB_FIRST_LAT", "median"), stand_lon=("ADSB_FIRST_LON", "median"),
                                                 stand_n=("ADSB_FIRST_LAT", "size"))
    return g[g["stand_n"] >= min_n].reset_index()


def _stand(d, table):
    s = d[["ADEP_mvt", "STAND_mvt"]].merge(table, on=["ADEP_mvt", "STAND_mvt"], how="left")
    la, lo = s["stand_lat"].to_numpy(float), s["stand_lon"].to_numpy(float)
    m = lambda lat, lon: 111_195.0 * np.hypot(np.asarray(lat, float) - la,  # noqa: E731
                                              (np.asarray(lon, float) - lo) * np.cos(np.radians(la)))
    return {"STAND_FIRST_M": m(d["ADSB_FIRST_LAT"], d["ADSB_FIRST_LON"]),
            "STAND_MOVE_M": m(d["ADSB_MOVE_LAT"], d["ADSB_MOVE_LON"]),
            "STAND_ROLL_M": m(d["ADSB_ROLL_LAT"], d["ADSB_ROLL_LON"]), "STAND_N": s["stand_n"].to_numpy(float)}


def matrix(d, stands, opdi, type_categories=None, stand_split=True, queue=True, opdi_airports=None):
    X = pd.DataFrame({"base": d["base"], "nm": d["NM_TAXI_DIFF_SEC"]})
    for c in TRACK:
        X[c] = d[c].astype(float)
    X["d_move"], X["d_first"] = d["ADSB_MOVE"] - d["base"], d["ADSB_FIRST"] - d["base"]
    X["d_move_c"], X["d_first_c"] = _centered(X["d_move"], d), _centered(X["d_first"], d)
    X["local"], X["local_n"] = _local(d, X["d_move_c"])
    X["base_std"] = d["base_std"].astype(float)
    for c, v in _stand(d, stands).items():
        X[c] = v
    X["STAND_FIRST_S"] = X["STAND_FIRST_M"] / 7.0
    if stand_split:
        for c in TRACK_SPLIT:
            X[c] = d[c].astype(float)
        X["d_taxi_start"], X["d_stand_next"] = X["ADSB_TAXI_START"] - d["base"], X["ADSB_STAND_NEXT"] - d["base"]
        X["d_taxi_start_c"] = _centered(X["d_taxi_start"], d)
    for c in (("LOG_SPLIT_AIRLINE", "LOG_SPLIT_STAND") if stand_split else ()) + ("LOG_AIRLINE", "LOG_STAND"):
        X[c] = d[c].to_numpy(float)
    actype = d["AIRCRAFT_TYPE_mvt"].astype(str)
    X["actype"] = pd.Categorical(actype, categories=type_categories) if type_categories is not None else actype.astype("category")
    X["d_nm"] = d["ADSB_MOVE"] - d["NM_TAXI_DIFF_SEC"]
    if queue:
        for c in T.QUEUE:
            X[c] = d[c].to_numpy(float)
    o = d[["MVT_ID_mvt"]].merge(opdi, on="MVT_ID_mvt", how="left")
    for c in OPDI:
        X[c] = o[c].to_numpy(float)
    X["d_opdi_stand"] = X["OPDI_STAND_EXIT"].to_numpy() - d["base"].to_numpy(float)
    X["d_opdi_taxi"] = X["OPDI_TAXI_ENTRY"].to_numpy() - d["base"].to_numpy(float)
    if opdi_airports:
        X.loc[~d["ADEP_mvt"].isin(opdi_airports).to_numpy(), OPDI + ["d_opdi_stand", "d_opdi_taxi"]] = np.nan
    return X


def train(log=print):
    t0 = time.time()
    DIR.mkdir(parents=True, exist_ok=True)
    stands, opdi = stand_table(), C.opdi_features()
    stands.to_parquet(DIR / "stands.parquet", index=False)
    d = logging_oof(training_data("adsb_tracks"))
    logging_tables(d).to_parquet(DIR / "logging.parquet", index=False)
    X = matrix(d, stands, opdi)
    type_categories = list(X["actype"].cat.categories)
    m = d["ADSB_MATCH"].fillna(0).to_numpy() > 0
    Xs, rs = [X[m].reset_index(drop=True)], [(d["y_taxi"] - d["base"]).to_numpy(float)[m]]
    for copy in COPIES:
        dc = logging_oof(training_data(f"adsb_tracks_{copy}"))
        Xc = matrix(dc, stands, opdi, type_categories)[list(X.columns)]
        mc = dc["ADSB_MATCH"].fillna(0).to_numpy() > 0
        Xs.append(Xc[mc].reset_index(drop=True))
        rs.append((dc["y_taxi"] - dc["base"]).to_numpy(float)[mc])
        log(f"[adsb] copy {copy}: {mc.sum():,} rows")
    Xt, rt = pd.concat(Xs, ignore_index=True), np.concatenate(rs)
    log(f"[adsb] training {len(Xt):,} rows ({m.sum():,} original), {X.shape[1]} features | {time.time() - t0:.0f} s")
    for s in SEEDS:
        lgb.train({**PARAMS, "seed": s}, lgb.Dataset(Xt, rt), ROUNDS).save_model(str(DIR / f"seed_{s}.txt"))
        log(f"[adsb] seed {s} | {time.time() - t0:.0f} s")
    C.save_json(DIR / "meta.json", {"columns": list(X.columns), "type_categories": type_categories})


def apply(a, log=print):
    meta = C.load_json(DIR / "meta.json")
    t = a[["MVT_ID_mvt", "NM_TAXI_DIFF_SEC"]].copy()
    t["base"] = a["taxi"].to_numpy(float)
    t["base_std"] = a[list(config.MODEL_A)].std(axis=1, ddof=0).to_numpy()
    t = t.merge(T.base("test")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"]], on="MVT_ID_mvt", how="left", validate="one_to_one")
    t = apply_logging(_frame(t, "adsb_tracks"), pd.read_parquet(DIR / "logging.parquet"))
    X = matrix(t, pd.read_parquet(DIR / "stands.parquet"), C.opdi_features(), meta["type_categories"])
    assert list(X.columns) == meta["columns"]
    m = t["ADSB_MATCH"].fillna(0).to_numpy() > 0
    rhat = np.zeros(len(t))
    rhat[m] = np.mean([lgb.Booster(model_file=str(DIR / f"seed_{s}.txt")).predict(X[m]) for s in SEEDS], axis=0)
    p = np.maximum(config.TAXI_MIN, t["base"].to_numpy() + rhat)
    h = data.departures("test", ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    h["g"] = (h["MVT_TIME_UTC_mvt"] - h["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    g = np.maximum(config.TAXI_MIN, t[["MVT_ID_mvt"]].merge(h[["MVT_ID_mvt", "g"]], on="MVT_ID_mvt", how="left")["g"]
                   .fillna(0).to_numpy(float))
    lirf = m & (t["ADEP_mvt"] == "LIRF").to_numpy()
    pl = np.nan_to_num(a["lock_p"].to_numpy(float))
    p = np.where(lirf, np.maximum(config.TAXI_MIN, p + ECHO_BETA * pl * (g - p)), p)
    log(f"[adsb] tracked {m.sum():,} / {len(t):,} rows | RMS change {np.sqrt(np.mean((p - t['base']) ** 2)):.2f}")
    return pd.DataFrame({"MVT_ID_mvt": t["MVT_ID_mvt"], "ADEP_mvt": t["ADEP_mvt"], "taxi_base": t["base"], "rhat": rhat,
                         "taxi": p, "ADSB_MATCH": t["ADSB_MATCH"], "ADSB_CATEGORY": C.adsb_category(t)})
