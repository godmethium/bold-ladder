import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from taxi import data
from taxi.correctors import common as C
from taxi.features import tables as T

OOF_B = config.ARTIFACTS / "oof" / "model_b_oof.parquet"
AIRPORTS = ("LTFM", "LFPG")
PA_AIRPORTS = ("EHAM", "LSZH", "EDDF", "LIRF")
ADSB_AIRPORTS = ("EDDF", "EDDM", "EHAM", "LEBL", "LSZH", "EGLL", "LEMD", "LFPG")
EXTRA_FILES = ["delay", "delay_runway", "arrivals_weather", "forward_windows"]
BASIC = ["delay_s", "hour", "weekday", "minute", "ADEP_mvt", "airline", "stand_group", "type_missing", "FLIGHT_RULE_mvt",
         "ades_country", "RUNWAY_mvt"]
RAW = ["ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "FLIGHT_mvt",
       "FLIGHT_RULE_mvt"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2)
ROUNDS, SEEDS, CLIP, R24 = 300, (1, 2, 3), 6000.0, 80_000.0


def _delay_columns(cols):
    return [c for c in cols if c.startswith(("DLY", "NMT_", "ARRV_", "WX_")) or c.endswith("_AFTER_10")]


def _epoch(s):
    s = pd.to_datetime(s)
    s = s.dt.tz_localize(None) if s.dt.tz is not None else s
    return (s - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()


def features(ids, split):
    t = data.departures(split, RAW)
    t = t[t["MVT_ID_mvt"].isin(ids)].drop_duplicates("MVT_ID_mvt")
    Tm, S = _epoch(t["MVT_TIME_UTC_mvt"]), _epoch(t["SCHED_TIME_UTC_mvt"])
    z = pd.to_datetime(Tm, unit="s")
    t["delay_s"] = Tm - S
    t["hour"], t["weekday"], t["month"], t["minute"] = z.hour, z.dayofweek, z.month, z.hour * 60 + z.minute
    t["type_missing"] = t["AIRCRAFT_TYPE_mvt"].isna() | t["AIRCRAFT_TYPE_mvt"].astype(str).str.strip().isin(["", "nan", "None"])
    t["airline"] = t["FLIGHT_mvt"].astype(str).str.extract(r"^([A-Z]{2,3}|[A-Z0-9]{2})", expand=False).fillna("?")
    t["stand_group"] = t["STAND_mvt"].astype(str).str[:1]
    t["ades_country"] = t["ADES_mvt"].astype(str).str[:2]
    t = t[["MVT_ID_mvt", "month"] + [c for c in BASIC if c != "ADEP_mvt"]]
    for f in EXTRA_FILES:
        g = T.table(f)
        g = g[g["MVT_ID_mvt"].isin(ids)]
        k = _delay_columns(g.columns)
        t = t.merge(g[["MVT_ID_mvt"] + k].astype({c: "float32" for c in k}), on="MVT_ID_mvt", how="left", validate="one_to_one")
    return pd.DataFrame({"MVT_ID_mvt": ids}).merge(t, on="MVT_ID_mvt", how="left", validate="one_to_one")


def _categorical(df, cols):
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(str).astype("category")
    return X


def training_rows():
    b = T.base("train")
    b = b[b["NM_TAXI_DIFF_SEC"].isna() & b["TAXITIME_SEC_mvt"].notna()][["MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt"]]
    b = b.reset_index(drop=True)
    b = b.merge(pd.read_parquet(OOF_B), on="MVT_ID_mvt", how="left", validate="one_to_one")
    tr = C.adsb_features("adsb_tracks", days=C.adsb_days("adsb_tracks", 2025), columns=["MVT_ID_mvt", "ADSB_MATCH"])
    b = b.merge(tr, on="MVT_ID_mvt", how="left", validate="one_to_one")
    base = np.where(b["ADEP_mvt"].isin(PA_AIRPORTS), b["base_pa"], b["base_all"])
    tracked = b["ADEP_mvt"].isin(ADSB_AIRPORTS) & (b["ADSB_MATCH"].fillna(0) > 0)
    b["base"] = np.maximum(config.TAXI_MIN, base + np.where(tracked, b["adsb_rhat"].fillna(0), 0.0))
    b["y"] = b["TAXITIME_SEC_mvt"].astype(float)
    f = features(b["MVT_ID_mvt"].to_numpy(), "train")
    return pd.concat([b[["MVT_ID_mvt", "ADEP_mvt", "y", "base"]], f.drop(columns="MVT_ID_mvt")], axis=1)


def apply(b, log=print, k=None):
    k = config.SCALE["model_b_residual"] if k is None else k
    d = training_rows()
    t = b[["MVT_ID_mvt", "ADEP_mvt", "taxi"]].rename(columns={"taxi": "base"}).reset_index(drop=True)
    t = pd.concat([t, features(t["MVT_ID_mvt"].to_numpy(), "test").drop(columns="MVT_ID_mvt")], axis=1)
    cols = ["base"] + BASIC + _delay_columns(d.columns)
    Xa = _categorical(pd.concat([d[cols], t[cols]], ignore_index=True), cols)
    X, Xt = Xa.iloc[:len(d)], Xa.iloc[len(d):]
    y, p0 = d["y"].to_numpy(), d["base"].to_numpy()
    tr = y < R24
    models = [lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X[tr], np.clip(y - p0, -CLIP, CLIP)[tr]), ROUNDS) for s in SEEDS]
    sel = t["ADEP_mvt"].isin(AIRPORTS).to_numpy()
    rhat = np.where(sel, k * np.mean([m.predict(Xt) for m in models], axis=0), 0.0)
    out = b.copy()
    tb = t["base"].to_numpy(float)
    out["taxi"] = np.where(sel, np.maximum(config.TAXI_MIN, tb + rhat), tb)
    out["rhat_residual"] = out["taxi"] - tb
    log(f"[model_b residual] training {tr.sum():,} | applied to {sel.sum():,} rows ({', '.join(AIRPORTS)}), k = {k}")
    return out
