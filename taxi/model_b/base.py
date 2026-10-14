import re

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from taxi import data
from taxi.correctors import common as C
from taxi.features import tables as T

DIR = config.MODELS / "model_b"
TARGET, DELAY = "TAXITIME_SEC_mvt", "TAKEOFF_VS_SCHED_SEC"
PA_AIRPORTS = ("EHAM", "LSZH", "EDDF")
PA_NM = ["CALLSIGN_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "ADES_FILED", "LOBT_DIFF_SEC",
         "IOBT_DIFF_SEC", "EOBT_DIFF_SEC", "AOBT_DIFF_SEC", "ARVT3_DIFF_SEC", "AOBT_MINUS_EOBT", "NM_TAXI_DIFF_SEC",
         "DEST_VISIBILITY", "QUEUE_DELAY_EST", "NM_OFFSET_SEC", "NM_MISSING", "FLIGHT_TYPE_flt", "PLANNED_AIR_TIME",
         "AOBT_MINUS_IOBT"]
PA_EXCLUDED = {"MVT_ID_mvt", TARGET, "VISIBILITY", "IS_LVP", "IS_LOW_VIS", "QUEUE_RATIO_TO_HOUR_MEAN",
               "SCHED_DIFF_SEC_HOUR_PCT", "delay", "ADEP_mvt"} | set(PA_NM)
PA_CATEGORICAL = ["ADES_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "FLIGHT_CODE", "AIRLINE_LETTERS", "STAND_TYPE",
                  "ROUTE_PAIR"]
ALL_CATEGORICAL = ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "STAND_TYPE", "AIRLINE_LETTERS"]
ALL_EXCLUDED = {"FLIGHT_CODE", "ROUTE_PAIR", "FLIGHT_NUMBER"}
PA_REG = dict(objective="regression", metric="rmse", learning_rate=0.04, num_leaves=31, max_depth=5, min_child_samples=10,
              feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1, random_state=42, n_jobs=8, verbosity=-1)
LIRF_CLS = dict(objective="multiclass", num_class=3, metric="multi_logloss", learning_rate=0.03, num_leaves=31, max_depth=5,
                min_child_samples=6, feature_fraction=0.70, random_state=42, n_jobs=8, verbosity=-1)
LIRF_R0 = dict(objective="regression", metric="rmse", learning_rate=0.04, num_leaves=15, max_depth=4, min_child_samples=6,
               verbosity=-1, n_jobs=8)
ALL_MU = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, n_jobs=8, verbosity=-1, random_state=42)
ALL_P = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=31, max_depth=5, min_child_samples=10,
             feature_fraction=0.7, bagging_fraction=0.85, bagging_freq=1, n_jobs=8, verbosity=-1, random_state=42)
ALL_PH = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, n_jobs=8, verbosity=-1, random_state=42)
CAP = 3500.0
RATE_DELAY, RATE_PRIOR = 2700.0, 10.0
RATE_KEYS = ("airline", "stand")


def departure_columns():
    s = list(T.base("train").columns)
    return s[:s.index("RWY_PHYSICAL_LENGTH_FT")]


def pa_features():
    return [c for c in departure_columns() if c not in PA_EXCLUDED and not c.startswith("DEST_") and "holiday" not in c.lower()]


def all_features():
    return [c for c in pa_features() if c not in ALL_EXCLUDED] + ["ADEP_mvt", "IS_B"]


def b_rows(split):
    d = T.base(split)
    d = d[d["NM_TAXI_DIFF_SEC"].isna()]
    d = d[d[TARGET].notna()] if split == "train" else d[d["MVT_MONTH"].isin(config.TEST_MONTHS)]
    return d.reset_index(drop=True)


def next_day_condition(df):
    at = df["AIRCRAFT_TYPE_mvt"]
    empty = (at.isna() | at.astype(str).str.strip().isin(["", "nan", "None"])).to_numpy()
    return (df[DELAY].to_numpy(float) > df["MVT_MINOFDAY_UTC"].to_numpy(float) * 60) & empty


def _pa_frame(df, features):
    X = df[features].copy()
    for c in features:
        X[c] = X[c].astype("category") if c in PA_CATEGORICAL else pd.to_numeric(X[c], errors="coerce").astype(np.float32)
    return X


def _all_frame(df, features, categories=None):
    X = df[features].copy()
    for c in features:
        if c in ALL_CATEGORICAL:
            s = X[c].fillna("nan").astype(str)
            X[c] = pd.Categorical(s, categories=categories[c]) if categories else s.astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32).replace([np.inf, -np.inf], np.nan)
    return X


def _all_regime(y, d):
    r = np.zeros(len(y), dtype=int)
    r[(np.abs(y - d) <= 60) & (y >= 2000)] = 1
    r[y >= 80000] = 2
    return r


def _pa_regime(y):
    r = np.zeros(len(y), dtype=int)
    r[(y >= 2000) & (y < 40000)] = 1
    r[y >= 40000] = 2
    return r


def _lirf_departures():
    d = data.departures("train", ["FLIGHT_mvt", "STAND_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                                  "SCHED_TIME_UTC_mvt"])
    d = d[d["ADEP_mvt"] == "LIRF"].copy()
    d["month"] = d["MVT_TIME_UTC_mvt"].dt.month
    d["delay"] = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    d["ph"] = ((d["BLOCK_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds().abs() <= 60).astype(float)
    return d[d["delay"] > RATE_DELAY]


def _rate_keys(df):
    fl = df["FLIGHT_mvt"].fillna("").astype(str)
    airline = fl.map(lambda s: (re.match(r"^([A-Z]{3}|[A-Z]{2}|[A-Z]\d|\d[A-Z])", s) or [None, "?"])[1] if s else "?")
    return {"airline": airline, "stand": df["STAND_mvt"].fillna("?").astype(str)}


def _rate(ref, target, ref_k):
    p0 = ref["ph"].mean()
    g = ref.groupby(ref_k.loc[ref.index])["ph"].agg(["sum", "size"])
    o = (g["sum"] + RATE_PRIOR * p0) / (g["size"] + RATE_PRIOR)
    return target.map(o).fillna(p0).to_numpy(float), target.map(g["size"]).fillna(0).to_numpy(float)


def rate_features(df, ref, oof):
    ka, kr = _rate_keys(df), _rate_keys(ref)
    out = {}
    for name in RATE_KEYS:
        if oof:
            v, n = np.zeros(len(df)), np.zeros(len(df))
            for m in np.unique(df["MVT_MONTH"]):
                i = (df["MVT_MONTH"] == m).to_numpy()
                v[i], n[i] = _rate(ref[ref["month"] != m], ka[name][i], kr[name])
        else:
            v, n = _rate(ref, ka[name], kr[name])
        out[f"PH_RATE_{name.upper()}"], out[f"PH_N_{name.upper()}"] = v, n
    return pd.DataFrame(out, index=df.index)


def _flight_numbers(df):
    h = pd.concat([data.departures(k, ["FLIGHT_mvt"]) for k in ("train", "test")], ignore_index=True)
    return df[["MVT_ID_mvt"]].merge(h.drop_duplicates("MVT_ID_mvt"), on="MVT_ID_mvt", how="left")["FLIGHT_mvt"].to_numpy()


def train(log=print):
    DIR.mkdir(parents=True, exist_ok=True)
    tr = b_rows("train")
    y = tr[TARGET].to_numpy(float)
    features = pa_features()
    cats = [c for c in PA_CATEGORICAL if c in features]
    X = _pa_frame(tr, features)
    for apt in PA_AIRPORTS:
        i = (tr["ADEP_mvt"] == apt).to_numpy()
        lgb.train(PA_REG, lgb.Dataset(X[i], label=y[i], categorical_feature=cats), 250).save_model(
            str(DIR / f"per_airport_{apt}.txt"))
    i = (tr["ADEP_mvt"] == "LIRF").to_numpy()
    li = tr[i].reset_index(drop=True)
    li["FLIGHT_mvt"] = _flight_numbers(li)
    r, yl = _pa_regime(y[i]), y[i]
    Xl = X[i].reset_index(drop=True)
    lgb.train(LIRF_CLS, lgb.Dataset(Xl, label=r, categorical_feature=cats), 200).save_model(str(DIR / "lirf_cls.txt"))
    ref = _lirf_departures()
    Xo = pd.concat([_pa_frame(li, features), rate_features(li, ref, oof=True).reset_index(drop=True)], axis=1)
    lgb.train(LIRF_CLS, lgb.Dataset(Xo, label=r, categorical_feature=cats), 200).save_model(str(DIR / "lirf_cls_rate.txt"))
    lgb.train(LIRF_R0, lgb.Dataset(Xl[r == 0], label=yl[r == 0], categorical_feature=cats), 100).save_model(
        str(DIR / "lirf_r0.txt"))
    ref[["FLIGHT_mvt", "STAND_mvt", "month", "ph"]].to_parquet(DIR / "lirf_rate_ref.parquet", index=False)
    log(f"[model_b] per-airport and LIRF: {len(tr):,} training rows without NM (LIRF {i.sum():,})")
    af = all_features()
    full = T.base("train")
    full = full[full[TARGET].notna()].reset_index(drop=True)
    full["IS_B"] = full["NM_TAXI_DIFF_SEC"].isna().astype(np.float32)
    Xt = _all_frame(full, af)
    yt, rt = full[TARGET].to_numpy(float), _all_regime(full[TARGET].to_numpy(float), full[DELAY].to_numpy(float))
    b = full["IS_B"].to_numpy() == 1
    lgb.train(ALL_MU, lgb.Dataset(Xt[rt == 0], label=yt[rt == 0]), 600).save_model(str(DIR / "all_mu.txt"))
    lgb.train(ALL_P, lgb.Dataset(Xt[b], label=rt[b]), 300).save_model(str(DIR / "all_p.txt"))
    keep = rt != 2
    lgb.train(ALL_PH, lgb.Dataset(Xt[keep], label=(rt[keep] == 1).astype(float)), 400).save_model(str(DIR / "all_ph.txt"))
    res24 = float(np.median(yt[rt == 2] - 86400)) if (rt == 2).any() else 1000.0
    C.save_json(DIR / "meta.json", {"per_airport_features": features, "all_features": af,
                                    "all_categories": {c: list(Xt[c].cat.categories) for c in ALL_CATEGORICAL},
                                    "res24": res24})
    log(f"[model_b] all-rows model: {len(full):,} rows (B {b.sum():,}), next-day residual median {res24:.0f}")


def apply(log=print):
    meta = C.load_json(DIR / "meta.json")
    ap = b_rows("test")
    d = ap[DELAY].to_numpy(float)
    X = _pa_frame(ap, meta["per_airport_features"])
    out = np.full(len(ap), np.nan)
    for apt in PA_AIRPORTS:
        i = (ap["ADEP_mvt"] == apt).to_numpy()
        if i.any():
            out[i] = np.clip(lgb.Booster(model_file=str(DIR / f"per_airport_{apt}.txt")).predict(X[i]), 60.0, CAP)
    a2 = ap.assign(IS_B=np.float32(1.0))
    Xt = _all_frame(a2, meta["all_features"], meta["all_categories"])
    mu = lgb.Booster(model_file=str(DIR / "all_mu.txt")).predict(Xt)
    p = lgb.Booster(model_file=str(DIR / "all_p.txt")).predict(Xt)
    ph = lgb.Booster(model_file=str(DIR / "all_ph.txt")).predict(Xt)
    p24, d24 = p[:, 2], np.where(d > 40000, d, 86400 + meta["res24"])
    full = np.maximum(60.0, np.minimum(p24 * d24 + (1 - p24) * (ph * np.maximum(60.0, d) + (1 - ph) * mu), CAP))
    other = ~ap["ADEP_mvt"].isin(PA_AIRPORTS + ("LIRF",)).to_numpy()
    out[other] = full[other]
    i = (ap["ADEP_mvt"] == "LIRF").to_numpy()
    if i.any():
        li = ap[i].reset_index(drop=True)
        li["FLIGHT_mvt"] = _flight_numbers(li)
        Xl = X[i].reset_index(drop=True)
        dl = d[i]
        p2, p1 = np.where(dl > 40000, dl, 87400.0), np.clip(dl, 60.0, None)
        r0 = lgb.Booster(model_file=str(DIR / "lirf_r0.txt")).predict(Xl)
        pv = lgb.Booster(model_file=str(DIR / "lirf_cls.txt")).predict(Xl)
        pa = np.clip(pv[:, 0] * r0 + pv[:, 1] * p1 + pv[:, 2] * p2, 60.0, None)
        ref = pd.read_parquet(DIR / "lirf_rate_ref.parquet")
        Xo = pd.concat([_pa_frame(li, meta["per_airport_features"]), rate_features(li, ref, oof=False).reset_index(drop=True)],
                       axis=1)
        po = lgb.Booster(model_file=str(DIR / "lirf_cls_rate.txt")).predict(Xo)
        new = np.clip(po[:, 0] * mu[i] + po[:, 1] * p1 + po[:, 2] * p2, 60.0, None)
        out[i] = np.where(next_day_condition(li), pa, new)
    assert np.isfinite(out).all()
    log(f"[model_b] base: {len(ap):,} test rows without NM | mean {out.mean():.0f} s")
    return pd.DataFrame({"MVT_ID_mvt": ap["MVT_ID_mvt"].to_numpy(), "ADEP_mvt": ap["ADEP_mvt"].to_numpy(), "base": out})
