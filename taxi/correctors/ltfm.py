import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from taxi.correctors import common as C
from taxi.features import tables as T

DIR = config.MODELS / "ltfm"
APT = "LTFM"
FILES = ["queue", "delay", "delay_runway", "forward_windows", "stand_arrivals"]
PARAMS = dict(objective="huber", alpha=300.0, learning_rate=0.03, num_leaves=63, min_data_in_leaf=300, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=12)
ROUNDS, SEEDS = 400, (1, 2, 3)
DEFAULTS = [(1134, 1148), (1194, 1208), (1254, 1266), (1374, 1386), (1434, 1448), (1494, 1508), (1734, 1746), (1794, 1806),
            (2636, 2646), (2694, 2706)]


def nm_features(df):
    nm = df["NM_TAXI_DIFF_SEC"].to_numpy(float)
    c = np.full(len(df), np.nan)
    for lo, hi in DEFAULTS:
        c[(nm >= lo) & (nm < hi)] = round((lo + hi) / 2 / 60) * 60
    return pd.DataFrame({"NM_TAXI": nm, "NM_IS_DEFAULT": np.isfinite(c).astype(np.float32), "NM_DEFAULT": c,
                         "NM_DEV": nm - c}, index=df.index).astype(np.float32)


def features(d, split):
    b = T.base(split)[["MVT_ID_mvt", "MVT_MINOFDAY_UTC", "SCHED_MINUS_AOBT_CDM"]]
    X = d.merge(b, on="MVT_ID_mvt", how="left", validate="one_to_one")
    for f in FILES:
        o = T.table(f)
        X = X.merge(o[o["MVT_ID_mvt"].isin(X["MVT_ID_mvt"])], on="MVT_ID_mvt", how="left", validate="one_to_one")
    X["base"] = np.maximum(config.TAXI_MIN, X["NM_TAXI_DIFF_SEC"] - X["offset_base"])
    X = pd.concat([X, nm_features(X)], axis=1)
    return X.drop(columns=["MVT_ID_mvt"]).astype("float32")


def train(log=print):
    DIR.mkdir(parents=True, exist_ok=True)
    d = C.oof_base()
    d = d[d["ADEP_mvt"] == APT].reset_index(drop=True)
    X = features(d[["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "offset_base", "base_std"]], "train")
    r = np.clip(d["y_taxi"].to_numpy() - X["base"].to_numpy(), -3000, 3000)
    for s in SEEDS:
        lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X, r), ROUNDS).save_model(str(DIR / f"seed_{s}.txt"))
    C.save_json(DIR / "meta.json", {"columns": list(X.columns)})
    log(f"[ltfm] training {len(d):,} rows")


def apply(a, s, log=print, k=None):
    k = config.SCALE["ltfm_second"] if k is None else k
    meta = C.load_json(DIR / "meta.json")
    if "ADEP_mvt" not in a:
        a = a.merge(s[["MVT_ID_mvt", "ADEP_mvt"]], on="MVT_ID_mvt", how="left", validate="one_to_one")
    h = a[a["ADEP_mvt"] == APT].rename(columns={"blend": "offset_base"}).reset_index(drop=True)
    h["base_std"] = h[list(config.MODEL_A)].std(axis=1, ddof=0)
    X = features(h[["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "offset_base", "base_std"]], "test")[meta["columns"]]
    r = pd.Series(np.mean([lgb.Booster(model_file=str(DIR / f"seed_{s_}.txt")).predict(X) for s_ in SEEDS], axis=0),
                  index=h["MVT_ID_mvt"])
    t = s.copy()
    t["rhat_ltfm"] = t["MVT_ID_mvt"].map(r).to_numpy(float)
    touch = ((t["ADEP_mvt"] == APT) & t["rhat_ltfm"].notna() & (t["ADSB_MATCH"].fillna(0) <= 0)
             & (t["opdi_rule"] == 0)).to_numpy()
    t["taxi"] = np.where(touch, np.maximum(config.TAXI_MIN, t["taxi"].to_numpy() + k * t["rhat_ltfm"].fillna(0).to_numpy()),
                         t["taxi"].to_numpy())
    log(f"[ltfm] applied to {touch.sum():,} rows (k = {k})")
    return t
