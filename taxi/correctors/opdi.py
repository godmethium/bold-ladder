import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from taxi.correctors import common as C

DIR = config.MODELS / "opdi"
FEATURES = ["OPDI_FIRST", "OPDI_FIRST_GROUND", "OPDI_STAND_EXIT", "OPDI_STAND_DUR", "OPDI_APRON_EXIT", "OPDI_TAXI_ENTRY",
            "OPDI_TAXI_TOTAL", "OPDI_TAXI_MAX", "OPDI_TWY_N", "OPDI_RWY_ENTRY", "OPDI_RWY_DUR", "OPDI_RWY_N",
            "OPDI_STAND_MATCH", "OPDI_RWY_MATCH", "OPDI_N"]
PARAMS = dict(objective="huber", alpha=600.0, learning_rate=0.03, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8)
ROUNDS, SEEDS = 600, (42, 2026, 7)


def _ground(d):
    return (d["OPDI_TAXI_ENTRY"].notna() | d["OPDI_STAND_EXIT"].notna()).to_numpy()


def matrix(d, airport_categories):
    X = pd.DataFrame({"apt": pd.Categorical(d["ADEP_mvt"], categories=airport_categories), "base": d["base"].to_numpy(float),
                      "nm": d["NM_TAXI_DIFF_SEC"].to_numpy(float), "base_std": d["base_std"].to_numpy(float)})
    for c in FEATURES:
        X[c] = d[c].to_numpy(float)
    return X


def train(log=print):
    DIR.mkdir(parents=True, exist_ok=True)
    d = C.oof_base().merge(C.opdi_features(), on="MVT_ID_mvt", how="left", validate="one_to_one")
    airport_categories = sorted(d["ADEP_mvt"].unique())
    g = _ground(d)
    r = np.clip((d["y_taxi"] - d["base"]).to_numpy(), -3000, 3000)
    X = matrix(d, airport_categories)
    for s in SEEDS:
        lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X[g], r[g]), ROUNDS).save_model(str(DIR / f"seed_{s}.txt"))
    C.save_json(DIR / "meta.json", {"airport_categories": airport_categories})
    log(f"[opdi] training {g.sum():,} rows (with OPDI ground events)")


def apply(a, s, log=print):
    meta = C.load_json(DIR / "meta.json")
    t = s.copy()
    t["base"] = t["taxi_base"]
    t["base_std"] = a[list(config.MODEL_A)].std(axis=1, ddof=0).to_numpy()
    t["NM_TAXI_DIFF_SEC"] = a["NM_TAXI_DIFF_SEC"].to_numpy(float)
    t = t.merge(C.opdi_features(), on="MVT_ID_mvt", how="left", validate="one_to_one")
    g = _ground(t)
    rh = np.full(len(t), np.nan)
    if g.any():
        X = matrix(t, meta["airport_categories"])[g]
        rh[g] = np.mean([lgb.Booster(model_file=str(DIR / f"seed_{s_}.txt")).predict(X) for s_ in SEEDS], axis=0)
    rule = g & np.isin(t["ADSB_CATEGORY"].to_numpy(), C.WEAK)
    t["rhat_opdi"], t["opdi_rule"] = rh, rule.astype(int)
    t["taxi"] = np.where(rule, np.maximum(config.TAXI_MIN, t["base"].to_numpy() + np.nan_to_num(rh)), t["taxi"].to_numpy())
    log(f"[opdi] rows with OPDI ground events {g.sum():,} | rule applied {rule.sum():,}")
    return t[list(s.columns) + ["rhat_opdi", "opdi_rule"]]
