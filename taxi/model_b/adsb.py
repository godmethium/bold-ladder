import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from taxi.correctors import adsb as D, common as C

AIRPORTS = ("EDDF", "EDDM", "EHAM", "LEBL", "LSZH", "EGLL", "LEMD", "LFPG")
OPDI_AIRPORTS = ["LSZH"]


def _matrix(frame, stands, opdi, type_categories):
    return D.matrix(frame, stands, opdi, type_categories, stand_split=False, queue=False, opdi_airports=OPDI_AIRPORTS)


def _joint(left, right, stands, opdi, type_categories):
    cols = sorted(set(left.columns) & set(right.columns))
    X = _matrix(pd.concat([left[cols], right[cols]], ignore_index=True), stands, opdi, type_categories)
    return X.iloc[:len(left)].reset_index(drop=True), X.iloc[len(left):].reset_index(drop=True)


def apply(b, a_test, log=print):
    stands, opdi = pd.read_parquet(D.DIR / "stands.parquet"), C.opdi_features()
    logging = pd.read_parquet(D.DIR / "logging.parquet")
    a = D.logging_oof(D.training_data("adsb_tracks"))
    bt = b[["MVT_ID_mvt", "ADEP_mvt", "base"]].copy()
    bt["MVT_MONTH"] = bt[["MVT_ID_mvt"]].merge(D.T.base("test")[["MVT_ID_mvt", "MVT_MONTH"]], on="MVT_ID_mvt",
                                                 how="left")["MVT_MONTH"].to_numpy()
    bt = D.apply_logging(D._frame(bt, "adsb_tracks"), logging)
    bt["NM_TAXI_DIFF_SEC"], bt["base_std"] = np.nan, np.nan
    t = a_test[["MVT_ID_mvt", "NM_TAXI_DIFF_SEC"]].copy()
    t["base"] = a_test["taxi"].to_numpy(float)
    t["base_std"] = a_test[list(config.MODEL_A)].std(axis=1, ddof=0).to_numpy()
    t = t.merge(D.T.base("test")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"]], on="MVT_ID_mvt", how="left", validate="one_to_one")
    t = D.apply_logging(D._frame(t, "adsb_tracks"), logging)
    Xa, _ = _joint(a, bt, stands, opdi, None)
    type_categories = list(Xa["actype"].cat.categories)
    Xa, _ = _joint(a, bt, stands, opdi, type_categories)
    _, Xb = _joint(t, bt, stands, opdi, type_categories)
    Xb = Xb[list(Xa.columns)]
    m = a["ADSB_MATCH"].fillna(0).to_numpy() > 0
    ra = (a["y_taxi"] - a["base"]).to_numpy(float)
    models = [lgb.train({**D.PARAMS, "seed": s}, lgb.Dataset(Xa[m], ra[m]), D.ROUNDS) for s in D.SEEDS]
    mt = (bt["ADSB_MATCH"].fillna(0).to_numpy() > 0) & bt["ADEP_mvt"].isin(AIRPORTS).to_numpy()
    r = np.zeros(len(bt))
    if mt.any():
        r[mt] = np.mean([model.predict(Xb[mt]) for model in models], axis=0)
    out = b.copy()
    rr = out[["MVT_ID_mvt"]].merge(pd.DataFrame({"MVT_ID_mvt": bt["MVT_ID_mvt"], "r": r}), on="MVT_ID_mvt", how="left")
    out["rhat_adsb"] = rr["r"].fillna(0).to_numpy()
    applied = out["ADEP_mvt"].isin(AIRPORTS).to_numpy()
    out["base_adsb"] = np.where(applied, np.maximum(config.TAXI_MIN, out["base"] + out["rhat_adsb"]), out["base"])
    log(f"[model_b adsb] training {m.sum():,} rows with NM | corrected rows without NM {mt.sum():,}")
    return out
