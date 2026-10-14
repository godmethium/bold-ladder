import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from taxi.correctors import adsb as D, common as C
from taxi.features import tables as T
from taxi.model_b import adsb as MBA, base as TB

QUARTERS = {"Q1": (1, 2, 3), "Q2": (4, 5, 6), "Q3": (7, 8, 9), "Q4": (10, 11, 12)}
ADSB_MONTHS = (1, 2, 6, 7, 8, 12)
DIR = config.WORK / "oof"


def _pa_predict(tr, va, features):
    cats = [c for c in TB.PA_CATEGORICAL if c in features]
    Xt, Xv = TB._pa_frame(tr, features), TB._pa_frame(va, features)
    out = np.full(len(va), np.nan)
    for apt in va["ADEP_mvt"].unique():
        ti, vi = (tr["ADEP_mvt"] == apt).to_numpy(), (va["ADEP_mvt"] == apt).to_numpy()
        y = tr.loc[ti, TB.TARGET].to_numpy(float)
        if apt == "LIRF":
            r = TB._pa_regime(y)
            cls = lgb.train(TB.LIRF_CLS, lgb.Dataset(Xt[ti], label=r, categorical_feature=cats), 200)
            r0 = lgb.train(TB.LIRF_R0, lgb.Dataset(Xt[ti][r == 0], label=y[r == 0], categorical_feature=cats), 100)
            p, d = cls.predict(Xv[vi]), va.loc[vi, TB.DELAY].to_numpy(float)
            pred = p[:, 0] * r0.predict(Xv[vi]) + p[:, 1] * np.clip(d, 60.0, None) + p[:, 2] * np.where(d > 40000, d, 87400.0)
            out[vi] = np.clip(pred, 60.0, None)
        else:
            m = lgb.train(TB.PA_REG, lgb.Dataset(Xt[ti], label=y, categorical_feature=cats), 250)
            out[vi] = np.clip(m.predict(Xv[vi]), 60.0, TB.CAP)
    return out


def _all_predict(tr, va, features):
    Xt = TB._all_frame(tr, features)
    Xv = TB._all_frame(va, features, {c: list(Xt[c].cat.categories) for c in TB.ALL_CATEGORICAL})
    y = tr[TB.TARGET].to_numpy(float)
    r = TB._all_regime(y, tr[TB.DELAY].to_numpy(float))
    b = tr["IS_B"].to_numpy() == 1
    mu = lgb.train(TB.ALL_MU, lgb.Dataset(Xt[r == 0], label=y[r == 0]), 600).predict(Xv)
    p = lgb.train(TB.ALL_P, lgb.Dataset(Xt[b], label=r[b]), 300).predict(Xv)
    keep = r != 2
    ph = lgb.train(TB.ALL_PH, lgb.Dataset(Xt[keep], label=(r[keep] == 1).astype(float)), 400).predict(Xv)
    res24 = float(np.median(y[r == 2] - 86400)) if (r == 2).any() else 1000.0
    d = va[TB.DELAY].to_numpy(float)
    p24, d24 = p[:, 2], np.where(d > 40000, d, 86400 + res24)
    pred = p24 * d24 + (1 - p24) * (ph * np.maximum(60.0, d) + (1 - ph) * mu)
    pred = np.where(va["ADEP_mvt"].to_numpy() == "LIRF", pred, np.minimum(pred, TB.CAP))
    return np.maximum(60.0, pred)


def base_oof(log=print):
    b = TB.b_rows("train")
    full = T.base("train")
    full = full[full[TB.TARGET].notna()].reset_index(drop=True)
    full["IS_B"] = full["NM_TAXI_DIFF_SEC"].isna().astype(np.float32)
    b["IS_B"] = np.float32(1.0)
    pa_features, all_features = TB.pa_features(), TB.all_features()
    out = b[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"]].assign(y=b[TB.TARGET].to_numpy(float), base_pa=np.nan, base_all=np.nan)
    for q, months in QUARTERS.items():
        va = b["MVT_MONTH"].isin(months).to_numpy()
        tr_b = b[~va].reset_index(drop=True)
        tr_all = full[~full["MVT_MONTH"].isin(months)].reset_index(drop=True)
        v = b[va].reset_index(drop=True)
        out.loc[va, "base_pa"] = _pa_predict(tr_b, v, pa_features)
        out.loc[va, "base_all"] = _all_predict(tr_all, v, all_features)
        log(f"[model_b oof] {q}: training without NM {len(tr_b):,} | all-rows training {len(tr_all):,} | validation {va.sum():,}")
    return out


def _logging_out_of_month(b, a):
    for prefix, ref in D._log_ref(a):
        for name, col in (("AIRLINE", "AIRLINE_LETTERS"), ("STAND", "STAND_mvt")):
            kb, kr, v = D._key(b, col), D._key(ref, col), np.full(len(b), np.nan)
            for month in np.unique(b["MVT_MONTH"]):
                rr = ref[ref["MVT_MONTH"] != month]
                prior, g = rr.groupby("ADEP_mvt")["u"].mean(), rr.groupby(kr.loc[rr.index])["u"].agg(["sum", "size"])
                i = (b["MVT_MONTH"] == month).to_numpy()
                o = b.loc[i, "ADEP_mvt"].map(prior).to_numpy(float)
                s_, n_ = kb[i].map(g["sum"]).fillna(0).to_numpy(float), kb[i].map(g["size"]).fillna(0).to_numpy(float)
                v[i] = (s_ + D.LOG_M * o) / (n_ + D.LOG_M)
            b[f"{prefix}_{name}"] = v
    return b


def adsb_oof(base, log=print):
    a = D.logging_oof(D.training_data("adsb_tracks"))
    b = base[base["MVT_MONTH"].isin(ADSB_MONTHS)].reset_index(drop=True)
    b["base"] = np.where(b["ADEP_mvt"].isin(TB.PA_AIRPORTS), b["base_pa"], b["base_all"])
    b = _logging_out_of_month(D._frame(b[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "base"]], "adsb_tracks"), a)
    b["NM_TAXI_DIFF_SEC"], b["base_std"] = np.nan, np.nan
    stands, opdi = D.stand_table(), C.opdi_features()
    Xa, _ = MBA._joint(a, b, stands, opdi, None)
    type_categories = list(Xa["actype"].cat.categories)
    Xa, Xb = MBA._joint(a, b, stands, opdi, type_categories)
    Xb = Xb[list(Xa.columns)]
    ra = (a["y_taxi"] - a["base"]).to_numpy(float)
    m_a, m_b = a["ADSB_MATCH"].fillna(0).to_numpy() > 0, b["ADSB_MATCH"].fillna(0).to_numpy() > 0
    month_a, month_b = a["MVT_MONTH"].to_numpy(), b["MVT_MONTH"].to_numpy()
    rhat = np.full(len(b), np.nan)
    for month in ADSB_MONTHS:
        tr, te = m_a & (month_a != month), m_b & (month_b == month)
        if te.any():
            models = [lgb.train({**D.PARAMS, "seed": s}, lgb.Dataset(Xa[tr], ra[tr]), D.ROUNDS) for s in D.SEEDS]
            rhat[te] = np.mean([m.predict(Xb[te]) for m in models], axis=0)
        log(f"[model_b oof] ADS-B month {month}: Model A training {tr.sum():,} -> without NM {te.sum():,}")
    return pd.DataFrame({"MVT_ID_mvt": b["MVT_ID_mvt"], "adsb_rhat": rhat})


def build(log=print):
    t = base_oof(log)
    out = t.merge(adsb_oof(t, log), on="MVT_ID_mvt", how="left", validate="one_to_one")
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / "model_b_oof.parquet"
    out[["MVT_ID_mvt", "base_pa", "base_all", "adsb_rhat"]].to_parquet(path, index=False)
    for name in ("base_pa", "base_all"):
        log(f"[model_b oof] {name} RMSE {np.sqrt(np.mean((out[name] - out['y']) ** 2)):.1f} s")
    return path


def copy():
    target = config.ARTIFACTS / "oof" / "model_b_oof.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.read_parquet(DIR / "model_b_oof.parquet").to_parquet(target, index=False)
    return target
