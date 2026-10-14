import json
import time

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

import config
from taxi.features import matrices as MX


def _folder(name, root=None, create=True):
    k = (root or config.MODELS / "model_a") / name
    if create:
        k.mkdir(parents=True, exist_ok=True)
    return k


def _airports(df):
    return sorted(df["ADEP_mvt"].unique())


def train(name, frame, features, log=print, root=None):
    c = config.MODEL_A[name]
    k = _folder(name, root)
    (k / "features.json").write_text(json.dumps(features, ensure_ascii=False), encoding="utf-8")
    y_all = frame[config.TARGET].to_numpy(float)
    t0 = time.time()
    if c["kind"] == "catboost":
        cats = MX.gcb_categorical(c["cat"])
        d = MX.gcb_frame(frame, features, c["cat"])
        y = np.clip(y_all, -config.GCB_TARGET_CLIP, config.GCB_TARGET_CLIP)
        for s in c["seeds"]:
            m = CatBoostRegressor(**config.cb_device(**c["params"], random_seed=s, verbose=1000, thread_count=config.CB_THREADS))
            m.fit(Pool(d, y, cat_features=cats))
            m.save_model(str(k / f"seed_{s}.cbm"))
            log(f"[{name}] seed {s} done | {time.time() - t0:.0f} s")
        return
    for apt in _airports(frame):
        i = (frame["ADEP_mvt"] == apt).to_numpy()
        e, y = frame[i], y_all[i]
        X = MX.lgb_matrix(e, features)
        if c["kind"] == "xgboost":
            n, _ = MX.numeric(X, features)
            for s in c["seeds"]:
                xgb.XGBRegressor(**c["params"], device=config.XGB_DEVICE, random_state=s).fit(n, y).save_model(
                    str(k / f"{apt}_seed_{s}.json"))
        else:
            mask = np.ones(len(y), bool)
            if c["kind"] == "lock_mixture":
                lock = np.abs(y - e[config.LOCK_COLUMN].to_numpy(float)) <= config.LOCK_TOL
                lgb.train(c["classifier"], lgb.Dataset(X, label=lock.astype(float)),
                          num_boost_round=c["classifier_rounds"]).save_model(str(k / f"{apt}_lock.txt"))
                mask = ~lock
            for s in c["seeds"]:
                lgb.train({**c["params"], "random_state": s}, lgb.Dataset(X[mask], label=y[mask]),
                          num_boost_round=c["rounds"]).save_model(str(k / f"{apt}_seed_{s}.txt"))
        log(f"[{name}] {apt}: {i.sum():,} rows | {time.time() - t0:.0f} s")


def lock_probability(test, root=None):
    name = next(n for n, c in config.MODEL_A.items() if c["kind"] == "lock_mixture")
    k = _folder(name, root, create=False)
    features = json.loads((k / "features.json").read_text(encoding="utf-8"))
    out = np.full(len(test), np.nan)
    for apt in _airports(test):
        i = (test["ADEP_mvt"] == apt).to_numpy()
        out[i] = lgb.Booster(model_file=str(k / f"{apt}_lock.txt")).predict(MX.lgb_matrix(test[i], features))
    return out


def predict(name, test, root=None):
    c = config.MODEL_A[name]
    k = _folder(name, root, create=False)
    features = json.loads((k / "features.json").read_text(encoding="utf-8"))
    missing = [f for f in features if f not in test.columns]
    assert not missing, f"[{name}] features missing in test: {missing[:5]}"
    if c["kind"] == "catboost":
        cats = MX.gcb_categorical(c["cat"])
        p = Pool(MX.gcb_frame(test, features, c["cat"]), cat_features=cats)
        return np.mean([CatBoostRegressor().load_model(str(k / f"seed_{s}.cbm")).predict(p) for s in c["seeds"]], axis=0)
    out = np.full(len(test), np.nan)
    for apt in _airports(test):
        i = (test["ADEP_mvt"] == apt).to_numpy()
        X = MX.lgb_matrix(test[i], features)
        if c["kind"] == "xgboost":
            n, _ = MX.numeric(X, features)
            out[i] = np.mean([_load_xgb(k / f"{apt}_seed_{s}.json").predict(n) for s in c["seeds"]], axis=0)
            continue
        mu = np.mean([lgb.Booster(model_file=str(k / f"{apt}_seed_{s}.txt")).predict(X) for s in c["seeds"]], axis=0)
        if c["kind"] == "lock_mixture":
            p = lgb.Booster(model_file=str(k / f"{apt}_lock.txt")).predict(X)
            mu = p * test.loc[i, config.LOCK_COLUMN].to_numpy(float) + (1 - p) * mu
        out[i] = mu
    assert np.isfinite(out).all(), f"[{name}] missing predictions"
    return out


def _load_xgb(path):
    m = xgb.XGBRegressor()
    m.load_model(str(path))
    return m
