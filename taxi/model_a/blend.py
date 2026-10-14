import numpy as np

import config


def blend(offsets):
    w = {k: config.MODEL_A[k]["weight"] for k in offsets}
    return sum(w[k] * np.asarray(v, float) for k, v in offsets.items()) / sum(w.values())


def taxi(nm_taxi, offset):
    return np.maximum(config.TAXI_MIN, np.asarray(nm_taxi, float) - np.asarray(offset, float))


def choose_weights(oof=None, splits=("split_1", "split_2")):
    import pandas as pd
    from scipy.optimize import minimize
    d = pd.read_parquet(config.ARTIFACTS / "oof" / "model_a_oof.parquet") if oof is None else oof
    if "y" not in d:
        from taxi.features import tables as T
        e = T.base("train")[["MVT_ID_mvt", "MVT_MONTH", config.TARGET]].rename(columns={config.TARGET: "y"})
        d = d.merge(e, on="MVT_ID_mvt", how="left", validate="one_to_one")
    names = list(config.MODEL_A)
    picks = []
    for b in splits:
        x = d[d["split"] == b]
        first, second = config.OOF_SPLITS[b]
        share = 0.4425
        w = x["MVT_MONTH"].map({first: share / (x["MVT_MONTH"] == first).sum(),
                                second: (1 - share) / (x["MVT_MONTH"] == second).sum()}).to_numpy(float)
        P, y = x[names].to_numpy(float), x["y"].to_numpy(float)
        r = minimize(lambda a: float(np.sum(w * (P @ a - y) ** 2)), np.full(len(names), 1 / len(names)), method="SLSQP",
                     bounds=[(0, 1)] * len(names), constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1}])
        picks.append(np.round(r.x, 3))
    a = np.mean(picks, axis=0)
    return dict(zip(names, np.round(a / a.sum(), 4).tolist()))
