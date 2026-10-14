import numpy as np
import pandas as pd

import config
from taxi.features import matrices as MX, tables as T
from taxi.model_a import components as B

DIR = config.WORK / "oof"


def frames(split):
    a = T.base("train")
    a = a[a[config.TARGET].notna()].reset_index(drop=True)
    out = a["MVT_MONTH"].isin(config.OOF_SPLITS[split]).to_numpy()
    tr = a[~out & T.clean_mask(a).to_numpy()].reset_index(drop=True)
    va = a[out].reset_index(drop=True)
    tr, va = T.add_groups(tr, va)
    T.add_te(tr, va)
    return tr, va


def build(split, components=None, log=print):
    components = components or tuple(config.MODEL_A)
    tr, va = frames(split)
    features = MX.feature_list(tr.columns)
    root = config.MODELS / "model_a_oof" / split
    log(f"[oof {split}] train {len(tr):,} | held out {len(va):,} ({config.OOF_SPLITS[split]})")
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / f"model_a_oof_{split}.parquet"
    out = pd.read_parquet(path) if path.exists() else va[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "MVT_DAY",
                                                          "NM_TAXI_DIFF_SEC"]].assign(y=va[config.TARGET].to_numpy(float))
    for name in components:
        B.train(name, tr, features, log, root=root)
        out[name] = B.predict(name, va, root=root)
        out.to_parquet(path, index=False)
        rmse = np.sqrt(np.mean((out[name] - out["y"]) ** 2))
        log(f"[oof {split}] {name}: offset RMSE {rmse:.1f} s -> {path.relative_to(config.ROOT)}")
    return out


def combine():
    parts = []
    for b in config.OOF_SPLITS:
        d = pd.read_parquet(DIR / f"model_a_oof_{b}.parquet")
        missing = [name for name in config.MODEL_A if name not in d]
        assert not missing, f"{b}: missing components {missing}"
        parts.append(d.assign(split=b))
    d = pd.concat(parts, ignore_index=True)
    path = config.ARTIFACTS / "oof" / "model_a_oof.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    d[["MVT_ID_mvt"] + list(config.MODEL_A) + ["split"]].to_parquet(path, index=False)
    return path
