import json

import numpy as np
import pandas as pd

import config
from taxi import data
from taxi.features import tables as T

OOF_A = config.ARTIFACTS / "oof" / "model_a_oof.parquet"
WEAK = ("not processed", "no track", "takeoff roll / airborne only")


def oof_base():
    d = pd.read_parquet(OOF_A)
    if "y" not in d:
        e = T.base("train")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "MVT_DAY", "NM_TAXI_DIFF_SEC", config.TARGET]]
        d = d.merge(e.rename(columns={config.TARGET: "y"}), on="MVT_ID_mvt", how="left", validate="one_to_one")
        d = d[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "MVT_DAY", "NM_TAXI_DIFF_SEC", "y"] + list(config.MODEL_A) + ["split"]]
    comps = [d[name].to_numpy(float) for name in config.MODEL_A]
    d = d.drop(columns=list(config.MODEL_A) + ["split"])
    d["offset_base"] = sum(config.MODEL_A[name]["weight"] * v for name, v in zip(config.MODEL_A, comps))
    d["base_std"] = np.std(np.vstack(comps), axis=0)
    d["base"] = np.maximum(config.TAXI_MIN, d["NM_TAXI_DIFF_SEC"] - d["offset_base"])
    d["y_taxi"] = d["NM_TAXI_DIFF_SEC"] - d["y"]
    return d


def _day_files(folder):
    k = config.WORK / folder
    return {f.stem: f for f in sorted(k.glob("*.parquet"))} if k.exists() else {}


def adsb_days(folder, year=None):
    return sorted(d for d in _day_files(folder) if year is None or d.startswith(str(year)))


def adsb_features(folder="adsb_tracks", days=None, columns=None):
    fs = _day_files(folder)
    if days is not None:
        fs = {d: f for d, f in fs.items() if d in set(days)}
    d = pd.concat([pd.read_parquet(f, columns=columns) for f in fs.values()], ignore_index=True)
    return d.drop_duplicates("MVT_ID_mvt", keep="last")


def opdi_features():
    return T.table("opdi")


def adsb_category(d):
    match = d["ADSB_MATCH"]
    tracked = (match.fillna(0) > 0).to_numpy()
    cat = np.full(len(d), "not processed", dtype=object)
    cat[match.notna().to_numpy() & ~tracked] = "no track"
    cat[tracked] = "seen mid-taxi"
    cat[tracked & ((d["ADSB_FIRST_GS"] >= 60) | (d["ADSB_FIRST"] < 60) | d["ADSB_FIRST"].isna()).to_numpy()] = \
        "takeoff roll / airborne only"
    cat[tracked & (d["ADSB_FIRST_GS"] < 3).to_numpy()] = "seen at stand"
    return cat


def mvt_epoch(ids):
    x = pd.concat([data.departures(k, ["MVT_TIME_UTC_mvt"]) for k in ("train", "test")], ignore_index=True)
    x = x[x["MVT_ID_mvt"].isin(ids)].drop_duplicates("MVT_ID_mvt")
    x["mvt"] = ((x["MVT_TIME_UTC_mvt"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).astype(float)
    return x[["MVT_ID_mvt", "mvt"]]


def base_columns(ids, columns):
    parts = [T.base(k)[["MVT_ID_mvt"] + columns] for k in ("train", "test")]
    x = pd.concat(parts, ignore_index=True).drop_duplicates("MVT_ID_mvt")
    return x[x["MVT_ID_mvt"].isin(ids)]


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
