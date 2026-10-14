import pandas as pd

import config


def _read(files, columns=None):
    columns = None if columns is None else list(dict.fromkeys(["MVT_ID_mvt", "PHASE_mvt"] + list(columns)))
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def movements(split, columns=None):
    if split == "train":
        return _read(config.TRAIN_FILES, columns)
    d = _read(config.TEST_FILES, columns)
    return d.drop_duplicates("MVT_ID_mvt", keep="last").reset_index(drop=True) if len(config.TEST_FILES) > 1 else d


def departures(split, columns=None):
    d = movements(split, columns)
    d = d[d["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    if split == "test" and "MVT_TIME_UTC_mvt" in d:
        d = d[d["MVT_TIME_UTC_mvt"].dt.month.isin(config.TEST_MONTHS)].reset_index(drop=True)
    return d


def template():
    return pd.read_parquet(config.TEMPLATE, columns=["MVT_ID_mvt"])
