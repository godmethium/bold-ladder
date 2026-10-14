import numpy as np
import pandas as pd

from taxi.features.tables import TE_COLUMNS

CATEGORICAL = ["AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt"]
EXCLUDED = {"MVT_ID_mvt", "TAXITIME_SEC_mvt", "NM_OFFSET_SEC", "NM_TAXI_DIFF_SEC", "ADEP_mvt", "VISIBILITY", "IS_LVP",
            "IS_LOW_VIS"}
GCB_EXTRA_CATEGORICAL = {"basic": [], "extended": ["ADES_mvt", "MARKET_SEGMENT_flt", "STAND_TYPE", "AIRLINE_LETTERS", "ROUTE_PAIR"]}


def feature_list(columns):
    c = [s for s in columns if s not in TE_COLUMNS]
    i = c.index("RWY_PHYSICAL_LENGTH_FT")
    c[i:i] = TE_COLUMNS[:-1]
    c.insert(c.index("RWY_DISPLACED_THRESH_FT") + 1, TE_COLUMNS[-1])
    out = [s for s in c if s not in EXCLUDED and not s.startswith("DEST_") and "holiday" not in s.lower()]
    assert len(out) == len(set(out)), "duplicate feature name"
    return out


def lgb_matrix(df, features):
    X = df[features].copy()
    for c in features:
        if c in CATEGORICAL:
            X[c] = X[c].fillna("nan").astype(str).astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(np.float32)
    return X


def numeric(X, features):
    num = [c for c in features if c not in CATEGORICAL]
    a = X[num].to_numpy(dtype=np.float32)
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6), num


def gcb_categorical(cat):
    return ["ADEP_mvt"] + CATEGORICAL + GCB_EXTRA_CATEGORICAL[cat]


def gcb_frame(df, features, cat):
    fn = [c for c in features if c not in GCB_EXTRA_CATEGORICAL[cat]]
    a, num = numeric(lgb_matrix(df, fn), fn)
    cats = gcb_categorical(cat)
    return pd.concat([pd.DataFrame(a, columns=num), df[cats].fillna("nan").astype(str).reset_index(drop=True)], axis=1)
