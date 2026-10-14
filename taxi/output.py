import csv
import hashlib
import re
import time

import numpy as np
import pandas as pd

import config
from taxi import data


def next_path():
    nums = [int(m.group(1)) for f in config.OUTPUT.glob("prediction_*.parquet")
            if (m := re.fullmatch(r"prediction_(\d+)\.parquet", f.name))]
    return config.OUTPUT / f"prediction_{max(nums, default=0) + 1}.parquet"


def check(prediction):
    s = data.template()
    d = s.merge(prediction[["MVT_ID_mvt", "taxi"]], on="MVT_ID_mvt", how="left", validate="one_to_one")
    t = d["taxi"].to_numpy(float)
    checks = {"row count": len(d) == len(s), "no missing": bool(np.isfinite(t).all()),
              "floor 60": bool((t >= config.TAXI_MIN - 1e-9).all()),
              "id order": bool(np.array_equal(d["MVT_ID_mvt"].to_numpy(), s["MVT_ID_mvt"].to_numpy()))}
    return d, checks


def write(prediction, note, log=print):
    d, checks = check(prediction)
    assert all(checks.values()), checks
    path = next_path()
    config.OUTPUT.mkdir(exist_ok=True)
    t = d["taxi"].to_numpy(float)
    pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "TAXITIME_SEC_mvt": t.astype(np.float32)}).to_parquet(path, index=False)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    config.LOGS.mkdir(exist_ok=True)
    ledger = config.LOGS / "output_ledger.csv"
    new = not ledger.exists()
    with open(ledger, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "file", "md5", "note"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), path.name, md5, note])
    log(f"-> {path.relative_to(config.ROOT)} | MD5 {md5} | checks ok | mean taxi {t.mean():.1f}")
    return path
