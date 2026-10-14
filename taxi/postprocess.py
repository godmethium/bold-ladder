import re
import zipfile

import numpy as np
import pandas as pd

import config
from taxi import data

RAW_COLUMNS = ["ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt", "AIRCRAFT_TYPE_mvt"]
MONTHLY_TABLE = config.EXTERNAL / "Taxi-Out_Additional_Time.xlsx"


def add_raw(df):
    h = data.departures("test", RAW_COLUMNS)
    return df.merge(h, on="MVT_ID_mvt", how="left", validate="one_to_one")


def time_shift(df):
    s = pd.read_parquet(config.FINAL_TEST_FILE, columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    s = s[s["PHASE_mvt"] == "DEP"]
    old = df["MVT_TIME_UTC_mvt"].reset_index(drop=True)
    new = df[["MVT_ID_mvt"]].merge(s, on="MVT_ID_mvt", how="left", validate="one_to_one")["MVT_TIME_UTC_mvt"].fillna(old)
    shift = (new - old).dt.total_seconds().to_numpy()
    df["MVT_TIME_UTC_mvt"] = new.set_axis(df.index)
    return df["taxi"].to_numpy(float) + shift, shift


def published():
    z = zipfile.ZipFile(MONTHLY_TABLE)
    ss = ["".join(re.findall(r"<t[^>]*>([^<]*)</t>", s))
          for s in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode("utf-8"), re.S)]
    rows = []
    for r in re.findall(r"<row[^>]*>(.*?)</row>", z.read("xl/worksheets/sheet4.xml").decode("utf-8"), re.S):
        h = []
        for c in re.findall(r"<c ([^>]*?)(?:/>|>(.*?)</c>)", r, re.S):
            t, v = re.search(r't="(\w+)"', c[0]), re.search(r"<v>([^<]*)</v>", c[1] or "")
            h.append((ss[int(v.group(1))] if t and t.group(1) == "s" else v.group(1)) if v else "")
        rows.append(h)
    d = pd.DataFrame([r[:len(rows[0])] for r in rows[1:]], columns=rows[0])
    d = d[d["APT_ICAO"].isin(config.AIRPORTS)].copy()
    for c in ["YEAR", "MONTH_NUM", "TF", "VALID_FL", "TOTAL_REF_NB_FL", "TOTAL_REF_TIME_MIN", "TOTAL_ADD_TIME_MIN"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["TOTAL_REF_NB_FL"] > 0]
    d["pub"] = (d["TOTAL_REF_TIME_MIN"] + d["TOTAL_ADD_TIME_MIN"]) * 60 / d["TOTAL_REF_NB_FL"]
    d["invalid"] = (d["TF"] - d["VALID_FL"]) / d["TF"]
    return d.set_index(["APT_ICAO", "YEAR", "MONTH_NUM"])


def _trimmed_mean(x, K):
    x, K = np.sort(np.asarray(x, float)), int(round(K))
    return float(x[:len(x) - K].mean()) if 0 <= K < len(x) else np.nan


def monthly_calibration(df, k=None):
    k = config.SCALE["monthly_calibration"] if k is None else k
    y = published()
    e = data.departures("train", ["ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
    e["month"] = e["MVT_TIME_UTC_mvt"].dt.month
    diffs = []
    for (a, m), g in e.groupby(["ADEP_mvt", "month"]):
        r = y.loc[(a, config.TRAIN_YEAR, m)]
        if r["invalid"] <= 0.10:
            diffs.append((a, r["pub"] - _trimmed_mean(g["TAXITIME_SEC_mvt"], (r["TF"] - r["VALID_FL"]) * len(g) / r["TF"])))
    fs = pd.DataFrame(diffs, columns=["apt", "diff"]).groupby("apt")["diff"].agg(["mean", "std"])
    p = df["taxi"].to_numpy(float)
    t = df["MVT_TIME_UTC_mvt"]
    shift, report = np.zeros(len(df)), []
    cell = pd.DataFrame({"a": df["ADEP_mvt"].to_numpy(), "y": t.dt.year.to_numpy(), "m": t.dt.month.to_numpy()})
    for (a, year, m), i in cell.groupby(["a", "y", "m"]).indices.items():
        r, f = y.loc[(a, year, m)], fs.loc[a]
        K = (r["TF"] - r["VALID_FL"]) * len(i) / r["TF"]
        bias = _trimmed_mean(p[i], K) - (r["pub"] - f["mean"])
        km = 0.0 if r["invalid"] > 0.10 else -np.sign(bias) * max(0.0, abs(bias) - 2 * f["std"])
        shift[i] = k * km
        report.append(dict(apt=a, month=m, rows=len(i), published=r["pub"], bias=bias, shift=k * km))
    return p + shift, pd.DataFrame(report)


def lobt_window(df):
    mid = (df["MVT_TIME_UTC_mvt"] - df["LOBT_flt"]).dt.total_seconds().to_numpy()
    p = df["taxi"].to_numpy(float)
    lo, hi = np.maximum(config.TAXI_MIN, mid - config.LOBT_WINDOW), np.maximum(config.TAXI_MIN, mid + config.LOBT_WINDOW)
    return np.maximum(config.TAXI_MIN, np.where(np.isfinite(mid), np.clip(p, lo, hi), p))


def apply(df, log=print, calibration=True):
    df = add_raw(df[["MVT_ID_mvt", "taxi"]].copy())
    df["taxi"], shift = time_shift(df)
    log(f"time shift: {int((shift != 0).sum()):,} rows, RMS {np.sqrt(np.mean(shift ** 2)):.2f} s")
    report = None
    if calibration:
        before = df["taxi"].to_numpy(float)
        df["taxi"], report = monthly_calibration(df)
        log(f"monthly calibration: shift in {int((report['shift'] != 0).sum())}/{len(report)} cells, RMS change "
            f"{np.sqrt(np.mean((df['taxi'] - before) ** 2)):.2f}")
    before = df["taxi"].to_numpy(float)
    df["taxi"] = lobt_window(df)
    log(f"LOBT window: {int((np.abs(df['taxi'] - before) > 1e-6).sum())} rows clipped")
    return df[["MVT_ID_mvt", "taxi"]], report
