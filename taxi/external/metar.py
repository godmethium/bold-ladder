import datetime as dt
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

import config

RAW = config.EXTERNAL / "metar_raw"
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIELDS = ["tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "wxcodes", "metar"]
CUT = re.compile(r"\s(?:TEMPO|BECMG|NOSIG|RMK)\b")
FLAGS = {
    "WX_TS": re.compile(r"\s(?:\+|-|VC)?TS|(?:\d{3}|///)CB"),
    "WX_FZ": re.compile(r"\sFZ(?:RA|DZ|FG)"),
    "WX_SN": re.compile(r"\s(?:\+|-|VC)?(?:SH|BL|DR)?(?:SN|SG|PL)\b"),
    "WX_FG": re.compile(r"\s(?:BC|MI|PR)?FG\b"),
    "WX_RA_HEAVY": re.compile(r"\s\+(?:SH)?RA"),
    "WX_TS_ONLY": re.compile(r"\s(?:\+|-|VC)?TS"),
}
WIND = re.compile(r"\s(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT\b")
VISIBILITY = re.compile(r"KT\s+(?:\d{3}V\d{3}\s+)?(CAVOK|\d{4})(?:NDV)?\b")
TEMPERATURE = re.compile(r"\s(M?\d{2})/(M?\d{2})(?=\s|$)")
VALUES = ["TEMP_C", "DEWPOINT_C", "VIS_M", "WIND_KT", "GUST_KT", "WIND_DIR"]


def _next(y, m):
    return (y + 1, 1) if m == 12 else (y, m + 1)


def _hours(y, m):
    y2, m2 = _next(y, m)
    return int((dt.datetime(y2, m2, 1) - dt.datetime(y, m, 1)).total_seconds() // 3600)


def _url(icao, y, m):
    y2, m2 = _next(y, m)
    q = [("station", icao)] + [("data", a) for a in FIELDS] + [
        ("year1", y), ("month1", m), ("day1", 1), ("year2", y2), ("month2", m2), ("day2", 1), ("tz", "Etc/UTC"),
        ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"), ("missing", "M"), ("trace", "T"), ("direct", "no"),
        ("report_type", 3), ("report_type", 4)]
    return URL + "?" + "&".join(f"{k}={v}" for k, v in q)


def _valid(path, y, m):
    if not path.exists() or path.stat().st_size < 1000:
        return False
    with open(path, encoding="utf-8", errors="replace") as f:
        first, n = f.readline(), sum(1 for _ in f)
    return first.startswith("station,valid") and n >= 0.5 * _hours(y, m)


def download(months, wait=5, attempts=5):
    RAW.mkdir(parents=True, exist_ok=True)
    for icao in config.AIRPORTS:
        for y, m in months:
            path = RAW / f"metar_{icao}_{y}{m:02d}.csv"
            for a in range(1, attempts + 1):
                if _valid(path, y, m):
                    break
                try:
                    req = urllib.request.Request(_url(icao, y, m), headers={"User-Agent": "taxi-out-prediction/1.0"})
                    with urllib.request.urlopen(req, timeout=180) as r:
                        path.write_bytes(r.read())
                except Exception as e:
                    print(icao, y, m, "error:", e, file=sys.stderr)
                time.sleep(wait * (1 if _valid(path, y, m) else 2 ** a))
            print(icao, f"{y}-{m:02d}", "ok" if _valid(path, y, m) else "MISSING", flush=True)


def _degrees(s):
    return np.nan if s is None else (-float(s[1:]) if s.startswith("M") else float(s))


def _parse(text):
    full = " " + str(text).strip() + " "
    k = CUT.search(full)
    obs = full[:k.start()] + " " if k else full
    r = {a: bool(p.search(obs)) for a, p in FLAGS.items()}
    w = WIND.search(obs)
    r["WIND_DIR"] = np.nan if (w is None or w.group(1) == "VRB") else float(w.group(1))
    r["WIND_KT"] = float(w.group(2)) if w else np.nan
    r["GUST_KT"] = float(w.group(3)) if (w and w.group(3)) else np.nan
    v = VISIBILITY.search(obs)
    r["VIS_M"] = np.nan if v is None else (10000.0 if v.group(1) in ("CAVOK", "9999") else float(v.group(1)))
    t = TEMPERATURE.search(obs)
    r["TEMP_C"], r["DEWPOINT_C"] = (_degrees(t.group(1)), _degrees(t.group(2))) if t else (np.nan, np.nan)
    return r


def hourly(months=None, name="metar_hourly_extra.parquet"):
    parts = []
    for f in sorted(RAW.glob("metar_*_*.csv")):
        icao, month = f.stem.split("_")[1], f.stem.split("_")[2]
        if months is not None and (int(month[:4]), int(month[4:])) not in months:
            continue
        d = pd.read_csv(f, usecols=["station", "valid", "metar"], dtype=str)
        d = d[d["metar"].notna() & d["valid"].notna()]
        p = pd.DataFrame([_parse(x) for x in d["metar"]], index=d.index)
        p["ICAO"], p["hour"], p["_month"] = icao, pd.to_datetime(d["valid"]).dt.floor("h"), month
        parts.append(p)
    r = pd.concat(parts, ignore_index=True)
    g = r.groupby(["ICAO", "hour"])
    s = g[list(FLAGS)].max()
    s["TEMP_C"], s["DEWPOINT_C"] = g["TEMP_C"].mean(), g["DEWPOINT_C"].mean()
    s["VIS_M"], s["WIND_KT"], s["GUST_KT"] = g["VIS_M"].min(), g["WIND_KT"].mean(), g["GUST_KT"].max()
    s["WIND_DIR"], s["REPORT_N"] = g["WIND_DIR"].last(), g.size()
    s = s.reset_index()
    grid = pd.concat([pd.DataFrame({"ICAO": i, "hour": pd.date_range(pd.Timestamp(f"{a[:4]}-{a[4:]}-01"),
                                                                      pd.Timestamp(f"{a[:4]}-{a[4:]}-01") + pd.offsets.MonthBegin(1),
                                                                      freq="h", inclusive="left")})
                      for i, a in r[["ICAO", "_month"]].drop_duplicates().itertuples(index=False)], ignore_index=True)
    t = grid.merge(s, on=["ICAO", "hour"], how="left")
    t["REPORT_N"] = t["REPORT_N"].fillna(0).astype("int16")
    for c in FLAGS:
        t[c] = t[c].astype("boolean")
    for c in VALUES:
        t[c] = t[c].astype("float32")
    t = t[["ICAO", "hour"] + list(FLAGS) + VALUES + ["REPORT_N"]].sort_values(["ICAO", "hour"]).reset_index(drop=True)
    t.to_parquet(config.EXTERNAL / name, index=False)
    gaps = t.assign(month=t["hour"].dt.strftime("%Y-%m")).groupby(["ICAO", "month"])["REPORT_N"].apply(lambda s: (s == 0).mean())
    print(f"-> {name}: {len(t):,} hours | max share of hours without a report {gaps.max():.1%}")


if __name__ == "__main__":
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    months = [tuple(int(x) for x in a.split("-")) for a in sys.argv[1:] if "=" not in a]
    download(months)
    hourly(months, opt.get("output", "metar_hourly_extra.parquet"))
