import datetime as dt
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config

URL = "https://www.eurocontrol.int/performance/data/download/OPDI/v002"
RAW = config.OPDI_ROOT / "raw"
FILTERED = config.OPDI_ROOT / "filtered"
CALENDAR = dt.date(2022, 1, 1)
REF = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4775, -0.4614), "EHAM": (52.3081, 4.7642),
       "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
       "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519)}
POINT_TYPES = ["first_seen", "take-off", "landing", "last_seen"]
INFO = (r'"osm_id":(?P<osm_id>\d+),"osm_aeroway":"(?P<osm_aeroway>[^"]*)","osm_ref":"(?P<osm_ref>[^"]*)",'
        r'"osm_airport":"(?P<osm_airport>[^"]*)","opdi_time_in_use_s":(?P<duration>[-0-9.]+)')
OSN = r'"osn_icao24":"(?P<icao24>[^"]*)","osn_flight_id":"(?P<callsign>[^"]*)"'
KM = 15.0


def files(months):
    events, lists = [], []
    for month in months:
        y, m = map(int, month.split("-"))
        start, end = dt.date(y, m, 1), dt.date(y + (m == 12), m % 12 + 1, 1)
        a = CALENDAR + dt.timedelta(days=10 * ((start - dt.timedelta(days=1) - CALENDAR).days // 10))
        while a < end:
            name = f"flight_events_{a:%Y%m%d}_{a + dt.timedelta(days=10):%Y%m%d}.parquet"
            events += [name] if name not in events else []
            a += dt.timedelta(days=10)
        for d in (start - dt.timedelta(days=1), start):
            name = f"flight_list_{d:%Y%m}.parquet"
            lists += [name] if name not in lists else []
    return [("flight_events", a) for a in events] + [("flight_list", a) for a in lists]


def _size(url, attempts=8):
    for i in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
                return int(r.headers["Content-Length"])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if (isinstance(e, urllib.error.HTTPError) and e.code == 404) or i == attempts - 1:
                raise
            time.sleep(min(300, 15 * 2 ** i))


def _download(kind, name):
    url, path = f"{URL}/{kind}/{name}", RAW / name
    n = _size(url)
    if path.exists() and path.stat().st_size == n:
        return name, n, "exists"
    part = path.with_suffix(".parquet.part")
    for attempt in range(5):
        before = part.stat().st_size if part.exists() else 0
        r = subprocess.run(["curl", "-sS", "-L", "--http1.1", "-C", "-", "--retry", "5", "--retry-delay", "10",
                            "-o", str(part), url])
        now = part.stat().st_size if part.exists() else 0
        if now == n:
            part.rename(path)
            return name, n, "downloaded"
        print(f"  {name}: attempt {attempt + 1} incomplete ({now:,} / {n:,}; curl {r.returncode})", flush=True)
        if now <= before or now > n:
            part.unlink(missing_ok=True)
        time.sleep(15)
    raise RuntimeError(f"{name} could not be downloaded")


def download(months, workers=4):
    RAW.mkdir(parents=True, exist_ok=True)
    todo = files(months)
    with ThreadPoolExecutor(workers) as ex:
        for name, n, status in ex.map(lambda x: _download(*x), todo):
            print(f"{time.strftime('%H:%M:%S')} {name:45s} {n / 1e6:7.0f} MB  {status}", flush=True)


def _i64(a):
    a = np.asarray(a)
    return a.view(np.int64) if a.dtype == np.uint64 else a.astype(np.int64)


def flight_list():
    parts = []
    for f in sorted(RAW.glob("flight_list_*.parquet")):
        x = pd.read_parquet(f, columns=["id", "adep", "ades", "flt_id", "icao24", "registration", "typecode"])
        x["id"] = _i64(x["id"].to_numpy())
        parts.append(x)
    fl = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    fl["callsign"] = fl.pop("flt_id").astype(str).str.strip().str.upper()
    return fl.rename(columns={"id": "flight_id"})


def _nearest(lat, lon):
    names = np.array(list(REF))
    la, lo = np.array([v[0] for v in REF.values()]), np.array([v[1] for v in REF.values()])
    d = 111.2 * np.hypot(lat[:, None] - la, (lon[:, None] - lo) * np.cos(np.radians(la)))
    i = np.argmin(d, axis=1)
    return names[i], d[np.arange(len(d)), i]


_FL = None


def _init(fl):
    global _FL
    _FL = fl


def _filter(path):
    target = FILTERED / path.name
    if target.exists():
        return path.name, "exists"
    t = pq.read_table(path, columns=["id", "flight_id", "type", "event_time", "longitude", "latitude", "altitude", "info"])
    info = pc.extract_regex(pc.fill_null(t["info"], ""), INFO)
    osn = pc.extract_regex(pc.fill_null(t["info"], ""), OSN)
    ground = pc.fill_null(pc.is_in(pc.struct_field(info, "osm_airport"), value_set=pa.array(list(REF))), False)
    flights = pc.unique(pc.filter(t["flight_id"], ground))
    keep = pc.or_(ground, pc.and_(pc.is_in(t["type"], value_set=pa.array(POINT_TYPES)),
                                  pc.is_in(t["flight_id"], value_set=flights)))
    t, info, osn = t.filter(keep), info.filter(keep), osn.filter(keep)
    field = lambda s, a: pc.struct_field(s, a).to_pandas()  # noqa: E731
    d = pd.DataFrame({
        "event_id": _i64(t["id"].to_numpy()), "flight_id": _i64(t["flight_id"].to_numpy()), "type": t["type"].to_pandas(),
        "t": (pd.to_datetime(t["event_time"].to_pandas()) - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64"),
        "lat": t["latitude"].to_numpy(zero_copy_only=False).astype("float32"),
        "lon": t["longitude"].to_numpy(zero_copy_only=False).astype("float32"),
        "altitude_ft": t["altitude"].to_numpy(zero_copy_only=False).astype("float32"),
        "osm_aeroway": field(info, "osm_aeroway"), "osm_ref": field(info, "osm_ref"), "osm_airport": field(info, "osm_airport"),
        "osm_id": pd.to_numeric(field(info, "osm_id"), errors="coerce").astype("Int64"),
        "duration_s": pd.to_numeric(field(info, "duration"), errors="coerce").astype("float32"),
        "callsign_i": field(osn, "callsign").str.strip().str.upper(), "icao24_i": field(osn, "icao24")})
    empty = d["osm_airport"].isna().to_numpy()
    names, km = _nearest(d.loc[empty, "lat"].to_numpy(float), d.loc[empty, "lon"].to_numpy(float))
    d.loc[empty, "osm_airport"] = names
    d = d[~empty | np.isin(np.arange(len(d)), np.flatnonzero(empty)[km <= KM])]
    d = d.merge(_FL, on="flight_id", how="left")
    for col, src in (("callsign", "callsign_i"), ("icao24", "icao24_i")):
        d[src] = d[src].replace("", np.nan)
        d[col] = d[col].replace("", np.nan).fillna(d.groupby("flight_id")[src].transform("first"))
    d = d.drop(columns=["callsign_i", "icao24_i"])
    d["direction"] = np.select([d["osm_airport"] == d["adep"], d["osm_airport"] == d["ades"]], ["departure", "arrival"],
                               "unknown")
    d.sort_values(["flight_id", "t"]).reset_index(drop=True).to_parquet(target, index=False)
    return path.name, f"{len(d):,} events"


def filter_events(workers=3):
    FILTERED.mkdir(parents=True, exist_ok=True)
    fs = sorted(RAW.glob("flight_events_*.parquet"))
    with Pool(workers, initializer=_init, initargs=(flight_list(),)) as pool:
        for name, status in pool.imap(_filter, fs):
            print(f"{name}: {status}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    download([a for a in sys.argv[1:] if "=" not in a], int(opt.get("workers", 4)))
    filter_events()
