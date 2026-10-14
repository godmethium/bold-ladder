import datetime as dt
import json
import re
import sys
import time

import numpy as np
import pandas as pd

import config
from taxi import data

POINTS = config.ADSB_ROOT / "points"
OUTPUT = config.WORK / "adsb_tracks"
ELEVATION = {"EDDF": 364, "EDDM": 1487, "EGLL": 83, "EHAM": -11, "LEBL": 12, "LEMD": 2000, "LFPG": 392, "LIRF": 13,
             "LSZH": 1416, "LTFM": 325}
GAP, STAND_REST, AIRBORNE = 600.0, 600.0, 300.0
STAND_RADIUS = 50.0
POINT_COLUMNS = ["hex", "apt", "t", "lat", "lon", "alt", "ground", "gs", "callsign"]
DEPARTURE_COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "STAND_mvt"]
FLIGHT_NO = re.compile(r"^([A-Z]{2,3}|[A-Z]\d|\d[A-Z])(\d+[A-Z]*)$")
_stands, _iata = None, None


def days(args):
    out = []
    for a in args:
        b, e = (dt.date.fromisoformat(x) for x in (a.split(":") if ":" in a else (a, a)))
        out += [(b + dt.timedelta(days=i)).isoformat() for i in range((e - b).days + 1)]
    return out


def _m(lat1, lon1, lat2, lon2):
    return 111_195.0 * np.hypot(lat2 - lat1, (lon2 - lon1) * np.cos(np.radians(lat1)))


def _stand_center(apt, stand):
    global _stands
    if _stands is None:
        m = pd.read_parquet(config.ARTIFACTS / "reference" / "stand_centers.parquet")
        _stands = {(a, s): (la, lo) for a, s, la, lo in m[["ADEP_mvt", "STAND_mvt", "stand_lat", "stand_lon"]].itertuples(index=False)}
    return _stands.get((apt, stand), (np.nan, np.nan))


def _callsign(r):
    global _iata
    if isinstance(r.CALLSIGN_flt, str) and r.CALLSIGN_flt.strip():
        return r.CALLSIGN_flt.strip()
    if not isinstance(r.FLIGHT_mvt, str):
        return None
    if _iata is None:
        path = config.ARTIFACTS / "reference" / "iata_to_icao.json"
        _iata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    m = FLIGHT_NO.match(r.FLIGHT_mvt.strip())
    if not m:
        return None
    prefix, number = m.group(1), m.group(2)
    if len(prefix) == 3:
        return prefix + number
    return (_iata[prefix] + number) if prefix in _iata else None


def _aircraft(p):
    if p.empty:
        return {}
    p = p.sort_values(["hex", "t"], kind="stable")
    hx = p["hex"].astype(str).to_numpy()
    bounds = np.flatnonzero(np.r_[True, hx[1:] != hx[:-1], True])
    cols = {c: p[c].to_numpy() for c in ("t", "lat", "lon", "alt", "ground", "gs")}
    cols["cs"] = p["callsign"].astype(object).to_numpy()
    return {hx[a]: {c: v[a:b] for c, v in cols.items()} for a, b in zip(bounds[:-1], bounds[1:])}


def _low(u, el):
    return u["ground"] | ((u["alt"] <= el + AIRBORNE) & (np.nan_to_num(u["gs"], nan=0.0) < 200))


def _takeoffs(u, el):
    air = (~u["ground"]) & (u["alt"] > el + AIRBORNE)
    return np.flatnonzero(air[1:] & _low(u, el)[:-1] & (np.diff(u["t"]) <= 180)) + 1


def _stand_split(t, lat, lon, gs, ks, mvt, slat, slon):
    g = gs[ks].astype(float)
    g0 = np.nan_to_num(g, nan=0.0)
    la, lo, tt = lat[ks], lon[ks], t[ks]
    k4 = np.flatnonzero((g0[:-1] >= 4) & (g0[1:] >= 4))
    i4 = k4[0] if len(k4) else len(ks)
    o = {"ADSB_TAXI_START": mvt - tt[k4[0]] if len(k4) else np.nan, "ADSB_STAND_LAST": np.nan, "ADSB_STAND_NEXT": np.nan,
         "ADSB_STAND_GAP": np.nan, "ADSB_STAND_FIRST": np.nan}
    step = np.r_[0.0, _m(la[:-1], lo[:-1], la[1:], lo[1:])] if len(ks) > 1 else np.zeros(len(ks))
    still = np.where(np.isnan(g), step < 10, g < 1)
    if np.isfinite(slat):
        alat, alon = slat, slon
    else:
        j = np.flatnonzero(still[:i4])
        alat, alon = (la[j[0]], lo[j[0]]) if len(j) else (np.nan, np.nan)
    if np.isfinite(alat):
        j = np.flatnonzero(still[:i4] & (_m(alat, alon, la[:i4], lo[:i4]) <= STAND_RADIUS))
        if len(j) and j[-1] + 1 < len(ks):
            o.update(ADSB_STAND_LAST=mvt - tt[j[-1]], ADSB_STAND_NEXT=mvt - tt[j[-1] + 1],
                     ADSB_STAND_GAP=tt[j[-1] + 1] - tt[j[-1]], ADSB_STAND_FIRST=mvt - tt[j[0]])
    return o


def _segment(u, i_air, mvt, slat, slon):
    t, lat, lon, gs, alt, ground = u["t"], u["lat"], u["lon"], u["gs"], u["alt"], u["ground"]
    last = j = i_air - 1
    arrival = False
    while j > 0 and t[j] - t[j - 1] <= GAP:
        if alt[j - 1] > 0 and not ground[j - 1] and alt[j - 1] > alt[last] + AIRBORNE:
            arrival = True
            break
        j -= 1
    idx = np.arange(j, last + 1)
    stopped = (np.nan_to_num(gs[idx], nan=0.0) < 2) & (np.r_[0.0, _m(lat[idx[:-1]], lon[idx[:-1]], lat[idx[1:]], lon[idx[1:]])] < 25)
    parked, start = False, 0
    if arrival:
        k = 0
        while k < len(idx) - 1:
            if stopped[k]:
                e = k
                while e + 1 < len(idx) and stopped[e + 1]:
                    e += 1
                if t[idx[e]] - t[idx[k]] >= STAND_REST:
                    parked, start = True, e
                    break
                k = e + 1
            else:
                k += 1
    ks = idx[start:]
    gk = np.nan_to_num(gs[ks], nan=0.0)
    moving = np.flatnonzero((gk >= 2) | (_m(lat[ks[0]], lon[ks[0]], lat[ks], lon[ks]) > 25))
    slow = np.flatnonzero(gk < 40)
    i_roll = ks[slow[-1]] if len(slow) else ks[-1]
    t_move = t[ks[moving[0]]] if len(moving) else np.nan
    if np.isfinite(t_move) and t_move > mvt - 60:
        t_move = np.nan
    first_stop = t[ks[moving[0]]] - t[ks[0]] if len(moving) else 0.0
    if np.isfinite(t_move):
        seg = np.arange(ks[moving[0]], i_roll + 1)
        step = _m(lat[seg[:-1]], lon[seg[:-1]], lat[seg[1:]], lon[seg[1:]])
        interval = np.diff(t[seg])
        still = (np.nan_to_num(gs[seg[1:]], nan=0.0) < 2) & (step < 25)
        path, stop = float(step.sum()), float(interval[still].sum())
        first_5min = t[seg[1:]] <= t_move + 300
        gap_5min = float(interval[first_5min].max()) if first_5min.any() else np.nan
    else:
        path = stop = gap_5min = np.nan
    i_move = ks[moving[0]] if len(moving) else ks[0]
    return {**_stand_split(t, lat, lon, gs, ks, mvt, slat, slon),
            "ADSB_FIRST": mvt - t[ks[0]], "ADSB_MOVE": mvt - t_move, "ADSB_TO_DIFF": t[i_air] - mvt, "ADSB_N": len(ks),
            "ADSB_GAP": float(np.max(np.diff(t[ks]))) if len(ks) > 1 else np.nan, "ADSB_GAP_5MIN": gap_5min,
            "ADSB_FIRST_GS": float(gs[ks[0]]), "ADSB_PARKED": parked, "ADSB_FIRST_STOP": float(first_stop),
            "ADSB_FIRST_DIST": float(_m(lat[ks[0]], lon[ks[0]], lat[i_roll], lon[i_roll])), "ADSB_PATH": path,
            "ADSB_STOP": stop, "ADSB_ROLL": mvt - t[i_roll],
            "ADSB_FIRST_LAT": float(lat[ks[0]]), "ADSB_FIRST_LON": float(lon[ks[0]]), "ADSB_MOVE_LAT": float(lat[i_move]),
            "ADSB_MOVE_LON": float(lon[i_move]), "ADSB_ROLL_LAT": float(lat[i_roll]), "ADSB_ROLL_LON": float(lon[i_roll])}


def _gapped(aircraft, cs_hex, cs, mvt, el):
    best = None
    for hx in cs_hex.get(cs, ()):
        u = aircraft[hx]
        t = u["t"]
        k = np.flatnonzero(_low(u, el) & (t < mvt + 60) & (t >= mvt - 3600))
        if len(k) == 0 or k[-1] + 1 >= len(t) or t[k[-1] + 1] > mvt + 900:
            continue
        candidate = (hx, int(k[-1] + 1), mvt - t[k[-1]])
        if best is None or candidate[2] < best[2]:
            best = candidate
    return None if best is None else best[:2]


def _points(day, density=1.0):
    p = pd.read_parquet(POINTS / f"{day}.parquet", columns=POINT_COLUMNS)
    prev = POINTS / f"{dt.date.fromisoformat(day) - dt.timedelta(days=1)}.parquet"
    if prev.exists():
        p0 = pd.read_parquet(prev, columns=POINT_COLUMNS)
        p = pd.concat([p0[p0["t"] >= p["t"].min() - 3 * 3600], p], ignore_index=True)
        for c in ("hex", "apt", "callsign"):
            p[c] = p[c].astype(str).replace({"None": None, "nan": None})
    if density < 1:
        ground = p["ground"].fillna(False).to_numpy(bool)
        keep = ~ground | (np.random.default_rng(int(day.replace("-", ""))).random(len(p)) < density)
        p = p[keep].reset_index(drop=True)
    return p


def folder(density=1.0):
    return config.WORK / ("adsb_tracks" if density >= 1 else f"adsb_tracks_s{round(100 * density)}")


def departure_table(ds):
    splits = {"train" if d.startswith(str(config.TRAIN_YEAR)) else "test" for d in ds}
    k = pd.concat([data.departures(x, DEPARTURE_COLUMNS) for x in sorted(splits)], ignore_index=True)
    k["_day"] = k["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    return k[k["_day"].isin(set(ds))].reset_index(drop=True)


def day(d, departures=None, output=OUTPUT, density=1.0):
    t0 = time.time()
    if departures is None:
        departures = departure_table([d])
    dep = departures[departures["_day"].eq(d)].reset_index(drop=True)
    dep["mvt"] = ((dep["MVT_TIME_UTC_mvt"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).astype(float)
    pts = _points(d, density)
    rows = []
    for apt, d_apt in dep.groupby("ADEP_mvt"):
        el = ELEVATION[apt]
        aircraft = _aircraft(pts[pts["apt"].eq(apt)])
        events = []
        for hx, u in aircraft.items():
            for i in _takeoffs(u, el):
                events.append((u["t"][i], hx, i, next((c for c in u["cs"][max(0, i - 50):i + 5][::-1] if c), None)))
        ev = pd.DataFrame(events, columns=["t", "hex", "i", "cs"])
        cs_hex = {}
        for hx, u in aircraft.items():
            for c in {str(c).strip() for c in u["cs"] if c}:
                cs_hex.setdefault(c, []).append(hx)
        used = set()
        for r in d_apt.itertuples(index=False):
            pick, kind = None, 0
            cs_r = _callsign(r)
            if cs_r and len(ev):
                c = ev[ev["cs"].eq(cs_r) & (ev["t"].sub(r.mvt).abs() <= 300)]
                if len(c):
                    pick, kind = c.iloc[(c["t"] - r.mvt).abs().argmin()], 1
            if pick is None and len(ev):
                c = ev[(ev["t"].sub(r.mvt).abs() <= 90) & ~ev.index.isin(used)]
                if len(c) == 1:
                    pick, kind = c.iloc[0], 2
            f = {"MVT_ID_mvt": r.MVT_ID_mvt, "ADSB_MATCH": kind}
            if pick is not None:
                used.add(pick.name)
                f.update(_segment(aircraft[pick["hex"]], int(pick["i"]), r.mvt, *_stand_center(apt, r.STAND_mvt)))
            elif cs_r:
                g3 = _gapped(aircraft, cs_hex, cs_r, r.mvt, el)
                if g3 is not None:
                    f["ADSB_MATCH"] = 3
                    f.update(_segment(aircraft[g3[0]], g3[1], r.mvt, *_stand_center(apt, r.STAND_mvt)))
            rows.append(f)
    out = pd.DataFrame(rows)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output / f"{d}.parquet", index=False)
    m = out["ADSB_MATCH"] if len(out) else pd.Series(dtype=float)
    print(f"{d}: departures {len(dep):,} | matched {int((m > 0).sum()):,} (callsign {int((m == 1).sum()):,}, "
          f"gapped {int((m == 3).sum()):,}) | {time.time() - t0:.0f} s", flush=True)
    return out


def _day_worker(d):
    return day(d, _day_worker.departures, folder(_day_worker.density), _day_worker.density)


def build(ds, workers=6, density=1.0):
    present = [d for d in ds if (POINTS / f"{d}.parquet").exists()]
    if len(present) < len(ds):
        print(f"[adsb] WARNING: no points for {len(ds) - len(present)} days: {', '.join(sorted(set(ds) - set(present)))}",
              flush=True)
    _day_worker.departures, _day_worker.density = departure_table(present), density
    if workers > 1:
        from multiprocessing import get_context
        with get_context("fork").Pool(workers) as pool:
            pool.map(_day_worker, present, chunksize=1)
    else:
        for d in present:
            _day_worker(d)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    build(days([a for a in sys.argv[1:] if "=" not in a]), int(opt.get("workers", 1)), float(opt.get("density", 1)))
