import json
import sys
import time
import urllib.parse
import urllib.request
from math import asin, cos, radians, sin, sqrt

import pandas as pd

import config
from taxi import data

REF = config.ARTIFACTS / "reference"
OURAIRPORTS = "https://davidmegginson.github.io/ourairports-data/runways.csv"
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter")
RUNWAYS = config.EXTERNAL / "ourairports_runways.csv"
OSM_STANDS = config.EXTERNAL / "osm_stands.csv"
BOX = {"EDDF": (49.95, 50.07, 8.47, 8.64), "EDDM": (48.34, 48.37, 11.75, 11.82), "EGLL": (51.45, 51.49, -0.51, -0.42),
       "EHAM": (52.28, 52.34, 4.70, 4.82), "LEBL": (41.27, 41.32, 2.03, 2.13), "LEMD": (40.45, 40.55, -3.63, -3.52),
       "LFPG": (48.98, 49.04, 2.51, 2.60), "LIRF": (41.77, 41.83, 12.21, 12.30), "LSZH": (47.44, 47.48, 8.53, 8.58),
       "LTFM": (41.25, 41.31, 28.70, 28.78)}


def stand_centers(min_n=5):
    from taxi.correctors import common
    from taxi.features.base import population_test_ids
    f = common.adsb_features("adsb_tracks", columns=["MVT_ID_mvt", "ADSB_FIRST_GS", "ADSB_FIRST_LAT", "ADSB_FIRST_LON"])
    k = pd.concat([data.departures("train", ["ADEP_mvt", "STAND_mvt"]), data.departures("test", ["ADEP_mvt", "STAND_mvt"])])
    k = k[~k["MVT_ID_mvt"].isin(data.departures("test", ["ADEP_mvt"])["MVT_ID_mvt"]) | k["MVT_ID_mvt"].isin(population_test_ids())]
    d = f.merge(k.drop_duplicates("MVT_ID_mvt"), on="MVT_ID_mvt", how="inner")
    d = d[(d["ADSB_FIRST_GS"] < 3) & d["ADSB_FIRST_LAT"].notna() & d["STAND_mvt"].notna()]
    g = d.groupby(["ADEP_mvt", "STAND_mvt"]).agg(stand_lat=("ADSB_FIRST_LAT", "median"), stand_lon=("ADSB_FIRST_LON", "median"),
                                                 stand_n=("ADSB_FIRST_LAT", "size"))
    return g[g["stand_n"] >= min_n].reset_index()


def iata_to_icao(min_n=20, share=0.5):
    e = data.departures("train", ["FLIGHT_mvt", "CALLSIGN_flt", "AOBT_3_flt"])
    e = e[e["AOBT_3_flt"].notna() & e["FLIGHT_mvt"].notna() & e["CALLSIGN_flt"].notna()]
    m = e["FLIGHT_mvt"].str.strip().str.extract(r"^([A-Z]\d|\d[A-Z]|[A-Z]{2})(\d+[A-Z]*)$")
    cs = e["CALLSIGN_flt"].str.strip().str.upper().str.extract(r"^([A-Z]{3})\d")[0]
    c = pd.DataFrame({"iata": m[0], "icao": cs}).dropna()
    n = c.groupby(["iata", "icao"]).size().rename("n").reset_index().sort_values(["iata", "n"], ascending=[True, False])
    top = n.groupby("iata").head(1).merge(n.groupby("iata")["n"].sum().rename("total").reset_index(), on="iata")
    top = top[(top["total"] >= min_n) & (top["n"] / top["total"] >= share)]
    return dict(zip(top["iata"], top["icao"]))


def _fetch(url, payload=None, attempts=4, timeout=180):
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, data=payload, headers={"User-Agent": "taxi-out-prediction/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            print(f"  {url[:60]}: {e}", file=sys.stderr)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"could not download: {url}")


def download_ourairports():
    config.EXTERNAL.mkdir(parents=True, exist_ok=True)
    RUNWAYS.write_bytes(_fetch(OURAIRPORTS))
    return RUNWAYS


def download_osm():
    folder = config.EXTERNAL / "osm"
    folder.mkdir(parents=True, exist_ok=True)
    for apt, (la0, la1, lo0, lo1) in BOX.items():
        path = folder / f"{apt}.json"
        if path.exists():
            continue
        q = (f'[out:json][timeout:60];(node["aeroway"="parking_position"]({la0},{lo0},{la1},{lo1});'
             f'way["aeroway"="parking_position"]({la0},{lo0},{la1},{lo1}););out center;')
        for url in OVERPASS:
            try:
                path.write_bytes(_fetch(url, urllib.parse.urlencode({"data": q}).encode(), attempts=2))
                break
            except Exception as e:
                print(f"  {apt}: {url} failed ({e})", file=sys.stderr)
        time.sleep(2)
    missing = [a for a in BOX if not (folder / f"{a}.json").exists()]
    if missing:
        raise RuntimeError(f"Overpass did not respond: {missing}; run again (downloaded airports are kept)")
    rows = []
    for apt in BOX:
        for el in json.loads((folder / f"{apt}.json").read_text(encoding="utf-8")).get("elements", []):
            ref = el.get("tags", {}).get("ref", el.get("tags", {}).get("name", ""))
            lat = el.get("lat", el.get("center", {}).get("lat"))
            lon = el.get("lon", el.get("center", {}).get("lon"))
            if lat and lon and ref:
                rows.append({"ADEP": apt, "STAND_ID": str(ref).strip(), "LAT": float(lat), "LON": float(lon)})
    pd.DataFrame(rows).to_csv(OSM_STANDS, index=False)
    return OSM_STANDS


def _runways():
    r = pd.read_csv(RUNWAYS)
    return r[r["airport_ident"].isin(config.AIRPORTS)]


def runways():
    out = {}
    for _, r in _runways().iterrows():
        for end in ("le", "he"):
            if pd.isna(r[f"{end}_ident"]):
                continue
            k = str(r[f"{end}_ident"]).strip()
            value = (r["length_ft"], r[f"{end}_heading_degT"], r[f"{end}_displaced_threshold_ft"])
            out[(r["airport_ident"], k)] = value
            if k.startswith("0"):
                out[(r["airport_ident"], k[1:])] = value
    return pd.DataFrame([(a, k, *v) for (a, k), v in out.items()],
                        columns=["ADEP_mvt", "RUNWAY_mvt", "RWY_PHYSICAL_LENGTH_FT", "RWY_HEADING_DEG_T",
                                 "RWY_DISPLACED_THRESH_FT"]).sort_values(["ADEP_mvt", "RUNWAY_mvt"])


def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = radians(lat1), radians(lat2)
    a = sin(radians(lat2 - lat1) / 2) ** 2 + cos(p1) * cos(p2) * sin(radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * asin(sqrt(a))


def stand_runway_distances():
    stands, thresholds = pd.read_csv(OSM_STANDS), []
    for _, r in _runways().iterrows():
        for end in ("le", "he"):
            k = str(r.get(f"{end}_ident", "")).strip()
            la, lo = r.get(f"{end}_latitude_deg"), r.get(f"{end}_longitude_deg")
            if k and k != "nan" and pd.notna(la) and pd.notna(lo):
                thresholds.append((r["airport_ident"], k, float(la), float(lo)))
    rows = []
    for apt, k, la, lo in thresholds:
        for p in stands[stands["ADEP"] == apt].itertuples(index=False):
            m = _haversine_m(p.LAT, p.LON, la, lo)
            rows.append({"ADEP": apt, "STAND_ID": p.STAND_ID, "RUNWAY": k, "DIST_M": m, "DIST_S": m / 5.0})
    return pd.DataFrame(rows)


def build(*parts):
    parts = parts or ("stand_centers", "iata_to_icao")
    REF.mkdir(parents=True, exist_ok=True)
    for p in parts:
        if p == "stand_centers":
            stand_centers().to_parquet(REF / "stand_centers.parquet", index=False)
        elif p == "iata_to_icao":
            (REF / "iata_to_icao.json").write_text(json.dumps(iata_to_icao(), ensure_ascii=False, sort_keys=True),
                                                   encoding="utf-8")
        elif p == "runways":
            runways().to_csv(REF / "runways.csv", index=False)
        elif p == "stand_runway_distances":
            stand_runway_distances().to_csv(REF / "stand_runway_distances.csv", index=False)
        print(f"[references] {p} -> artifacts/reference", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    build(*sys.argv[1:])
