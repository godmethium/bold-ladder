import json
import sys

import numpy as np
import pandas as pd

import config
from taxi import data
from taxi.external.opdi import FILTERED
from taxi.features.adsb_tracks import FLIGHT_NO

BEFORE, AFTER, TOL = 3 * 3600, 300, 45.0
COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt"]


def _callsigns(d):
    nm = d["CALLSIGN_flt"].astype("string").str.strip().str.upper()
    iata = json.loads((config.ARTIFACTS / "reference" / "iata_to_icao.json").read_text(encoding="utf-8"))
    p = d["FLIGHT_mvt"].astype("string").str.strip().str.extract(FLIGHT_NO.pattern)
    prefix = p[0].where(p[0].str.len() == 3, p[0].map(iata))
    return nm.where(nm.notna() & (nm != ""), prefix + p[1])


def departures():
    d = pd.concat([data.departures(k, COLUMNS) for k in ("train", "test")], ignore_index=True).drop_duplicates("MVT_ID_mvt")
    d["MVT_ID_mvt"] = d["MVT_ID_mvt"].astype("int64")
    d["T"] = (d["MVT_TIME_UTC_mvt"].dt.tz_localize(None) - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64")
    d["cs"] = _callsigns(d)
    return d.drop(columns=["MVT_TIME_UTC_mvt"]).reset_index(drop=True)


def events():
    cols = ["flight_id", "type", "t", "osm_aeroway", "osm_ref", "osm_airport", "osm_id", "duration_s", "direction", "callsign"]
    e = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(FILTERED.glob("*.parquet"))], ignore_index=True)
    e = e[e["osm_airport"].isin(config.AIRPORTS) & e["callsign"].notna()]
    landing = e.loc[e["type"] == "landing", ["flight_id", "osm_airport"]].drop_duplicates().assign(arrival=True)
    e = e.merge(landing, on=["flight_id", "osm_airport"], how="left")
    e = e[(e["direction"] != "arrival") & e["arrival"].isna()].drop(columns=["arrival", "direction"])
    e["osm_id"] = e["osm_id"].fillna(-1).astype("int64")
    e = e.drop_duplicates(["callsign", "osm_airport", "type", "t", "osm_id"]).reset_index(drop=True)
    e["eid"], e["day"] = np.arange(len(e)), e["t"] // 86400
    return e


def _match(d, e):
    parts = []
    for shift in (0, BEFORE):
        k = d.loc[d["cs"].notna(), ["MVT_ID_mvt", "ADEP_mvt", "cs", "T"]].copy()
        k["day"] = (k["T"] - shift) // 86400
        parts.append(k.merge(e, left_on=["ADEP_mvt", "cs", "day"], right_on=["osm_airport", "callsign", "day"]))
    m = pd.concat(parts, ignore_index=True).drop_duplicates(["MVT_ID_mvt", "eid"])
    return m[(m["t"] >= m["T"] - BEFORE) & (m["t"] <= m["T"] + AFTER)]


def _match_time(d, e, m1, tol):
    rest = d[~d["MVT_ID_mvt"].isin(m1["MVT_ID_mvt"])][["MVT_ID_mvt", "ADEP_mvt", "T", "RUNWAY_mvt"]]
    used = set(zip(m1["osm_airport"], m1["callsign"], m1["T"] // 86400))
    ex = e[e["type"] == "exit-runway"][["osm_airport", "callsign", "t", "osm_ref"]].copy()
    ex = ex[[(a, c, t // 86400) not in used for a, c, t in zip(ex["osm_airport"], ex["callsign"], ex["t"])]]
    ex["mn"] = ex["t"] // 60
    c = pd.concat([rest.assign(mn=rest["T"] // 60 + k).merge(ex, left_on=["ADEP_mvt", "mn"], right_on=["osm_airport", "mn"])
                   for k in (-1, 0, 1)], ignore_index=True)
    c = c[(c["t"] - c["T"]).abs() <= tol]
    runway = c["RUNWAY_mvt"].astype("string").str.strip().str.upper().str.lstrip("0")
    ref = c["osm_ref"].astype("string").str.upper().str.split("/")
    same = [p in [x.lstrip("0") for x in r] if isinstance(r, list) and isinstance(p, str) else True for p, r in zip(runway, ref)]
    c = c[same].drop_duplicates(["MVT_ID_mvt", "callsign"])
    c = c[c.groupby("MVT_ID_mvt")["callsign"].transform("size") == 1]
    c = c[c.groupby(["osm_airport", "callsign", c["T"] // 86400])["MVT_ID_mvt"].transform("size") == 1]
    d2 = d.merge(c[["MVT_ID_mvt", "callsign"]], on="MVT_ID_mvt").assign(cs=lambda x: x["callsign"]).drop(columns="callsign")
    return _match(d2, e)


def _features(m, d):
    g = m.groupby("MVT_ID_mvt")
    T = g["T"].first()
    o = pd.DataFrame({"OPDI_N": g.size(), "OPDI_FIRST": T - g["t"].min()})
    o["OPDI_FIRST_GROUND"] = T - m[m["osm_aeroway"].notna()].groupby("MVT_ID_mvt")["t"].min()
    pk = (m[m["type"] == "exit-parking_position"].sort_values(["MVT_ID_mvt", "duration_s"], ascending=[True, False])
          .drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt"))
    o["OPDI_STAND_EXIT"], o["OPDI_STAND_DUR"] = T - pk["t"], pk["duration_s"]
    o["OPDI_APRON_EXIT"] = T - m[m["type"] == "exit-apron"].groupby("MVT_ID_mvt")["t"].max()
    gt = m[m["type"] == "entry-taxiway"].groupby("MVT_ID_mvt")
    o["OPDI_TAXI_ENTRY"] = T - gt["t"].min()
    o["OPDI_TAXI_TOTAL"], o["OPDI_TAXI_MAX"], o["OPDI_TWY_N"] = gt["duration_s"].sum(), gt["duration_s"].max(), gt["osm_id"].nunique()
    rw = m[(m["type"] == "entry-runway") & (m["t"] <= m["T"] + 60)].sort_values(["MVT_ID_mvt", "t"])
    last = rw.drop_duplicates("MVT_ID_mvt", keep="last").set_index("MVT_ID_mvt")
    o["OPDI_RWY_ENTRY"], o["OPDI_RWY_DUR"] = T - last["t"], last["duration_s"]
    o["OPDI_RWY_N"] = rw.groupby("MVT_ID_mvt")["osm_id"].nunique()
    o = o.join(pk["osm_ref"].rename("stand_ref")).join(last["osm_ref"].rename("runway_ref"))
    o = o.join(d.set_index("MVT_ID_mvt")[["STAND_mvt", "RUNWAY_mvt"]])
    st = o["STAND_mvt"].astype("string").str.strip().str.upper()
    o["OPDI_STAND_MATCH"] = np.where(o["stand_ref"].notna() & st.notna(), o["stand_ref"].astype("string").str.upper() == st, np.nan)
    runway = o["RUNWAY_mvt"].astype("string").str.strip().str.upper().str.lstrip("0")
    o["OPDI_RWY_MATCH"] = [float(p in [x.lstrip("0") for x in str(r).upper().split("/")])
                           if isinstance(r, str) and isinstance(p, str) else np.nan for p, r in zip(runway, o["runway_ref"])]
    return o.drop(columns=["stand_ref", "runway_ref", "STAND_mvt", "RUNWAY_mvt"]).astype("float32")


def compute(tol=TOL):
    d, e = departures(), events()
    m1 = _match(d, e)
    m = pd.concat([m1.assign(OPDI_MATCH=1), _match_time(d, e, m1, tol).assign(OPDI_MATCH=2)], ignore_index=True)
    o = _features(m, d)
    o["OPDI_MATCH"] = m.groupby("MVT_ID_mvt")["OPDI_MATCH"].first().astype("float32")
    o = o.reset_index()
    print(f"[opdi] departures {len(d):,} | matched {len(o):,} (time match {int((o['OPDI_MATCH'] == 2).sum()):,})", flush=True)
    return o


def build():
    from taxi.features.tables import GROUP_DIR
    o = compute()
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    o.to_parquet(GROUP_DIR / "opdi.parquet", index=False)
    print(f"[opdi] {len(o):,} departures -> work/groups/opdi.parquet", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    build()
