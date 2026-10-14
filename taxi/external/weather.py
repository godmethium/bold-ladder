import json
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

import config
from taxi.features.base import WEATHER, WEATHER_COLUMNS, WEATHER_KEYS

RAW = config.EXTERNAL / "weather_raw"
URL = "https://archive-api.open-meteo.com/v1/archive"
LOCATION = {"EDDF": (50.0333, 8.5705), "EDDM": (48.3537, 11.7860), "EGLL": (51.4700, -0.4543), "EHAM": (52.3105, 4.7683),
            "LEBL": (41.2974, 2.0833), "LEMD": (40.4839, -3.5680), "LFPG": (49.0097, 2.5479), "LIRF": (41.7999, 12.2462),
            "LTFM": (41.2590, 28.7424), "LSZH": (47.4647, 8.5492)}
VARIABLES = {"temperature_2m": "TEMPERATURE", "dew_point_2m": "DEW_POINT", "relative_humidity_2m": "HUMIDITY",
             "precipitation": "PRECIPITATION", "snowfall": "SNOWFALL", "snow_depth": "SNOW_DEPTH",
             "wind_speed_10m": "WIND_SPEED", "wind_direction_10m": "WIND_DIR", "cloud_cover": "CLOUD_COVER"}


def _valid(path):
    try:
        return "hourly" in json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def download(start, end, attempts=4):
    RAW.mkdir(parents=True, exist_ok=True)
    for icao, (lat, lon) in LOCATION.items():
        path = RAW / f"{icao}_{start}_{end}.json"
        for a in range(1, attempts + 1):
            if _valid(path):
                break
            url = (f"{URL}?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
                   f"&hourly={','.join(VARIABLES)}")
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "taxi-out-prediction/1.0"}),
                                            timeout=90) as r:
                    path.write_bytes(r.read())
            except Exception as e:
                print(icao, "error:", e, file=sys.stderr)
            time.sleep(1 if _valid(path) else 5 * a)
        print(icao, start, end, "ok" if _valid(path) else "MISSING", flush=True)


def derive(h):
    h = h.sort_values(["ADEP_mvt", "hour"]).reset_index(drop=True)
    h["FROST_RISK"] = ((h["TEMPERATURE"] <= 3) & ((h["TEMPERATURE"] - h["DEW_POINT"]).abs() <= 3)).astype(int)
    h["IS_FREEZING"] = (h["TEMPERATURE"] <= 0).astype(int)
    g = h.groupby("ADEP_mvt", sort=False)
    h["SNOWFALL_3H"] = g["SNOWFALL"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    h["SNOWFALL_6H"] = g["SNOWFALL"].transform(lambda s: s.rolling(6, min_periods=1).sum())
    h["PRECIP_3H"] = g["PRECIPITATION"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    h["TEMP_DROP_3H"] = g["TEMPERATURE"].transform(lambda s: s.shift(3) - s).fillna(0)
    r = np.deg2rad(h["WIND_DIR"])
    h["WIND_U"], h["WIND_V"] = -h["WIND_SPEED"] * np.sin(r), -h["WIND_SPEED"] * np.cos(r)
    h["MVT_YEAR"], h["MVT_MONTH"], h["MVT_DAY"], h["MVT_HOUR"] = (h["hour"].dt.year, h["hour"].dt.month, h["hour"].dt.day,
                                                                  h["hour"].dt.hour)
    return h


def hourly():
    parts = []
    for f in sorted(RAW.glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))["hourly"]
        d = pd.DataFrame(j).rename(columns=VARIABLES)
        d["hour"], d["ADEP_mvt"] = pd.to_datetime(d.pop("time")), f.stem.split("_")[0]
        parts.append(d)
    h = pd.concat(parts, ignore_index=True).drop_duplicates(["ADEP_mvt", "hour"], keep="last")
    h = derive(h)
    missing = h.groupby("ADEP_mvt")["TEMPERATURE"].apply(lambda s: s.isna().mean())
    h[WEATHER_KEYS + WEATHER_COLUMNS].to_parquet(WEATHER, index=False)
    print(f"-> {WEATHER.name}: {len(h):,} hours ({h['hour'].min()} - {h['hour'].max()}) | "
          f"max missing temperature {missing.max():.1%}")


if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2])
    hourly()
