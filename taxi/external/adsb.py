import gzip
import io
import json
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

import config
from taxi.features.adsb_tracks import POINTS, days

RAW = config.ADSB_ROOT / "raw"
REF = {"EDDF": (50.0333, 8.5706, 364), "EDDM": (48.3538, 11.7861, 1487), "EGLL": (51.4775, -0.4614, 83),
       "EHAM": (52.3081, 4.7642, -11), "LEBL": (41.2971, 2.0785, 12), "LEMD": (40.4719, -3.5626, 2000),
       "LFPG": (49.0097, 2.5479, 392), "LIRF": (41.8003, 12.2389, 13), "LSZH": (47.4647, 8.5492, 1416),
       "LTFM": (41.2753, 28.7519, 325)}
DLAT = 0.10
DLON = {k: 0.10 / np.cos(np.radians(v[0])) for k, v in REF.items()}
CEILING = 3000.0
BATCH = 1500


def _preferred(year):
    f = RAW / f"PREFERRED_RELEASES_{year}.txt"
    if not f.exists():
        url = f"https://raw.githubusercontent.com/adsblol/globe_history_{year}/main/PREFERRED_RELEASES.txt"
        f.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return [s.strip() for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]


def _curl(folder, urls):
    for url in urls:
        subprocess.run(["curl", "-L", "--http1.1", "--fail", "--retry", "8", "--retry-delay", "10", "-C", "-", "-sS",
                        "-o", str(folder / url.rsplit("/", 1)[1]), url], check=True)


def _other_copy(urls):
    old, new = ("staging", "prod") if "-staging-" in urls[0] else ("prod", "staging")
    tag = urls[0].rsplit("/", 2)[1].replace(f"-{old}-", f"-{new}-")
    repo = urls[0].split("/")[4]
    api = f"https://api.github.com/repos/adsblol/{repo}/releases/tags/{tag}"
    assets = json.loads(urllib.request.urlopen(api, timeout=60).read())["assets"]
    return [a["browser_download_url"] for a in sorted(assets, key=lambda a: a["name"]) if ".tar" in a["name"]]


def download(d, other=False):
    k = RAW / d
    if (k / ".done").exists() and not other:
        return
    tag = f"/v{d.replace('-', '.')}-planes-readsb-"
    year = int(d[:4])
    line = [s for s in _preferred(year) if tag in s] or [s for s in _preferred(year + 1) if tag in s]
    if len(line) != 1:
        raise RuntimeError(f"{d}: PREFERRED_RELEASES line not found ({len(line)})")
    k.mkdir(parents=True, exist_ok=True)
    urls = line[0].split(",")
    if other and (k / ".done").exists():
        urls = (k / ".done").read_text(encoding="utf-8").split(",")
        (k / ".done").unlink()
    try:
        if other:
            raise subprocess.CalledProcessError(1, "other")
        _curl(k, urls)
    except subprocess.CalledProcessError:
        urls = _other_copy(urls)
        for p in k.glob("*.tar*"):
            p.unlink()
        print(f"{d}: trying the other copy ({len(urls)} parts)", flush=True)
        _curl(k, urls)
    (k / ".done").write_text(",".join(urls), encoding="utf-8")
    print(f"{d}: downloaded, {sum(p.stat().st_size for p in k.glob('*.tar*')) / 1e9:.2f} GB", flush=True)


class _Chain(io.RawIOBase):

    def __init__(self, paths):
        self.fs, self.i, self.read_bytes = [open(p, "rb") for p in paths], 0, 0
        self.total = sum(Path(p).stat().st_size for p in paths)

    def readable(self):
        return True

    def readinto(self, b):
        while self.i < len(self.fs):
            n = self.fs[self.i].readinto(b)
            if n:
                self.read_bytes += n
                return n
            self.fs[self.i].close()
            self.i += 1
        return 0


def _trace(raw):
    try:
        o = orjson.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    except Exception:
        return None
    tr = o.get("trace")
    if not tr:
        return None
    n = len(tr)
    lat = np.fromiter((p[1] for p in tr), float, n)
    lon = np.fromiter((p[2] for p in tr), float, n)
    hits = [(a, (np.abs(lat - la) < DLAT) & (np.abs(lon - lo) < DLON[a]), el) for a, (la, lo, el) in REF.items()]
    hits = [x for x in hits if x[1].any()]
    if not hits:
        return None
    alts = [p[3] for p in tr]
    ground = np.fromiter((a == "ground" for a in alts), bool, n)
    alt = np.fromiter(((a if isinstance(a, (int, float)) else np.nan) for a in alts), float, n)
    parts, cs = [], None
    for apt, m, el in hits:
        keep = m & (ground | (alt <= el + CEILING))
        if not keep.any():
            continue
        if cs is None:
            cs, last = np.empty(n, dtype=object), None
            for i, p in enumerate(tr):
                if len(p) > 8 and isinstance(p[8], dict) and p[8].get("flight"):
                    last = p[8]["flight"].strip()
                cs[i] = last
        idx = np.nonzero(keep)[0]
        parts.append(pd.DataFrame({
            "hex": o.get("icao"), "reg": o.get("r"), "type": o.get("t"), "apt": apt,
            "t": float(o["timestamp"]) + np.fromiter((tr[i][0] for i in idx), float, len(idx)),
            "lat": lat[idx].astype(np.float32), "lon": lon[idx].astype(np.float32), "alt": alt[idx].astype(np.float32),
            "ground": ground[idx],
            "gs": np.array([tr[i][4] if tr[i][4] is not None else np.nan for i in idx], dtype=np.float32),
            "track": np.array([tr[i][5] if tr[i][5] is not None else np.nan for i in idx], dtype=np.float32),
            "flags": np.array([tr[i][6] or 0 for i in idx], dtype=np.int16),
            "source": [tr[i][9] if len(tr[i]) > 9 else None for i in idx], "callsign": cs[idx]}))
    return pd.concat(parts, ignore_index=True) if parts else None


def extract(d, pool, partial=False):
    parts = sorted((RAW / d).glob("*.tar.a?")) or sorted((RAW / d).glob("*.tar"))
    if not (RAW / d / ".done").exists() or not parts:
        raise RuntimeError(f"{d}: archive missing or incomplete")
    t0 = time.time()
    chain = _Chain(parts)
    result, batch, n_traces = [], [], 0

    def flush():
        nonlocal batch, n_traces
        result.extend(x for x in pool.map(_trace, batch, chunksize=25) if x is not None)
        n_traces += len(batch)
        batch = []

    with tarfile.open(fileobj=io.BufferedReader(chain, buffer_size=16 << 20), mode="r|") as tar:
        for u in tar:
            if u.isfile() and "trace_full_" in u.name:
                batch.append(tar.extractfile(u).read())
                if len(batch) >= BATCH:
                    flush()
        if batch:
            flush()
    if chain.read_bytes < chain.total - (64 << 20):
        message = f"{d}: tar stream ended early ({chain.read_bytes / 1e9:.2f} / {chain.total / 1e9:.2f} GB)"
        if not (partial and chain.read_bytes >= 0.5 * chain.total):
            raise RuntimeError(message)
        print(f"WARNING {message}; writing points of the readable part", flush=True)
    df = pd.concat(result, ignore_index=True)
    for c in ("hex", "reg", "type", "apt", "source", "callsign"):
        df[c] = df[c].astype("category")
    POINTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(POINTS / f"{d}.parquet", index=False)
    print(f"{d}: traces {n_traces:,} | aircraft-airport {len(result):,} | points {len(df):,} | {time.time() - t0:.0f} s",
          flush=True)


def _try_download(d, gate):
    gate.acquire()
    for attempt in range(3):
        try:
            download(d)
            return True
        except Exception as e:
            print(f"{d}: download error ({e}), attempt {attempt + 1}", flush=True)
    return False


def pipeline(ds, workers=6, delete=False, streams=2, pending=4):
    RAW.mkdir(parents=True, exist_ok=True)
    ds = [d for d in ds if not (POINTS / f"{d}.parquet").exists()]
    gate = threading.Semaphore(pending)
    with ThreadPoolExecutor(streams) as downloader, Pool(workers) as pool:
        jobs = {d: downloader.submit(_try_download, d, gate) for d in ds}
        for d in ds:
            try:
                if not jobs[d].result():
                    continue
                try:
                    extract(d, pool)
                except Exception as e:
                    print(f"{d}: extraction error ({e}); other copy", flush=True)
                    try:
                        download(d, other=True)
                        extract(d, pool)
                    except Exception as e2:
                        print(f"{d}: other copy failed too ({e2})", flush=True)
                if delete and (POINTS / f"{d}.parquet").exists():
                    for p in (RAW / d).glob("*.tar*"):
                        p.unlink()
            finally:
                gate.release()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    pipeline(days([a for a in sys.argv[1:] if "=" not in a]), int(opt.get("workers", 6)), opt.get("delete") == "1",
             int(opt.get("streams", 2)))
