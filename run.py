import calendar
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from taxi import data  # noqa: E402

ADSB_TRAIN_MONTHS = (1, 2, 6, 7, 8, 12)
START = time.time()


def log(m):
    print(f"[{time.time() - START:7.0f} s] {m}", flush=True)


def status():
    where = lambda p: str(p.relative_to(config.ROOT)) if p.is_relative_to(config.ROOT) else str(p)  # noqa: E731
    print(f"raw data   {where(config.RAW)} | training files {len(config.TRAIN_FILES)} | "
          f"test {[f.name for f in config.TEST_FILES]}")
    print(f"ADS-B {where(config.ADSB_ROOT)} | OPDI {where(config.OPDI_ROOT)}")
    print(f"test months {config.TEST_MONTHS} | template {config.TEMPLATE.name} | GPU {config.GPU}")
    missing = [f for f in config.TRAIN_FILES + config.TEST_FILES + [config.TEMPLATE] if not f.exists()]
    print("missing inputs: " + (", ".join(map(str, missing)) if missing else "none"))
    e = data.departures("train", ["MVT_TIME_UTC_mvt", "AOBT_3_flt"])
    t = data.departures("test", ["MVT_TIME_UTC_mvt", "AOBT_3_flt"])
    print(f"departures: train {len(e):,} (with NM {e['AOBT_3_flt'].notna().sum():,}) | test {len(t):,} "
          f"(with NM {t['AOBT_3_flt'].notna().sum():,}) | template {len(data.template()):,}")
    print("scales: " + ", ".join(f"{k}={v}" for k, v in config.SCALE.items()))


def days(year, months):
    return [f"{year}-{m:02d}-{d:02d}" for m in months for d in range(1, calendar.monthrange(year, m)[1] + 1)]


def features(*parts):
    parts = parts or ("base", "groups", "adsb", "opdi")
    if "base" in parts:
        from taxi.features import base
        base.build(log)
    if "groups" in parts:
        from taxi.features import groups
        groups.build(log)
    if "adsb" in parts or "adsb_sparse" in parts:
        from taxi.features import adsb_tracks
        train_days = days(config.TRAIN_YEAR, ADSB_TRAIN_MONTHS)
        if "adsb" in parts:
            ds = train_days + days(config.TRAIN_YEAR + 1, config.TEST_MONTHS)
            adsb_tracks.build(ds, workers=6)
            log(f"[adsb] {len(ds)} days -> work/adsb_tracks")
        if "adsb_sparse" in parts:
            for p in (0.3, 0.1):
                adsb_tracks.build(train_days, workers=6, density=p)
                log(f"[adsb] density {p}: {len(train_days)} days -> {adsb_tracks.folder(p).relative_to(config.ROOT)}")
    if "opdi" in parts:
        from taxi.features import opdi_events
        opdi_events.build()


def train_a(*components):
    from taxi.features import matrices as MX, tables as T
    from taxi.model_a import components as B
    components = components or tuple(config.MODEL_A)
    t0 = time.time()
    e, _ = T.model_a_frames()
    feats = MX.feature_list(e.columns)
    log(f"frame ready: train {len(e):,} rows, {len(feats)} features | {time.time() - t0:.0f} s")
    for name in components:
        B.train(name, e, feats, log)
        log(f"[{name}] saved")


def train_a_oof(*splits):
    from taxi.model_a import oof
    for b in splits or tuple(config.OOF_SPLITS):
        oof.build(b, log=log)
    if all((oof.DIR / f"model_a_oof_{b}.parquet").exists() for b in config.OOF_SPLITS):
        log(f"-> {oof.combine().relative_to(config.ROOT)}")


def weights():
    from taxi.model_a import blend as H
    new = H.choose_weights()
    log("chosen weights: " + ", ".join(f"{k} {v:.4f}" for k, v in new.items()))
    log("config.MODEL_A: " + ", ".join(f"{k} {v['weight']:.4f}" for k, v in config.MODEL_A.items()))
    return new


def train_b_oof():
    from taxi.model_b import oof
    oof.build(log)
    log(f"-> {oof.copy().relative_to(config.ROOT)}")


def references(*parts):
    from taxi import references as R
    parts = parts or ("stand_centers", "iata_to_icao", "runways", "stand_runway_distances")
    if "runways" in parts or "stand_runway_distances" in parts:
        R.download_ourairports()
    if "stand_runway_distances" in parts:
        R.download_osm()
    R.build(*parts)


def train_correctors(*names):
    import importlib
    for name in names or ("adsb", "opdi", "ltfm"):
        importlib.import_module(f"taxi.correctors.{name}").train(log)
        log(f"[{name}] saved")


def train_b():
    from taxi.model_b import base
    base.train(log)


def predict_a(*components):
    import pandas as pd
    from taxi.features import tables as T
    from taxi.model_a import blend as H, components as B
    components = components or tuple(config.MODEL_A)
    _, t = T.model_a_frames()
    out = pd.DataFrame({"MVT_ID_mvt": t["MVT_ID_mvt"].to_numpy(), "ADEP_mvt": t["ADEP_mvt"].to_numpy(),
                        "MVT_MONTH": t["MVT_MONTH"].to_numpy(), "NM_TAXI_DIFF_SEC": t["NM_TAXI_DIFF_SEC"].to_numpy()})
    for name in components:
        out[name] = B.predict(name, t)
        log(f"[{name}] prediction: mean offset {out[name].mean():.1f}")
    if any(config.MODEL_A[c]["kind"] == "lock_mixture" for c in components):
        out["lock_p"] = B.lock_probability(t)
    if set(components) == set(config.MODEL_A):
        out["blend"] = H.blend({k: out[k] for k in components})
        out["taxi"] = H.taxi(out["NM_TAXI_DIFF_SEC"], out["blend"])
    config.WORK.mkdir(exist_ok=True)
    path = config.WORK / ("model_a_test.parquet" if "blend" in out else f"model_a_test_{'_'.join(components)}.parquet")
    out.to_parquet(path, index=False)
    log(f"-> {path.relative_to(config.ROOT)} ({len(out):,} rows)")


def predict():
    import pandas as pd
    from taxi.correctors import adsb, ltfm, opdi
    from taxi.model_b import adsb as b_adsb, base, residual
    a = pd.read_parquet(config.WORK / "model_a_test.parquet")
    s = ltfm.apply(a, opdi.apply(a, adsb.apply(a, log), log), log)
    b = b_adsb.apply(base.apply(log), a, log)
    b = residual.apply(b.assign(taxi=b["base_adsb"]), log)
    out = pd.concat([s[["MVT_ID_mvt", "ADEP_mvt", "taxi", "taxi_base"]].assign(model="A"),
                     b[["MVT_ID_mvt", "ADEP_mvt", "taxi", "base"]].rename(columns={"base": "taxi_base"}).assign(model="B")],
                    ignore_index=True)
    path = config.WORK / "prediction_test.parquet"
    out.to_parquet(path, index=False)
    log(f"-> {path.relative_to(config.ROOT)}: {len(out):,} rows (A {int((out['model'] == 'A').sum()):,}, "
        f"B {int((out['model'] == 'B').sum()):,})")


def output(*note):
    import pandas as pd
    from taxi import output as O, postprocess as P
    t = pd.read_parquet(config.WORK / "prediction_test.parquet")
    final, report = P.apply(t, log)
    log("monthly calibration shifts:\n" + report[report["shift"] != 0].round(1).to_string(index=False))
    return O.write(final, " ".join(note) or "full pipeline (Model A + correctors, Model B) + post-processing", log)


STAGES = {"status": status, "features": features, "references": references, "train_a": train_a,
          "train_a_oof": train_a_oof, "weights": weights, "train_correctors": train_correctors, "train_b": train_b,
          "train_b_oof": train_b_oof, "predict_a": predict_a, "predict": predict, "output": output}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    stage = sys.argv[1] if len(sys.argv) > 1 else "status"
    if stage not in STAGES:
        sys.exit(f"unknown stage '{stage}'; available: {', '.join(STAGES)}")
    STAGES[stage](*sys.argv[2:])
