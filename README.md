# Departure taxi-out time prediction

This project predicts the **taxi-out time** of every departure: the time between leaving the stand (BLOCK) and take-off (MVT), in seconds.
- **Training data:** 12 months of 2025. Departure and arrival movements at 10 European airports: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM.
- **Test data:** departures in January, February, June and July 2026 (670,790 rows).

## Quick start
1. Create the environment:
   ```
   conda env create -f environment.yml && conda activate taxi
   python -m ipykernel install --user --name taxi
   ```
2. Put the raw movement files in `data/` (see [Usage](#usage)).
3. Download `models.zip` and `features.zip` from the **Releases** page of this repository and unzip both in the repository root.
4. Run `jupyter lab notebooks/` and start with `00_overview`. For prediction only, `04_prediction` is enough.

| Notebook | Purpose | Time |
|---|---|---|
| `00_overview` | Environment, inputs, a first look at the data | seconds |
| `01_external_data` | Downloads external data; all flags off by default | ADS-B takes hours, the rest minutes |
| `02_features` | Feature tables | ~6 min |
| `03_training` | Retraining and regeneration of the intermediate artifacts; with all flags off, the pretrained models are used | Model A ~4–5 h (CPU) |
| `04_prediction` | Test prediction and the output file `output/prediction_<N>.parquet` | ~4 min |
| `05_evaluation` | Evaluation on held-out months of the training year | seconds |

The same steps run from the command line:
```
python run.py status | features | references | train_a | train_a_oof | weights | train_b | train_b_oof | train_correctors | predict_a | predict | output
```

## Usage
### Folder layout
```
data/                       raw movement files (not in the repository)
work/                       generated and downloaded data (not in the repository; features.zip fills it)
  base_train.parquet, base_test.parquet
  groups/                   group feature tables
  adsb_tracks/              ADS-B track features per day (adsb_tracks_s30/, adsb_tracks_s10/: thinned copies)
  external/                 weather, METAR, monthly taxi-out table, OurAirports, OpenStreetMap
  adsb/raw/, adsb/points/   adsb.lol archives and extracted points
  opdi/raw/, opdi/filtered/ OPDI flight events and the filtered ground events
models/                     trained models (models.zip)
artifacts/oof/              out-of-fold predictions of the training year
artifacts/reference/        reference tables
output/                     prediction files
```

### Required files and where to save them
**To predict with the pretrained models**, you need the raw files and the two release archives:

| File | Where to get it | Save to |
|---|---|---|
| `training_2025-*.parquet` (12 monthly files) | provided with the dataset | `data/` |
| `ranking.parquet` (January and July 2026, first release) | provided with the dataset | `data/` |
| `final_ranking.parquet` (all four test months) | provided with the dataset | `data/` |
| `final_submitting.parquet` (output template: row order of the output file) | provided with the dataset | `data/` |
| `models.zip` | Releases page of this repository | unzip in the repository root → `models/` |
| `features.zip` | Releases page of this repository | unzip in the repository root → `work/` |

`features.zip` contains `work/base_*.parquet`, `work/groups/`, `work/adsb_tracks*/` and, in `work/external/`, the hourly weather and METAR tables and the monthly taxi-out table. With it, no external download is needed.

The raw folder can be moved with the environment variable `TAXI_RAW_DIR`; the ADS-B and OPDI roots with `TAXI_ADSB_ROOT` and `TAXI_OPDI_ROOT` (defaults `work/adsb`, `work/opdi`).

**To rebuild the features from scratch** (instead of `features.zip`), download the external data. Every downloader skips files that are already present and resumes partial downloads.

| Data | Source | Command | Saved to |
|---|---|---|---|
| Hourly weather | Open-Meteo archive API, `https://archive-api.open-meteo.com/v1/archive` | `python -m taxi.external.weather 2025-01-01 2025-12-31` and `python -m taxi.external.weather 2025-12-31 2026-07-31` | raw: `work/external/weather_raw/<ICAO>_<start>_<end>.json`; table: `work/external/weather_hourly.parquet` |
| METAR | Iowa Environmental Mesonet ASOS archive, `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` | `python -m taxi.external.metar 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12 2026-01 2026-07 output=metar_hourly.parquet` and `python -m taxi.external.metar 2026-02 2026-06` | raw: `work/external/metar_raw/metar_<ICAO>_<YYYYMM>.csv`; tables: `work/external/metar_hourly.parquet` (training year, January and July 2026), `work/external/metar_hourly_extra.parquet` (February and June 2026) |
| Monthly taxi-out table | ansperformance.eu (EUROCONTROL), Airport → Taxi-out additional time, Excel download | manual | `work/external/Taxi-Out_Additional_Time.xlsx` |
| ADS-B traces | adsb.lol globe_history releases, `https://github.com/adsblol/globe_history_2025` and `https://github.com/adsblol/globe_history_2026` (ODbL-1.0) | `python -m taxi.external.adsb 2025-01-01:2025-02-28 2025-05-31:2025-08-31 2025-12-01:2026-02-28 2026-05-31:2026-07-31 workers=6 delete=1` | raw: `work/adsb/raw/<day>/` (deleted after extraction with `delete=1`); points: `work/adsb/points/<day>.parquet` |
| OPDI flight events | EUROCONTROL Open Performance Data Initiative v0.0.2, `https://www.eurocontrol.int/performance/data/download/OPDI/v002/` | `python -m taxi.external.opdi 2025-01 2025-02 2025-06 2025-07 2025-08 2025-12 2026-01 2026-02 2026-06 2026-07` | raw: `work/opdi/raw/flight_events_*.parquet`, `flight_list_*.parquet`; filtered: `work/opdi/filtered/` |
| Runways | OurAirports, `https://davidmegginson.github.io/ourairports-data/runways.csv` | `python run.py references runways` | `work/external/ourairports_runways.csv` → `artifacts/reference/runways.csv` |
| Stand positions | OpenStreetMap Overpass API (`aeroway=parking_position`) | `python run.py references stand_runway_distances` | `work/external/osm/<ICAO>.json`, `work/external/osm_stands.csv` → `artifacts/reference/stand_runway_distances.csv` |

Notes on the downloads:
- One ADS-B day is a 2–3 part tar archive of about 3–4 GB covering the whole world. Only points near the 10 airports are kept.
- The ADS-B days include the day before each month: departures shortly after midnight use the last 3 hours of the previous day.
- OPDI events come in 10-day files; the month list also fetches the file that overlaps the day before each month.
- The METAR months for `metar_hourly.parquet` are all 12 months of 2025 plus `2026-01` and `2026-07`.

Then build the features:
```
python run.py features base groups adsb opdi
```
Add `adsb_sparse` to also build the thinned ADS-B copies; they are needed only to retrain the ADS-B corrector.

### Prediction
```
python run.py predict_a     # Model A for rows with an NM off-block time -> work/model_a_test.parquet
python run.py predict       # correctors and Model B -> work/prediction_test.parquet
python run.py output        # post-processing and checks -> output/prediction_<N>.parquet
```
The output has the columns `MVT_ID_mvt` and `TAXITIME_SEC_mvt`, in the row order of `final_submitting.parquet`. Each written file is logged with its MD5 in `logs/output_ledger.csv`.

## Regenerating everything
The code that produces every model and artifact is in this repository:

| Output | Code | Stage |
|---|---|---|
| `work/` feature tables | `taxi/features/*` | `features` |
| `artifacts/reference/stand_centers.parquet`, `iata_to_icao.json` | `taxi/references.py` | `references` |
| `artifacts/reference/runways.csv`, `stand_runway_distances.csv` | `taxi/references.py` (downloads OurAirports and OpenStreetMap) | `references` |
| `models/model_a` | `taxi/model_a/components.py` | `train_a` |
| `artifacts/oof/model_a_oof.parquet` | `taxi/model_a/oof.py` | `train_a_oof` |
| blend weights in `config.MODEL_A` | `taxi/model_a/blend.py: choose_weights` | `weights` |
| `models/model_b` | `taxi/model_b/base.py` | `train_b` |
| `artifacts/oof/model_b_oof.parquet` | `taxi/model_b/oof.py` | `train_b_oof` |
| `models/adsb`, `models/opdi`, `models/ltfm` | `taxi/correctors/*` | `train_correctors` |
| `artifacts/reference/base_schema.json` | column order and types of the base table (definition file) | — |

Regenerating the artifacts gives close but not identical files:
- **Reference tables.** OpenStreetMap and OurAirports change over time. Regenerated tables agree with the provided ones on common rows but differ in coverage:
  - `runways.csv`: 101 runway ends regenerated vs 71 provided; the 63 common rows are identical.
  - `stand_runway_distances.csv`: 31,476 vs 26,634 rows.
  - `stand_centers.parquet`: 16 of 1,564 stands differ.
  - `iata_to_icao.json`: 118 vs 126 entries.
- **Model B OOF:** the correlation with the provided file is 0.9998 for the per-airport method, 0.997 for the all-rows model and 0.97 for the ADS-B corrector.
- **Retrained models.**
  - Retraining the correctors moves the test taxi times by 9 s RMS (correlation 0.9997).
  - Retraining Model B moves them by 78 s RMS on its 9,077 rows (correlation 0.9997).
  - On small data sets LightGBM chooses col-wise or row-wise histogram construction by a timing test, so a retrain can give a slightly different model. Set `force_col_wise=True` in the parameters for bit-exact reruns.
- **Blend weights:** the `weights` stage re-runs the selection; on the provided OOF it returns [0.195, 0.148, 0.220, 0.129, 0.307]. `config.MODEL_A` holds the weights used for the provided models.

## Structure
```
config.py        paths, test files, model settings, scales (single place for settings)
run.py           stages (the notebooks call these)
taxi/
  data.py        raw movements (train / test), output template
  external/      downloaders: weather (Open-Meteo), METAR (IEM), ADS-B (adsb.lol), OPDI
  features/      base table, group features, ADS-B tracks, OPDI events, target encoding, model matrices
  model_a/       5 components (train / save / predict), blend, OOF generation
  correctors/    ADS-B, OPDI, LTFM second stage
  model_b/       rows without an NM off-block time: base, ADS-B corrector, residual corrector
  references.py  reference tables
  postprocess.py time shift, monthly calibration, LOBT window
  output.py      template order, checks, output ledger
notebooks/       00–05
artifacts/       oof/ (held-out predictions of the training year), reference/ (runway dimensions, stand–runway distances,
                 stand centres, IATA→ICAO, base table schema)
```

## Method
1. **Model A** (rows with an NM off-block time AOBT_3, 98.6% of the test rows).
   - Target offset = BLOCK − AOBT_3. Prediction: taxi = max(60, (MVT − AOBT_3) − offset).
   - Five components:
     - XGBoost (per airport)
     - symmetric CatBoost
     - Depthwise CatBoost
     - Huber LightGBM (per airport)
     - locked-row mixture: probability that BLOCK equals the schedule × schedule + the rest × LightGBM
   - The blend weights were chosen by constrained least squares on held-out predictions.
2. **Correctors.** They learn the residual of the held-out Model A prediction (`artifacts/oof`):
   - **ADS-B:** track features such as taxi start, stand split and first movement.
   - **OPDI:** ground events, used when ADS-B does not see the aircraft on the ground.
   - **LTFM second stage:** LTFM rows that the other correctors leave untouched (scale 0.3).
3. **Model B** (rows without an NM off-block time):
   - **EHAM / LSZH / EDDF:** per-airport LightGBM.
   - **LIRF:** a regime mixture of normal taxi, schedule echo and a 24-hour shift.
   - **Other airports:** a mixture trained on all rows.
   - Then an ADS-B corrector and a residual corrector (LTFM / LFPG) are applied.
4. **Post-processing:**
   - time shift (see the notes)
   - shift of the airport × month mean to the published monthly taxi-out average (trimmed mean for the valid-flight definition)
   - LOBT window |BLOCK − LOBT| ≤ 3,606 s
   - 60 s floor

## Evaluation (training year, 6 held-out months)
In each split the Model A components are trained on 10 months and predict the remaining 2. Taxi RMSE (s):

| | NM baseline (MVT − AOBT_3) | Model A blend |
|---|---|---|
| 6 months (1,015,188 departures) | 398.5 | 206.0 |

Breakdowns by airport and month are in `notebooks/05_evaluation.ipynb`.

## Notes
- **Two test files.**
  - In `final_ranking.parquet`, the MVT times of the January/July rows (and the BLOCK times of arrivals) carry ±12 s noise; the same rows are noise-free in `ranking.parquet`.
  - Features use the noise-free times; when an ID is in both files, the later file wins.
  - The predicted BLOCK is kept fixed and the taxi time is moved to the MVT of the final file: `postprocess.time_shift`.
- **Population.** Group statistics (median NM taxi time and counts per stand–runway, aircraft type and operator) are computed on a fixed population: the training year plus the departures in `ranking.parquet`. The models were trained with this definition.
- **Five window columns.** ARR_TAXI_MED_30/60, ARR_TAXI_MAX_30 and DEP_TAXI_MED_15/30 are in practice noise drawn from the month × airport distribution.
  - Cause: the results are in time order while the rows are matched in file order.
  - They are kept because the models were trained with this definition. Test rows are arranged in month blocks, so the mixing stays within a month, as in training.
  - Details: `taxi/features/base.py`, `_window_stat`.
- **CPU / GPU.** Without a GPU, CatBoost trains on the CPU with `border_count=128`. The provided Model A was trained on a CPU.
- **Intermediate artifacts.**
  - `artifacts/oof/model_a_oof.parquet`: held-out predictions of Model A over 3 splits; the training base of the correctors. A few hours per split on a CPU.
  - `artifacts/oof/model_b_oof.parquet`: 4-quarter CV predictions of Model B; the training base of the Model B residual corrector. ~15 min.
