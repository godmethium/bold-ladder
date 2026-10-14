import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
EXTERNAL = WORK / "external"
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "output"
LOGS = ROOT / "logs"

RAW = Path(os.environ.get("TAXI_RAW_DIR", ROOT / "data"))
TRAIN_FILES = sorted(RAW.glob("training_2025-*.parquet"))
TEST_FILES = [RAW / "final_ranking.parquet", RAW / "ranking.parquet"]
FINAL_TEST_FILE = RAW / "final_ranking.parquet"
TEMPLATE = RAW / "final_submitting.parquet"
TEST_MONTHS = (1, 2, 6, 7)
POPULATION_TEST_FILE = RAW / "ranking.parquet"

ADSB_ROOT = Path(os.environ.get("TAXI_ADSB_ROOT", WORK / "adsb"))
OPDI_ROOT = Path(os.environ.get("TAXI_OPDI_ROOT", WORK / "opdi"))

GPU = os.environ.get("TAXI_GPU", "1" if shutil.which("nvidia-smi") else "0") == "1"
XGB_DEVICE = "cuda" if GPU else "cpu"
CB_THREADS = int(os.environ.get("TAXI_CB_THREADS", "-1"))


def cb_device(**params):
    return {"task_type": "GPU", **params} if GPU else {"task_type": "CPU", "border_count": 128, **params}


AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
TARGET = "NM_OFFSET_SEC"
TAXI_MIN = 60.0
LOBT_WINDOW = 3606.0
SEED = 42
TRAIN_YEAR = 2025
OOF_SPLITS = {"split_1": (1, 7), "split_2": (2, 8), "split_3": (12, 6)}

LGB_COMMON = dict(objective="huber", alpha=1000.0, metric="rmse", learning_rate=0.035, num_leaves=200, max_depth=-1,
                  min_child_samples=25, feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1, n_jobs=8,
                  verbosity=-1)
MODEL_A = {
    "xgboost": dict(kind="xgboost", weight=0.1021, seeds=(42, 999),
                params=dict(n_estimators=1500, learning_rate=0.04, max_depth=7, tree_method="hist", verbosity=0)),
    "catboost_symmetric": dict(kind="catboost", weight=0.2036, seeds=(42, 2026, 7), cat="basic",
                params=dict(iterations=6000, learning_rate=0.06, depth=9)),
    "lightgbm_lock": dict(kind="lock_mixture", weight=0.2211, seeds=(42, 123, 777), rounds=1500, params=LGB_COMMON,
               classifier=dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
                               feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1, n_jobs=8, verbosity=-1,
                               random_state=42), classifier_rounds=400),
    "lightgbm_huber": dict(kind="lightgbm", weight=0.1236, seeds=(42, 123, 777), rounds=1500, params=LGB_COMMON),
    "catboost_depthwise": dict(kind="catboost", weight=0.3497, seeds=(42, 2026, 7, 11), cat="extended",
               params=dict(iterations=9000, learning_rate=0.03, depth=9, grow_policy="Depthwise", min_data_in_leaf=50,
                           l2_leaf_reg=10)),
}
LOCK_COLUMN = "SCHED_MINUS_AOBT_CDM"
LOCK_TOL = 60.0
GCB_TARGET_CLIP = 7200.0

SCALE = {"ltfm_second": 0.3, "model_b_residual": 1.0, "monthly_calibration": 1.0}
