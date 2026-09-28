"""Shared configuration for the STADIO 1 / KKBox churn pipeline."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("RAW_DIR", ROOT / "data" / "raw"))
INTERIM = Path(os.environ.get("INTERIM_DIR", ROOT / "data" / "interim"))
PROCESSED = Path(os.environ.get("PROCESSED_DIR", ROOT / "data" / "processed"))
MODELS = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
SETUP = ROOT / "experiments" / "setup"
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "experiments" / "results"))
FIGURES = RESULTS / "figures"
DOCS_OUT = Path(os.environ.get("DOCS_OUT", ROOT))  # where the *.md result files are written

SEED = 42

# Time-based split: a cohort = users whose membership expires in that month.
# The prediction date is the first day of the cohort month; features use data BEFORE it only.
COHORTS = {
    "train": {"label_file": "train.csv", "ref_date": "2017-02-01"},     # Feb 2017 expiries
    "test": {"label_file": "train_v2.csv", "ref_date": "2017-03-01"},   # Mar 2017 expiries
}
LOG_LOOKBACK_DAYS = 180
SAMPLE_USERS_PER_COHORT = int(os.environ.get("SAMPLE_USERS", 200_000))  # 0 = all users
VALID_FRACTION = 0.2      # random slice of the TRAIN cohort for tuning / early stopping
BOOTSTRAPS = int(os.environ.get("BOOTSTRAPS", 1000))

CATEGORICAL = ["city", "registered_via", "gender", "last_payment_method_id"]

for _p in (INTERIM, PROCESSED, MODELS, SETUP, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
