"""Model 1: L2-regularised logistic regression (baseline).

Usage:
    python src/pipeline/model1.py train      # tune C on the validation slice, save model + config
    python src/pipeline/model1.py predict    # score the test cohort -> experiments/results/model1_predictions.parquet
    python src/pipeline/model1.py predict --input path/to/features.parquet --output out.parquet
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from pipeline.common import cast_categoricals, feature_columns, load_cohort, logistic_preprocessor, split_train_valid, xy  # noqa: E402
from stats.metrics import best_f1_threshold  # noqa: E402

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
MODEL_PATH = C.MODELS / "model1_logreg.joblib"
CONFIG_PATH = C.SETUP / "model1_config.json"


def train():
    df = load_cohort("train")
    nums, cats = feature_columns(df)
    tr, va = split_train_valid(df)
    Xtr, ytr = xy(tr, nums, cats)
    Xva, yva = xy(va, nums, cats)
    best = None
    grid_log = []
    for c in C_GRID:
        pipe = Pipeline([("prep", logistic_preprocessor(nums, cats)),
                         ("clf", LogisticRegression(C=c, solver="lbfgs", max_iter=2000))])
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        ll, ap = log_loss(yva, p), average_precision_score(yva, p)
        grid_log.append({"C": c, "valid_log_loss": ll, "valid_pr_auc": ap})
        print(f"C={c:<7} valid log loss={ll:.4f}  PR-AUC={ap:.4f}")
        if best is None or ll < best[0]:
            best = (ll, c, pipe, p)
    ll, c, pipe, pva = best
    thr = best_f1_threshold(yva, pva)
    joblib.dump({"pipeline": pipe, "nums": nums, "cats": cats}, MODEL_PATH)
    json.dump({"model": "LogisticRegression (L2, lbfgs, max_iter=2000)", "selection_metric": "validation log loss",
               "grid": grid_log, "best_params": {"C": c}, "f1_threshold": thr,
               "n_train": len(tr), "n_valid": len(va)}, open(CONFIG_PATH, "w"), indent=2)
    print(f"saved {MODEL_PATH.name}; best C={c}; F1-optimal threshold={thr:.3f}")


def predict(input_path=None, output_path=None):
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_parquet(input_path) if input_path else load_cohort("test")
    df = cast_categoricals(df)
    X = df[bundle["nums"] + bundle["cats"]]
    out = pd.DataFrame({"msno": df["msno"], "y_true": df.get("is_churn"),
                        "y_score": bundle["pipeline"].predict_proba(X)[:, 1]})
    out_path = Path(output_path) if output_path else C.RESULTS / "model1_predictions.parquet"
    out.to_parquet(out_path, index=False)
    print(f"wrote {len(out):,} predictions to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--input"); ap.add_argument("--output")
    a = ap.parse_args()
    train() if a.mode == "train" else predict(a.input, a.output)
