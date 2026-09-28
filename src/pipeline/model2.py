"""Model 2: XGBoost gradient-boosted trees.

Usage:
    python src/pipeline/model2.py train      # small depth grid, early stopping on validation log loss
    python src/pipeline/model2.py predict    # score the test cohort -> experiments/results/model2_predictions.parquet
    python src/pipeline/model2.py predict --input path/to/features.parquet --output out.parquet

Fixed hyper-parameters: learning_rate=0.05, n_estimators<=2000 (early stopping, 50 rounds), min_child_weight=5,
subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist', no class re-weighting
(Ahmad et al. 2019 found boosted trees did best without re-balancing).  Grid: max_depth in {4, 6, 8}.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from pipeline.common import cast_categoricals, feature_columns, load_cohort, split_train_valid, tree_preprocessor, xy  # noqa: E402
from stats.metrics import best_f1_threshold  # noqa: E402

DEPTH_GRID = [4, 6, 8]
FIXED = dict(learning_rate=0.05, n_estimators=2000, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=1.0, tree_method="hist", eval_metric="logloss", early_stopping_rounds=50,
             random_state=C.SEED, n_jobs=-1)
MODEL_PATH = C.MODELS / "model2_xgboost.joblib"
CONFIG_PATH = C.SETUP / "model2_config.json"


def train():
    df = load_cohort("train")
    nums, cats = feature_columns(df)
    tr, va = split_train_valid(df)
    Xtr, ytr = xy(tr, nums, cats)
    Xva, yva = xy(va, nums, cats)
    prep = tree_preprocessor(nums, cats).fit(Xtr)
    Ptr, Pva = prep.transform(Xtr), prep.transform(Xva)
    best, grid_log = None, []
    for d in DEPTH_GRID:
        clf = XGBClassifier(max_depth=d, **FIXED)
        clf.fit(Ptr, ytr, eval_set=[(Pva, yva)], verbose=False)
        p = clf.predict_proba(Pva)[:, 1]
        ll, ap = log_loss(yva, p), average_precision_score(yva, p)
        grid_log.append({"max_depth": d, "best_iteration": int(clf.best_iteration), "valid_log_loss": ll,
                         "valid_pr_auc": ap})
        print(f"max_depth={d} trees={clf.best_iteration} valid log loss={ll:.4f}  PR-AUC={ap:.4f}")
        if best is None or ll < best[0]:
            best = (ll, d, clf, p)
    ll, d, clf, pva = best
    thr = best_f1_threshold(yva, pva)
    joblib.dump({"prep": prep, "clf": clf, "nums": nums, "cats": cats}, MODEL_PATH)
    imp = pd.Series(clf.feature_importances_, index=nums + cats).sort_values(ascending=False)
    imp.to_csv(C.RESULTS / "model2_feature_importance.csv", header=["gain_importance"])
    json.dump({"model": "XGBClassifier", "selection_metric": "validation log loss", "fixed_params":
               {k: v for k, v in FIXED.items() if k not in ("random_state", "n_jobs")}, "grid": grid_log,
               "best_params": {"max_depth": d, "n_estimators_used": int(clf.best_iteration) + 1},
               "f1_threshold": thr, "n_train": len(tr), "n_valid": len(va)}, open(CONFIG_PATH, "w"), indent=2)
    print(f"saved {MODEL_PATH.name}; best max_depth={d}; F1-optimal threshold={thr:.3f}")


def predict(input_path=None, output_path=None):
    b = joblib.load(MODEL_PATH)
    df = pd.read_parquet(input_path) if input_path else load_cohort("test")
    df = cast_categoricals(df)
    X = b["prep"].transform(df[b["nums"] + b["cats"]])
    out = pd.DataFrame({"msno": df["msno"], "y_true": df.get("is_churn"), "y_score": b["clf"].predict_proba(X)[:, 1]})
    out_path = Path(output_path) if output_path else C.RESULTS / "model2_predictions.parquet"
    out.to_parquet(out_path, index=False)
    print(f"wrote {len(out):,} predictions to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--input"); ap.add_argument("--output")
    a = ap.parse_args()
    train() if a.mode == "train" else predict(a.input, a.output)
