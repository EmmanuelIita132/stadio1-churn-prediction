"""Evaluation + statistical comparison. Writes the Part C markdown files with the REAL numbers.

Usage (after `model1.py predict` and `model2.py predict`):
    python src/stats/evaluate.py model1     # -> Model1Performance.md
    python src/stats/evaluate.py model2     # -> Model2Performance.md
    python src/stats/evaluate.py compare    # -> Comparison.md (+ experiments/results/comparison.json)
    python src/stats/evaluate.py all
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from stats.metrics import all_metrics, bootstrap_ci, mcnemar, paired_bootstrap, point_metrics  # noqa: E402
from visualisation import plots  # noqa: E402

NAMES = {"model1": "Model 1 (logistic regression)", "model2": "Model 2 (XGBoost)"}
FIG = "experiments/results/figures"


def load_preds(key):
    return pd.read_parquet(C.RESULTS / f"{key}_predictions.parquet")


def rule_scores(test_msno):
    """Baseline rule: predict churn when auto-renew was OFF on the last transaction before the prediction date."""
    f = pd.read_parquet(C.PROCESSED / "features_test.parquet").set_index("msno")
    return (f.loc[test_msno, "last_auto_renew"].fillna(0) == 0).astype(float).to_numpy()


def fmt(v, d=4):
    return f"{v:.{d}f}"


def metric_table(m, ci):
    rows = [("Log loss (lower is better)", "log_loss"), ("ROC AUC", "roc_auc"), ("PR AUC (average precision)", "pr_auc"),
            ("Brier score (lower is better)", "brier"), ("Top-10% lift", "top10_lift")]
    out = "| Metric | Value | 95% bootstrap CI |\n|---|---|---|\n"
    for label, k in rows:
        out += f"| {label} | {fmt(m[k])} | [{fmt(ci[k][0])}, {fmt(ci[k][1])}] |\n"
    out += f"| Expected calibration error | {fmt(m['ece'])} | n/a |\n"
    out += f"| Recall in top 10% of risk scores | {fmt(m['top10_recall'])} | n/a |\n"
    out += f"| Recall in top 20% of risk scores | {fmt(m['top20_recall'])} | n/a |\n"
    return out


def performance(key):
    p = load_preds(key)
    y, s = p.y_true.astype(int).to_numpy(), p.y_score.to_numpy()
    cfg = json.load(open(C.SETUP / f"{key}_config.json"))
    thr = cfg["f1_threshold"]
    m = all_metrics(y, s, thr)
    ci = bootstrap_ci(y, s, C.BOOTSTRAPS)
    rule = point_metrics(y, rule_scores(p.msno), 0.5)
    json.dump({"metrics": m, "ci": ci, "rule_baseline": rule, "n_test": len(y), "churn_rate": float(y.mean())},
              open(C.RESULTS / f"{key}_metrics.json", "w"), indent=2)
    plots.pr_curve({NAMES[key]: s}, y, C.FIGURES / f"{key}_pr.png")
    plots.calibration({NAMES[key]: s}, y, C.FIGURES / f"{key}_calibration.png")
    plots.decile_lift({NAMES[key]: s}, y, C.FIGURES / f"{key}_lift.png")
    extra = ""
    if key == "model2":
        imp = pd.read_csv(C.RESULTS / "model2_feature_importance.csv", index_col=0)["gain_importance"]
        plots.importance(imp, C.FIGURES / "model2_importance.png")
        top = ", ".join(f"`{i}`" for i in imp.index[:8])
        extra = (f"\n## Feature importance\n\nTop features by XGBoost importance: {top}.\n\n"
                 f"![importance]({FIG}/model2_importance.png)\n")
    calib = "well calibrated (ECE < 0.02)" if m["ece"] < 0.02 else "imperfectly calibrated (ECE >= 0.02)"
    script = "src/pipeline/model1.py" if key == "model1" else "src/pipeline/model2.py"
    md = f"""# {NAMES[key]}: Performance

**Related code**
- Training / inference: `{script}` (`python {script} train`, then `python {script} predict`)
- Metrics and this report: `src/stats/evaluate.py` (`python src/stats/evaluate.py {key}`), metric functions in `src/stats/metrics.py`
- Plots: `src/visualisation/plots.py`
- Hyper-parameter search log: `experiments/setup/{key}_config.json` (selected values: `{json.dumps(cfg['best_params'])}`)
- Raw predictions: `experiments/results/{key}_predictions.parquet`

## Evaluation set-up
- Dataset: WSDM KKBox churn data. **Training cohort**: users expiring in Feb 2017 (`train.csv`, prediction date 2017-02-01); **test cohort**: users expiring in Mar 2017 (`train_v2.csv`, prediction date 2017-03-01). Users in both cohorts were removed from training. This is a time-based split, so the test is a genuine future period.
- Validation: a random {int(C.VALID_FRACTION*100)}% slice of the training cohort ({cfg['n_valid']:,} users) for tuning and early stopping; model selection metric: {cfg['selection_metric']}.
- Test users: {len(y):,}; test churn rate: {y.mean():.2%}.
- Classification threshold ({thr:.3f}) was chosen on the validation slice to maximise F1, then applied unchanged to the test set.

## Results on the test cohort

{metric_table(m, ci)}
Confidence intervals come from {C.BOOTSTRAPS} bootstrap resamples of the test users.

### Classification at the validation-chosen threshold

| Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|
| {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {m['tp']:,} | {m['fp']:,} | {m['fn']:,} | {m['tn']:,} |

Rule baseline ("auto-renew off on last transaction"): precision {fmt(rule['precision'])}, recall {fmt(rule['recall'])}, F1 {fmt(rule['f1'])}.

## Interpretation
- The model is {calib}.
- Targeting the top 10% of risk scores captures {m['top10_recall']:.1%} of all churners (lift {m['top10_lift']:.2f}x over random targeting).
- Because churners are a minority class, PR AUC ({fmt(m['pr_auc'])}) should be compared with the no-skill level of {y.mean():.4f}, not with 0.5.

## Figures
![PR]({FIG}/{key}_pr.png) ![calibration]({FIG}/{key}_calibration.png) ![lift]({FIG}/{key}_lift.png)
{extra}"""
    (C.DOCS_OUT / f"{'Model1' if key=='model1' else 'Model2'}Performance.md").write_text(md)
    print(f"wrote {key} performance md; log loss={m['log_loss']:.4f} PR-AUC={m['pr_auc']:.4f}")


def compare():
    p1, p2 = load_preds("model1"), load_preds("model2")
    assert (p1.msno.values == p2.msno.values).all(), "prediction files are not aligned"
    y = p1.y_true.astype(int).to_numpy(); s1, s2 = p1.y_score.to_numpy(), p2.y_score.to_numpy()
    c1, c2 = (json.load(open(C.SETUP / f"{k}_config.json")) for k in ("model1", "model2"))
    m1, m2 = all_metrics(y, s1, c1["f1_threshold"]), all_metrics(y, s2, c2["f1_threshold"])
    rule = rule_scores(p1.msno); mr = point_metrics(y, rule, 0.5)
    pb = paired_bootstrap(y, s1, s2, C.BOOTSTRAPS)
    mc = mcnemar(y, (s1 >= c1["f1_threshold"]).astype(int), (s2 >= c2["f1_threshold"]).astype(int))
    curves = {"Model 1: LogReg": s1, "Model 2: XGBoost": s2}
    plots.pr_curve(curves, y, C.FIGURES / "compare_pr.png"); plots.roc_plot(curves, y, C.FIGURES / "compare_roc.png")
    plots.calibration(curves, y, C.FIGURES / "compare_calibration.png"); plots.decile_lift(curves, y, C.FIGURES / "compare_lift.png")
    better_ll = "Model 2" if m2["log_loss"] < m1["log_loss"] else "Model 1"
    better_ap = "Model 2" if m2["pr_auc"] > m1["pr_auc"] else "Model 1"
    winner = better_ll if better_ll == better_ap else better_ap
    sig = pb["pr_auc"]["p_two_sided"] < 0.05 and pb["log_loss"]["p_two_sided"] < 0.05
    json.dump({"model1": m1, "model2": m2, "rule": mr, "paired_bootstrap": pb, "mcnemar": mc,
               "winner": winner, "significant": bool(sig)}, open(C.RESULTS / "comparison.json", "w"), indent=2)
    rows = [("Log loss (lower better)", "log_loss"), ("ROC AUC", "roc_auc"), ("PR AUC", "pr_auc"),
            ("Brier (lower better)", "brier"), ("ECE (lower better)", "ece"), ("Top-10% lift", "top10_lift"),
            ("Top-10% recall", "top10_recall"), ("Precision @ threshold", "precision"), ("Recall @ threshold", "recall"),
            ("F1 @ threshold", "f1")]
    tab = "| Metric | Model 1 (LogReg) | Model 2 (XGBoost) | Rule baseline |\n|---|---|---|---|\n"
    for label, k in rows:
        rv = fmt(mr[k]) if k in mr else "n/a"
        tab += f"| {label} | {fmt(m1[k])} | {fmt(m2[k])} | {rv} |\n"
    pbt = "| Metric (Model 2 minus Model 1) | Mean difference | 95% CI | p (two-sided) |\n|---|---|---|---|\n"
    for k, lab in [("log_loss", "Log loss"), ("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("brier", "Brier"), ("top10_lift", "Top-10% lift")]:
        d = pb[k]; pbt += f"| {lab} | {fmt(d['mean_diff'],5)} | [{fmt(d['ci'][0],5)}, {fmt(d['ci'][1],5)}] | {d['p_two_sided']:.3f} |\n"
    md = f"""# Comparison of Model 1 and Model 2

**Related code**
- Comparison: `src/stats/evaluate.py` (`python src/stats/evaluate.py compare`); tests in `src/stats/metrics.py`
- Inputs: `experiments/results/model1_predictions.parquet` and `model2_predictions.parquet` (same test users, same order)
- Plots: `src/visualisation/plots.py`; numeric output: `experiments/results/comparison.json`

Both models were evaluated on the same {len(y):,} test users (March 2017 expiries, churn rate {y.mean():.2%}), trained only on the earlier February 2017 cohort.

## Side-by-side metrics

{tab}
## Statistical tests
**Paired bootstrap** ({C.BOOTSTRAPS} resamples; the same resampled users are scored by both models):

{pbt}
**McNemar test** at each model's validation-chosen threshold: only Model 1 correct = {mc['b']:,}; only Model 2 correct = {mc['c']:,}; chi-square = {mc['statistic']:.1f}; p = {mc['p_value']:.3g}.

## Conclusion
On log loss and PR AUC the better model is **{winner}** ({'the difference is statistically significant at the 5% level' if sig else 'the difference is NOT statistically significant at the 5% level on both key metrics'}).

## Figures
![PR]({FIG}/compare_pr.png) ![ROC]({FIG}/compare_roc.png)
![calibration]({FIG}/compare_calibration.png) ![lift]({FIG}/compare_lift.png)
"""
    (C.DOCS_OUT / "Comparison.md").write_text(md)
    print("wrote Comparison.md; winner:", winner, "| significant:", sig)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("model1", "all"): performance("model1")
    if what in ("model2", "all"): performance("model2")
    if what in ("compare", "all"): compare()
