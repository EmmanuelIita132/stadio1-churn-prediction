"""Metrics and statistical tests used for Model 1 / Model 2 evaluation and comparison."""
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_recall_curve,
                             roc_auc_score)


def ece(y, p, bins=10):
    """Expected calibration error with equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
               for b in range(bins) if (idx == b).any())


def top_frac(y, p, frac=0.1):
    """Precision, recall and lift among the top `frac` highest-risk users."""
    k = max(1, int(len(y) * frac))
    top = np.argsort(-p)[:k]
    prec = y[top].mean()
    return prec, y[top].sum() / max(y.sum(), 1), prec / y.mean()


def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[np.argmax(f1)])


def point_metrics(y, p, threshold):
    yhat = (p >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(threshold=threshold, precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12), tp=tp, fp=fp, fn=fn, tn=tn)


def all_metrics(y, p, threshold):
    p10 = top_frac(y, p, 0.10)
    p20 = top_frac(y, p, 0.20)
    return {"log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "ece": ece(y, p),
            "top10_precision": p10[0], "top10_recall": p10[1], "top10_lift": p10[2],
            "top20_recall": p20[1], "top20_lift": p20[2], **point_metrics(y, p, threshold)}


BOOT_KEYS = ["log_loss", "roc_auc", "pr_auc", "brier", "top10_lift"]


def _boot_metrics(y, p):
    return {"log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "top10_lift": top_frac(y, p, 0.1)[2]}


def bootstrap_ci(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    res = {k: [] for k in BOOT_KEYS}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        m = _boot_metrics(y[i], p[i])
        for k in BOOT_KEYS:
            res[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in res.items()}


def paired_bootstrap(y, p1, p2, n=1000, seed=42):
    """Paired bootstrap of metric(model2) - metric(model1); same resample for both models."""
    rng = np.random.default_rng(seed)
    d = {k: [] for k in BOOT_KEYS}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        m1, m2 = _boot_metrics(y[i], p1[i]), _boot_metrics(y[i], p2[i])
        for k in BOOT_KEYS:
            d[k].append(m2[k] - m1[k])
    out = {}
    for k, v in d.items():
        v = np.array(v)
        out[k] = {"mean_diff": float(v.mean()), "ci": (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))),
                  "p_two_sided": float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))}
    return out


def mcnemar(y, yhat1, yhat2):
    """McNemar test (continuity-corrected chi-square) on correctness of the two classifiers."""
    c1, c2 = yhat1 == y, yhat2 == y
    b = int((c1 & ~c2).sum()); c = int((~c1 & c2).sum())  # b: only model1 right, c: only model2 right
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1))}
