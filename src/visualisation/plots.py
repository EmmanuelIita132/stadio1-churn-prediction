"""Plot helpers (PR / ROC curves, calibration, decile lift, feature importance). Used by src/stats/evaluate.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def pr_curve(curves: dict, y, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, p in curves.items():
        pr, rc, _ = precision_recall_curve(y, p); ax.plot(rc, pr, label=name)
    ax.axhline(y.mean(), ls=":", c="grey", label="No skill"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve"); ax.legend(); _save(fig, path)


def roc_plot(curves: dict, y, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], ls=":", c="grey"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(); _save(fig, path)


def calibration(curves: dict, y, path, bins=10):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    edges = np.linspace(0, 1, bins + 1)
    for name, p in curves.items():
        idx = np.minimum(np.digitize(p, edges) - 1, bins - 1)
        xs = [p[idx == b].mean() for b in range(bins) if (idx == b).sum() > 20]
        ys = [y[idx == b].mean() for b in range(bins) if (idx == b).sum() > 20]
        ax.plot(xs, ys, marker="o", label=name)
    ax.plot([0, 1], [0, 1], ls=":", c="grey"); ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn rate"); ax.set_title("Calibration"); ax.legend(); _save(fig, path)


def decile_lift(curves: dict, y, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    w = 0.8 / len(curves)
    for j, (name, p) in enumerate(curves.items()):
        order = np.argsort(-p); chunks = np.array_split(y[order], 10)
        ax.bar(np.arange(1, 11) + j * w - 0.4 + w / 2, [c.mean() / y.mean() for c in chunks], w, label=name)
    ax.axhline(1, ls=":", c="grey"); ax.set_xlabel("Risk decile (1 = highest risk)"); ax.set_ylabel("Lift")
    ax.set_title("Lift by risk decile"); ax.legend(); _save(fig, path)


def importance(series, path, top=15):
    s = series.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.barh(s.index, s.values); ax.set_title("XGBoost feature importance (top 15)")
    _save(fig, path)
