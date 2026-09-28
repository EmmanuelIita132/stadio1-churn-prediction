"""Feature engineering: data/interim/*.parquet -> data/processed/features_{train,test}.parquet.

Usage:
    python src/pipeline/feature_engineering.py

Every feature is computed RELATIVE to the cohort's prediction date (ref_date) using only data
strictly before it ("relative refactoring", Gregory 2018). A feature such as `days_since_last_log`
therefore means the same thing in the training and test cohorts.

Feature groups
  transactions : counts, cancel history, last plan / price / auto-renew flag, discount, days to expiry
  members      : age, gender, city, registration channel, registration tenure
  activity     : per-window (7/14/30/60/90/180 days) active days, seconds, unique songs, completion
                 and skip ratios, recency, and short-vs-long-term trend ratios
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

WINDOWS = [7, 14, 30, 60, 90, 180]


def transaction_features(tx: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    t = tx[tx.transaction_date < ref].sort_values(["msno", "transaction_date", "membership_expire_date"])
    g = t.groupby("msno")
    f = pd.DataFrame({
        "n_tx": g.size(),
        "n_cancel": g["is_cancel"].sum(),
        "share_auto_renew": g["is_auto_renew"].mean(),
        "n_payment_methods": g["payment_method_id"].nunique(),
        "n_plan_lengths": g["payment_plan_days"].nunique(),
        "total_paid": g["actual_amount_paid"].sum(),
        "first_tx": g["transaction_date"].min(),
        "last_tx": g["transaction_date"].max(),
        "max_expire": g["membership_expire_date"].max(),
    })
    last = g.tail(1).set_index("msno")
    f["last_payment_method_id"] = last["payment_method_id"]
    f["last_plan_days"] = last["payment_plan_days"]
    f["last_list_price"] = last["plan_list_price"]
    f["last_paid"] = last["actual_amount_paid"]
    f["last_auto_renew"] = last["is_auto_renew"]
    f["last_is_cancel"] = last["is_cancel"]
    f["days_to_expiry"] = (f["max_expire"] - ref).dt.days
    f["days_since_last_tx"] = (ref - f["last_tx"]).dt.days
    f["tenure_days"] = (ref - f["first_tx"]).dt.days
    f["avg_days_between_tx"] = f["tenure_days"] / f["n_tx"].clip(lower=1)
    f["last_discount"] = f["last_list_price"] - f["last_paid"]
    f["last_price_per_day"] = f["last_paid"] / f["last_plan_days"].replace(0, np.nan)
    f["cancel_rate"] = f["n_cancel"] / f["n_tx"]
    return f.drop(columns=["first_tx", "last_tx", "max_expire"])


def member_features(m: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    f = m.set_index("msno")[["city", "bd", "gender", "registered_via"]].copy()
    f = f.rename(columns={"bd": "age"})
    f["days_since_registration"] = (ref - m.set_index("msno")["registration_init_time"]).dt.days
    return f


def log_features(logs: pd.DataFrame, users: pd.Index, ref: pd.Timestamp) -> pd.DataFrame:
    l = logs[logs.msno.isin(users)].copy()
    l["age"] = (ref - l["date"]).dt.days  # 1 = day before prediction date
    l["all_songs"] = l[["num_25", "num_50", "num_75", "num_985", "num_100"]].sum(axis=1)
    out = pd.DataFrame(index=pd.Index(users, name="msno"))
    for w in WINDOWS:
        s = l[l.age <= w].groupby("msno").agg(
            **{f"active_days_{w}": ("date", "nunique"), f"secs_{w}": ("total_secs", "sum"),
               f"unq_{w}": ("num_unq", "sum"), f"n100_{w}": ("num_100", "sum"),
               f"n25_{w}": ("num_25", "sum"), f"songs_{w}": ("all_songs", "sum")})
        out = out.join(s)
    cols = [c for c in out.columns]
    out[cols] = out[cols].fillna(0)  # no logs in the window = genuinely no activity
    for w in (14, 30, 90):
        out[f"completion_{w}"] = out[f"n100_{w}"] / out[f"songs_{w}"].replace(0, np.nan)
        out[f"skip_{w}"] = out[f"n25_{w}"] / out[f"songs_{w}"].replace(0, np.nan)
        out[f"secs_per_active_day_{w}"] = out[f"secs_{w}"] / out[f"active_days_{w}"].replace(0, np.nan)
    # trend: last 14 days vs the previous 30 days (days 15-44), per-day rates
    prev = l[(l.age > 14) & (l.age <= 44)].groupby("msno").agg(prev_secs=("total_secs", "sum"),
                                                              prev_days=("date", "nunique"))
    out = out.join(prev).fillna({"prev_secs": 0, "prev_days": 0})
    out["trend_secs"] = (out["secs_14"] / 14 + 1) / (out["prev_secs"] / 30 + 1)
    out["trend_active_days"] = (out["active_days_14"] / 14 + 0.01) / (out["prev_days"] / 30 + 0.01)
    out["trend_secs_30_vs_90"] = (out["secs_30"] / 30 + 1) / (out["secs_90"] / 90 + 1)
    out = out.drop(columns=["prev_secs", "prev_days"])
    rec = l.groupby("msno")["age"].min().rename("days_since_last_log")
    out = out.join(rec)
    out["has_logs"] = out["days_since_last_log"].notna().astype(int)
    out["days_since_last_log"] = out["days_since_last_log"].fillna(C.LOG_LOOKBACK_DAYS + 1)
    return out


def main():
    members = pd.read_parquet(C.INTERIM / "members.parquet")
    tx = pd.read_parquet(C.INTERIM / "transactions.parquet")
    logs = pd.read_parquet(C.INTERIM / "logs.parquet")
    for name, cfg in C.COHORTS.items():
        ref = pd.Timestamp(cfg["ref_date"])
        lab = pd.read_parquet(C.INTERIM / f"labels_{name}.parquet").set_index("msno")
        users = lab.index
        feats = (lab[["is_churn"]]
                 .join(transaction_features(tx[tx.msno.isin(users)], ref))
                 .join(member_features(members[members.msno.isin(users)], ref))
                 .join(log_features(logs, users, ref)))
        feats["ref_date"] = ref
        feats.reset_index().to_parquet(C.PROCESSED / f"features_{name}.parquet", index=False)
        print(f"{name}: {feats.shape[0]:,} users x {feats.shape[1]-2} features, churn={feats.is_churn.mean():.3%}")


if __name__ == "__main__":
    main()
