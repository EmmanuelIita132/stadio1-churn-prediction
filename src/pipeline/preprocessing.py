"""Preprocessing: raw KKBox CSVs -> cleaned per-cohort parquet tables in data/interim/.

Usage:
    python src/pipeline/preprocessing.py                # uses data/raw
    RAW_DIR=/path/to/kkbox python src/pipeline/preprocessing.py

Steps
1. Read labels for each cohort (train.csv = Feb 2017 expiries, train_v2.csv = Mar 2017 expiries).
2. Remove from the TRAIN cohort any user who is also in the TEST cohort (no user overlap).
3. Optionally sample users per cohort (config.SAMPLE_USERS_PER_COHORT) to keep memory manageable.
4. Clean members (implausible ages, missing gender, date parsing).
5. Clean transactions (date parsing, de-duplication, keep only rows BEFORE the cohort prediction date).
6. Stream user_logs.csv in chunks and keep only sampled users, only days inside the look-back window
   and strictly before the prediction date (prevents leakage).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def parse_yyyymmdd(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def load_labels() -> dict:
    labels = {name: pd.read_csv(C.RAW / cfg["label_file"]).drop_duplicates("msno")
              for name, cfg in C.COHORTS.items()}
    # Time-based split hygiene: a user in the test cohort must not appear in training.
    n_before = len(labels["train"])
    labels["train"] = labels["train"][~labels["train"].msno.isin(labels["test"].msno)]
    print(f"train cohort: dropped {n_before - len(labels['train']):,} users that also appear in test")
    if C.SAMPLE_USERS_PER_COHORT:
        for k in labels:
            n = min(C.SAMPLE_USERS_PER_COHORT, len(labels[k]))
            labels[k] = labels[k].sample(n, random_state=C.SEED)
    for k, v in labels.items():
        print(f"{k}: {len(v):,} users, churn rate {v.is_churn.mean():.3%}")
    return labels


def clean_members(users: set) -> pd.DataFrame:
    m = pd.read_csv(C.RAW / "members_v3.csv")
    m = m[m.msno.isin(users)].drop_duplicates("msno").copy()
    m["bd"] = m["bd"].where(m["bd"].between(10, 90))  # 0 / negative / >90 are placeholders -> NaN
    m["gender"] = m["gender"].fillna("unknown")
    m["registration_init_time"] = parse_yyyymmdd(m["registration_init_time"])
    return m


def clean_transactions(users: set) -> pd.DataFrame:
    parts = []
    for f in ("transactions.csv", "transactions_v2.csv"):
        p = C.RAW / f
        if p.exists():
            for ch in pd.read_csv(p, chunksize=2_000_000):
                parts.append(ch[ch.msno.isin(users)])
    t = pd.concat(parts, ignore_index=True).drop_duplicates()
    t["transaction_date"] = parse_yyyymmdd(t["transaction_date"])
    t["membership_expire_date"] = parse_yyyymmdd(t["membership_expire_date"])
    t = t.dropna(subset=["transaction_date", "membership_expire_date"])
    # implausible expiry earlier than transaction -> drop
    t = t[t.membership_expire_date >= t.transaction_date]
    return t


def filter_logs(users: set, ref_dates: dict) -> pd.DataFrame:
    """Stream user_logs.csv; users belong to exactly one cohort, so one ref date each."""
    user_ref = pd.concat([pd.DataFrame({"msno": list(u), "ref": pd.Timestamp(ref_dates[k])})
                          for k, u in users.items()])
    ref_map = user_ref.set_index("msno")["ref"]
    all_users = set(ref_map.index)
    parts = []
    log_file = C.RAW / "user_logs.csv"
    if not log_file.exists():
        print("WARNING: user_logs.csv not found - falling back to user_logs_v2.csv (March only). "
              "Test-cohort log features will be empty; results will be worse.")
        log_file = C.RAW / "user_logs_v2.csv"
    for i, ch in enumerate(pd.read_csv(log_file, chunksize=5_000_000)):
        ch = ch[ch.msno.isin(all_users)]
        if ch.empty:
            continue
        ch["date"] = parse_yyyymmdd(ch["date"])
        ch["ref"] = ch.msno.map(ref_map)
        age = (ch["ref"] - ch["date"]).dt.days
        ch = ch[(age >= 1) & (age <= C.LOG_LOOKBACK_DAYS)]  # strictly before the prediction date
        parts.append(ch.drop(columns="ref"))
        if i % 10 == 0:
            print(f"  log chunk {i}: kept {sum(len(p) for p in parts):,} rows")
    logs = pd.concat(parts, ignore_index=True)
    for c in ("num_25", "num_50", "num_75", "num_985", "num_100", "num_unq"):
        logs[c] = logs[c].astype("int32")
    # total_secs contains extreme / negative outliers -> clip to [0, 24h]
    logs["total_secs"] = logs["total_secs"].clip(0, 86_400).astype("float32")
    return logs


def main():
    labels = load_labels()
    users = {k: set(v.msno) for k, v in labels.items()}
    everyone = set().union(*users.values())

    members = clean_members(everyone)
    tx = clean_transactions(everyone)
    logs = filter_logs(users, {k: c["ref_date"] for k, c in C.COHORTS.items()})

    for k, v in labels.items():
        v.to_parquet(C.INTERIM / f"labels_{k}.parquet", index=False)
    members.to_parquet(C.INTERIM / "members.parquet", index=False)
    tx.to_parquet(C.INTERIM / "transactions.parquet", index=False)
    logs.to_parquet(C.INTERIM / "logs.parquet", index=False)
    print(f"saved interim tables to {C.INTERIM}: members={len(members):,} tx={len(tx):,} logs={len(logs):,}")


if __name__ == "__main__":
    main()
