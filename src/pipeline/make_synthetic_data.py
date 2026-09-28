"""Create a SMALL synthetic dataset in the exact KKBox file format, for smoke-testing the pipeline only.
NOT real data - never report results from it.   Usage: python src/pipeline/make_synthetic_data.py OUT_DIR [n_users]"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
rng = np.random.default_rng(0)
ymd = lambda s: pd.Series(s).dt.strftime("%Y%m%d").astype(int).to_numpy()

def cohort(prefix, month_start, offset):
    ids = [f"{prefix}{i:06d}=" for i in range(n)]
    eng = rng.normal(size=n); auto = rng.random(n) < 0.7
    p = 1 / (1 + np.exp(-(-1.2 - 1.0 * eng + 1.5 * (~auto))))
    churn = (rng.random(n) < p).astype(int)
    exp = month_start + pd.to_timedelta(rng.integers(0, 27, n), "D")
    plan = rng.choice([30, 30, 30, 7, 90], n)
    tx_date = exp - pd.to_timedelta(plan, "D")
    tx = pd.DataFrame({"msno": ids, "payment_method_id": rng.choice([38, 39, 40, 41], n), "payment_plan_days": plan,
                       "plan_list_price": plan * 5, "actual_amount_paid": plan * 5 - (rng.random(n) < 0.1) * 30,
                       "is_auto_renew": auto.astype(int), "transaction_date": ymd(tx_date),
                       "membership_expire_date": ymd(exp), "is_cancel": (rng.random(n) < 0.05).astype(int)})
    mem = pd.DataFrame({"msno": ids, "city": rng.integers(1, 20, n), "bd": rng.choice([0, 22, 30, 41, 150], n),
                        "gender": rng.choice(["male", "female", None], n), "registered_via": rng.choice([3, 4, 7, 9], n),
                        "registration_init_time": ymd(month_start - pd.to_timedelta(rng.integers(60, 1500, n), "D"))})
    logs = []
    ref = month_start
    for i, uid in enumerate(ids):
        k = rng.poisson(max(0.5, 12 + 6 * eng[i]))
        days = ref - pd.to_timedelta(rng.integers(1, 120, k), "D")
        logs.append(pd.DataFrame({"msno": uid, "date": ymd(days), "num_25": rng.poisson(5, k),
                                  "num_50": rng.poisson(2, k), "num_75": rng.poisson(2, k), "num_985": rng.poisson(2, k),
                                  "num_100": rng.poisson(20, k), "num_unq": rng.poisson(25, k),
                                  "total_secs": rng.gamma(2, 3000, k)}))
    return pd.DataFrame({"msno": ids, "is_churn": churn}), tx, mem, pd.concat(logs)

l1, t1, m1, g1 = cohort("a", pd.Timestamp("2017-02-01"), 0)
l2, t2, m2, g2 = cohort("b", pd.Timestamp("2017-03-01"), 0)
l1.to_csv(out / "train.csv", index=False); l2.to_csv(out / "train_v2.csv", index=False)
pd.concat([m1, m2]).to_csv(out / "members_v3.csv", index=False)
t1.to_csv(out / "transactions.csv", index=False); t2.to_csv(out / "transactions_v2.csv", index=False)
pd.concat([g1, g2]).to_csv(out / "user_logs.csv", index=False)
print("synthetic data written to", out)
