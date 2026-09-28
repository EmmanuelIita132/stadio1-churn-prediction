# Feature Engineering (Part B)

**Code:** `src/pipeline/feature_engineering.py`  |  **Input:** `data/interim/`  |  **Output:** `data/processed/features_train.parquet`, `features_test.parquet`

## How to run
`python src/pipeline/feature_engineering.py` (run `Preprocessing.md` steps first).

## Design principle
Every feature is computed **relative to the cohort's prediction date** and only from earlier data (the "relative refactoring" idea in Gregory, 2018). For example, `days_since_last_log` means the same thing for the February and the March cohort.

## Feature groups (about 70 features)
| Group | Features | Rationale |
|---|---|---|
| Transaction / plan | `n_tx`, `n_cancel`, `cancel_rate`, `share_auto_renew`, `n_payment_methods`, `n_plan_lengths`, `total_paid`, `tenure_days`, `days_since_last_tx`, `avg_days_between_tx`, `days_to_expiry` | Cancellation history and subscription state were strong predictors in Gregory (2018). |
| Last plan | `last_plan_days`, `last_list_price`, `last_paid`, `last_discount`, `last_price_per_day`, `last_auto_renew`, `last_is_cancel`, `last_payment_method_id` | Auto-renew off or a cancel flag is the clearest early warning; discounts flag promotional users. |
| Member profile | `age`, `gender`, `city`, `registered_via`, `days_since_registration` | Tenure and acquisition channel. |
| Activity windows | For 7/14/30/60/90/180 days before the prediction date: `active_days_w`, `secs_w`, `unq_w`, `n100_w`, `n25_w`, `songs_w` | Recency, frequency and volume of use (analogue of STADIO 1 viewing sessions). |
| Behaviour ratios | `completion_w` (full plays / plays), `skip_w` (25% plays / plays), `secs_per_active_day_w` for w = 14, 30, 90 | Depth of engagement. |
| Trends | `trend_secs` (last 14 days vs the previous 30 days), `trend_active_days`, `trend_secs_30_vs_90` | Engagement trend was the strongest signal in Gregory (2018) and matches the STADIO 1 assumption that engagement drops before cancelling. |
| Recency | `days_since_last_log`, `has_logs` | No activity is set to 181 days plus a flag. |

Window sums and counts are set to 0 when no logs exist (genuine inactivity); ratios stay missing when their denominator is 0.

## Not built (possible extensions)
Survival-style targets (Kvamme et al., 2019), social/relational features (Ahmad et al., 2019) and different look-back windows.
