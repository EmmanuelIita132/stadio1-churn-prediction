# SS3 Part B: applying the SS2 churn method to the STADIO 1 client data

**Notebook:** `SS3_PartB.ipynb` (five marked components: 1 Dataset loading, 2 Dataset preprocessing, 3 Model training, 4 Model validation, 5 Results visualisation).

## How to run
1. `pip install -r requirements.txt`
2. Put the client CSVs in `data/raw/stadioone/` (or set `STADIO_DATA_DIR`). Do not commit raw data.
3. `jupyter nbconvert --to notebook --execute SS3_PartB.ipynb --inplace` (about 30 s). Figures and metrics go to `outputs/` (set `STADIO_OUT_DIR` to change).

## What was done
- **Method:** XGBoost (SS2 final model) with logistic-regression and "no viewing in 30 days" baselines.
- **Data:** tables 1-7, 9a, 9b, 10 plus the race calendar; table 8 unused (sample only). Churn = Cancelled / Payment failed.
- **Task:** monthly score per running subscription; label = cancels in the next 30 days (~3.8% positives).
- **Leakage controls:** features strictly before the observation date; `status`, `end_datetime`, `cancellation_*`, `auto_renew`, `notification_setting` excluded; retention-offer fans excluded; identity links below 0.9 confidence ignored.
- **Split:** by time. Train Jan-Dec 2024, validation Jan-Jun 2025, test Jul 2025 - Jul 2026.

## Headline results (unseen test months)
| Model | ROC-AUC | PR-AUC | Top-10% lift |
|---|---|---|---|
| XGBoost | 0.75 | 0.21 | 3.8x |
| Logistic regression | 0.72 | 0.17 | 3.7x |
| Rule: no viewing 30 days | 0.44 | 0.03 | 0.4x |

The top 10% riskiest subscriber-months contain 38% of churners. The rule is worse than chance here: churners are not quieter than stayers in the last 30 days.

## Video
Part C video link: _add here_
