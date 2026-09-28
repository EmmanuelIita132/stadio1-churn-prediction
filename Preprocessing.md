# Data Preprocessing (Part B)

**Dataset:** WSDM KKBox's Churn Prediction Challenge (Kaggle; KKBOX, 2018). See the Part A literature review.
**Code:** `src/pipeline/preprocessing.py`  |  **Output:** `data/interim/*.parquet`

## How to run
1. Accept the competition rules on Kaggle and download `train.csv`, `train_v2.csv`, `members_v3.csv`, `transactions.csv`, `transactions_v2.csv` and `user_logs.csv` into `data/raw/` (or point `RAW_DIR` at another folder). `user_logs.csv` is about 30 GB; it is streamed in chunks and never loaded whole.
2. `pip install -r requirements.txt`
3. `python src/pipeline/preprocessing.py`
   Optional: `SAMPLE_USERS=200000` (default) samples users per cohort to keep memory low; `SAMPLE_USERS=0` uses every user.

## Techniques used
| Step | Technique | Reason |
|---|---|---|
| Time-based cohorts | Train = users expiring Feb 2017 (`train.csv`, prediction date 2017-02-01); test = users expiring Mar 2017 (`train_v2.csv`, prediction date 2017-03-01) | Mirrors deployment (train on the past, score the future); follows the time-based split of Gregory (2018). |
| No user overlap | Users present in the test cohort are removed from training | Prevents the same subscriber leaking across the split. |
| Sampling | Random sample of up to 200,000 users per cohort (seed 42) | Keeps the 30 GB log file tractable; the sample keeps the natural churn rate. |
| Member cleaning | Ages outside 10-90 set to missing; missing gender set to `unknown`; `registration_init_time` parsed to dates | Age contains placeholder values (0, negative, over 100). |
| Transaction cleaning | Integer dates converted to dates; duplicates dropped; rows with expiry before purchase dropped | Basic validity checks. |
| Leakage control | Transactions and logs are kept only if dated **before** the cohort's prediction date; logs only within 180 days before it | Ensures features never contain information from the label period. |
| Log outliers | `total_secs` clipped to [0, 86,400] seconds | The raw column has negative and impossible values. |
| Missing values | Left as missing at this stage; handled per model (see `Model1.md`, `Model2.md`) | Missingness is informative (e.g. no logs). |

## Outputs
`labels_train/test.parquet`, `members.parquet`, `transactions.parquet`, `logs.parquet` in `data/interim/`.

Note: only `user_logs.csv` (data up to 28 Feb 2017) is needed, because both prediction dates are on or before 1 March 2017. If it is missing the script falls back to `user_logs_v2.csv` and prints a warning; that fallback leaves the test-cohort log features empty and should not be used for reported results.
