# Model 1: Logistic Regression (baseline) (Part B)

**Code:** `src/pipeline/model1.py`  |  **Shared helpers:** `src/pipeline/common.py`
**Inputs:** `data/processed/features_train.parquet`, `features_test.parquet`

## How to run
```bash
python src/pipeline/model1.py train      # tunes C, saves models/model1_logreg.joblib + experiments/setup/model1_config.json
python src/pipeline/model1.py predict    # scores the test cohort -> experiments/results/model1_predictions.parquet
python src/pipeline/model1.py predict --input other_features.parquet --output out.parquet   # score new data
```

## Technique
L2-regularised logistic regression (scikit-learn, `lbfgs`, `max_iter=2000`). It is the interpretable baseline planned in Assignment 1.

## Model-specific preprocessing (inside one scikit-learn `Pipeline`, fitted on training data only)
- Numeric: median imputation with missing-value indicators, signed `log1p` (many counts and durations are heavily skewed), standardisation.
- Categorical (`city`, `registered_via`, `gender`, `last_payment_method_id`): fill missing with `missing`, one-hot encode, rare levels (<200 users) grouped as "infrequent"; unseen levels are ignored at prediction time.

## Hyper-parameters
| Parameter | Value |
|---|---|
| Regularisation strength `C` | grid {0.001, 0.01, 0.1, 1, 10}; the chosen value is stored in `experiments/setup/model1_config.json` and shown in `Model1Performance.md` |
| Penalty / solver | L2 / lbfgs |
| Class weights | none (no re-balancing, so probabilities stay calibrated) |
| Selection | lowest log loss on a stratified random 20% validation slice of the training cohort |
| Threshold | F1-optimal on the validation slice; used for the threshold-based metrics |
