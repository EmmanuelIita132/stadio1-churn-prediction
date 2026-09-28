# Model 2: XGBoost (Part B)

**Code:** `src/pipeline/model2.py`  |  **Shared helpers:** `src/pipeline/common.py`
**Inputs:** `data/processed/features_train.parquet`, `features_test.parquet`

## How to run
```bash
python src/pipeline/model2.py train      # depth grid + early stopping; saves models/model2_xgboost.joblib,
                                         # experiments/setup/model2_config.json, experiments/results/model2_feature_importance.csv
python src/pipeline/model2.py predict    # scores the test cohort -> experiments/results/model2_predictions.parquet
python src/pipeline/model2.py predict --input other_features.parquet --output out.parquet
```

## Technique
Gradient-boosted decision trees (XGBoost, Chen and Guestrin, 2016). Gregory (2018) won the KKBox challenge with XGBoost and Ahmad et al. (2019) found it the best tree ensemble.

## Model-specific preprocessing
No scaling or imputation: XGBoost handles missing values natively. Categorical columns are ordinal-encoded (unseen levels get -1).

## Hyper-parameters
| Parameter | Value |
|---|---|
| `learning_rate` | 0.05 |
| `n_estimators` | up to 2000, stopped early (50 rounds without improvement on validation log loss) |
| `max_depth` | grid {4, 6, 8}; chosen value stored in `experiments/setup/model2_config.json` |
| `min_child_weight` | 5 |
| `subsample` / `colsample_bytree` | 0.8 / 0.8 |
| `reg_lambda` | 1.0 |
| `tree_method` | `hist` |
| Class re-weighting | none (Ahmad et al., 2019: boosted trees did best without re-balancing) |
| Selection | lowest validation log loss (the KKBox competition metric) |
| Threshold | F1-optimal on the validation slice |
