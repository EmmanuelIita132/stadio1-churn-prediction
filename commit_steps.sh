#!/usr/bin/env bash
# Creates a sensible multi-commit history (the brief penalises a single commit). Run once from the repo root.
# Run the pipeline first so the result files exist, then: bash commit_steps.sh
set -e
[ -d .git ] || git init
c() { git add "${@:2}" 2>/dev/null && git commit -q -m "$1" && echo "committed: $1" || echo "skipped: $1"; }
c "Add repository structure, README and data request" README.md docs/data_request.pdf .gitignore
c "Add Part A literature review (three publications and KKBox dataset)" docs/literature_review.pdf
c "Add requirements.txt" requirements.txt
c "Add shared config" src/__init__.py src/config.py
c "Add preprocessing script and documentation" src/pipeline/__init__.py src/pipeline/preprocessing.py Preprocessing.md
c "Add feature engineering script and documentation" src/pipeline/feature_engineering.py FeatureEngineering.md
c "Add shared modelling helpers" src/pipeline/common.py
c "Add Model 1 (logistic regression) script and documentation" src/pipeline/model1.py Model1.md
c "Add Model 2 (XGBoost) script and documentation" src/pipeline/model2.py Model2.md
c "Add metrics and statistical tests" src/stats/__init__.py src/stats/metrics.py
c "Add plotting helpers" src/visualisation
c "Add evaluation script" src/stats/evaluate.py
c "Add synthetic data generator for smoke tests" src/pipeline/make_synthetic_data.py
c "Add experiment setup (hyper-parameter search logs)" experiments/setup
c "Add Model 1 performance results" Model1Performance.md experiments/results/model1_metrics.json experiments/results/figures/model1_*
c "Add Model 2 performance results" Model2Performance.md experiments/results/model2_metrics.json experiments/results/model2_feature_importance.csv experiments/results/figures/model2_*
c "Add model comparison" Comparison.md experiments/results/comparison.json experiments/results/figures/compare_*
c "Add trained models" models
c "Add report builder" src/stats/build_report.py
c "Add client recommendations report" reports
git add -A && git commit -q -m "Final tidy-up" || true
