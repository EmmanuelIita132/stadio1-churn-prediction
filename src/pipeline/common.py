"""Shared data loading and preprocessing-for-modelling helpers used by model1.py and model2.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

NON_FEATURES = ["msno", "is_churn", "ref_date"]


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Treat codes as categories (strings, NaN kept as np.nan so sklearn imputers work)."""
    for c in C.CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    return df


def load_cohort(name: str) -> pd.DataFrame:
    df = pd.read_parquet(C.PROCESSED / f"features_{name}.parquet")
    return cast_categoricals(df)


def feature_columns(df: pd.DataFrame):
    cats = [c for c in C.CATEGORICAL if c in df.columns]
    nums = [c for c in df.columns if c not in NON_FEATURES + cats]
    return nums, cats


def split_train_valid(df: pd.DataFrame):
    """Random stratified validation slice of the TRAIN cohort (used for tuning / early stopping only)."""
    tr, va = train_test_split(df, test_size=C.VALID_FRACTION, stratify=df.is_churn, random_state=C.SEED)
    return tr, va


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def logistic_preprocessor(nums, cats):
    """Model 1: median impute (+missing indicators), signed log1p, standardise; impute + one-hot categoricals."""
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("log", FunctionTransformer(_signed_log1p)),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200,
                                             sparse_output=False))])
    return ColumnTransformer([("num", num, nums), ("cat", cat, cats)])


def tree_preprocessor(nums, cats):
    """Model 2: no scaling / imputation needed (XGBoost handles NaN); ordinal-encode categoricals."""
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    return ColumnTransformer([("num", "passthrough", nums), ("cat", cat, cats)])


def xy(df, nums, cats):
    return df[nums + cats], df["is_churn"].astype(int).to_numpy()
