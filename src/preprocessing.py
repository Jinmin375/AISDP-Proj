"""
Data loading, cleaning and feature engineering for the predictive-maintenance pipeline.

Every step here comes from a finding in eda.ipynb:
  * vibration values below zero are a sign error  -> take the absolute value   (EDA 2.3)
  * temperature is missing completely at random    -> fill with the median     (EDA 2.2)
  * temperature, vibration and remaining life act
    as thresholds, not straight lines              -> add yes/no flag features (EDA 5, 9)
  * some derived columns leak the target           -> drop them                (EDA 6)
  * machines already in failure look normal to
    every sensor                                   -> split them off, use a rule (EDA 6.4)

Typical use inside the pipeline:

    config = load_config("config.yaml")
    df = load_data(config["data"]["raw_path"])
    df = clean_data(df, config)
    model_df, failed_df = split_failed_machines(df, config)

    X, y = model_df.drop(columns=target), model_df[target]
    # ... train/test split ...
    prep = FeatureEngineer(config)
    X_train = prep.fit_transform(X_train)   # learns the median from training data only
    X_test  = prep.transform(X_test)        # reuses the training median (no leakage)

Run this file directly to write the cleaned CSV:

    python src/preprocessing.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin


def load_config(path: str | Path) -> dict:
    """Read the YAML configuration file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw sensor readings CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Fix data errors and drop leaky / useless columns.

    These steps do not learn anything from the data, so they are safe to run
    on the full dataset before the train/test split.
    """
    df = df.copy()

    # Negative vibration is a sign error, not a real reading (EDA 2.3).
    df["vibration"] = df["vibration"].abs()

    # Remove columns that leak the target or carry no signal (EDA 6.5, 7).
    to_drop = [c for c in config["preprocessing"]["drop_columns"] if c in df.columns]
    df = df.drop(columns=to_drop)

    return df


def split_failed_machines(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate machines that have already failed from the rest.

    Returns (model_df, failed_df):
      * model_df  - idle/running machines; the ML model predicts these.
                    machine_status is removed because it is no longer informative.
      * failed_df - machines already in failure; they always need maintenance,
                    so a simple rule handles them instead of the model.
    """
    failed_value = config["preprocessing"]["failed_status_value"]
    is_failed = df["machine_status"] == failed_value

    model_df = df.loc[~is_failed].drop(columns="machine_status").reset_index(drop=True)
    failed_df = df.loc[is_failed].reset_index(drop=True)
    return model_df, failed_df


def predict_failed_rule(failed_df: pd.DataFrame) -> pd.Series:
    """Rule for already-failed machines: maintenance is always required."""
    return pd.Series(1, index=failed_df.index, name="maintenance_required")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Fill missing temperature and add threshold flag features.

    Works like any scikit-learn transformer, so it can sit inside a Pipeline:
      * fit()       learns the fill value for temperature from the TRAINING data only
      * transform() applies that value and builds the flag columns
    """

    def __init__(self, config: dict):
        self.config = config

    def fit(self, X: pd.DataFrame, y=None):
        strategy = self.config["preprocessing"]["impute_strategy"]
        if strategy == "median":
            self.temperature_fill_ = X["temperature"].median()
        elif strategy == "mean":
            self.temperature_fill_ = X["temperature"].mean()
        else:
            raise ValueError(f"Unknown impute_strategy: {strategy!r} (use 'median' or 'mean')")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "temperature_fill_"):
            raise RuntimeError("FeatureEngineer must be fitted before transform().")

        t = self.config["preprocessing"]["thresholds"]
        X = X.copy()

        # Flags are built BEFORE filling, so a missing temperature never counts as "too hot".
        X["temp_missing"] = X["temperature"].isna().astype(int)
        X["temp_over_90"] = (X["temperature"] > t["temperature_high"]).astype(int)
        X["vibration_over_80"] = (X["vibration"] > t["vibration_high"]).astype(int)
        X["rul_under_20"] = (X["predicted_remaining_life"] < t["rul_critical"]).astype(int)
        X["rul_under_50"] = (X["predicted_remaining_life"] < t["rul_warning"]).astype(int)
        X["sensor_alarm"] = (
            (X["temp_over_90"] | X["vibration_over_80"]) & X["rul_under_50"]
        ).astype(int)

        X["temperature"] = X["temperature"].fillna(self.temperature_fill_)
        return X


def main() -> None:
    """Clean the raw CSV and save it, leaving temperature unfilled for the pipeline to impute."""
    parser = argparse.ArgumentParser(description="Clean the raw machine readings CSV.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    raw = load_data(config["data"]["raw_path"])
    clean = clean_data(raw, config)

    out_path = Path(config["data"]["clean_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    print(f"Saved {len(clean):,} rows x {clean.shape[1]} columns to {out_path}")


if __name__ == "__main__":
    main()
