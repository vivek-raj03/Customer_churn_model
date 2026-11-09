"""
Train a customer-churn model and save it for serving.

Usage:
    python train.py                                   # demo data (auto-generated if missing)
    python train.py --data data/my_real_data.csv      # your own CSV
    python train.py --model random_forest --threshold 0.4

Outputs (in artifacts/):
    churn_model.joblib  - full sklearn Pipeline (preprocessing + classifier)
    metadata.json       - features, metrics, threshold, version, timestamp
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from generate_data import generate_demo_data


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load CSV; if it's the default demo path and missing, generate it."""
    path = Path(path)
    if not path.exists():
        if path == config.DEFAULT_DATA_PATH:
            print("Demo data not found - generating it...")
            generate_demo_data(output_path=path)
        else:
            raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> None:
    """Fail early with a clear message if the CSV doesn't match config.py."""
    required = config.FEATURES + [config.TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing columns defined in config.py: {missing}. "
            "Update config.py or your dataset."
        )
    if df[config.TARGET].isna().any():
        raise ValueError("Target column contains missing values.")
    if not set(df[config.TARGET].unique()) <= {0, 1}:
        raise ValueError("Target must be binary (0/1).")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_pipeline(model_name: str, seed: int) -> Pipeline:
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
        ]
    )

    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }
    if model_name not in classifiers:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(classifiers)}")

    return Pipeline([("preprocess", preprocessor), ("model", classifiers[model_name])])


def evaluate(pipeline: Pipeline, X_test, y_test, threshold: float) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    print("\n=== Test-set evaluation (threshold = {:.2f}) ===".format(threshold))
    print(classification_report(y_test, pred, target_names=["stayed", "churned"]))
    print(f"ROC-AUC: {metrics['roc_auc']}   PR-AUC: {metrics['pr_auc']}")
    return metrics


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATA_PATH)
    parser.add_argument(
        "--model", default="gradient_boosting",
        choices=["logistic_regression", "random_forest", "gradient_boosting"],
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability cut-off to label a customer as churn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--metadata-out", type=Path, default=config.METADATA_PATH)
    args = parser.parse_args()

    df = load_data(args.data)
    validate_data(df)
    print(f"Loaded {len(df):,} rows | churn rate: {df[config.TARGET].mean():.1%}")

    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    pipeline = build_pipeline(args.model, args.seed)
    print(f"Training {args.model} on {len(X_train):,} rows...")
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test, args.threshold)

    # Refit on ALL data is optional; we keep the train-only model so the
    # reported metrics match the saved artifact exactly.
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)

    metadata = {
        "model_name": args.model,
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "data_source": str(args.data),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "threshold": args.threshold,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "metrics": metrics,
    }
    args.metadata_out.write_text(json.dumps(metadata, indent=2))
    print(f"\nModel saved    -> {args.model_out}")
    print(f"Metadata saved -> {args.metadata_out}")


if __name__ == "__main__":
    main()
