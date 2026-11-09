"""
Central configuration for the churn project.

To use REAL data, this is the main file you edit:
  1. Set TARGET / ID_COLUMN to match your dataset.
  2. List your numeric and categorical columns below.
Everything else (training, validation, API schema) adapts automatically.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

DEFAULT_DATA_PATH = DATA_DIR / "customers.csv"
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

# ---- Dataset schema -------------------------------------------------------
ID_COLUMN = "customer_id"   # identifier, never used as a feature
TARGET = "churn"            # 1 = churned, 0 = stayed

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "avg_monthly_usage_gb",
    "late_payments",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
    "has_tech_support",
    "paperless_billing",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---- Serving --------------------------------------------------------------
# Probability bands used by the API to label risk.
RISK_BANDS = [(0.7, "high"), (0.4, "medium"), (0.0, "low")]
