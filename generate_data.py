"""
Generate synthetic customer-churn data and save it as CSV.

Usage:
    python generate_data.py                       # 5000 rows -> data/customers.csv
    python generate_data.py --rows 20000 --output data/big.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_DATA_PATH


def generate_demo_data(
    n_rows: int = 5000,
    output_path: str | Path | None = DEFAULT_DATA_PATH,
    seed: int = 42,
    missing_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Create a synthetic churn dataset with realistic relationships:
    short tenure, month-to-month contracts, high bills, many support tickets
    and late payments all increase churn probability.

    Args:
        n_rows: number of customers to generate.
        output_path: where to save the CSV. Pass None to skip saving.
        seed: random seed for reproducibility.
        missing_rate: fraction of values blanked out in a few columns, so the
            training pipeline's imputation is exercised like it would be on
            real data.

    Returns:
        The generated DataFrame.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, n_rows)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"], n_rows, p=[0.55, 0.25, 0.20]
    )
    internet = rng.choice(["fiber", "dsl", "none"], n_rows, p=[0.45, 0.40, 0.15])
    payment = rng.choice(
        ["credit_card", "bank_transfer", "electronic_check", "mailed_check"],
        n_rows,
        p=[0.30, 0.25, 0.30, 0.15],
    )
    tech_support = rng.choice(["yes", "no"], n_rows, p=[0.35, 0.65])
    paperless = rng.choice(["yes", "no"], n_rows, p=[0.6, 0.4])

    base_price = np.select(
        [internet == "fiber", internet == "dsl"], [80, 55], default=25
    )
    monthly = np.round(base_price + rng.normal(0, 8, n_rows), 2).clip(15, None)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.05, n_rows), 2)
    tickets = rng.poisson(1.2, n_rows)
    usage = np.round(rng.gamma(4, 25, n_rows), 1)
    late = rng.poisson(0.6, n_rows)

    # Hidden "true" churn logic (log-odds)
    z = (
        -0.8
        - 0.045 * tenure
        + 1.3 * (contract == "month-to-month")
        - 0.9 * (contract == "two-year")
        + 0.015 * (monthly - 60)
        + 0.30 * tickets
        + 0.35 * late
        + 0.5 * (internet == "fiber")
        - 0.5 * (tech_support == "yes")
        + 0.4 * (payment == "electronic_check")
        - 0.004 * (usage - 100)
        + rng.normal(0, 0.5, n_rows)
    )
    prob = 1 / (1 + np.exp(-z))
    churn = (rng.uniform(size=n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(1, n_rows + 1)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_support_tickets": tickets,
            "avg_monthly_usage_gb": usage,
            "late_payments": late,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            "has_tech_support": tech_support,
            "paperless_billing": paperless,
            "churn": churn,
        }
    )

    # Inject some missing values to mimic messy real-world data
    if missing_rate > 0:
        for col in ["total_charges", "avg_monthly_usage_gb", "payment_method"]:
            mask = rng.uniform(size=n_rows) < missing_rate
            df.loc[mask, col] = np.nan

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df):,} rows -> {output_path}  (churn rate: {df['churn'].mean():.1%})")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demo churn data as CSV")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--output", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_demo_data(args.rows, args.output, args.seed)
