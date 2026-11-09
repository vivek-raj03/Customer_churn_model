# Customer Churn Project

Train a churn model and serve it with FastAPI. Works on generated demo data now,
and on your real CSV later with minimal changes.

## Structure
| File | Purpose |
|---|---|
| `config.py` | Paths + column definitions (the only file to edit for real data) |
| `generate_data.py` | `generate_demo_data()` -> writes demo CSV |
| `train.py` | Trains, evaluates, saves `artifacts/churn_model.joblib` + `metadata.json` |
| `app.py` | FastAPI service (`/predict`, `/predict/batch`, `/model-info`, `/health`) |

## Quick start
```bash
pip install -r requirements.txt
python generate_data.py            # creates data/customers.csv
python train.py                    # trains + saves model
uvicorn app:app --reload           # http://127.0.0.1:8000/docs
```

## Example request
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "tenure_months": 5, "monthly_charges": 85.5, "total_charges": 427.5,
  "num_support_tickets": 3, "avg_monthly_usage_gb": 60, "late_payments": 2,
  "contract_type": "month-to-month", "payment_method": "electronic_check",
  "internet_service": "fiber", "has_tech_support": "no", "paperless_billing": "yes"}'
```

## Switching to real data
1. Put your CSV somewhere (one row per customer, binary 0/1 churn column).
2. Edit `config.py`: `TARGET`, `ID_COLUMN`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`.
3. `python train.py --data path/to/your.csv`
4. Restart the API. The request schema updates automatically from `config.py`.

Training options: `--model {logistic_regression,random_forest,gradient_boosting}`,
`--threshold 0.4` (lower = catch more churners, more false alarms), `--test-size`, `--seed`.

## Real-world notes
- Avoid **target leakage**: don't include columns known only after a customer leaves
  (e.g. cancellation date, exit survey).
- With real data, prefer a **time-based split** (train on older, test on newer customers).
- Tune the threshold using business cost (cost of a retention offer vs. value of a saved customer).
- Retrain regularly and monitor drift; `metadata.json` records version and metrics.
