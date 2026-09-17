from typing import Any

from src.ml_model.service import CreditRiskPredictor

if __name__ == "__main__":
    # --- Isolated smoke test --------------------------------------------
    # Raw client record shaped like the feature contract used by the
    # preprocessing pipeline (see notebooks/01_data_preparation.ipynb).
    sample_client_features: dict[str, Any] = {
        "loan_amnt": 15000.0,
        "term": "36 months",
        "int_rate": 13.56,
        "installment": 456.78,
        "emp_length": 5,
        "home_ownership": "MORTGAGE",
        "annual_inc": 72000.0,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": 18.4,
        "fico_range_low": 690.0,
        "open_acc": 8.0,
        "pub_rec": 0.0,
        "revol_util": 42.3,
        "mort_acc": 1.0,
    }

    service = CreditRiskPredictor()
    default_probability = service.predict_proba(sample_client_features)
    print(f"Predicted probability of default: {default_probability:.4f}")
