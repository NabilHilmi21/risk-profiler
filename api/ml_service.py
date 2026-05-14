import joblib
import pandas as pd

from pathlib import Path
from api.schemas import RiskInput

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "risk_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
encoder = joblib.load(BASE_DIR / "models" / "category_encoder.pkl")

RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}


def predict_risk(data: RiskInput):

    df = pd.DataFrame([
        {
            "merchant_id": data.merchant_id,
            "business_category": data.business_category,
            "business_age_month": data.business_age_month,
            "purchase_volume_monthly": data.purchase_volume_monthly,
            "purchase_active_days": data.purchase_active_days,
            "ecommerce_rating": data.ecommerce_rating,
            "pln_delay_days": data.pln_delay_days,
        }
    ])

    # Encode category string -> number
    df["business_category"] = encoder.transform(
        df["business_category"]
    )

    # Scale
    scaled_data = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data).max()

    return {
        "risk_level": RISK_LABELS.get(int(prediction), "Unknown"),
        "probability": round(float(probability), 4)
    }