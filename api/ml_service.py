import joblib
import pandas as pd

from pathlib import Path
from api.schemas import RiskInput

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "risk_model.pkl")
features = joblib.load(BASE_DIR / "models" / "features.pkl")

RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}


def predict_risk(data: RiskInput):
    category = data.business_category.lower().strip()

    correction = {
        "f & b": "fnb",
        "f&b": "fnb",
    }

    category = correction.get(category, category)

    df = pd.DataFrame([
        {
            "business_age_months": data.business_age_months,
            "qris_volume_monthly": data.qris_volume_monthly,
            "qris_active_days": data.qris_active_days,
            "pln_delay_days": data.pln_delay_days,
            "ecommerce_rating": data.ecommerce_rating,
            "business_category": category,
        }
    ])

    df = pd.get_dummies(
        df,
        columns=["business_category"],
        drop_first=True,
        dtype=int
    )

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df).max()

    return {
        "risk_level": RISK_LABELS.get(int(prediction), "Unknown"),
        "probability": round(float(probability), 4)
    }