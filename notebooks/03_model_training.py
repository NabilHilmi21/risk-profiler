import os
import random
import joblib
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# =========================
# ROOT BACKEND DIRECTORY
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# CREATE DUMMY DATA
# =========================

categories = [
    "Food",
    "Fashion",
    "Technology",
    "Beauty",
    "Automotive"
]

data = []

for i in range(1000):
    merchant_id = i + 1
    business_category = random.choice(categories)
    business_age_month = random.randint(1, 120)
    purchase_volume_monthly = random.randint(500, 50000)
    purchase_active_days = random.randint(1, 30)
    ecommerce_rating = round(random.uniform(2.5, 5.0), 1)
    pln_delay_days = random.randint(0, 30)

    if pln_delay_days > 15 or ecommerce_rating < 3.5:
        risk = 2
    elif pln_delay_days > 5:
        risk = 1
    else:
        risk = 0

    data.append({
        "merchant_id": merchant_id,
        "business_category": business_category,
        "business_age_month": business_age_month,
        "purchase_volume_monthly": purchase_volume_monthly,
        "purchase_active_days": purchase_active_days,
        "ecommerce_rating": ecommerce_rating,
        "pln_delay_days": pln_delay_days,
        "risk": risk
    })

# =========================
# DATAFRAME
# =========================

df = pd.DataFrame(data)

# =========================
# ENCODE CATEGORY
# =========================

encoder = LabelEncoder()

df["business_category"] = encoder.fit_transform(
    df["business_category"]
)

# =========================
# FEATURES & TARGET
# =========================

X = df.drop(columns=["risk"])
y = df["risk"]

# =========================
# SCALING
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = RandomForestClassifier()

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")

# =========================
# SAVE PKL TO backend/models
# =========================

MODELS_DIR = BASE_DIR / "models"

os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(model, MODELS_DIR / "risk_model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(encoder, MODELS_DIR / "category_encoder.pkl")

print("PKL files saved to backend/models!")