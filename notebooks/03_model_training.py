import os
import joblib
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ini cuman buat lokasi root backend
BASE_DIR = Path(__file__).resolve().parent.parent

# PAKE DATA YANG UDH DICLEAN NI BOSS
CLEANED_DATA = BASE_DIR / "data" / "processed" / "cleaned_risk_profiler.csv"

df = pd.read_csv(CLEANED_DATA)

# target n features
TARGET = "risk_level"

features = [col for col in df.columns if col not in [TARGET, "merchant_id"]]

X = df[features]
y = df[TARGET]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# KERJA KERJA AI KERJA
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# hitung evaluasi
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, predictions))

# save pkl files ke ./models/
MODELS_DIR = BASE_DIR / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(model, MODELS_DIR / "risk_model.pkl")
joblib.dump(features, MODELS_DIR / "features.pkl")

print("Model and feature list saved to backend/models!")