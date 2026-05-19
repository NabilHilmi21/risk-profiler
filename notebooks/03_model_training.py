#%%
from xml.parsers.expat import model

import pandas as pd
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

# =========================================================
# LOAD DATA
# =========================================================

X_train = pd.read_csv('../data/processed/X_train.csv')
y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
X_test  = pd.read_csv('../data/processed/X_test.csv')
y_test  = pd.read_csv('../data/processed/y_test.csv').values.ravel()

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")


# =========================================================
# DEFINE MODELS
# Nama dict dibedakan dari loop variable
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators     = 500,
    random_state     = 42,
    class_weight     = 'balanced',
    max_depth        = 8,
    min_samples_leaf = 5
)

# =========================================================
# TRAIN & EVALUATE
# =========================================================

rf_model.fit(X_train, y_train)
  
y_pred = rf_model.predict(X_test)

# Metrics
accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
recall    = recall_score(y_test, y_pred, average='macro')
cm        = confusion_matrix(y_test, y_pred)

print(f'accuracy :  {accuracy:.4f}')
print(f'f1_macro :  {f1:.4f}')
print(f'precision : {precision:.4f}')   
print(f'recall    : {recall:.4f}')

print(classification_report(
    y_test, y_pred, 
    target_names = [
        'Low Risk (0)', 
        'Medium Risk (1)', 
        'High Risk (2)']
))

print(f'conf_matrix:\n{cm}')

joblib.dump(rf_model, '../models/random_forest_model.joblib')

# =========================================================
# SHAP ANALYSIS — hanya untuk tree-based model
# =========================================================

explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer(X_test)

# Summary plot untuk High Risk (class 2)
shap_high_risk = shap_values[:, :, 2]

plt.figure()
shap.summary_plot(
    shap_high_risk,
    X_test,
    title='SHAP — Feature Importance for High Risk',
    show=False
)

plt.tight_layout()
plt.show()
# %%