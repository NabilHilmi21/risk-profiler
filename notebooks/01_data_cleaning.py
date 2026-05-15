#%%
import importlib.util
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

# so ini buat import file feature engineering
# jadi panjang cuz penamaan file awalnya ada angka
module_path = Path("../notebooks/02_feature_engineering.py")

spec = importlib.util.spec_from_file_location(
    "feature_module",
    module_path
)

feature_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_module)

# ini buat import functionnya di dalem file feature engineering
add_featured_engineering = feature_module.add_featured_engineering

df = pd.read_csv('../data/raw/messy_risk_profiler.csv')

df.info()

#%%

missing_values = df.isnull().sum()

numeric = df.select_dtypes(include = ['number']).columns
categorical = df.select_dtypes(include = ['object']).columns

df['pln_delay_days'] = df['pln_delay_days'].fillna(0) 

df['ecommerce_rating_isna'] = df['ecommerce_rating'].isna().astype(int)
df['ecommerce_rating'] = df['ecommerce_rating'].fillna(df['ecommerce_rating'].median())

df['business_category'] = df['business_category'].astype(str).str.lower()
df['business_category'] = df['business_category'].str.strip()

df['qris_volume_monthly'] = df['qris_volume_monthly'].astype(str).str.replace(r'[^\d]','',regex=True)
df['qris_volume_monthly'] = pd.to_numeric(df['qris_volume_monthly'], errors='coerce')
df['qris_volume_monthly'] = df['qris_volume_monthly'].fillna(df['qris_volume_monthly'].median())

df['pdam_bill_avg'] = df['pdam_bill_avg'].fillna(0)
df['pdam_late_payments'] = df['pdam_late_payments'].fillna(0)

df['pdam_bill_avg'] = df['pdam_bill_avg'].abs()

duplikat = df['merchant_id'].duplicated().sum()

df['business_age_months'] = df['business_age_months'].abs()

df = df[df['qris_active_days'] <= 31]

correction = {
    'f & b': 'fnb',
    'f&b': 'fnb',
}

df['business_category'] = df['business_category'].replace(correction)

df = pd.get_dummies(df, columns = ['business_category'], drop_first = True, dtype = int)

# bagian ini dipindahin jd ke featured_engineering
df = add_featured_engineering(df)

# buat target label biar AI bisa taro hasilnya 
# 0 = low risk
# 1 = medium risk
# 2 = NONO YAH (high risk)
df["risk_level"] = 0

df.loc[(df["pln_delay_days"] > 5) | (df["ecommerce_rating"] < 4) | (df["pdam_late_payments"] > 1), "risk_level"] = 1
df.loc[(df["pln_delay_days"] > 14) | (df["ecommerce_rating"] < 3.5) | (df["pdam_late_payments"] > 3), "risk_level"] = 2

scaler = StandardScaler()

numeric_features = [
    'business_age_months', 
    'qris_volume_monthly', 
    'qris_active_days', 
    'pln_delay_days',
    'volume_per_active_day',
    'pln_delay_ratio',
    'volume_to_age_ratio',
    'pdam_bill_avg',
    'pdam_late_payments'
    ]

df[numeric_features] = scaler.fit_transform(df[numeric_features])   

df.info()
df[numeric_features].describe()
df.head()

df.to_csv('../data/processed/cleaned_risk_profiler.csv', index=False)
# %%

#ini data splitting ok

from sklearn.model_selection import train_test_split

X = df.drop(columns = ['merchant_id', 'risk_level'], errors = 'ignore')
y = df['risk_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

print(len(df))
print(len(X_train))
print(len(X_test))

X_train.to_csv('../data/processed/X_train.csv', index=False)
X_test.to_csv('../data/processed/X_test.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)
# %%
