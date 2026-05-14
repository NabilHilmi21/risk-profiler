#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
df['qris_volume_monthly'] = df['qris_volume_monthly'].fillna(df['qris_volume_monthly'].median)

duplikat = df['merchant_id'].duplicated().sum()

df['business_age_months'] = df['business_age_months'].abs()

df = df[df['qris_active_days'] <= 31]

correction = {
    'f & b': 'fnb',
    'f&b': 'fnb',
}

df['business_category'] = df['business_category'].replace(correction)

df = pd.get_dummies(df, columns = ['business_category'], drop_first = True, dtype = int)

df['volume_per_active_day'] = df['qris_volume_monthly'] / (df['qris_active_days'] + 1)
df['pln_delay_ratio'] = df['pln_delay_days'] / (df['business_age_months'] + 1)
df['volume_to_age_ratio'] = df['qris_volume_monthly'] / (df['business_age_months'] + 1)
df['chronic_pln_delay'] = (df['pln_delay_days'] > 14).astype(int)

scaler = StandardScaler()

numeric_features = [
    'business_age_months', 
    'qris_volume_monthly', 
    'qris_active_days', 
    'pln_delay_days',
    'volume_per_active_day',
    'pln_delay_ratio',
    'volume_to_age_ratio'
    ]

df[numeric_features] = scaler.fit_transform(df[numeric_features])   

df.info()
df[numeric_features].describe()
df.head()
    
# %%
