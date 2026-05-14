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

scaler = StandardScaler()

numeric_features = ['business_age_months', 'qris_volume_monthly', 'qris_active_days', 'pln_delay_days']

df[numeric_features] = scaler.fit_transform(df[numeric_features])   

df[numeric_features].describe()
        
# %%

df.describe()

# %%
