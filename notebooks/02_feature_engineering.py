def add_featured_engineering(df):
    df = df.copy()
    
    df['volume_per_active_day'] = df['qris_volume_monthly'] / (df['qris_active_days'] + 1)
    df['pln_delay_ratio'] = df['pln_delay_days'] / (df['business_age_months'] + 1)
    df['volume_to_age_ratio'] = df['qris_volume_monthly'] / (df['business_age_months'] + 1)
    df['chronic_pln_delay'] = (df['pln_delay_days'] > 14).astype(int)

    return df