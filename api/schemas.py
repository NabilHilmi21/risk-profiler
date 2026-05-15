from pydantic import BaseModel

# JANGAN MASIH BANYAK BANGET YANG KURANG JSON RESPONSENYA, LIAT PROPOSAL OR DESIGN OR WHATEVER
# BUAT SEMENTARA GINI DULU AJA YA 
# LAST UPDATED: 15/05/2026

class RiskInput(BaseModel):
    merchant_id: int
    business_category: str
    business_age_months: int
    qris_volume_monthly: float
    qris_active_days: int
    ecommerce_rating: float
    pln_delay_days: int


class PredictionResponse(BaseModel):
    risk_level: str
    probability: float