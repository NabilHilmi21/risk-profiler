from pydantic import BaseModel

class RiskInput(BaseModel):
    merchant_id: int
    business_category: str
    business_age_month: int
    purchase_volume_monthly: float
    purchase_active_days: int
    ecommerce_rating: float
    pln_delay_days: int


class PredictionResponse(BaseModel):
    risk_level: str
    probability: float