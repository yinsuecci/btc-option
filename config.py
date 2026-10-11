from dataclasses import dataclass

@dataclass
class Config:
    decision_frequency: int = 1
    threshold: float = 0.02
    min_train_weekends: int = 6
    rolling_weekends: int | None = None
    max_position: float = 0.1
    gross_limit: float = 0.4
    delta_limit: float = 0.08
    gamma_limit: float = 0.001
    vega_limit: float = 5000
    capital_limit: float = 10000
    risk_penalty: float = 0.001
    fee_rate: float = 0.0003
    slippage: float = 0.5
    max_quote_age_seconds: int = 120
