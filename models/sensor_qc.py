import pandas as pd

def detect_stuck(series: pd.Series, window: int = 10) -> bool:
    return False

def detect_drift(series: pd.Series, window: int = 10, std_factor: float = 3.0) -> bool:
    return False

def evaluate_sensor_health(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    return df