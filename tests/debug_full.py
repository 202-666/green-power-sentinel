"""在完整数据上验证波动率检测"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from core.data_cleaner import clean_data
from models.volatility_detector import detect_volatility_batch

SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_data")

for fname, param in [
    ("fault_grate_jam.csv", "grate_speed"),
    ("fault_bearing_overheat.csv", "bearing_vibration"),
]:
    print(f"\n=== {fname} ===")
    df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, fname), low_memory=False)
    df = clean_data(df)
    mask = df["data_quality_flag"] == "故障注入"

    for cw, bw, mults, label in [
        (10, 1440, {"yellow": 2.0, "orange": 2.5, "red": 3.5}, "cw10_bw1440_strict"),
        (30, 1440, {"yellow": 1.5, "orange": 2.0, "red": 2.5}, "cw30_bw1440_loose"),
        (30, 1440, {"yellow": 1.7, "orange": 2.2, "red": 3.0}, "cw30_bw1440_mid"),
        (20, 1440, {"yellow": 1.6, "orange": 2.0, "red": 2.8}, "cw20_bw1440"),
    ]:
        r = detect_volatility_batch(df[param], current_window=cw, baseline_window=bw, multipliers=mults)
        det = r.loc[mask, "detected"].mean() if mask.sum() > 0 else 0
        max_ratio = r.loc[mask, "ratio"].max() if mask.sum() > 0 else 0
        print(f"  {label}: 检出率={det:.1%}, 最大ratio={max_ratio:.2f}")

# 正常数据误报
print("\n=== normal_30days.csv ===")
df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, "normal_30days.csv"), low_memory=False)
df = clean_data(df)
for cw, bw, mults, label in [
    (10, 1440, {"yellow": 2.0, "orange": 2.5, "red": 3.5}, "cw10_bw1440_strict"),
    (30, 1440, {"yellow": 1.5, "orange": 2.0, "red": 2.5}, "cw30_bw1440_loose"),
    (30, 1440, {"yellow": 1.7, "orange": 2.2, "red": 3.0}, "cw30_bw1440_mid"),
]:
    fp = 0
    for p in ["grate_speed", "bearing_vibration", "furnace_temperature", "so2_concentration", "nox_concentration"]:
        r = detect_volatility_batch(df[p], current_window=cw, baseline_window=bw, multipliers=mults)
        fp += r["detected"].sum()
    print(f"  {label}: 误报={fp}/43200")
