"""诊断烟气超标关联检测"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import yaml
from core.data_cleaner import clean_data
from models.trend_detector import detect_trend_multi_params

SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_data")

df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, "fault_emission_exceed.csv"), low_memory=False)
df = clean_data(df)
mask = df["data_quality_flag"] == "故障注入"

skip_cols = {"timestamp", "device_id", "device_name", "device_type", "data_quality_flag", "qc_note"}
param_cols = [c for c in df.columns if c not in skip_cols]

trend_results = detect_trend_multi_params(df, param_cols)

print("SO2故障区间趋势:")
so2_trend = trend_results["so2_concentration"]
print(so2_trend.loc[mask, ["slope_10", "slope_30", "slope_60", "detected_10", "detected_30", "detected_60"]].describe())

print("\nNOX故障区间趋势:")
nox_trend = trend_results["nox_concentration"]
print(nox_trend.loc[mask, ["slope_10", "slope_30", "slope_60", "detected_10", "detected_30", "detected_60"]].describe())

print("\nOxygen故障区间:")
print(df.loc[mask, "oxygen_content"].describe())

# 查看R2规则
with open(os.path.join(PROJECT_ROOT, "config", "rules.yaml"), "r", encoding="utf-8") as f:
    rules = yaml.safe_load(f).get("rules", [])
r2 = [r for r in rules if r["rule_id"] == "R2"][0]
print(f"\nR2规则: {r2}")

# 手动检查故障点
matched = 0
for i in df[mask].index:
    so2_slope = trend_results["so2_concentration"].loc[i, "slope_30"]
    nox_slope = trend_results["nox_concentration"].loc[i, "slope_30"]
    oxygen = df.loc[i, "oxygen_content"]
    if so2_slope > 0.5 and nox_slope > 0.5 and oxygen < 7.0:
        matched += 1

print(f"\n手动统计满足R2条件的点数: {matched}/{mask.sum()}")

# 查看前10个故障点
for i in df[mask].index[:10]:
    so2_slope = trend_results["so2_concentration"].loc[i, "slope_30"]
    nox_slope = trend_results["nox_concentration"].loc[i, "slope_30"]
    oxygen = df.loc[i, "oxygen_content"]
    print(f"i={i}: so2_slope={so2_slope:.3f}, nox_slope={nox_slope:.3f}, oxygen={oxygen:.2f}")
