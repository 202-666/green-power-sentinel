"""快速诊断波动率检测效果"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from core.data_cleaner import clean_data
from models.volatility_detector import detect_volatility_batch

SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_data")

# 1. 炉排卡滞诊断
print("="*60)
print("炉排卡滞诊断")
print("="*60)
df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, "fault_grate_jam.csv"), low_memory=False)
df = clean_data(df)
mask = df["data_quality_flag"] == "故障注入"
print(f"总行数: {len(df)}, 故障行数: {mask.sum()}")

fault_idx = df[mask].index
fault_start, fault_end = fault_idx[0], fault_idx[-1]
window_start = max(0, fault_start - 500)
window_end = min(len(df), fault_end + 500)
sub_df = df.iloc[window_start:window_end].copy()
sub_mask = sub_df["data_quality_flag"] == "故障注入"
print(f"诊断窗口: [{window_start}, {window_end}], 故障行数: {sub_mask.sum()}")

result = detect_volatility_batch(sub_df["grate_speed"], current_window=10, baseline_window=200)
print(f"grate_speed 故障区间平均ratio: {result.loc[sub_mask, 'ratio'].mean():.3f}")
print(f"grate_speed 故障区间最大ratio: {result.loc[sub_mask, 'ratio'].max():.3f}")
print(f"grate_speed 故障区间检出率: {result.loc[sub_mask, 'detected'].mean():.2%}")
print(f"grate_speed 正常区间检出率: {result.loc[~sub_mask, 'detected'].mean():.2%}")

# 2. 轴承过热诊断
print("\n" + "="*60)
print("轴承过热诊断")
print("="*60)
df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, "fault_bearing_overheat.csv"), low_memory=False)
df = clean_data(df)
mask = df["data_quality_flag"] == "故障注入"
fault_idx = df[mask].index
fault_start, fault_end = fault_idx[0], fault_idx[-1]
window_start = max(0, fault_start - 500)
window_end = min(len(df), fault_end + 500)
sub_df = df.iloc[window_start:window_end].copy()
sub_mask = sub_df["data_quality_flag"] == "故障注入"

result = detect_volatility_batch(sub_df["bearing_vibration"], current_window=10, baseline_window=200)
print(f"bearing_vibration 故障区间平均ratio: {result.loc[sub_mask, 'ratio'].mean():.3f}")
print(f"bearing_vibration 故障区间最大ratio: {result.loc[sub_mask, 'ratio'].max():.3f}")
print(f"bearing_vibration 故障区间检出率: {result.loc[sub_mask, 'detected'].mean():.2%}")

# 3. 正常数据误报诊断
print("\n" + "="*60)
print("正常数据误报诊断 (前3000点)")
print("="*60)
df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, "normal_30days.csv"), low_memory=False)
df = clean_data(df)
sub_df = df.head(3000).copy()
for param in ["grate_speed", "bearing_vibration", "furnace_temperature"]:
    result = detect_volatility_batch(sub_df[param], current_window=10, baseline_window=200)
    fp = result["detected"].sum()
    print(f"{param}: 误报 {fp}次, 最大ratio: {result['ratio'].max():.3f}")
