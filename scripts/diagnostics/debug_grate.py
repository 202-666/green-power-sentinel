"""调试并标定最佳参数"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from core.data_cleaner import clean_data
from models.volatility_detector import detect_volatility_batch

SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_data")

def test_config(current_window, baseline_window, mults, name):
    # 炉排卡滞
    df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, "fault_grate_jam.csv"), low_memory=False)
    df = clean_data(df)
    mask = df["data_quality_flag"] == "故障注入"
    fault_idx = df[mask].index
    fs, fe = fault_idx[0], fault_idx[-1]
    sub = df.iloc[max(0, fs-500):min(len(df), fe+500)].copy()
    sub_mask = sub["data_quality_flag"] == "故障注入"
    r = detect_volatility_batch(sub["grate_speed"], current_window, baseline_window, mults)
    det_grate = r.loc[sub_mask, "detected"].mean()

    # 轴承过热
    df2 = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, "fault_bearing_overheat.csv"), low_memory=False)
    df2 = clean_data(df2)
    mask2 = df2["data_quality_flag"] == "故障注入"
    fault_idx2 = df2[mask2].index
    fs2, fe2 = fault_idx2[0], fault_idx2[-1]
    sub2 = df2.iloc[max(0, fs2-500):min(len(df2), fe2+500)].copy()
    sub_mask2 = sub2["data_quality_flag"] == "故障注入"
    r2 = detect_volatility_batch(sub2["bearing_vibration"], current_window, baseline_window, mults)
    det_vib = r2.loc[sub_mask2, "detected"].mean()

    # 正常数据误报 (前5000点)
    ndf = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, "normal_30days.csv"), low_memory=False)
    ndf = clean_data(ndf)
    sub_n = ndf.head(5000).copy()
    fp = 0
    for p in ["grate_speed", "bearing_vibration", "furnace_temperature", "so2_concentration", "nox_concentration"]:
        rn = detect_volatility_batch(sub_n[p], current_window, baseline_window, mults)
        fp += rn["detected"].sum()

    print(f"{name}: grate检出={det_grate:.1%}, vib检出={det_vib:.1%}, 误报={fp}/5000")

# 测试多种配置
test_config(10, 200, {"yellow":2.0,"orange":2.5,"red":3.5}, "cw10_bw200_strict")
test_config(10, 200, {"yellow":1.7,"orange":2.2,"red":3.0}, "cw10_bw200_mid")
test_config(10, 1440, {"yellow":1.7,"orange":2.2,"red":3.0}, "cw10_bw1440_mid")
test_config(30, 200, {"yellow":1.5,"orange":2.0,"red":2.5}, "cw30_bw200_loose")
test_config(30, 1440, {"yellow":1.5,"orange":2.0,"red":2.5}, "cw30_bw1440_loose")
test_config(30, 1440, {"yellow":1.7,"orange":2.2,"red":3.0}, "cw30_bw1440_mid")
test_config(20, 1440, {"yellow":1.6,"orange":2.0,"red":2.8}, "cw20_bw1440")
