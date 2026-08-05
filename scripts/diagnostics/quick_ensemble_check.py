"""快速验证综合评分效果"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import yaml
from core.data_cleaner import clean_data
from models.trend_detector import detect_trend_multi_params
from models.volatility_detector import detect_volatility_multi_params
from models.correlation_detector import detect_correlation_batch
from models.ensemble_scorer import compute_risk_score
from models.threshold_detector import detect_threshold, load_thresholds_from_yaml

SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_data")
SKIP_COLS = {"timestamp", "device_id", "device_name", "device_type", "data_quality_flag", "qc_note"}

with open(os.path.join(PROJECT_ROOT, "config", "rules.yaml"), "r", encoding="utf-8") as f:
    RULES = yaml.safe_load(f).get("rules", [])

THRESHOLDS_CFG = load_thresholds_from_yaml(
    os.path.join(PROJECT_ROOT, "config", "thresholds.yaml")
)

def quick_check(fname, fault_type):
    print(f"\n{'='*60}")
    print(f"检查: {fault_type}")
    print('='*60)

    df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, fname), low_memory=False)
    df = clean_data(df)
    mask = df["data_quality_flag"] == "故障注入"
    param_cols = [c for c in df.columns if c not in SKIP_COLS]

    # 批量检测
    print("  计算趋势...")
    trend_results = detect_trend_multi_params(df, param_cols)
    print("  计算波动率...")
    vol_results = detect_volatility_multi_params(df, param_cols)
    print("  计算关联...")
    corr_batch = detect_correlation_batch(df, RULES, trend_results, vol_results)

    # 逐点评分（只评故障区间前后100点以加速）
    fault_idx = df[mask].index
    check_start = max(0, fault_idx[0] - 100)
    check_end = min(len(df), fault_idx[-1] + 100)

    scores = []
    for i in range(check_start, check_end):
        row = df.iloc[i]
        current_values = {col: row[col] for col in param_cols if col in row}

        # threshold（使用 thresholds.yaml 方向性阈值）
        th_hits = detect_threshold(current_values, THRESHOLDS_CFG)

        # trend state
        tr_state = {}
        for col in param_cols:
            if col in trend_results and i in trend_results[col].index:
                trow = trend_results[col].loc[i]
                any_detected = any(trow.get(f"detected_{w}", False) for w in [10,30,60])
                max_level = "green"
                for w in [60, 30, 10]:
                    if trow.get(f"detected_{w}", False):
                        max_level = "yellow" if w == 10 else "orange" if w == 30 else "red"
                        break
                tr_state[col] = {
                    "param": col, "any_detected": any_detected, "max_level": max_level,
                    "window_10": {"slope": trow.get("slope_10", 0.0), "detected": trow.get("detected_10", False)},
                    "window_30": {"slope": trow.get("slope_30", 0.0), "detected": trow.get("detected_30", False)},
                    "window_60": {"slope": trow.get("slope_60", 0.0), "detected": trow.get("detected_60", False)},
                }

        # vol state
        vol_state = {}
        for col in param_cols:
            if col in vol_results and i in vol_results[col].index:
                vrow = vol_results[col].loc[i]
                vol_state[col] = {"param": col, "ratio": vrow.get("ratio", 1.0), "level": vrow.get("level"), "detected": vrow.get("detected", False)}

        # correlation
        corr_result = corr_batch[i]["matched_rules"] if i < len(corr_batch) else []

        score = compute_risk_score(th_hits, tr_state, vol_state, corr_result)
        scores.append((i, score))

    # 统计故障区间
    fault_scores = [s for i, s in scores if i in fault_idx]
    detected = sum(1 for s in fault_scores if s["level"] in ["yellow", "orange", "red"])
    print(f"  故障区间检出率: {detected}/{len(fault_scores)} = {detected/len(fault_scores):.1%}")
    print(f"  平均score: {sum(s['score'] for s in fault_scores)/len(fault_scores):.3f}")
    print(f"  最大score: {max(s['score'] for s in fault_scores):.3f}")
    levels = {}
    for s in fault_scores:
        levels[s['level']] = levels.get(s['level'], 0) + 1
    print(f"  level分布: {levels}")

    # 统计正常区间
    normal_scores = [s for i, s in scores if i not in fault_idx]
    fp = sum(1 for s in normal_scores if s["level"] in ["yellow", "orange", "red"])
    print(f"  检查窗口内正常区间误报: {fp}/{len(normal_scores)}")

# 3类故障
quick_check("fault_bearing_overheat.csv", "轴承过热")
quick_check("fault_emission_exceed.csv", "烟气超标")
quick_check("fault_grate_jam.csv", "炉排卡滞")

# 正常数据（只测前5000点）
print(f"\n{'='*60}")
print("检查: 正常数据")
print('='*60)
df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, "normal_30days.csv"), low_memory=False)
df = clean_data(df)
sub = df.head(5000).copy()
param_cols = [c for c in sub.columns if c not in SKIP_COLS]
print("  计算趋势...")
trend_results = detect_trend_multi_params(sub, param_cols)
print("  计算波动率...")
vol_results = detect_volatility_multi_params(sub, param_cols)
print("  计算关联...")
corr_batch = detect_correlation_batch(sub, RULES, trend_results, vol_results)

fp = 0
for i in range(len(sub)):
    row = sub.iloc[i]
    current_values = {col: row[col] for col in param_cols if col in row}
    th_hits = detect_threshold(current_values, THRESHOLDS_CFG)

    tr_state = {}
    for col in param_cols:
        if col in trend_results and i in trend_results[col].index:
            trow = trend_results[col].loc[i]
            any_detected = any(trow.get(f"detected_{w}", False) for w in [10,30,60])
            max_level = "green"
            for w in [60, 30, 10]:
                if trow.get(f"detected_{w}", False):
                    max_level = "yellow" if w == 10 else "orange" if w == 30 else "red"
                    break
            tr_state[col] = {
                "param": col, "any_detected": any_detected, "max_level": max_level,
                "window_10": {"slope": trow.get("slope_10", 0.0), "detected": trow.get("detected_10", False)},
                "window_30": {"slope": trow.get("slope_30", 0.0), "detected": trow.get("detected_30", False)},
                "window_60": {"slope": trow.get("slope_60", 0.0), "detected": trow.get("detected_60", False)},
            }

    vol_state = {}
    for col in param_cols:
        if col in vol_results and i in vol_results[col].index:
            vrow = vol_results[col].loc[i]
            vol_state[col] = {"param": col, "ratio": vrow.get("ratio", 1.0), "level": vrow.get("level"), "detected": vrow.get("detected", False)}

    corr_result = corr_batch[i]["matched_rules"] if i < len(corr_batch) else []
    score = compute_risk_score(th_hits, tr_state, vol_state, corr_result)
    if score["level"] in ["yellow", "orange", "red"]:
        fp += 1

print(f"  正常数据前5000点误报: {fp}/5000")
