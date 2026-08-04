"""
W7 FP 诊断脚本：定位正常数据上的误报来源
输出每个 FP 的：idx, timestamp, score, level, 各模块分数, 匹配的 correlation 规则, primary_cause
"""
import os
import sys
import json
from collections import Counter

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data_cleaner import clean_data
from models.threshold_detector import detect_threshold, load_thresholds_from_yaml
from models.trend_detector import detect_trend_multi_params
from models.volatility_detector import detect_volatility_multi_params
from models.correlation_detector import detect_correlation_batch
from models.ensemble_scorer import compute_risk_score_batch

SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_data")
SKIP_COLS = {"timestamp", "device_id", "device_name", "device_type",
             "data_quality_flag", "qc_note"}

with open(os.path.join(PROJECT_ROOT, "config", "rules.yaml"), "r", encoding="utf-8") as f:
    RULES = yaml.safe_load(f).get("rules", [])
THRESHOLDS_CFG = load_thresholds_from_yaml(
    os.path.join(PROJECT_ROOT, "config", "thresholds.yaml")
)


def _batch_threshold(df, param_cols):
    results = []
    for _, row in df.iterrows():
        cv = {c: row[c] for c in param_cols if c in row}
        results.append(detect_threshold(cv, THRESHOLDS_CFG))
    return results


def _batch_trend_states(trend_results, n, index):
    states = []
    for i in range(n):
        idx = index[i]
        st = {}
        for col, tdf in trend_results.items():
            if idx not in tdf.index:
                continue
            row = tdf.loc[idx]
            any_det = any(row.get(f"detected_{w}", False) for w in [10, 30, 60])
            max_level = "green"
            for w in [60, 30, 10]:
                if row.get(f"detected_{w}", False):
                    max_level = ("yellow" if w == 10
                                 else "orange" if w == 30 else "red")
                    break
            st[col] = {
                "param": col, "any_detected": any_det, "max_level": max_level,
                "window_10": {"slope": row.get("slope_10", 0.0),
                              "detected": row.get("detected_10", False)},
                "window_30": {"slope": row.get("slope_30", 0.0),
                              "detected": row.get("detected_30", False)},
                "window_60": {"slope": row.get("slope_60", 0.0),
                              "detected": row.get("detected_60", False)},
            }
        states.append(st)
    return states


def _batch_vol_states(vol_results, n, index):
    states = []
    for i in range(n):
        idx = index[i]
        st = {}
        for col, vdf in vol_results.items():
            if idx not in vdf.index:
                continue
            row = vdf.loc[idx]
            st[col] = {"param": col, "ratio": row.get("ratio", 1.0),
                       "level": row.get("level"),
                       "detected": row.get("detected", False)}
        states.append(st)
    return states


def diagnose(fname, label):
    print(f"\n{'='*72}")
    print(f"  诊断: {label}  ({fname})")
    print('='*72)
    df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, fname), low_memory=False)
    df = clean_data(df)
    n = len(df)
    index = df.index.tolist()
    param_cols = [c for c in df.columns if c not in SKIP_COLS]

    th_results = _batch_threshold(df, param_cols)
    trend_results = detect_trend_multi_params(df, param_cols)
    vol_results = detect_volatility_multi_params(df, param_cols)
    corr_batch = detect_correlation_batch(df, RULES, trend_results, vol_results)
    tr_states = _batch_trend_states(trend_results, n, index)
    vol_states = _batch_vol_states(vol_results, n, index)
    corr_results = [item["matched_rules"] for item in corr_batch]
    scores = compute_risk_score_batch(th_results, tr_states, vol_states, corr_results)

    yellow_plus = np.array([s["level"] in ("yellow", "orange", "red") for s in scores])
    fault_mask = (df["data_quality_flag"] == "故障注入").to_numpy()

    # FP: yellow+ 且非故障
    fp_idx = np.where(yellow_plus & ~fault_mask)[0]
    print(f"  FP 数量: {len(fp_idx)}")

    if len(fp_idx) == 0:
        return

    # 统计各模块触发情况
    th_trigger = Counter()
    tr_trigger = Counter()
    vol_trigger = Counter()
    corr_trigger = Counter()
    causes = Counter()
    for i in fp_idx:
        s = scores[i]
        ms = s["module_scores"]
        if ms["threshold"]["score"] > 0:
            th_trigger[round(ms["threshold"]["score"], 2)] += 1
        if ms["trend"]["score"] > 0:
            tr_trigger[round(ms["trend"]["score"], 2)] += 1
        if ms["volatility"]["score"] > 0:
            vol_trigger[round(ms["volatility"]["score"], 2)] += 1
        if ms["correlation"]["score"] > 0:
            corr_trigger[round(ms["correlation"]["score"], 2)] += 1
        cause = s.get("primary_cause", "")
        # 截取根因前缀
        causes[cause.split(":")[0][:30] if cause else "(none)"] += 1

    print(f"\n  [FP 触发模块分布]")
    print(f"    threshold:  触发 {sum(th_trigger.values())} 次, 分数分布: {dict(th_trigger.most_common(5))}")
    print(f"    trend:      触发 {sum(tr_trigger.values())} 次, 分数分布: {dict(tr_trigger.most_common(5))}")
    print(f"    volatility: 触发 {sum(vol_trigger.values())} 次, 分数分布: {dict(vol_trigger.most_common(5))}")
    print(f"    correlation:触发 {sum(corr_trigger.values())} 次, 分数分布: {dict(corr_trigger.most_common(5))}")
    print(f"\n  [FP 根因分布 Top10]:")
    for cause, cnt in causes.most_common(10):
        print(f"    {cnt:3d}  {cause}")

    # 打印前 20 个 FP 的详情
    print(f"\n  [前 20 个 FP 详情]:")
    print(f"  {'idx':>6} {'ts':<20} {'score':>6} {'level':<8} {'th':>5} {'tr':>5} {'vol':>5} {'corr':>5}  cause")
    for i in fp_idx[:20]:
        s = scores[i]
        ms = s["module_scores"]
        ts = df.iloc[i]["timestamp"]
        cf = s["details"].get("correlation_faults", [])
        cause = (s.get("primary_cause", "") or "")[:60]
        print(f"  {i:6d} {str(ts)[:19]:<20} {s['score']:6.3f} {s['level']:<8} "
              f"{ms['threshold']['score']:5.2f} {ms['trend']['score']:5.2f} "
              f"{ms['volatility']['score']:5.2f} {ms['correlation']['score']:5.2f}  "
              f"cf={cf} {cause}")


if __name__ == "__main__":
    diagnose("normal_30days.csv", "正常数据")
    # 也诊断故障文件中的 FP（正常区间的误报）
    diagnose("fault_bearing_overheat.csv", "轴承过热(正常区间FP)")
