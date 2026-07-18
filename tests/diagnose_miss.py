"""
W7 漏检诊断：定位故障区间内未被检出的点
重点诊断炉排卡滞（Recall 43%）和烟气超标（Recall 55%）
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# 读 config.yaml 保持与 benchmark 同配置
with open(os.path.join(PROJECT_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f) or {}
_DET_CFG = _CFG.get("detection", {})
WEIGHTS = _DET_CFG.get("weights", None)
RISK_LEVELS = _DET_CFG.get("risk_levels", None)
_ENS_CFG = _DET_CFG.get("ensemble", {})
ALLOW_SINGLE_MODULE_ALERT = _ENS_CFG.get("allow_single_module_alert", False)
PERSISTENCE_FILTER = _ENS_CFG.get("persistence_filter", False)
PERSISTENCE_N = int(_ENS_CFG.get("persistence_n", 3))
PERSISTENCE_MODE = str(_ENS_CFG.get("persistence_mode", "consecutive"))
PERSISTENCE_WINDOW = int(_ENS_CFG.get("persistence_window", 30))
PERSISTENCE_MIN_COUNT = int(_ENS_CFG.get("persistence_min_count", 8))
PERSISTENCE_BACKFILL = int(_ENS_CFG.get("persistence_backfill", 0))


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


def diagnose_miss(fname, label, focus_params):
    print(f"\n{'='*72}")
    print(f"  漏检诊断: {label}  ({fname})")
    print(f"  关注参数: {focus_params}")
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
    scores = compute_risk_score_batch(
        th_results, tr_states, vol_states, corr_results,
        weights=WEIGHTS, risk_levels=RISK_LEVELS,
        allow_single_module_alert=ALLOW_SINGLE_MODULE_ALERT,
        persistence_filter=PERSISTENCE_FILTER,
        persistence_n=PERSISTENCE_N,
        persistence_mode=PERSISTENCE_MODE,
        persistence_window=PERSISTENCE_WINDOW,
        persistence_min_count=PERSISTENCE_MIN_COUNT,
        persistence_backfill=PERSISTENCE_BACKFILL,
    )

    yellow_plus = np.array([s["level"] in ("yellow", "orange", "red") for s in scores])
    fault_mask = (df["data_quality_flag"] == "故障注入").to_numpy()

    fn_idx = np.where(~yellow_plus & fault_mask)[0]
    tp_idx = np.where(yellow_plus & fault_mask)[0]
    print(f"  TP={len(tp_idx)}, FN={len(fn_idx)}, fault_total={fault_mask.sum()}")

    if len(fn_idx) == 0:
        return

    # 故障区间内各参数的统计
    print(f"\n  [故障区间内关注参数统计]")
    for p in focus_params:
        if p not in df.columns:
            continue
        vals_fault = df.loc[fault_mask, p].astype(float)
        vals_normal = df.loc[~fault_mask, p].astype(float)
        print(f"    {p:<25} fault: min={vals_fault.min():.2f} max={vals_fault.max():.2f} "
              f"mean={vals_fault.mean():.2f} std={vals_fault.std():.2f} | "
              f"normal: mean={vals_normal.mean():.2f} std={vals_normal.std():.2f}")

    # 故障区间内各模块触发率
    th_in_fault = sum(1 for i in tp_idx if scores[i]["module_scores"]["threshold"]["score"] > 0)
    tr_in_fault = sum(1 for i in tp_idx if scores[i]["module_scores"]["trend"]["score"] > 0)
    vol_in_fault = sum(1 for i in tp_idx if scores[i]["module_scores"]["volatility"]["score"] > 0)
    corr_in_fault = sum(1 for i in tp_idx if scores[i]["module_scores"]["correlation"]["score"] > 0)
    print(f"\n  [TP 中各模块触发率]")
    print(f"    threshold:  {th_in_fault}/{len(tp_idx)} = {th_in_fault/max(len(tp_idx),1):.1%}")
    print(f"    trend:      {tr_in_fault}/{len(tp_idx)} = {tr_in_fault/max(len(tp_idx),1):.1%}")
    print(f"    volatility: {vol_in_fault}/{len(tp_idx)} = {vol_in_fault/max(len(tp_idx),1):.1%}")
    print(f"    correlation:{corr_in_fault}/{len(tp_idx)} = {corr_in_fault/max(len(tp_idx),1):.1%}")

    # 故障区间内趋势检出情况（按参数）
    print(f"\n  [故障区间内趋势检出 by 参数]")
    for p in focus_params:
        if p not in trend_results:
            continue
        tdf = trend_results[p]
        # 对齐故障区间
        fault_idx = df.index[fault_mask]
        det10 = tdf.loc[fault_idx, "detected_10"].sum() if "detected_10" in tdf.columns else 0
        det30 = tdf.loc[fault_idx, "detected_30"].sum() if "detected_30" in tdf.columns else 0
        det60 = tdf.loc[fault_idx, "detected_60"].sum() if "detected_60" in tdf.columns else 0
        any_det = tdf.loc[fault_idx, "any_detected"].sum() if "any_detected" in tdf.columns else 0
        slope30 = tdf.loc[fault_idx, "slope_30"].abs().mean() if "slope_30" in tdf.columns else 0
        print(f"    {p:<25} det10={det10:3d} det30={det30:3d} det60={det60:3d} "
              f"any={any_det:3d}/{len(fault_idx)}  mean|slope30|={slope30:.4f}")

    # 故障区间内波动率检出
    print(f"\n  [故障区间内波动率检出 by 参数]")
    for p in focus_params:
        if p not in vol_results:
            continue
        vdf = vol_results[p]
        fault_idx = df.index[fault_mask]
        det = vdf.loc[fault_idx, "detected"].sum() if "detected" in vdf.columns else 0
        ratio_mean = vdf.loc[fault_idx, "ratio"].mean() if "ratio" in vdf.columns else 1.0
        ratio_max = vdf.loc[fault_idx, "ratio"].max() if "ratio" in vdf.columns else 1.0
        print(f"    {p:<25} detected={det:3d}/{len(fault_idx)}  ratio_mean={ratio_mean:.2f} ratio_max={ratio_max:.2f}")

    # 关联规则在故障区间的匹配率
    print(f"\n  [故障区间内关联规则匹配率]")
    rule_match = Counter()
    for i in tp_idx:
        for r in corr_results[i]:
            if r.get("matched"):
                rule_match[r["rule_id"]] += 1
    for rid, cnt in sorted(rule_match.items()):
        print(f"    {rid}: {cnt}/{len(tp_idx)} = {cnt/max(len(tp_idx),1):.1%}")

    # 前 15 个 FN 详情
    print(f"\n  [前 15 个 FN 详情]")
    print(f"  {'idx':>6} {'ts':<20} {'score':>6} {'level':<8} {'th':>5} {'tr':>5} {'vol':>5} {'corr':>5}  focus值")
    for i in fn_idx[:15]:
        s = scores[i]
        ms = s["module_scores"]
        ts = df.iloc[i]["timestamp"]
        focus_vals = {p: round(float(df.iloc[i][p]), 2) for p in focus_params if p in df.columns}
        cf = s["details"].get("correlation_faults", [])
        print(f"  {i:6d} {str(ts)[:19]:<20} {s['score']:6.3f} {s['level']:<8} "
              f"{ms['threshold']['score']:5.2f} {ms['trend']['score']:5.2f} "
              f"{ms['volatility']['score']:5.2f} {ms['correlation']['score']:5.2f}  "
              f"{focus_vals} cf={cf}")


if __name__ == "__main__":
    # 炉排卡滞：grate_speed, feed_rate, furnace_pressure
    diagnose_miss("fault_grate_jam.csv", "炉排卡滞",
                  ["grate_speed", "feed_rate", "furnace_pressure",
                   "furnace_temperature", "oxygen_content"])
    # 烟气超标：so2, nox, oxygen
    diagnose_miss("fault_emission_exceed.csv", "烟气超标",
                  ["so2_concentration", "nox_concentration", "oxygen_content",
                   "furnace_temperature", "flue_gas_temperature"])
