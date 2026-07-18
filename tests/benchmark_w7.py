"""
绿电哨兵 — W7 性能与效果基准测试

输出指标：
- 每个故障数据集的 Recall（故障区间内 yellow+ 检出率）
- 正常数据的 FP 次数与 Precision
- 综合的 F1（故障点为正样本，正常点为负样本）
- 各检测阶段耗时分布

用法：
    python tests/benchmark_w7.py
    python tests/benchmark_w7.py --quick     # 仅前 5000 行快速验证
"""
import argparse
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data_cleaner import clean_data, PARAM_COLUMNS
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

# 从 config.yaml 读取检测配置（weights / risk_levels / ensemble 策略）
with open(os.path.join(PROJECT_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f) or {}
_DET_CFG = _CFG.get("detection", {})
WEIGHTS = _DET_CFG.get("weights", None)
RISK_LEVELS = _DET_CFG.get("risk_levels", None)
_ENS_CFG = _DET_CFG.get("ensemble", {})
ALLOW_SINGLE_MODULE_ALERT = _ENS_CFG.get("allow_single_module_alert", False)
SINGLE_MODULE_THRESHOLD = float(_ENS_CFG.get("single_module_threshold", 0.8))
SINGLE_MODULE_SCORE_RATIO = float(_ENS_CFG.get("single_module_score_ratio", 0.5))
PERSISTENCE_FILTER = _ENS_CFG.get("persistence_filter", False)
PERSISTENCE_N = int(_ENS_CFG.get("persistence_n", 3))
PERSISTENCE_MODE = str(_ENS_CFG.get("persistence_mode", "consecutive"))
PERSISTENCE_WINDOW = int(_ENS_CFG.get("persistence_window", 30))
PERSISTENCE_MIN_COUNT = int(_ENS_CFG.get("persistence_min_count", 8))
PERSISTENCE_BACKFILL = int(_ENS_CFG.get("persistence_backfill", 0))


def _batch_threshold(df, param_cols):
    """逐行阈值检测（保留原逻辑以便公平对比）"""
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


def run_pipeline_timed(df, param_cols):
    """执行完整检测并返回 (scores_df, timing_dict)"""
    n = len(df)
    index = df.index.tolist()
    timing = {}

    t0 = time.time()
    th_results = _batch_threshold(df, param_cols)
    timing["threshold_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    trend_results = detect_trend_multi_params(df, param_cols)
    timing["trend_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    vol_results = detect_volatility_multi_params(df, param_cols)
    timing["volatility_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    corr_batch = detect_correlation_batch(df, RULES, trend_results, vol_results)
    timing["correlation_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    tr_states = _batch_trend_states(trend_results, n, index)
    vol_states = _batch_vol_states(vol_results, n, index)
    corr_results = [item["matched_rules"] for item in corr_batch]
    scores = compute_risk_score_batch(
        th_results, tr_states, vol_states, corr_results,
        weights=WEIGHTS, risk_levels=RISK_LEVELS,
        allow_single_module_alert=ALLOW_SINGLE_MODULE_ALERT,
        single_module_threshold=SINGLE_MODULE_THRESHOLD,
        single_module_score_ratio=SINGLE_MODULE_SCORE_RATIO,
        persistence_filter=PERSISTENCE_FILTER,
        persistence_n=PERSISTENCE_N,
        persistence_mode=PERSISTENCE_MODE,
        persistence_window=PERSISTENCE_WINDOW,
        persistence_min_count=PERSISTENCE_MIN_COUNT,
        persistence_backfill=PERSISTENCE_BACKFILL,
    )
    timing["ensemble_s"] = round(time.time() - t0, 3)

    timing["total_s"] = round(sum(timing.values()), 3)
    return pd.DataFrame(scores), timing


def compute_metrics(scores_df, fault_mask):
    """计算 Recall / FP / Precision / F1"""
    yellow_plus = scores_df["level"].isin(["yellow", "orange", "red"])

    tp = int((yellow_plus & fault_mask).sum())
    fn = int((~yellow_plus & fault_mask).sum())
    fp = int((yellow_plus & ~fault_mask).sum())
    tn = int((~yellow_plus & ~fault_mask).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "fault_total": int(fault_mask.sum()),
        "normal_total": int((~fault_mask).sum()),
    }


def benchmark_fault(fname, fault_name, quick=False):
    print(f"\n{'='*64}")
    print(f"  故障: {fault_name}  ({fname})")
    print('='*64)

    df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, fname), low_memory=False)
    df = clean_data(df)
    if quick:
        # 保留故障区间 ± 500 行
        mask = df["data_quality_flag"] == "故障注入"
        if mask.any():
            fault_idx = df[mask].index
            lo = max(0, fault_idx[0] - 500)
            hi = min(len(df), fault_idx[-1] + 500)
            df = df.iloc[lo:hi].reset_index(drop=True)
            print(f"  [quick] 截取 {lo}~{hi} 共 {len(df)} 行")

    fault_mask = (df["data_quality_flag"] == "故障注入").to_numpy()
    param_cols = [c for c in df.columns if c not in SKIP_COLS]

    scores_df, timing = run_pipeline_timed(df, param_cols)
    metrics = compute_metrics(scores_df, fault_mask)

    print(f"  Recall:   {metrics['recall']:.2%}  (TP={metrics['tp']}/"
          f"fault={metrics['fault_total']})")
    print(f"  Precision:{metrics['precision']:.2%}  (FP={metrics['fp']}/"
          f"normal={metrics['normal_total']})")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  耗时分布: th={timing['threshold_s']}s tr={timing['trend_s']}s "
          f"vol={timing['volatility_s']}s corr={timing['correlation_s']}s "
          f"ens={timing['ensemble_s']}s total={timing['total_s']}s")
    level_dist = scores_df["level"].value_counts().to_dict()
    print(f"  level分布: {level_dist}")

    return {
        "fault_name": fault_name,
        "rows": len(df),
        "metrics": metrics,
        "timing": timing,
        "level_dist": level_dist,
    }


def benchmark_normal(quick=False):
    print(f"\n{'='*64}")
    print(f"  正常数据  (normal_30days.csv)")
    print('='*64)
    df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, "normal_30days.csv"),
                     low_memory=False)
    df = clean_data(df)
    if quick:
        df = df.head(5000).reset_index(drop=True)
        print(f"  [quick] 仅前 5000 行")

    fault_mask = np.zeros(len(df), dtype=bool)
    param_cols = [c for c in df.columns if c not in SKIP_COLS]

    scores_df, timing = run_pipeline_timed(df, param_cols)
    metrics = compute_metrics(scores_df, fault_mask)

    print(f"  误报次数(FP): {metrics['fp']}  / {metrics['normal_total']} 行")
    print(f"  误报率: {metrics['fp']/metrics['normal_total']:.4%}")
    print(f"  耗时分布: th={timing['threshold_s']}s tr={timing['trend_s']}s "
          f"vol={timing['volatility_s']}s corr={timing['correlation_s']}s "
          f"ens={timing['ensemble_s']}s total={timing['total_s']}s")
    print(f"  吞吐: {len(df)/timing['total_s']:.0f} records/s")
    print(f"  level分布: {scores_df['level'].value_counts().to_dict()}")

    return {
        "fault_name": "正常数据",
        "rows": len(df),
        "metrics": metrics,
        "timing": timing,
        "level_dist": scores_df["level"].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="快速模式（截取少量数据）")
    parser.add_argument("--out", type=str, default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    quick = args.quick
    print(f"W7 基准测试  quick={quick}")
    results = []
    results.append(benchmark_fault("fault_bearing_overheat.csv", "轴承过热", quick))
    results.append(benchmark_fault("fault_emission_exceed.csv", "烟气超标", quick))
    results.append(benchmark_fault("fault_grate_jam.csv", "炉排卡滞", quick))
    results.append(benchmark_normal(quick))

    # 综合指标：故障点合并计算
    all_tp = sum(r["metrics"]["tp"] for r in results[:3])
    all_fn = sum(r["metrics"]["fn"] for r in results[:3])
    all_fp = sum(r["metrics"]["fp"] for r in results)
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_f1 = (2 * overall_prec * overall_recall / (overall_prec + overall_recall)
                  if (overall_prec + overall_recall) > 0 else 0)

    print(f"\n{'='*64}")
    print(f"  W7 综合指标")
    print('='*64)
    print(f"  Overall Recall:    {overall_recall:.2%}  (TP={all_tp}/"
          f"fault={all_tp+all_fn})")
    print(f"  Overall Precision: {overall_prec:.2%}  (FP={all_fp})")
    print(f"  Overall F1:        {overall_f1:.4f}")
    print(f"  W7 目标:           Recall>90%, F1>0.7")
    verdict = "PASS" if (overall_recall > 0.9 and overall_f1 > 0.7) else "FAIL"
    print(f"  验收: {verdict}")

    summary = {
        "results": results,
        "overall": {
            "recall": round(overall_recall, 4),
            "precision": round(overall_prec, 4),
            "f1": round(overall_f1, 4),
            "tp": all_tp, "fn": all_fn, "fp": all_fp,
            "verdict": verdict,
        },
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {args.out}")


if __name__ == "__main__":
    main()
