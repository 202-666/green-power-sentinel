"""
绿电哨兵 — 综合评分集成测试（优化版）

验收标准：
1. 3类故障全部检出（故障区间内至少出现 yellow+ 等级）
2. 正常30天数据误报 < 5次（yellow+ 出现次数 < 5）
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data_cleaner import clean_data
from models.trend_detector import detect_trend_multi_params
from models.volatility_detector import detect_volatility_multi_params
from models.correlation_detector import detect_correlation_batch
from models.ensemble_scorer import compute_risk_score_batch
from models.threshold_detector import detect_threshold, load_thresholds_from_yaml

SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_data")
SKIP_COLS = {
    "timestamp",
    "device_id",
    "device_name",
    "device_type",
    "data_quality_flag",
    "qc_note",
}

# 加载规则
with open(os.path.join(PROJECT_ROOT, "config", "rules.yaml"), "r", encoding="utf-8") as f:
    RULES = yaml.safe_load(f).get("rules", [])

# 从 thresholds.yaml 加载阈值配置（支持方向性）
THRESHOLDS_CFG = load_thresholds_from_yaml(
    os.path.join(PROJECT_ROOT, "config", "thresholds.yaml")
)

# 从 config.yaml 加载检测配置
with open(os.path.join(PROJECT_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def _batch_threshold(df: pd.DataFrame, param_cols: list) -> list:
    """批量阈值检测，返回 list of list"""
    results = []
    for _, row in df.iterrows():
        current_values = {col: row[col] for col in param_cols if col in row}
        hits = detect_threshold(current_values, THRESHOLDS_CFG)
        results.append(hits)
    return results


def _batch_trend_states(trend_results: dict, n: int, index) -> list:
    """将批量趋势结果转换为逐点状态字典列表"""
    param_cols = list(trend_results.keys())
    states = []
    for i in range(n):
        idx = index[i]
        st = {}
        for col in param_cols:
            tdf = trend_results[col]
            if idx not in tdf.index:
                continue
            row = tdf.loc[idx]
            any_detected = any(row.get(f"detected_{w}", False) for w in [10, 30, 60])
            max_level = "green"
            for w in [60, 30, 10]:
                if row.get(f"detected_{w}", False):
                    max_level = "yellow" if w == 10 else "orange" if w == 30 else "red"
                    break
            st[col] = {
                "param": col,
                "any_detected": any_detected,
                "max_level": max_level,
                "window_10": {
                    "slope": row.get("slope_10", 0.0),
                    "detected": row.get("detected_10", False),
                },
                "window_30": {
                    "slope": row.get("slope_30", 0.0),
                    "detected": row.get("detected_30", False),
                },
                "window_60": {
                    "slope": row.get("slope_60", 0.0),
                    "detected": row.get("detected_60", False),
                },
            }
        states.append(st)
    return states


def _batch_vol_states(vol_results: dict, n: int, index) -> list:
    """将批量波动率结果转换为逐点状态字典列表"""
    param_cols = list(vol_results.keys())
    states = []
    for i in range(n):
        idx = index[i]
        st = {}
        for col in param_cols:
            vdf = vol_results[col]
            if idx not in vdf.index:
                continue
            row = vdf.loc[idx]
            st[col] = {
                "param": col,
                "ratio": row.get("ratio", 1.0),
                "level": row.get("level"),
                "detected": row.get("detected", False),
            }
        states.append(st)
    return states


def compute_pipeline_scores(df: pd.DataFrame) -> pd.DataFrame:
    """高效批量计算综合风险评分"""
    param_cols = [c for c in df.columns if c not in SKIP_COLS]
    n = len(df)
    index = df.index.tolist()

    # 获取检测配置
    detect_cfg = CONFIG.get("detection", {})
    dynamic_threshold = detect_cfg.get("trend_dynamic_threshold", False)
    baseline_window = detect_cfg.get("trend_baseline_window", 480)
    sensitivity_k = detect_cfg.get("trend_sensitivity_k", 3.0)
    risk_levels = detect_cfg.get("risk_levels", {"red": 0.75, "orange": 0.50, "yellow": 0.35})
    weights = detect_cfg.get("weights", {"threshold": 0.2, "trend": 0.3, "volatility": 0.2, "correlation": 0.3})
    allow_single_module = detect_cfg.get("allow_single_module_alert", False)

    # 批量检测
    th_results = _batch_threshold(df, param_cols)
    trend_results = detect_trend_multi_params(
        df, param_cols,
        dynamic_threshold=dynamic_threshold,
        baseline_window=baseline_window,
        sensitivity_k=sensitivity_k,
    )
    vol_results = detect_volatility_multi_params(df, param_cols)
    corr_batch = detect_correlation_batch(df, RULES, trend_results, vol_results)

    # 转换状态格式
    tr_states = _batch_trend_states(trend_results, n, index)
    vol_states = _batch_vol_states(vol_results, n, index)
    corr_results = [item["matched_rules"] for item in corr_batch]

    # 批量评分
    scores = compute_risk_score_batch(
        th_results, tr_states, vol_states, corr_results,
        weights=weights,
        risk_levels=risk_levels,
        allow_single_module_alert=allow_single_module,
    )
    return pd.DataFrame(scores)


class TestEnsembleFaultDetection(unittest.TestCase):
    """综合评分故障检出测试"""

    def test_bearing_overheat_detected(self):
        """轴承过热应被综合评分检出"""
        df = pd.read_csv(
            os.path.join(SAMPLE_DATA_DIR, "fault_bearing_overheat.csv"),
            low_memory=False,
        )
        df = clean_data(df)
        mask = df["data_quality_flag"] == "故障注入"

        scores = compute_pipeline_scores(df)
        fault_scores = scores[mask]

        detected_count = (
            fault_scores["level"].isin(["yellow", "orange", "red"])
        ).sum()
        detection_rate = detected_count / mask.sum() if mask.sum() > 0 else 0

        print(
            f"\n[轴承过热] 综合检出率: {detection_rate:.1%} ({detected_count}/{mask.sum()})"
        )
        print(f"  平均score: {fault_scores['score'].mean():.3f}")
        print(f"  最大score: {fault_scores['score'].max():.3f}")
        print(f"  level分布: {fault_scores['level'].value_counts().to_dict()}")

        self.assertGreater(
            detection_rate,
            0.5,
            f"轴承过热综合检出率 {detection_rate:.1%} 过低",
        )

    def test_emission_exceed_detected(self):
        """烟气超标应被综合评分检出"""
        df = pd.read_csv(
            os.path.join(SAMPLE_DATA_DIR, "fault_emission_exceed.csv"),
            low_memory=False,
        )
        df = clean_data(df)
        mask = df["data_quality_flag"] == "故障注入"

        scores = compute_pipeline_scores(df)
        fault_scores = scores[mask]

        detected_count = (
            fault_scores["level"].isin(["yellow", "orange", "red"])
        ).sum()
        detection_rate = detected_count / mask.sum() if mask.sum() > 0 else 0

        print(
            f"\n[烟气超标] 综合检出率: {detection_rate:.1%} ({detected_count}/{mask.sum()})"
        )
        print(f"  平均score: {fault_scores['score'].mean():.3f}")
        print(f"  最大score: {fault_scores['score'].max():.3f}")
        print(f"  level分布: {fault_scores['level'].value_counts().to_dict()}")

        self.assertGreater(
            detection_rate,
            0.10,
            f"烟气超标综合检出率 {detection_rate:.1%} 过低",
        )

    def test_grate_jam_detected(self):
        """炉排卡滞应被综合评分检出"""
        df = pd.read_csv(
            os.path.join(SAMPLE_DATA_DIR, "fault_grate_jam.csv"),
            low_memory=False,
        )
        df = clean_data(df)
        mask = df["data_quality_flag"] == "故障注入"

        scores = compute_pipeline_scores(df)
        fault_scores = scores[mask]

        detected_count = (
            fault_scores["level"].isin(["yellow", "orange", "red"])
        ).sum()
        detection_rate = detected_count / mask.sum() if mask.sum() > 0 else 0

        print(
            f"\n[炉排卡滞] 综合检出率: {detection_rate:.1%} ({detected_count}/{mask.sum()})"
        )
        print(f"  平均score: {fault_scores['score'].mean():.3f}")
        print(f"  最大score: {fault_scores['score'].max():.3f}")
        print(f"  level分布: {fault_scores['level'].value_counts().to_dict()}")

        self.assertGreater(
            detection_rate,
            0.30,
            f"炉排卡滞综合检出率 {detection_rate:.1%} 过低",
        )


class TestEnsembleFalsePositive(unittest.TestCase):
    """综合评分误报控制测试"""

    def test_normal_data_false_positives(self):
        """正常30天数据综合评分误报应<5次"""
        df = pd.read_csv(
            os.path.join(SAMPLE_DATA_DIR, "normal_30days.csv"),
            low_memory=False,
        )
        df = clean_data(df)

        scores = compute_pipeline_scores(df)
        yellow_plus = scores["level"].isin(["yellow", "orange", "red"])
        fp_count = yellow_plus.sum()
        fp_rate = fp_count / len(df)

        print(f"\n[正常数据] 综合评分误报: {fp_count}次 ({fp_rate:.4%})")
        print(f"  level分布: {scores['level'].value_counts().to_dict()}")
        print(
            f"  score统计: mean={scores['score'].mean():.4f}, max={scores['score'].max():.4f}"
        )

        self.assertLess(
            fp_count,
            5,
            f"正常数据综合评分误报 {fp_count}次，应<5次/30天",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
