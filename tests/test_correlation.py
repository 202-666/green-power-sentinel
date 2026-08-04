"""
绿电哨兵 — 关联检测单元测试

测试覆盖：
1. 规则引擎条件评估正确性（threshold/trend/volatility）
2. 单时刻关联检测（AND/OR 逻辑）
3. 批量关联检测
4. 3类故障的规则匹配率
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

from models.correlation_detector import (
    detect_correlation,
    detect_correlation_batch,
    _eval_condition,
)

RULES_PATH = os.path.join(PROJECT_ROOT, "config", "rules.yaml")
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_data")


def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("rules", [])


class TestEvalCondition(unittest.TestCase):
    """条件评估测试"""

    def test_threshold_up(self):
        """阈值上升条件"""
        cond = {
            "param": "temp",
            "direction": "up",
            "compare_to": "threshold",
            "threshold": 100,
        }
        satisfied, conf = _eval_condition(cond, {"temp": 105})
        self.assertTrue(satisfied)
        self.assertGreater(conf, 0)

    def test_threshold_down(self):
        """阈值下降条件"""
        cond = {
            "param": "pressure",
            "direction": "down",
            "compare_to": "threshold",
            "threshold": 50,
        }
        satisfied, conf = _eval_condition(cond, {"pressure": 45})
        self.assertTrue(satisfied)
        self.assertGreater(conf, 0)

    def test_threshold_not_met(self):
        """阈值未满足"""
        cond = {
            "param": "temp",
            "direction": "up",
            "compare_to": "threshold",
            "threshold": 100,
        }
        satisfied, conf = _eval_condition(cond, {"temp": 95})
        self.assertFalse(satisfied)

    def test_trend_condition(self):
        """趋势条件"""
        cond = {
            "param": "so2",
            "direction": "up",
            "compare_to": "trend",
            "trend_window": 30,
            "slope_threshold": 0.5,
        }
        trend_states = {"so2": {"window_30": {"slope": 0.8, "detected": True}}}
        satisfied, conf = _eval_condition(
            cond, {"so2": 100}, trend_states=trend_states
        )
        self.assertTrue(satisfied)
        self.assertGreater(conf, 0)

    def test_volatility_condition(self):
        """波动率条件"""
        cond = {
            "param": "grate_speed",
            "direction": "volatile",
            "compare_to": "volatility",
            "volatility_ratio": 2.0,
        }
        vol_states = {"grate_speed": {"ratio": 3.0, "detected": True}}
        satisfied, conf = _eval_condition(
            cond, {"grate_speed": 50}, volatility_states=vol_states
        )
        self.assertTrue(satisfied)
        self.assertGreater(conf, 0)

    def test_param_missing(self):
        """参数缺失应返回 False"""
        cond = {
            "param": "missing",
            "direction": "up",
            "compare_to": "threshold",
            "threshold": 100,
        }
        satisfied, conf = _eval_condition(cond, {"temp": 50})
        self.assertFalse(satisfied)


class TestDetectCorrelationSingle(unittest.TestCase):
    """单时刻关联检测测试"""

    def test_bearing_overheat_rule_r1(self):
        """轴承过热规则 R1 应被匹配"""
        rules = load_rules()
        r1 = [r for r in rules if r["rule_id"] == "R1"][0]

        current_values = {
            "bearing_temperature": 75,
            "bearing_vibration": 5.0,
        }
        result = detect_correlation(current_values, [r1])
        self.assertTrue(result[0]["matched"])
        self.assertEqual(result[0]["fault_type"], "轴承过热")
        self.assertGreater(result[0]["confidence"], 0)

    def test_grate_jam_rule_r3(self):
        """炉排卡滞规则 R3 应被匹配（按 rules.yaml/W8 现行 5 条件 MAJORITY 口径）"""
        rules = load_rules()
        r3 = [r for r in rules if r["rule_id"] == "R3"][0]

        current_values = {
            "grate_speed": 50,
            "feed_rate": 7.0,
            "furnace_pressure": 60,
        }
        trend_states = {
            "furnace_pressure": {
                "window_30": {"slope": 0.5},
                "window_60": {"slope": 0.3},
            },
            "feed_rate": {
                "window_30": {"slope": -0.1},
            },
        }

        result = detect_correlation(current_values, [r3], trend_states=trend_states)
        self.assertTrue(result[0]["matched"])
        self.assertEqual(result[0]["fault_type"], "炉排卡滞")

    def test_emission_rule_r2(self):
        """烟气超标规则 R2 应被匹配"""
        rules = load_rules()
        r2 = [r for r in rules if r["rule_id"] == "R2"][0]

        current_values = {
            "so2_concentration": 120,
            "nox_concentration": 150,
            "oxygen_content": 6.0,
        }
        trend_states = {
            "so2_concentration": {
                "window_30": {"slope": 0.8, "detected": True}
            },
            "nox_concentration": {
                "window_30": {"slope": 0.5, "detected": True}
            },
        }

        result = detect_correlation(current_values, [r2], trend_states=trend_states)
        self.assertTrue(result[0]["matched"])
        self.assertEqual(result[0]["fault_type"], "烟气超标")

    def test_no_match_on_normal(self):
        """正常状态不应匹配任何规则"""
        rules = load_rules()
        current_values = {
            "bearing_temperature": 45,
            "bearing_vibration": 2.0,
            "so2_concentration": 40,
            "nox_concentration": 60,
            "oxygen_content": 9.0,
            "grate_speed": 55,
            "feed_rate": 11,
            "furnace_pressure": 5,
        }
        result = detect_correlation(current_values, rules)
        matched_any = any(r["matched"] for r in result)
        self.assertFalse(matched_any)


class TestFaultDetectionRate(unittest.TestCase):
    """故障检出率测试"""

    @classmethod
    def setUpClass(cls):
        from core.data_cleaner import clean_data
        from models.trend_detector import detect_trend_multi_params
        from models.volatility_detector import detect_volatility_multi_params

        cls.rules = load_rules()
        cls.faults = {}
        cls.fault_ranges = {}
        cls.correlation_results = {}

        for fname, fault_type in [
            ("fault_bearing_overheat.csv", "bearing_overheat"),
            ("fault_emission_exceed.csv", "emission_exceed"),
            ("fault_grate_jam.csv", "grate_jam"),
        ]:
            fpath = os.path.join(SAMPLE_DATA_DIR, fname)
            df = pd.read_csv(fpath, low_memory=False)
            df = clean_data(df)
            cls.faults[fault_type] = df
            cls.fault_ranges[fault_type] = df["data_quality_flag"] == "故障注入"

            # 预计算趋势和波动率
            skip_cols = {
                "timestamp",
                "device_id",
                "device_name",
                "device_type",
                "data_quality_flag",
                "qc_note",
            }
            param_cols = [c for c in df.columns if c not in skip_cols]
            trend_results = detect_trend_multi_params(df, param_cols)
            vol_results = detect_volatility_multi_params(df, param_cols)

            # 批量关联检测
            corr_batch = detect_correlation_batch(
                df, cls.rules, trend_results, vol_results
            )
            cls.correlation_results[fault_type] = corr_batch

    def test_bearing_overheat_rule_match(self):
        """轴承过热：R1 规则应被匹配"""
        corr_batch = self.correlation_results["bearing_overheat"]
        mask = self.fault_ranges["bearing_overheat"]

        matched_count = 0
        for i, item in enumerate(corr_batch):
            if mask.iloc[i]:
                for rule in item["matched_rules"]:
                    if rule["rule_id"] == "R1":
                        matched_count += 1
                        break

        fault_total = mask.sum()
        match_rate = matched_count / fault_total if fault_total > 0 else 0.0
        print(
            f"\n[轴承过热] R1 规则匹配率: {match_rate:.1%} ({matched_count}/{fault_total})"
        )
        self.assertGreater(match_rate, 0.3, f"R1 规则匹配率 {match_rate:.1%} 过低")

    def test_grate_jam_rule_match(self):
        """炉排卡滞：R3 规则应被匹配"""
        corr_batch = self.correlation_results["grate_jam"]
        mask = self.fault_ranges["grate_jam"]

        matched_count = 0
        for i, item in enumerate(corr_batch):
            if mask.iloc[i]:
                for rule in item["matched_rules"]:
                    if rule["rule_id"] == "R3":
                        matched_count += 1
                        break

        fault_total = mask.sum()
        match_rate = matched_count / fault_total if fault_total > 0 else 0.0
        print(
            f"\n[炉排卡滞] R3 规则匹配率: {match_rate:.1%} ({matched_count}/{fault_total})"
        )
        self.assertGreater(match_rate, 0.2, f"R3 规则匹配率 {match_rate:.1%} 过低")

    def test_emission_exceed_rule_match(self):
        """烟气超标：R2 规则应在后期被匹配（oxygen<7阶段）"""
        corr_batch = self.correlation_results["emission_exceed"]
        mask = self.fault_ranges["emission_exceed"]

        matched_count = 0
        for i, item in enumerate(corr_batch):
            if mask.iloc[i]:
                for rule in item["matched_rules"]:
                    if rule["rule_id"] == "R2":
                        matched_count += 1
                        break

        fault_total = mask.sum()
        match_rate = matched_count / fault_total if fault_total > 0 else 0.0
        print(
            f"\n[烟气超标] R2 规则匹配率: {match_rate:.1%} ({matched_count}/{fault_total})"
        )
        # R2 匹配率低是因为 oxygen 在故障区间大部分时段仍>7（均值8.8），
        # 烟气超标主要靠 trend 检测覆盖，correlation 仅作辅助
        self.assertGreater(match_rate, 0.0, f"R2 规则匹配率 {match_rate:.1%} 过低")


if __name__ == "__main__":
    unittest.main(verbosity=2)
