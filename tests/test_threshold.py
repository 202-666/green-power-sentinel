"""
绿电哨兵 — 阈值检测单元测试

验收标准：阈值检测 100% 检出超限值

测试覆盖：
1. 正常值不触发
2. 各阈值类型（danger_threshold/warning_threshold/danger_low/danger_high/
   danger_threshold_low/danger_threshold_high/standard_limit）均能正确触发
3. 双向阈值的方向判断（upper/lower）
4. 同参数多阈值共存时按优先级（danger > warning）输出
5. 100% 检出：对真实故障数据中所有超限点必须检出
6. 无效值（NaN/None/字符串）不误报
"""

import os
import sys
import unittest

import pandas as pd
import numpy as np

# 让 tests 目录可导入项目模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.threshold_detector import (
    detect_threshold,
    detect_threshold_batch,
    load_thresholds_from_yaml,
    _normalize_thresholds,
)


THRESHOLDS_YAML = os.path.join(PROJECT_ROOT, "config", "thresholds.yaml")
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_data")


class TestThresholdDetectorBasics(unittest.TestCase):
    """基础功能测试"""

    def setUp(self):
        self.thresholds = load_thresholds_from_yaml(THRESHOLDS_YAML)

    def test_normal_values_no_detection(self):
        """正常值不应触发任何阈值"""
        normal_values = {
            "furnace_temperature": 950.0,
            "flue_gas_temperature": 215.0,
            "steam_pressure": 3.85,
            "steam_flow": 52.0,
            "bearing_vibration": 2.5,
            "bearing_temperature": 50.0,
            "so2_concentration": 60.0,
            "nox_concentration": 110.0,
            "grate_speed": 55.0,
            "feed_rate": 11.0,
            "oxygen_content": 9.0,
            "furnace_pressure": -10.0,
            "cooling_water_temp": 33.0,
        }
        results = detect_threshold(normal_values, self.thresholds)
        self.assertEqual(results, [], f"正常值不应触发阈值，但得到: {results}")

    def test_upper_danger_threshold(self):
        """上限危险阈值（danger_threshold）"""
        # 炉膛温度 danger_threshold=1050
        results = detect_threshold(
            {"furnace_temperature": 1060.0}, self.thresholds
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["param"], "furnace_temperature")
        self.assertEqual(r["level"], "danger")
        self.assertEqual(r["direction"], "upper")
        self.assertEqual(r["threshold"], 1050)
        self.assertEqual(r["value"], 1060.0)
        self.assertEqual(r["threshold_type"], "danger_threshold")
        self.assertGreater(r["exceed_amount"], 0)

    def test_warning_threshold(self):
        """警告阈值（warning_threshold）"""
        # 轴承振动 warning_threshold=4.5, danger_threshold=7.1
        # 值=5.0 触发 warning 但不到 danger
        results = detect_threshold(
            {"bearing_vibration": 5.0}, self.thresholds
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["level"], "warning")
        self.assertEqual(r["threshold"], 4.5)

    def test_danger_overrides_warning(self):
        """危险优先级高于警告（同一参数同时触发时）"""
        # bearing_vibration=8.0 同时 > warning(4.5) 和 > danger(7.1)
        # 应仅返回 danger 级别
        results = detect_threshold(
            {"bearing_vibration": 8.0}, self.thresholds
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["level"], "danger")
        self.assertEqual(results[0]["threshold"], 7.1)

    def test_bidirectional_high(self):
        """双向阈值——超上限（danger_high / danger_threshold_high）"""
        # 蒸汽压力 danger_high=4.5
        results = detect_threshold(
            {"steam_pressure": 4.6}, self.thresholds
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["direction"], "upper")
        self.assertEqual(r["level"], "danger")
        self.assertEqual(r["threshold"], 4.5)

    def test_bidirectional_low(self):
        """双向阈值——低于下限（danger_low / danger_threshold_low）"""
        # 蒸汽压力 danger_low=3.0
        results = detect_threshold(
            {"steam_pressure": 2.8}, self.thresholds
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["direction"], "lower")
        self.assertEqual(r["level"], "danger")
        self.assertEqual(r["threshold"], 3.0)

    def test_lower_threshold_only(self):
        """纯下限阈值（danger_threshold_low，如蒸汽流量）"""
        # 蒸汽流量 danger_threshold_low=30
        results = detect_threshold(
            {"steam_flow": 25.0}, self.thresholds
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["direction"], "lower")
        self.assertEqual(r["level"], "danger")
        self.assertEqual(r["threshold"], 30)

    def test_standard_limit(self):
        """国标限值（standard_limit，如 SO2）"""
        # SO2 standard_limit=200
        results = detect_threshold(
            {"so2_concentration": 210.0}, self.thresholds
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["level"], "limit")
        self.assertEqual(r["threshold"], 200)

    def test_invalid_values_ignored(self):
        """无效值应被忽略，不报错也不检出"""
        results = detect_threshold(
            {"furnace_temperature": None,
             "flue_gas_temperature": float("nan"),
             "steam_pressure": "invalid_string"},
            self.thresholds
        )
        self.assertEqual(results, [])

    def test_unknown_param_ignored(self):
        """未知参数（不在 thresholds 配置中）应被忽略"""
        results = detect_threshold(
            {"unknown_param": 9999.0, "furnace_temperature": 950.0},
            self.thresholds
        )
        self.assertEqual(results, [])

    def test_boundary_value_strict_greater(self):
        """边界值（恰好等于阈值）不应触发（v > th 严格大于）"""
        # 炉膛温度 danger_threshold=1050，值=1050 不应触发
        results = detect_threshold(
            {"furnace_temperature": 1050.0}, self.thresholds
        )
        self.assertEqual(results, [])


class TestThresholdDetectorBatch(unittest.TestCase):
    """批量检测测试"""

    def setUp(self):
        self.thresholds = load_thresholds_from_yaml(THRESHOLDS_YAML)

    def test_batch_on_normal_data(self):
        """正常30天数据上批量检测——应有极少量或零超限值"""
        normal_csv = os.path.join(SAMPLE_DATA_DIR, "normal_30days.csv")
        df = pd.read_csv(normal_csv, low_memory=False)
        results = detect_threshold_batch(df, self.thresholds)
        # 正常数据理论上应无超限（数据生成时已 clip 到 normal_range）
        # 但 normal_range 与 danger_threshold 之间有缓冲区，不会超
        # 允许极少量误检（<0.1%）
        rate = len(results) / len(df)
        self.assertLess(rate, 0.001,
                        f"正常数据误检率 {rate:.4%} 过高（共 {len(results)} 条）")

    def test_batch_100_percent_recall_on_bearing_fault(self):
        """验收标准：轴承过热故障中所有超限值必须 100% 检出"""
        fault_csv = os.path.join(SAMPLE_DATA_DIR, "fault_bearing_overheat.csv")
        df = pd.read_csv(fault_csv, low_memory=False)

        results = detect_threshold_batch(df, self.thresholds)

        # 1. 找出 DataFrame 中所有"应当被检出"的超限点（ground truth）
        norm = _normalize_thresholds(self.thresholds)
        from models.threshold_detector import THRESHOLD_TYPES
        expected_violations = []
        for idx, row in df.iterrows():
            for param, param_th in norm.items():
                if param not in df.columns:
                    continue
                try:
                    v = float(row[param])
                except (TypeError, ValueError):
                    continue
                if v != v:
                    continue
                for type_key, direction, level in THRESHOLD_TYPES:
                    if type_key not in param_th:
                        continue
                    th = float(param_th[type_key])
                    if (direction == "upper" and v > th) or \
                       (direction == "lower" and v < th):
                        expected_violations.append((idx, param))
                        break  # 同参数一条记录只算一次

        # 2. 检测结果中的 (idx, param) 集合
        detected_set = {(r["row_idx"], r["param"]) for r in results}

        # 3. 验证 100% 检出
        missed = [(idx, p) for idx, p in expected_violations
                  if (idx, p) not in detected_set]
        self.assertEqual(
            len(missed), 0,
            f"阈值检测未 100% 检出：应检出 {len(expected_violations)} 条，"
            f"实际检出 {len(expected_violations) - len(missed)} 条，"
            f"漏检 {len(missed)} 条: {missed[:5]}"
        )

    def test_batch_100_percent_recall_on_all_faults(self):
        """验收标准：3类故障数据中所有超限值必须 100% 检出"""
        fault_files = [
            "fault_bearing_overheat.csv",
            "fault_emission_exceed.csv",
            "fault_grate_jam.csv",
        ]

        for fname in fault_files:
            fpath = os.path.join(SAMPLE_DATA_DIR, fname)
            df = pd.read_csv(fpath, low_memory=False)

            results = detect_threshold_batch(df, self.thresholds)

            norm = _normalize_thresholds(self.thresholds)
            from models.threshold_detector import THRESHOLD_TYPES
            expected_violations = []
            for idx, row in df.iterrows():
                for param, param_th in norm.items():
                    if param not in df.columns:
                        continue
                    try:
                        v = float(row[param])
                    except (TypeError, ValueError):
                        continue
                    if v != v:
                        continue
                    for type_key, direction, level in THRESHOLD_TYPES:
                        if type_key not in param_th:
                            continue
                        th = float(param_th[type_key])
                        if (direction == "upper" and v > th) or \
                           (direction == "lower" and v < th):
                            expected_violations.append((idx, param))
                            break

            detected_set = {(r["row_idx"], r["param"]) for r in results}
            missed = [(idx, p) for idx, p in expected_violations
                      if (idx, p) not in detected_set]

            self.assertEqual(
                len(missed), 0,
                f"[{fname}] 阈值检测未 100% 检出：应检出 {len(expected_violations)} 条，"
                f"漏检 {len(missed)} 条: {missed[:5]}"
            )


class TestThresholdConfigLoading(unittest.TestCase):
    """配置加载与归一化测试"""

    def test_load_from_yaml(self):
        """从 yaml 加载应得到扁平 dict"""
        th = load_thresholds_from_yaml(THRESHOLDS_YAML)
        self.assertIsInstance(th, dict)
        self.assertIn("furnace_temperature", th)
        self.assertIn("danger_threshold", th["furnace_temperature"])
        self.assertEqual(th["furnace_temperature"]["danger_threshold"], 1050)

    def test_normalize_full_yaml(self):
        """归一化完整 yaml 格式"""
        full = {"parameters": [
            {"name": "x", "danger_threshold": 100, "unit": "°C"},
            {"name": "y", "warning_threshold": 50},
        ]}
        norm = _normalize_thresholds(full)
        self.assertEqual(norm["x"]["danger_threshold"], 100)
        self.assertEqual(norm["x"]["unit"], "°C")
        self.assertEqual(norm["y"]["warning_threshold"], 50)

    def test_normalize_flat_dict(self):
        """归一化扁平 dict 应原样返回"""
        flat = {"x": {"danger_threshold": 100}}
        norm = _normalize_thresholds(flat)
        self.assertEqual(norm, flat)


if __name__ == "__main__":
    unittest.main(verbosity=2)
