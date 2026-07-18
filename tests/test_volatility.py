"""
绿电哨兵 — 波动率检测单元测试

验收标准：
1. 炉排卡滞前期（grate_speed 波动增大）应被检出
2. 轴承过热（bearing_vibration 噪声增大）应被检出
3. 正常数据误报 < 5次/30天
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.volatility_detector import (
    detect_volatility,
    detect_volatility_batch,
    detect_volatility_multi_params,
    DEFAULT_MULTIPLIERS,
    DEFAULT_CURRENT_WINDOW,
)

SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_data")


class TestVolatilityBasic(unittest.TestCase):
    """基础功能测试"""

    def test_constant_series_no_detection(self):
        """常数序列不应检出"""
        series = pd.Series(np.full(3000, 50.0), name="test_param")
        result = detect_volatility(series)
        self.assertFalse(result["detected"])
        self.assertIsNone(result["level"])

    def test_volatile_series_detected(self):
        """波动增大序列应被检出"""
        np.random.seed(42)
        stable = np.random.normal(50, 1.0, 2000)
        volatile = np.random.normal(50, 6.0, 30)
        series = pd.Series(np.concatenate([stable, volatile]), name="test_param")
        result = detect_volatility(series)
        self.assertTrue(result["detected"])
        self.assertIsNotNone(result["level"])
        self.assertGreaterEqual(result["ratio"], DEFAULT_MULTIPLIERS["yellow"])

    def test_insufficient_data(self):
        """数据不足应返回 reason"""
        series = pd.Series(np.array([1.0, 2.0, 3.0]), name="test_param")
        result = detect_volatility(series)
        self.assertFalse(result["detected"])
        self.assertIn("reason", result)

    def test_batch_shape_matches_input(self):
        """批量检测输出长度应与输入一致"""
        np.random.seed(42)
        series = pd.Series(np.random.randn(500), name="test_param")
        result_df = detect_volatility_batch(series)
        self.assertEqual(len(result_df), 500)
        self.assertIn("detected", result_df.columns)
        self.assertIn("ratio", result_df.columns)

    def test_batch_detects_volatile_segment(self):
        """批量检测应识别出波动增大段"""
        np.random.seed(42)
        stable = np.random.normal(50, 1.0, 300)
        volatile = np.random.normal(50, 6.0, 50)
        series = pd.Series(np.concatenate([stable, volatile]), name="test_param")

        result = detect_volatility_batch(series, current_window=DEFAULT_CURRENT_WINDOW, baseline_window=100)

        # 平稳段（除去预热期）应几乎无检出
        warmup = DEFAULT_CURRENT_WINDOW * 3 + 100
        detected_in_flat = result["detected"].iloc[warmup:300].sum()
        self.assertLess(detected_in_flat, 5, f"平稳段不应检出，实际 {detected_in_flat}")

        # 波动段应大量检出
        detected_in_volatile = result["detected"].iloc[330:].sum()
        self.assertGreater(
            detected_in_volatile, 10, f"波动段应大量检出，实际 {detected_in_volatile}"
        )


class TestFaultDetection(unittest.TestCase):
    """故障检出率测试"""

    @classmethod
    def setUpClass(cls):
        from core.data_cleaner import clean_data

        cls.faults = {}
        cls.fault_ranges = {}
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

    def test_grate_jam_volatility_detection(self):
        """炉排卡滞：grate_speed 波动增大应被检出"""
        df = self.faults["grate_jam"]
        mask = self.fault_ranges["grate_jam"]

        vol_results = detect_volatility_multi_params(
            df, ["grate_speed", "feed_rate", "furnace_pressure"]
        )
        grate_vol = vol_results["grate_speed"]

        fault_detected = grate_vol.loc[mask, "detected"].sum()
        fault_total = mask.sum()
        detection_rate = fault_detected / fault_total if fault_total > 0 else 0.0

        print(
            f"\n[炉排卡滞] grate_speed 波动检出率: {detection_rate:.1%} ({fault_detected}/{fault_total})"
        )
        print(f"  故障区间平均 ratio: {grate_vol.loc[mask, 'ratio'].mean():.2f}")

        self.assertGreater(
            detection_rate, 0.30, f"炉排卡滞波动检出率 {detection_rate:.1%} 过低"
        )

    def test_bearing_overheat_volatility_detection(self):
        """轴承过热：bearing_vibration 噪声增大应被检出"""
        df = self.faults["bearing_overheat"]
        mask = self.fault_ranges["bearing_overheat"]

        vol_results = detect_volatility_multi_params(
            df, ["bearing_vibration", "bearing_temperature"]
        )
        vib_vol = vol_results["bearing_vibration"]

        fault_detected = vib_vol.loc[mask, "detected"].sum()
        fault_total = mask.sum()
        detection_rate = fault_detected / fault_total if fault_total > 0 else 0.0

        print(
            f"\n[轴承过热] bearing_vibration 波动检出率: {detection_rate:.1%} ({fault_detected}/{fault_total})"
        )
        print(f"  故障区间平均 ratio: {vib_vol.loc[mask, 'ratio'].mean():.2f}")

        self.assertGreater(
            detection_rate, 0.05, f"轴承过热波动检出率 {detection_rate:.1%} 过低"
        )


class TestFalsePositiveControl(unittest.TestCase):
    """误报控制测试"""

    def test_normal_data_false_positives(self):
        """正常30天数据波动率误报应<5次"""
        normal_csv = os.path.join(SAMPLE_DATA_DIR, "normal_30days.csv")
        df = pd.read_csv(normal_csv, low_memory=False)
        from core.data_cleaner import clean_data

        df = clean_data(df)

        # 仅测试关键参数以加速
        key_params = [
            "grate_speed",
            "bearing_vibration",
            "furnace_temperature",
            "so2_concentration",
            "nox_concentration",
        ]
        vol_results = detect_volatility_multi_params(df, key_params)

        any_detected = np.zeros(len(df), dtype=bool)
        for col, vdf in vol_results.items():
            any_detected |= vdf["detected"].to_numpy()

        fp_count = int(any_detected.sum())
        fp_rate = fp_count / len(df)

        print(f"\n[正常数据] 波动率误报: {fp_count}次 ({fp_rate:.4%})")

        self.assertLess(
            fp_count, 5, f"正常数据波动率误报 {fp_count}次，应<5次/30天"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
