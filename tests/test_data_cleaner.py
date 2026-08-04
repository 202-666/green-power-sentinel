"""
绿电哨兵 — 数据清洗单元测试

测试覆盖：
1. 去重（同一timestamp保留最后一条）
2. 时间对齐（四舍五入到整分钟）
3. 缺失值线性插值
4. 物理范围截断（标记不删除）
5. 严重缺失处理（>50%参数缺失跳过）
6. 传感器健康检查（卡死/漂移检测）
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data_cleaner import (
    clean_data,
    check_sensor_health,
    PARAM_COLUMNS,
    PHYSICAL_RANGES,
)


class TestDataCleanerBasic(unittest.TestCase):
    """基础清洗功能测试"""

    def test_drop_duplicate_timestamp(self):
        """同一timestamp仅保留最后一条"""
        data = {
            "timestamp": ["2026-07-01 10:00:00", "2026-07-01 10:00:00", "2026-07-01 10:01:00"],
            "furnace_temperature": [950.0, 960.0, 955.0],
        }
        df = pd.DataFrame(data)
        cleaned = clean_data(df)
        self.assertEqual(len(cleaned), 2)
        # 保留最后一条（960.0）
        self.assertEqual(cleaned["furnace_temperature"].iloc[0], 960.0)

    def test_dedup_after_sort_keeps_latest_actual_time(self):
        """L6：去重前先按时间排序，同一（取整后）分钟保留实际时间最新的一条"""
        data = {
            "timestamp": ["2026-07-01 10:00:00", "2026-07-01 10:01:00", "2026-07-01 10:00:59"],
            "furnace_temperature": [950.0, 960.0, 955.0],
        }
        df = pd.DataFrame(data)
        cleaned = clean_data(df)
        self.assertEqual(len(cleaned), 2)
        # 10:00:59 与 10:01:00 都取整到 10:01，应保留实际时间更晚的 10:01:00（960.0）
        row_1001 = cleaned[cleaned["timestamp"] == pd.Timestamp("2026-07-01 10:01:00")]
        self.assertEqual(row_1001["furnace_temperature"].iloc[0], 960.0)

    def test_time_alignment(self):
        """时间戳对齐到整分钟"""
        data = {
            "timestamp": ["2026-07-01 10:00:31", "2026-07-01 10:00:29"],
            "furnace_temperature": [950.0, 960.0],
        }
        df = pd.DataFrame(data)
        cleaned = clean_data(df)
        # 31秒进上去为 10:01，29秒舍去为 10:00
        self.assertEqual(str(cleaned["timestamp"].iloc[0]), "2026-07-01 10:00:00")
        self.assertEqual(str(cleaned["timestamp"].iloc[1]), "2026-07-01 10:01:00")

    def test_linear_interpolation(self):
        """缺失值线性插值"""
        data = {
            "timestamp": ["2026-07-01 10:00:00", "2026-07-01 10:01:00", "2026-07-01 10:02:00"],
            "furnace_temperature": [100.0, np.nan, 200.0],
            "flue_gas_temperature": [200.0, 210.0, 220.0],
            "steam_pressure": [3.8, 3.85, 3.9],
        }
        df = pd.DataFrame(data)
        cleaned = clean_data(df)
        self.assertFalse(cleaned["furnace_temperature"].isna().any())
        self.assertEqual(cleaned["furnace_temperature"].iloc[1], 150.0)

    def test_physical_range_mark(self):
        """超出物理范围的值应标记为异常但不删除"""
        data = {
            "timestamp": ["2026-07-01 10:00:00"],
            "furnace_temperature": [3000.0],  # 超出物理范围 [-50, 2000]
        }
        df = pd.DataFrame(data)
        cleaned = clean_data(df)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned["data_quality_flag"].iloc[0], "异常")
        self.assertIn("超出物理范围", cleaned["qc_note"].iloc[0])

    def test_severe_missing_skip(self):
        """超过50%参数缺失的记录应跳过"""
        data = {
            "timestamp": ["2026-07-01 10:00:00", "2026-07-01 10:01:00"],
            "furnace_temperature": [950.0, np.nan],
            "flue_gas_temperature": [210.0, np.nan],
            "steam_pressure": [3.8, np.nan],
            "steam_flow": [52.0, np.nan],
            "bearing_vibration": [2.5, np.nan],
            "bearing_temperature": [50.0, np.nan],
            "so2_concentration": [60.0, np.nan],
        }
        df = pd.DataFrame(data)
        cleaned = clean_data(df)
        # 第2条记录所有参数列都缺失（7/7=100% > 50%），应被跳过
        self.assertEqual(len(cleaned), 1)

    def test_not_severe_missing_kept(self):
        """未超过50%参数缺失的记录应保留"""
        data = {
            "timestamp": ["2026-07-01 10:00:00"],
            "furnace_temperature": [np.nan],
            "flue_gas_temperature": [np.nan],
            "steam_pressure": [3.8],
            "steam_flow": [52.0],
            "bearing_vibration": [2.5],
            "bearing_temperature": [50.0],
            "so2_concentration": [60.0],
        }
        df = pd.DataFrame(data)
        cleaned = clean_data(df)
        # 2/7≈28.6% < 50%，应保留
        self.assertEqual(len(cleaned), 1)

    def test_empty_df(self):
        """空DataFrame应返回空DataFrame"""
        df = pd.DataFrame(columns=["timestamp"] + PARAM_COLUMNS[:3])
        cleaned = clean_data(df)
        self.assertTrue(cleaned.empty)


class TestSensorHealthCheck(unittest.TestCase):
    """传感器健康检查"""

    def test_stuck_detection(self):
        """连续10个点值完全不变应标记为卡死"""
        data = {
            "timestamp": pd.date_range("2026-07-01", periods=15, freq="1min"),
            "furnace_temperature": [950.0] * 15,
        }
        df = pd.DataFrame(data)
        flags = check_sensor_health(df, window=10)
        # 从第10个点开始（索引9）应标记为卡死
        self.assertEqual(flags.iloc[9], "卡死")
        self.assertEqual(flags.iloc[14], "卡死")

    def test_not_stuck(self):
        """有波动但无趋势的序列不应标记为卡死或漂移"""
        np.random.seed(42)
        data = {
            "timestamp": pd.date_range("2026-07-01", periods=20, freq="1min"),
            "furnace_temperature": 950.0 + np.random.randn(20) * 2.0,
        }
        df = pd.DataFrame(data)
        flags = check_sensor_health(df, window=10)
        self.assertTrue((flags == "正常").all())

    def test_drift_detection(self):
        """单调变化超过3σ应标记为漂移"""
        np.random.seed(42)
        # 前半段平稳（正常波动），后半段单调递增（漂移）
        # 这样 rolling_sigma 能正确反映正常波动水平
        stable_part = np.full(10, 950.0) + np.random.randn(10) * 2.0
        drift_part = 950.0 + np.arange(10) * 10.0  # 每分钟上升10°C
        data = {
            "timestamp": pd.date_range("2026-07-01", periods=20, freq="1min"),
            "furnace_temperature": np.concatenate([stable_part, drift_part]),
        }
        df = pd.DataFrame(data)
        flags = check_sensor_health(df, window=10)
        # 后半段应有漂移标记
        drift_count = (flags == "漂移").sum()
        self.assertGreater(drift_count, 3, f"漂移检测应标记多个点，实际 {drift_count}")

    def test_short_series(self):
        """数据不足窗口大小不应触发检测"""
        data = {
            "timestamp": pd.date_range("2026-07-01", periods=5, freq="1min"),
            "furnace_temperature": [950.0] * 5,
        }
        df = pd.DataFrame(data)
        flags = check_sensor_health(df, window=10)
        self.assertTrue((flags == "正常").all())


class TestCleanerIntegration(unittest.TestCase):
    """集成测试：使用真实样本数据"""

    def test_clean_real_normal_data(self):
        """清洗正常30天数据不应报错"""
        normal_path = os.path.join(PROJECT_ROOT, "data", "sample_data", "normal_30days.csv")
        df = pd.read_csv(normal_path, low_memory=False)
        cleaned = clean_data(df)
        self.assertEqual(len(cleaned), len(df))
        self.assertIn("data_quality_flag", cleaned.columns)
        self.assertIn("qc_note", cleaned.columns)

    def test_clean_real_fault_data(self):
        """清洗故障数据应保留故障标记"""
        fault_path = os.path.join(PROJECT_ROOT, "data", "sample_data", "fault_bearing_overheat.csv")
        df = pd.read_csv(fault_path, low_memory=False)
        cleaned = clean_data(df)
        # 故障注入标记不应被覆盖
        fault_mask = cleaned["data_quality_flag"] == "故障注入"
        self.assertGreater(fault_mask.sum(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
