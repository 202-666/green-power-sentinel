import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.pipeline import SentinelPipeline
from core.data_cleaner import PARAM_COLUMNS
from feishu.bitable_client import BitableClient as RealBitableClient

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestPipelineInit(unittest.TestCase):
    def test_pipeline_init(self):
        pipeline = SentinelPipeline()
        self.assertIsNotNone(pipeline.config)


class TestPipelineOfflineW4(unittest.TestCase):
    """W4 离线模式联调测试"""

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
        cls.pipeline = SentinelPipeline(config_path=cls.config_path)

    def test_normal_data_zero_false_positive(self):
        """正常30天数据应0误报"""
        data_path = os.path.join(PROJECT_ROOT, "data", "sample_data", "normal_30days.csv")
        result = self.pipeline.run_full_pipeline(data_path, "INC-01-BRG-01")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["records_collected"], 43200)
        self.assertEqual(result["anomalies_detected"], 0)

        summary = self.pipeline.get_detection_summary()
        self.assertEqual(summary["total_rows"], 43200)
        self.assertEqual(len(summary["anomalies"]), 0)

        # 验证所有评分为 green
        scores = summary["scores"]
        self.assertEqual(len(scores), 43200)
        non_green = [s for s in scores if s["level"] != "green"]
        self.assertEqual(len(non_green), 0, f"正常数据出现 {len(non_green)} 个非 green 评分")

    def test_fault_bearing_overheat_detected(self):
        """轴承过热故障数据应检出异常"""
        data_path = os.path.join(PROJECT_ROOT, "data", "sample_data", "fault_bearing_overheat.csv")
        result = self.pipeline.run_full_pipeline(data_path, "INC-01-BRG-01")

        self.assertEqual(result["status"], "success")
        self.assertGreater(result["anomalies_detected"], 0)

        summary = self.pipeline.get_detection_summary()
        anomalies = summary["anomalies"]
        self.assertGreater(len(anomalies), 0)

        # 验证异常覆盖故障区间（第15天 10:00 ~ 12:00，即 row 20760 ~ 20880）
        row_indices = [a["row_idx"] for a in anomalies]
        self.assertTrue(
            any(20760 <= idx <= 20880 for idx in row_indices),
            "异常应覆盖轴承过热故障区间"
        )

        # 验证 primary_cause 已填充
        for a in anomalies[:5]:
            self.assertTrue(a["primary_cause"], "primary_cause 不应为空")

    def test_detection_summary_interface(self):
        """get_detection_summary 公共接口应返回结构化数据"""
        data_path = os.path.join(PROJECT_ROOT, "data", "sample_data", "normal_30days.csv")
        self.pipeline.run_full_pipeline(data_path, "INC-01-BRG-01")
        summary = self.pipeline.get_detection_summary()

        self.assertIn("scores", summary)
        self.assertIn("anomalies", summary)
        self.assertIn("total_rows", summary)
        self.assertIn("columns", summary)
        self.assertIsInstance(summary["scores"], list)
        self.assertIsInstance(summary["anomalies"], list)
        self.assertIsInstance(summary["total_rows"], int)


class TestBitableCollection(unittest.TestCase):
    """H1：Bitable 读取分支（mock 网络层，不发起真实请求）"""

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")

    def _pipeline_with_bitable_config(self, feishu_cfg: dict):
        pipeline = SentinelPipeline(config_path=self.config_path)
        pipeline.config["data_source"] = {"type": "bitable"}
        pipeline.config["feishu"] = feishu_cfg
        return pipeline

    @patch("feishu.bitable_client.BitableClient")
    def test_bitable_branch_no_file_not_found(self, mock_client_cls):
        """在线模式走 Bitable 分支，不应再抛 FileNotFoundError"""
        # 仅 mock 网络层，保留真实字段展平逻辑
        mock_client_cls.flatten_fields = RealBitableClient.flatten_fields
        mock_client = mock_client_cls.return_value
        mock_client.get_records.side_effect = [
            {
                "code": 0,
                "data": {
                    "items": [
                        {
                            "fields": {
                                "timestamp": 1782921600000,  # 2026-07-01 08:00:00 (+08)
                                "furnace_temperature": 950.5,
                                "device_id": "INC-01-BRG-01",
                                "device_name": "1#焚烧炉-引风机-前轴承",
                                "device_type": "焚烧炉",
                            }
                        }
                    ],
                    "has_more": False,
                },
            }
        ]
        pipeline = self._pipeline_with_bitable_config(
            {
                "app_id": "cli_x",
                "app_secret": "secret",
                "app_token": "appX",
                "tables": {"runtime_data": "tblRuntime"},
            }
        )

        n = pipeline.run_data_collection("bitable")
        self.assertEqual(n, 1)
        self.assertIsNotNone(pipeline._cached_df)
        # 飞书日期字段（毫秒时间戳）应转为字符串，clean_data 可正常解析
        self.assertRegex(
            str(pipeline._cached_df["timestamp"].iloc[0]),
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
        )
        self.assertEqual(pipeline._cached_df["furnace_temperature"].iloc[0], 950.5)

    def test_bitable_missing_config_raises_clear_error(self):
        """缺配置时应给出明确 RuntimeError，而不是 FileNotFoundError"""
        pipeline = self._pipeline_with_bitable_config(
            {
                "app_id": "",
                "app_secret": "",
                "app_token": "",
                "tables": {"runtime_data": ""},
            }
        )
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.run_data_collection("bitable")
        self.assertIn("FEISHU_APP_ID", str(ctx.exception))
        self.assertIn("FEISHU_TABLE_RUNTIME", str(ctx.exception))

    def test_bitable_field_flatten(self):
        """单选/多选对象应展平为纯文本，不臆造字段名"""
        import datetime

        from core.pipeline import SentinelPipeline

        pipeline = SentinelPipeline(config_path=self.config_path)
        fields = {
            "timestamp": 1782921600000,
            "device_name": {"text": "1#焚烧炉"},
            "data_quality_flag": {"text": "正常"},
            "qc_note": "",
        }
        flat = pipeline._bitable_fields_to_runtime(fields)
        self.assertEqual(flat["device_name"], "1#焚烧炉")
        self.assertEqual(flat["data_quality_flag"], "正常")
        expected = datetime.datetime.fromtimestamp(
            1782921600000 / 1000.0
        ).strftime("%Y-%m-%d %H:%M:%S")
        self.assertEqual(flat["timestamp"], expected)


class TestIncrementalDetectionIsolation(unittest.TestCase):
    """H2：增量检测逐模块 try/except，任一模块异常不中断增量流程"""

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")

    def setUp(self):
        self.pipeline = SentinelPipeline(config_path=self.config_path)
        # 小规模合成数据（13 参数列 + 元数据），避免全量 43200 行耗时
        n = 500
        rng = np.random.default_rng(42)
        data = {
            "timestamp": pd.date_range("2026-07-01", periods=n, freq="1min"),
            "device_id": "INC-01-BRG-01",
            "device_name": "1#焚烧炉-引风机-前轴承",
            "device_type": "焚烧炉",
            "data_quality_flag": "正常",
            "qc_note": "",
        }
        for col in PARAM_COLUMNS:
            data[col] = 100.0 + rng.normal(0, 1.0, n)
        self.csv_path = os.path.join(
            tempfile.gettempdir(), "gps_incremental_isolation_test.csv"
        )
        pd.DataFrame(data).to_csv(self.csv_path, index=False)

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_threshold_module_failure_does_not_abort(self):
        with patch.object(
            self.pipeline, "_detect_threshold_batch_func",
            side_effect=RuntimeError("threshold boom"),
        ):
            result = self.pipeline.run_incremental_detection(
                self.csv_path, "INC-01-BRG-01", last_row_idx=0
            )
        self.assertIn("new_rows", result)
        self.assertIn("last_row_idx", result)
        self.assertEqual(result["last_row_idx"], 500)

    def test_trend_module_failure_does_not_abort(self):
        with patch.object(
            self.pipeline, "_detect_trend_multi_func",
            side_effect=RuntimeError("trend boom"),
        ):
            result = self.pipeline.run_incremental_detection(
                self.csv_path, "INC-01-BRG-01", last_row_idx=0
            )
        self.assertIn("new_rows", result)
        self.assertEqual(result["last_row_idx"], 500)


class TestYellowTrackerPersistence(unittest.TestCase):
    """L7：黄色升级计数器持久化，进程重启不清零"""

    def setUp(self):
        probe = SentinelPipeline()
        self.state_dir = os.path.dirname(probe._yellow_tracker_path())
        if os.path.exists(self.state_dir):
            shutil.rmtree(self.state_dir)

    def tearDown(self):
        if os.path.exists(self.state_dir):
            shutil.rmtree(self.state_dir)

    def test_tracker_persists_across_instances(self):
        p1 = SentinelPipeline()
        p1._yellow_tracker["INC-01-BRG-01_轴承过热"] = 3
        p1._save_yellow_tracker()

        p2 = SentinelPipeline()
        self.assertEqual(p2._yellow_tracker.get("INC-01-BRG-01_轴承过热"), 3)

    def test_corrupt_state_file_falls_back_to_empty(self):
        os.makedirs(self.state_dir, exist_ok=True)
        with open(
            os.path.join(self.state_dir, "yellow_tracker.json"), "w", encoding="utf-8"
        ) as f:
            f.write("{not-json")
        pipeline = SentinelPipeline()
        self.assertEqual(pipeline._yellow_tracker, {})


if __name__ == "__main__":
    unittest.main()
