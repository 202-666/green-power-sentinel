import os
import unittest
from core.pipeline import SentinelPipeline

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


if __name__ == "__main__":
    unittest.main()