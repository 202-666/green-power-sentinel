"""M4：最小调度器测试（once=True 单轮执行，不进入 while True 循环）"""

import os
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.pipeline import SentinelPipeline
from core.scheduler import run_scheduler


class TestSchedulerOnce(unittest.TestCase):
    """单轮调度：采集 + 检测 + 按 agent3_mode 推送"""

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")

    def test_once_cycle_runs_and_returns_summary(self):
        pipeline = SentinelPipeline(config_path=self.config_path)
        with patch.object(pipeline, "run_data_collection", return_value=42) as m_collect, \
             patch.object(pipeline, "run_anomaly_detection", return_value=[]) as m_detect, \
             patch.object(pipeline, "run_alert_push", return_value=True) as m_push:
            summary = run_scheduler(
                pipeline, data_source="dummy.csv", device_id="DEV-1", once=True
            )
        self.assertEqual(summary["cycles"], 1)
        self.assertEqual(summary["records_collected"], 42)
        self.assertEqual(summary["anomalies_detected"], 0)
        m_collect.assert_called_once_with("dummy.csv")
        m_detect.assert_called_once_with("DEV-1")
        m_push.assert_not_called()

    def test_realtime_mode_pushes_anomalies(self):
        pipeline = SentinelPipeline(config_path=self.config_path)
        alerts = [{"alert_id": "ALT_1", "risk_level": "red"}]
        with patch.object(pipeline, "run_data_collection", return_value=1), \
             patch.object(pipeline, "run_anomaly_detection", return_value=alerts), \
             patch.object(pipeline, "run_alert_push", return_value=True) as m_push:
            summary = run_scheduler(
                pipeline, data_source="dummy.csv", device_id="DEV-1", once=True
            )
        self.assertEqual(summary["anomalies_detected"], 1)
        self.assertEqual(summary["alerts_pushed"], 1)
        m_push.assert_called_once_with(alerts[0])


if __name__ == "__main__":
    unittest.main()
