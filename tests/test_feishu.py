"""
W5 飞书API集成 + Agent 3 预警推送 — 验收测试

验收标准（技术框架 §9 W5）：
1. 预警事件写入多维表格（bitable_client.py 字段映射 + 批量写入）
2. 消息卡片推送成功（message_sender.py 标准卡片 + 风险分级策略）

测试覆盖：
- BitableClient.alert_to_fields 字段映射正确性
- MessageSender.create_alert_card 飞书标准卡片结构
- MessageSender.send_alert 风险分级推送策略（红/橙即时、黄色跳过）
- SentinelPipeline.run_alert_push 离线模式降级（卡片写入本地）
- 全链路：故障数据 → 异常检测 → 预警推送 → 本地卡片文件生成
"""

import json
import os
import shutil
import sys
import unittest
from unittest.mock import patch, MagicMock

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feishu.bitable_client import BitableClient, RISK_LEVEL_CN, ALERT_STATUS_CN
from feishu.message_sender import (
    MessageSender,
    RISK_CARD_TEMPLATE,
    INSTANT_PUSH_LEVELS,
)


def _make_alert(risk_level="red", **overrides):
    """构造一个标准的预警事件字典（来自 pipeline.run_anomaly_detection 输出）"""
    alert = {
        "alert_id": "ALT_20260715_100000_001",
        "trigger_time": "2026-07-15 10:00:00",
        "device_id": "INC-01-BRG-01",
        "device_name": "1#焚烧炉-引风机-前轴承",
        "fault_type": "轴承过热",
        "risk_level": risk_level,
        "confidence": 0.85,
        "score": 0.82,
        "detection_method": "阈值检测+趋势检测",
        "abnormal_params": "bearing_temperature, bearing_vibration",
        "param_values": {"bearing_temperature": 78.5, "bearing_vibration": 6.2},
        "alert_status": "待推送",
        "primary_cause": "趋势异常",
        "row_idx": 20760,
    }
    alert.update(overrides)
    return alert


# ============================================================
# 1. BitableClient.alert_to_fields 字段映射测试
# ============================================================

class TestAlertToFields(unittest.TestCase):
    """测试预警事件字典 → 飞书多维表格 fields 的映射"""

    def test_risk_level_mapping(self):
        """风险等级英文应正确映射为中文单选值"""
        for en, cn in [("red", "红"), ("orange", "橙"), ("yellow", "黄")]:
            fields = BitableClient.alert_to_fields(_make_alert(risk_level=en))
            self.assertEqual(fields["risk_level"], cn, f"{en} 应映射为 {cn}")

    def test_param_values_to_json(self):
        """param_values dict 应转为 JSON 字符串"""
        fields = BitableClient.alert_to_fields(_make_alert())
        self.assertIsInstance(fields["param_values"], str)
        parsed = json.loads(fields["param_values"])
        self.assertEqual(parsed["bearing_temperature"], 78.5)

    def test_timestamp_to_millis(self):
        """trigger_time 字符串应转为毫秒时间戳"""
        fields = BitableClient.alert_to_fields(_make_alert())
        self.assertIsInstance(fields["trigger_time"], int)
        # 2026-07-15 10:00:00 的毫秒时间戳应大于 1.7e12
        self.assertGreater(fields["trigger_time"], 1_700_000_000_000)

    def test_default_alert_status(self):
        """缺省 alert_status 应置为「待推送」"""
        alert = _make_alert()
        del alert["alert_status"]
        fields = BitableClient.alert_to_fields(alert)
        self.assertEqual(fields["alert_status"], "待推送")

    def test_alert_status_english_to_chinese(self):
        """英文状态应映射为中文"""
        fields = BitableClient.alert_to_fields(_make_alert(alert_status="pushed"))
        self.assertEqual(fields["alert_status"], "已推送")

    def test_required_fields_present(self):
        """多维表格「预警事件」表实际存在的必填字段都应存在"""
        fields = BitableClient.alert_to_fields(_make_alert())
        required = [
            "alert_id", "trigger_time", "device_id",
            "fault_type", "risk_level", "confidence", "detection_method",
            "abnormal_params", "param_values", "alert_status",
        ]
        for key in required:
            self.assertIn(key, fields, f"必填字段 {key} 缺失")

    def test_primary_cause_appended_to_method(self):
        """根因信息应追加到 detection_method 字段"""
        fields = BitableClient.alert_to_fields(_make_alert(primary_cause="趋势异常"))
        self.assertIn("根因", fields["detection_method"])
        self.assertIn("趋势异常", fields["detection_method"])

    def test_confidence_is_float(self):
        """confidence 应为浮点数（多维表格数字字段要求）"""
        fields = BitableClient.alert_to_fields(_make_alert(confidence="0.85"))
        self.assertIsInstance(fields["confidence"], float)


# ============================================================
# 2. MessageSender.create_alert_card 卡片结构测试
# ============================================================

class TestCreateAlertCard(unittest.TestCase):
    """测试飞书标准 interactive 消息卡片生成"""

    def setUp(self):
        # 不需要真实凭据，仅用卡片生成方法
        self.sender = MessageSender.__new__(MessageSender)

    def test_card_has_standard_structure(self):
        """卡片应包含 config / header / elements 三大顶层字段"""
        card = self.sender.create_alert_card(_make_alert())
        self.assertIn("config", card)
        self.assertIn("header", card)
        self.assertIn("elements", card)

    def test_header_template_matches_level(self):
        """header.template 应与风险等级颜色对应"""
        for level, template in [("red", "red"), ("orange", "orange"), ("yellow", "yellow")]:
            card = self.sender.create_alert_card(_make_alert(risk_level=level))
            self.assertEqual(card["header"]["template"], template)

    def test_header_title_contains_device_and_fault(self):
        """卡片标题应包含设备名和故障类型"""
        card = self.sender.create_alert_card(_make_alert())
        title_content = card["header"]["title"]["content"]
        self.assertIn("1#焚烧炉-引风机-前轴承", title_content)
        self.assertIn("轴承过热", title_content)

    def test_red_card_has_action_buttons(self):
        """红色预警卡片应包含「确认预警」「标记误报」操作按钮"""
        card = self.sender.create_alert_card(_make_alert(risk_level="red"))
        action_blocks = [e for e in card["elements"] if e.get("tag") == "action"]
        self.assertEqual(len(action_blocks), 1)
        actions = action_blocks[0]["actions"]
        contents = [a["text"]["content"] for a in actions]
        self.assertTrue(any("确认预警" in c for c in contents))
        self.assertTrue(any("标记误报" in c for c in contents))

    def test_orange_card_has_action_buttons(self):
        """橙色预警卡片也应包含操作按钮"""
        card = self.sender.create_alert_card(_make_alert(risk_level="orange"))
        action_blocks = [e for e in card["elements"] if e.get("tag") == "action"]
        self.assertEqual(len(action_blocks), 1)

    def test_yellow_card_no_action_buttons(self):
        """黄色预警卡片不应包含操作按钮（仅汇总至日报）"""
        card = self.sender.create_alert_card(_make_alert(risk_level="yellow"))
        action_blocks = [e for e in card["elements"] if e.get("tag") == "action"]
        self.assertEqual(len(action_blocks), 0)

    def test_card_contains_key_info(self):
        """卡片正文应包含设备编号、置信度、异常参数等关键信息"""
        card = self.sender.create_alert_card(_make_alert())
        all_text = json.dumps(card, ensure_ascii=False)
        self.assertIn("INC-01-BRG-01", all_text)
        self.assertIn("85%", all_text)  # 置信度 0.85 → 85%
        self.assertIn("bearing_temperature", all_text)

    def test_card_uses_lark_md(self):
        """卡片文本应使用 lark_md 富文本标签"""
        card = self.sender.create_alert_card(_make_alert())
        div_blocks = [e for e in card["elements"] if e.get("tag") == "div"]
        self.assertTrue(any(d["text"]["tag"] == "lark_md" for d in div_blocks))


# ============================================================
# 3. MessageSender.send_alert 风险分级推送策略测试
# ============================================================

class TestSendAlertStrategy(unittest.TestCase):
    """测试风险分级推送策略（mock 网络层）"""

    def setUp(self):
        self.sender = MessageSender.__new__(MessageSender)

    def _mock_send_card(self, ret=None):
        """mock send_card_message，返回成功响应"""
        ret = ret or {"code": 0, "data": {"message_id": "om_test"}}
        return patch.object(self.sender, "send_card_message", return_value=ret)

    def test_red_alert_is_pushed(self):
        """红色预警应即时推送"""
        with self._mock_send_card():
            result = self.sender.send_alert(_make_alert(risk_level="red"), "chat_xxx")
        self.assertTrue(result["pushed"])
        self.assertEqual(result["code"], 0)

    def test_orange_alert_is_pushed(self):
        """橙色预警应即时推送"""
        with self._mock_send_card():
            result = self.sender.send_alert(_make_alert(risk_level="orange"), "chat_xxx")
        self.assertTrue(result["pushed"])

    def test_yellow_alert_is_skipped(self):
        """黄色预警应跳过即时推送"""
        with self._mock_send_card() as mock_send:
            result = self.sender.send_alert(_make_alert(risk_level="yellow"), "chat_xxx")
        self.assertFalse(result["pushed"])
        self.assertEqual(result["msg"], "yellow_skipped")
        mock_send.assert_not_called()

    def test_red_alert_with_mentions(self):
        """红色预警应 @提醒所有配置的用户"""
        with self._mock_send_card() as mock_send:
            self.sender.send_alert(
                _make_alert(risk_level="red"),
                "chat_xxx",
                mention_open_ids=["ou_user1", "ou_user2"],
            )
        mock_send.assert_called_once()
        sent_card = mock_send.call_args[0][1]
        all_text = json.dumps(sent_card, ensure_ascii=False)
        self.assertIn("ou_user1", all_text)
        self.assertIn("ou_user2", all_text)

    def test_push_failure_returns_error(self):
        """推送失败应返回 code=-1, pushed=False"""
        with self._mock_send_card(ret={"code": 230002, "msg": "invalid chat_id"}):
            result = self.sender.send_alert(_make_alert(risk_level="red"), "bad_chat")
        self.assertFalse(result["pushed"])
        self.assertEqual(result["code"], -1)


# ============================================================
# 4. SentinelPipeline.run_alert_push 离线模式测试
# ============================================================

class TestPipelineAlertPushOffline(unittest.TestCase):
    """测试 pipeline.run_alert_push 离线降级模式"""

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
        cls.alert_cards_dir = os.path.join(
            PROJECT_ROOT, "demo", "demo_output", "alert_cards"
        )

    def setUp(self):
        # 每个测试前清空 alert_cards 目录，避免历史文件干扰
        if os.path.exists(self.alert_cards_dir):
            shutil.rmtree(self.alert_cards_dir)

    def tearDown(self):
        if os.path.exists(self.alert_cards_dir):
            shutil.rmtree(self.alert_cards_dir)

    def test_offline_push_saves_card_json(self):
        """离线模式应将卡片 JSON 保存到 demo_output/alert_cards/"""
        from core.pipeline import SentinelPipeline

        pipeline = SentinelPipeline(config_path=self.config_path)
        alert = _make_alert(risk_level="red")
        ok = pipeline.run_alert_push(alert)

        self.assertTrue(ok)
        files = os.listdir(self.alert_cards_dir)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(".json"))

        # 验证卡片文件内容
        with open(os.path.join(self.alert_cards_dir, files[0]), encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["alert_id"], alert["alert_id"])
        self.assertEqual(payload["risk_level"], "red")
        self.assertIn("card", payload)
        self.assertEqual(payload["card"]["header"]["template"], "red")

    def test_yellow_alert_not_pushed(self):
        """黄色预警在离线模式下也不即时推送"""
        from core.pipeline import SentinelPipeline

        pipeline = SentinelPipeline(config_path=self.config_path)
        ok = pipeline.run_alert_push(_make_alert(risk_level="yellow"))
        self.assertFalse(ok)
        # 不应生成任何卡片文件
        if os.path.exists(self.alert_cards_dir):
            self.assertEqual(len(os.listdir(self.alert_cards_dir)), 0)


# ============================================================
# 5. 全链路集成测试：故障数据 → 检测 → 推送
# ============================================================

class TestFullPipelineWithPush(unittest.TestCase):
    """全链路集成：运行故障数据，验证预警推送产出"""

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
        cls.alert_cards_dir = os.path.join(
            PROJECT_ROOT, "demo", "demo_output", "alert_cards"
        )

    def setUp(self):
        if os.path.exists(self.alert_cards_dir):
            shutil.rmtree(self.alert_cards_dir)

    def tearDown(self):
        if os.path.exists(self.alert_cards_dir):
            shutil.rmtree(self.alert_cards_dir)

    def test_bearing_overheat_pipeline_pushes_alerts(self):
        """轴承过热故障应触发预警推送，生成至少1张本地卡片"""
        from core.pipeline import SentinelPipeline

        pipeline = SentinelPipeline(config_path=self.config_path)
        data_path = os.path.join(
            PROJECT_ROOT, "data", "sample_data", "fault_bearing_overheat.csv"
        )
        report = pipeline.run_full_pipeline(data_path, "INC-01-BRG-01")

        self.assertEqual(report["status"], "success")
        self.assertGreater(report["anomalies_detected"], 0)
        # 离线模式下，红/橙预警应被推送（生成卡片文件）
        self.assertGreater(report["alerts_pushed"], 0)

        # 验证本地卡片文件已生成
        files = os.listdir(self.alert_cards_dir)
        self.assertGreater(len(files), 0)
        # 每个文件应为合法 JSON 且包含标准卡片结构
        for fname in files[:3]:
            with open(os.path.join(self.alert_cards_dir, fname), encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIn("card", payload)
            self.assertIn(payload["card"]["header"]["template"], ("red", "orange"))


# ============================================================
# 6. 在线模式 BitableClient 集成测试（mock 网络）
# ============================================================

class TestBitableClientOnline(unittest.TestCase):
    """测试 BitableClient 预警事件写入与状态更新（mock 网络层）"""

    def _make_client(self):
        """构造 BitableClient（mock 鉴权，不发起真实网络请求）"""
        client = BitableClient.__new__(BitableClient)
        client.auth = MagicMock()
        client.auth.get_tenant_access_token.return_value = "t-xxx"
        client.app_token = "appXXX"
        client.base_url = "https://open.feishu.cn/open-apis/bitable/v1"
        return client

    @patch("feishu.bitable_client.requests.post")
    def test_append_alert_events_success(self, mock_post):
        """批量写入预警事件应调用 batch_create 接口"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "code": 0,
            "data": {
                "records": [
                    {"record_id": "rec1"},
                    {"record_id": "rec2"},
                ]
            },
        }
        mock_post.return_value = mock_resp

        client = self._make_client()
        alerts = [_make_alert(risk_level="red"), _make_alert(risk_level="orange")]
        result = client.append_alert_events("tblAlerts", alerts)

        self.assertEqual(result["code"], 0)
        self.assertEqual(len(result["data"]["record_ids"]), 2)
        mock_post.assert_called_once()
        # 验证请求 URL 包含 batch_create
        called_url = mock_post.call_args[0][0]
        self.assertIn("batch_create", called_url)

    @patch("feishu.bitable_client.requests.put")
    def test_update_alert_status_success(self, mock_put):
        """更新预警状态应调用 update 接口"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"code": 0}
        mock_put.return_value = mock_resp

        client = self._make_client()
        result = client.update_alert_status(
            "tblAlerts", "rec1", "pushed", push_time="2026-07-15 10:05:00"
        )

        self.assertEqual(result["code"], 0)
        # 验证请求体中 alert_status 已转为中文
        sent_payload = mock_put.call_args[1]["json"]
        self.assertEqual(sent_payload["fields"]["alert_status"], "已推送")
        self.assertIn("push_time", sent_payload["fields"])

    def test_append_alert_events_empty(self):
        """空列表应直接返回成功，不发起网络请求"""
        client = self._make_client()
        result = client.append_alert_events("tblAlerts", [])
        self.assertEqual(result["code"], 0)
        self.assertEqual(result["data"]["record_ids"], [])


if __name__ == "__main__":
    unittest.main()
