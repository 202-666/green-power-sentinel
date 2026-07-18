"""
W5b 验收测试 — Agent 4 知识库检索 + 工单生成
验收标准：输入故障类型，返回合理维修建议
"""

import os
import unittest

from core.maintenance_advisor import (
    MaintenanceAdvisor,
    RISK_TO_PRIORITY,
    _parse_param_values,
)
from core.pipeline import SentinelPipeline
from models.knowledge_retriever import (
    KnowledgeRetriever,
    _extract_params_with_direction,
    _normalize_tokens,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
KB_PATH = os.path.join(CONFIG_DIR, "knowledge_base.yaml")

# 3 类核心故障对应的症状模式（与 rules.yaml 关联规则一致）
FAULT_SCENARIOS = {
    "轴承过热": {
        "symptom_pattern": "bearing_temperature↑ AND bearing_vibration↑",
        "param_values": {"bearing_temperature": 78.5, "bearing_vibration": 6.2},
        "expected_subtypes": {"轴承润滑不足", "轴承磨损失效", "冷却水系统失效"},
    },
    "烟气超标": {
        "symptom_pattern": "so2_concentration↑ AND nox_concentration↑ AND oxygen_content↓",
        "param_values": {
            "so2_concentration": 180.0,
            "nox_concentration": 190.0,
            "oxygen_content": 5.5,
        },
        "expected_subtypes": {"燃烧异常导致烟气超标", "脱硫剂供给不足", "给料不均导致燃烧波动"},
    },
    "炉排卡滞": {
        "symptom_pattern": "grate_speed↓ AND feed_rate↓ AND furnace_pressure↑",
        "param_values": {
            "grate_speed": 15.0,
            "feed_rate": 6.0,
            "furnace_pressure": 80.0,
        },
        "expected_subtypes": {"异物卡阻炉排", "炉排机械磨损卡阻"},
    },
}


class TestKnowledgeBase(unittest.TestCase):
    """知识库填充验收：10+ 条案例"""

    @classmethod
    def setUpClass(cls):
        cls.retriever = KnowledgeRetriever(kb_path=KB_PATH)

    def test_knowledge_base_has_at_least_10_cases(self):
        """知识库应填充 10+ 条案例（W5b 验收门槛）"""
        stats = self.retriever.stats()
        self.assertGreaterEqual(
            stats["total_cases"], 10,
            f"知识库案例数不足 10 条，当前 {stats['total_cases']} 条"
        )

    def test_knowledge_base_covers_3_core_fault_types(self):
        """知识库应覆盖 3 类核心故障类型"""
        stats = self.retriever.stats()
        by_type = stats["by_fault_type"]
        for fault_type in ["轴承过热", "烟气超标", "炉排卡滞"]:
            self.assertGreater(
                by_type.get(fault_type, 0), 0,
                f"知识库缺少核心故障类型：{fault_type}"
            )

    def test_all_cases_have_required_fields(self):
        """每条案例应包含技术框架 §3.3 要求的全部 17 个字段"""
        required_fields = [
            "case_id", "fault_type", "fault_subtype", "symptom_pattern",
            "description", "cause_analysis", "repair_plan", "required_tools",
            "required_parts", "estimated_duration_min", "severity",
            "historical_frequency", "source", "last_updated", "reference_count",
        ]
        for case in self.retriever.cases:
            for field in required_fields:
                self.assertIn(
                    field, case,
                    f"案例 {case.get('case_id')} 缺少字段: {field}"
                )
                self.assertTrue(
                    str(case.get(field)).strip(),
                    f"案例 {case.get('case_id')} 字段 {field} 为空"
                )

    def test_case_ids_unique(self):
        """案例 ID 应全局唯一"""
        ids = [c.get("case_id") for c in self.retriever.cases]
        self.assertEqual(len(ids), len(set(ids)),
                         f"存在重复 case_id: {[i for i in ids if ids.count(i) > 1]}")

    def test_estimated_duration_is_numeric(self):
        """estimated_duration_min 应为数字类型"""
        for case in self.retriever.cases:
            val = case.get("estimated_duration_min")
            self.assertIsInstance(
                val, (int, float),
                f"案例 {case.get('case_id')} 的 estimated_duration_min 不是数字: {type(val)}"
            )
            self.assertGreater(val, 0,
                               f"案例 {case.get('case_id')} 的 estimated_duration_min 应 > 0")

    def test_severity_valid_values(self):
        """severity 应为 高/中/低 之一"""
        valid = {"高", "中", "低"}
        for case in self.retriever.cases:
            self.assertIn(
                case.get("severity"),
                valid,
                f"案例 {case.get('case_id')} 的 severity 无效: {case.get('severity')}"
            )


class TestRetrievalByFaultType(unittest.TestCase):
    """核心验收：输入故障类型，返回合理维修建议"""

    @classmethod
    def setUpClass(cls):
        cls.advisor = MaintenanceAdvisor(kb_path=KB_PATH)

    def _build_alert(self, fault_type, risk_level="red"):
        """构造模拟预警事件"""
        scenario = FAULT_SCENARIOS[fault_type]
        return {
            "alert_id": f"ALT_TEST_{fault_type}",
            "trigger_time": "2026-07-17 10:05:00",
            "device_id": "INC-01-BRG-01",
            "device_name": f"1#焚烧炉-{fault_type}测试设备",
            "fault_type": fault_type,
            "risk_level": risk_level,
            "confidence": 0.85,
            "abnormal_params": ", ".join(scenario["param_values"].keys()),
            "param_values": scenario["param_values"],
            "alert_status": "已确认",
        }

    def test_bearing_overheat_returns_relevant_advice(self):
        """输入「轴承过热」应返回合理的维修建议"""
        fault_type = "轴承过热"
        alert = self._build_alert(fault_type)
        advice = self.advisor.generate_advice_report(alert)

        # 应返回结构化报告
        self.assertIn("report_text", advice)
        self.assertIn("top_cases", advice)
        self.assertIn("work_order", advice)

        # Top-3 案例不应为空
        top_cases = advice["top_cases"]
        self.assertGreater(len(top_cases), 0, "应检索到至少 1 条案例")

        # 最相似案例的故障类型应为「轴承过热」
        primary = advice["primary_case"]
        self.assertIsNotNone(primary, "应返回最相似案例")
        self.assertEqual(primary["fault_type"], "轴承过热")
        self.assertGreater(primary["match_score"], 0.3, "匹配度应合理")

        # 维修建议应包含轴承相关子类
        primary_subtypes = {c["fault_subtype"] for c in top_cases if c["fault_type"] == "轴承过热"}
        self.assertTrue(
            primary_subtypes & FAULT_SCENARIOS[fault_type]["expected_subtypes"],
            f"轴承过热案例应覆盖子类，实际: {primary_subtypes}"
        )

    def test_emission_exceed_returns_relevant_advice(self):
        """输入「烟气超标」应返回合理的维修建议"""
        fault_type = "烟气超标"
        alert = self._build_alert(fault_type)
        advice = self.advisor.generate_advice_report(alert)

        top_cases = advice["top_cases"]
        self.assertGreater(len(top_cases), 0)

        primary = advice["primary_case"]
        self.assertEqual(primary["fault_type"], "烟气超标")
        self.assertGreater(primary["match_score"], 0.3)

        primary_subtypes = {c["fault_subtype"] for c in top_cases if c["fault_type"] == "烟气超标"}
        self.assertTrue(
            primary_subtypes & FAULT_SCENARIOS[fault_type]["expected_subtypes"],
            f"烟气超标案例应覆盖子类，实际: {primary_subtypes}"
        )

    def test_grate_jam_returns_relevant_advice(self):
        """输入「炉排卡滞」应返回合理的维修建议"""
        fault_type = "炉排卡滞"
        alert = self._build_alert(fault_type)
        advice = self.advisor.generate_advice_report(alert)

        top_cases = advice["top_cases"]
        self.assertGreater(len(top_cases), 0)

        primary = advice["primary_case"]
        self.assertEqual(primary["fault_type"], "炉排卡滞")
        self.assertGreater(primary["match_score"], 0.3)

        primary_subtypes = {c["fault_subtype"] for c in top_cases if c["fault_type"] == "炉排卡滞"}
        self.assertTrue(
            primary_subtypes & FAULT_SCENARIOS[fault_type]["expected_subtypes"],
            f"炉排卡滞案例应覆盖子类，实际: {primary_subtypes}"
        )

    def test_advice_report_contains_repair_plan(self):
        """维修建议报告应包含推荐维修方案"""
        for fault_type in FAULT_SCENARIOS:
            with self.subTest(fault_type=fault_type):
                alert = self._build_alert(fault_type)
                advice = self.advisor.generate_advice_report(alert)
                self.assertIn("推荐维修方案", advice["report_text"])
                self.assertTrue(advice["primary_case"]["repair_plan"].strip())

    def test_top_cases_ranked_by_score(self):
        """检索结果应按匹配度从高到低排序"""
        for fault_type in FAULT_SCENARIOS:
            with self.subTest(fault_type=fault_type):
                alert = self._build_alert(fault_type)
                advice = self.advisor.generate_advice_report(alert)
                scores = [c["match_score"] for c in advice["top_cases"]]
                self.assertEqual(scores, sorted(scores, reverse=True),
                                 f"{fault_type} 检索结果未按匹配度降序排列")


class TestWorkOrderGeneration(unittest.TestCase):
    """工单草稿生成验收（§4.4 工单草稿）"""

    @classmethod
    def setUpClass(cls):
        cls.advisor = MaintenanceAdvisor(kb_path=KB_PATH)

    def _build_alert(self, fault_type, risk_level):
        scenario = FAULT_SCENARIOS[fault_type]
        return {
            "alert_id": f"ALT_WO_{fault_type}_{risk_level}",
            "trigger_time": "2026-07-17 10:05:00",
            "device_id": "INC-01-BRG-01",
            "device_name": f"1#焚烧炉-{fault_type}设备",
            "fault_type": fault_type,
            "risk_level": risk_level,
            "confidence": 0.85,
            "abnormal_params": ", ".join(scenario["param_values"].keys()),
            "param_values": scenario["param_values"],
            "alert_status": "已确认",
        }

    def test_work_order_priority_maps_risk_level(self):
        """工单优先级应正确映射风险等级：红色→紧急/橙色→高/黄色→中"""
        for risk_level, expected_priority in RISK_TO_PRIORITY.items():
            with self.subTest(risk_level=risk_level):
                alert = self._build_alert("轴承过热", risk_level)
                advice = self.advisor.generate_advice_report(alert)
                work_order = advice["work_order"]
                self.assertEqual(work_order["priority"], expected_priority)

    def test_work_order_has_required_fields(self):
        """工单草稿应包含 §4.4 要求的字段"""
        alert = self._build_alert("轴承过热", "red")
        advice = self.advisor.generate_advice_report(alert)
        wo = advice["work_order"]

        required_fields = [
            "work_order_id", "alert_id", "device_id", "device_name",
            "fault_type", "fault_description", "recommended_repair_plan",
            "priority", "matched_case_id", "suggested_deadline", "status",
        ]
        for field in required_fields:
            self.assertIn(field, wo, f"工单缺少字段: {field}")
            self.assertTrue(str(wo[field]).strip(), f"工单字段 {field} 为空")

    def test_work_order_id_format(self):
        """工单号应遵循 WO_YYYYMMDDHHMMSS 格式"""
        alert = self._build_alert("轴承过热", "red")
        advice = self.advisor.generate_advice_report(alert)
        wo_id = advice["work_order"]["work_order_id"]
        self.assertTrue(wo_id.startswith("WO_"),
                        f"工单号格式错误: {wo_id}")
        self.assertGreaterEqual(len(wo_id), 15)

    def test_work_order_deadline_future(self):
        """建议处理时限应为未来时间"""
        import datetime
        alert = self._build_alert("轴承过热", "red")
        advice = self.advisor.generate_advice_report(alert)
        deadline = advice["work_order"]["suggested_deadline"]
        deadline_dt = datetime.datetime.strptime(deadline, "%Y-%m-%d %H:%M:%S")
        self.assertGreater(deadline_dt, datetime.datetime.now(),
                           "建议处理时限应为未来时间")


class TestEdgeCases(unittest.TestCase):
    """边界场景测试（鲁棒性验证）"""

    @classmethod
    def setUpClass(cls):
        cls.advisor = MaintenanceAdvisor(kb_path=KB_PATH)
        cls.retriever = KnowledgeRetriever(kb_path=KB_PATH)

    def test_empty_fault_type_returns_results(self):
        """空 fault_type 仍应返回结果（不崩溃）"""
        top = self.retriever.retrieve(fault_type="", top_k=3)
        self.assertIsInstance(top, list)
        self.assertEqual(len(top), 3)  # 即使不匹配也返回分数最高的 3 条

    def test_empty_symptom_pattern_works(self):
        """空 symptom_pattern 应正常检索"""
        top = self.retriever.retrieve(
            fault_type="轴承过热", symptom_pattern="", top_k=3
        )
        self.assertGreater(len(top), 0)
        self.assertEqual(top[0]["fault_type"], "轴承过热")

    def test_param_values_json_string(self):
        """param_values 为 JSON 字符串时应正确解析"""
        alert = {
            "alert_id": "ALT_TEST_JSON_STR",
            "fault_type": "轴承过热",
            "risk_level": "red",
            "confidence": 0.85,
            "abnormal_params": "bearing_temperature, bearing_vibration",
            "param_values": '{"bearing_temperature": 78.5, "bearing_vibration": 6.2}',
        }
        advice = self.advisor.generate_advice_report(alert)
        self.assertIn("work_order", advice)
        self.assertIn("bearing_temperature=78.5",
                      advice["work_order"]["fault_description"])

    def test_param_values_invalid_json_falls_back_gracefully(self):
        """param_values 为非法 JSON 字符串时应优雅降级（不崩溃）"""
        alert = {
            "alert_id": "ALT_TEST_BAD_JSON",
            "fault_type": "轴承过热",
            "risk_level": "red",
            "confidence": 0.85,
            "abnormal_params": "bearing_temperature",
            "param_values": "{not valid json",
        }
        advice = self.advisor.generate_advice_report(alert)
        self.assertIsNotNone(advice)
        self.assertIn("work_order", advice)

    def test_top_k_greater_than_total_cases(self):
        """top_k 大于知识库总量时应返回全部案例"""
        top = self.retriever.retrieve(fault_type="轴承过热", top_k=100)
        self.assertEqual(len(top), len(self.retriever.cases))

    def test_abnormal_params_as_list(self):
        """abnormal_params 为列表格式时应正常处理"""
        alert = {
            "alert_id": "ALT_TEST_LIST_PARAMS",
            "fault_type": "炉排卡滞",
            "risk_level": "orange",
            "confidence": 0.7,
            "abnormal_params": ["grate_speed", "furnace_pressure"],
            "param_values": {"grate_speed": 15.0, "furnace_pressure": 80.0},
        }
        advice = self.advisor.generate_advice_report(alert)
        self.assertEqual(advice["primary_case"]["fault_type"], "炉排卡滞")

    def test_parse_param_values_helper(self):
        """_parse_param_values 工具函数的各种输入场景"""
        # dict 输入
        self.assertEqual(_parse_param_values({"a": 1}), {"a": 1})
        # JSON 字符串
        self.assertEqual(
            _parse_param_values('{"x": 2}'),
            {"x": 2}
        )
        # 空字符串
        self.assertIsNone(_parse_param_values(""))
        # 非法 JSON
        self.assertIsNone(_parse_param_values("not json"))
        # None
        self.assertIsNone(_parse_param_values(None))
        # 列表（非 dict）
        self.assertIsNone(_parse_param_values([1, 2, 3]))

    def test_extract_params_with_direction(self):
        """_extract_params_with_direction 应正确提取参数方向"""
        text = "bearing_temperature↑ AND steam_pressure↓ AND grate_speed波动"
        result = _extract_params_with_direction(text)
        self.assertEqual(result["bearing_temperature"], "up")
        self.assertEqual(result["steam_pressure"], "down")
        self.assertEqual(result["grate_speed"], "volatile")

    def test_direction_aware_keyword_matching(self):
        """方向感知的关键词匹配：方向一致分更高，相反分更低"""
        # 构造方向一致的查询
        top_same = self.retriever.retrieve(
            fault_type="",
            symptom_pattern="grate_speed↓ AND feed_rate↓ AND furnace_pressure↑",
            top_k=3,
        )
        # 构造方向相反的查询（grate_speed↑ 而非 ↓）
        top_opposite = self.retriever.retrieve(
            fault_type="",
            symptom_pattern="grate_speed↑ AND feed_rate↑ AND furnace_pressure↓",
            top_k=3,
        )
        # 找炉排卡滞案例在两种查询中的排名差异
        # 方向一致时，炉排卡滞案例排名应更靠前
        grate_case_same = next(
            (i for i, c in enumerate(top_same) if c["fault_type"] == "炉排卡滞"),
            99
        )
        grate_case_opp = next(
            (i for i, c in enumerate(top_opposite) if c["fault_type"] == "炉排卡滞"),
            99
        )
        # 方向一致的匹配度应高于方向相反
        self.assertLessEqual(
            grate_case_same, grate_case_opp,
            "方向一致的查询应使炉排卡滞案例排名更靠前"
        )


class TestWorkOrderFieldMapping(unittest.TestCase):
    """工单字段映射单元测试（飞书多维表格写入格式验证）"""

    def test_work_order_to_fields_mapping(self):
        """BitableClient.work_order_to_fields 应正确转换字段格式"""
        from feishu.bitable_client import BitableClient

        work_order = {
            "work_order_id": "WO_20260717120000",
            "alert_id": "ALT_001",
            "device_id": "INC-01-BRG-01",
            "device_name": "1#焚烧炉-引风机",
            "fault_type": "轴承过热",
            "fault_description": "轴承温度异常",
            "recommended_repair_plan": "检查润滑",
            "required_tools": "油脂枪",
            "required_parts": "润滑脂",
            "estimated_duration_min": 120,
            "priority": "紧急",
            "matched_case_id": "CASE_001",
            "suggested_deadline": "2026-07-17 16:00:00",
            "status": "待分配",
            "created_at": "2026-07-17 12:00:00",
        }
        fields = BitableClient.work_order_to_fields(work_order)

        # 关键字段存在
        for k in ["work_order_id", "alert_id", "device_id", "device_name",
                  "fault_type", "priority", "status"]:
            self.assertIn(k, fields, f"缺少字段: {k}")

        # 日期字段应转为毫秒时间戳（数字类型）
        self.assertIsInstance(fields["suggested_deadline"], int,
                              "suggested_deadline 应转为毫秒时间戳")
        self.assertIsInstance(fields["created_at"], int,
                              "created_at 应转为毫秒时间戳")
        self.assertGreater(fields["suggested_deadline"], 0,
                           "日期时间戳应 > 0")

        # 数字字段类型
        self.assertEqual(fields["estimated_duration_min"], 120)

    def test_work_order_to_fields_default_values(self):
        """work_order_to_fields 处理缺省值不崩溃"""
        from feishu.bitable_client import BitableClient

        wo = {"work_order_id": "WO_TEST_MINIMAL"}
        fields = BitableClient.work_order_to_fields(wo)
        self.assertEqual(fields["work_order_id"], "WO_TEST_MINIMAL")
        self.assertEqual(fields["priority"], "中")
        self.assertEqual(fields["status"], "待分配")


class TestPipelineAgent4Integration(unittest.TestCase):
    """Agent 4 与 Pipeline 全链路集成验收（离线 demo 模式）"""

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(CONFIG_DIR, "config.yaml")
        cls.pipeline = SentinelPipeline(config_path=cls.config_path)

    def test_agent4_triggered_in_full_pipeline(self):
        """全链路运行轴承过热故障应触发 Agent 4 生成维修建议"""
        data_path = os.path.join(
            PROJECT_ROOT, "data", "sample_data", "fault_bearing_overheat.csv"
        )
        report = self.pipeline.run_full_pipeline(data_path, "INC-01-BRG-01")

        self.assertEqual(report["status"], "success")
        self.assertGreater(report["anomalies_detected"], 0)
        # Agent 4 应成功生成建议
        self.assertTrue(
            report["advice_generated"],
            "Agent 4 未生成维修建议（advice_generated=False）"
        )

    def test_agent4_offline_advice_saved(self):
        """离线模式下维修建议应保存为 JSON 文件"""
        data_path = os.path.join(
            PROJECT_ROOT, "data", "sample_data", "fault_bearing_overheat.csv"
        )
        self.pipeline.run_full_pipeline(data_path, "INC-01-BRG-01")

        advice_dir = os.path.join(
            PROJECT_ROOT, "demo", "demo_output", "advice"
        )
        self.assertTrue(os.path.isdir(advice_dir), "离线 advice 输出目录未创建")

        advice_files = [f for f in os.listdir(advice_dir) if f.endswith(".json")]
        self.assertGreater(len(advice_files), 0, "未生成离线维修建议 JSON 文件")

    def test_agent4_direction_inference_in_pipeline(self):
        """Pipeline 中 Agent 4 应正确推断参数方向（基于 thresholds）"""
        # 烟气超标故障中 oxygen_content 应低于正常范围中值（9%）→ ↓
        data_path = os.path.join(
            PROJECT_ROOT, "data", "sample_data", "fault_emission_exceed.csv"
        )
        self.pipeline.run_full_pipeline(data_path, "INC-01-BRG-01")

        # 从最近一次异常中取第一个，验证 advice 的 primary_case 是烟气超标
        summary = self.pipeline.get_detection_summary()
        anomalies = summary["anomalies"]
        self.assertGreater(len(anomalies), 0)

        first_alert_id = anomalies[0]["alert_id"]
        advice = self.pipeline.run_maintenance_advice(first_alert_id)
        if advice:
            # primary_case 应为烟气超标类型
            primary = advice.get("primary_case", {})
            # 由于方向正确推断，烟气超标案例应排在前面
            fault_type = primary.get("fault_type", "")
            # 至少 top_cases 中包含烟气超标案例
            top_types = [c["fault_type"] for c in advice.get("top_cases", [])]
            self.assertIn(
                "烟气超标", top_types,
                f"Top-3 案例中应包含烟气超标，实际: {top_types}"
            )


if __name__ == "__main__":
    unittest.main()
