"""
绿电哨兵 — Agent 4 运维建议哨兵
实现技术框架 §4.4「运维建议哨兵」：
  - 从故障知识库检索 Top-3 相似案例
  - 生成结构化维修建议报告
  - 生成工单草稿（多维表格「工单」表字段）

触发条件：预警事件被责任人确认后（alert_status=已确认）触发；
离线 demo 模式下可由 run_maintenance_advice 直接调用验收。
"""

import datetime
import json
import logging
import os
from typing import Optional

from models.knowledge_retriever import KnowledgeRetriever

logger = logging.getLogger(__name__)


# 风险等级 → 工单优先级映射（技术框架 §4.4 工单草稿）
RISK_TO_PRIORITY = {
    "red": "紧急",
    "orange": "高",
    "yellow": "中",
}

# 严重程度 → 处理时限（小时）映射
SEVERITY_TO_DEADLINE_HOURS = {
    "高": 4,    # 高严重度 4 小时内处理
    "中": 24,   # 中严重度 24 小时内处理
    "低": 72,   # 低严重度 72 小时内处理
}


def _parse_param_values(raw) -> Optional[dict]:
    """
    解析 param_values，统一返回 dict。
    支持：dict（直接返回）、JSON 字符串（解析后返回）、其他（返回 None）。
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


class MaintenanceAdvisor:
    """Agent 4 运维建议哨兵"""

    def __init__(self, kb_path: str = "config/knowledge_base.yaml",
                 weights: Optional[dict] = None,
                 param_ranges: Optional[dict] = None):
        """
        初始化运维建议生成器。

        Args:
            kb_path: 知识库 YAML 路径
            weights: 检索权重，传递给 KnowledgeRetriever
            param_ranges: 参数正常范围字典 {param_name: [low, high]}，
                          用于推断异常参数的方向（↑/↓）。
                          为 None 时退化为统一标记 ↑。
        """
        self.retriever = KnowledgeRetriever(kb_path=kb_path, weights=weights)
        self.param_ranges = param_ranges or {}
        logger.info(
            f"MaintenanceAdvisor initialized, "
            f"param_ranges={len(self.param_ranges)} params"
        )

    # ==================================================================
    # 从预警事件构造查询输入
    # ==================================================================

    def _build_symptom_from_alert(self, alert: dict) -> str:
        """
        从预警事件的 abnormal_params + param_values 推导症状模式。
        方向推断规则：
          - 有 param_ranges 且 param_values 为数值：高于范围中值 → ↑，低于 → ↓
          - 无 param_ranges 或无法推断：默认 ↑（保守策略，保证关键词能匹配到）

        例：bearing_temperature=78.5, 正常范围 [30,70] → 高于中值 50 → bearing_temperature↑
        """
        params = alert.get("abnormal_params", "")
        if not params:
            return ""
        # 兼容字符串与列表
        if isinstance(params, list):
            tokens = [str(p).strip() for p in params if str(p).strip()]
        else:
            tokens = [p.strip() for p in str(params).split(",") if p.strip()]

        # 解析 param_values 用于方向推断
        param_values = _parse_param_values(alert.get("param_values", {}))

        parts = []
        for t in tokens:
            direction = "↑"  # 默认
            if self.param_ranges and param_values and t in param_values:
                val = param_values[t]
                rng = self.param_ranges.get(t)
                if rng and isinstance(val, (int, float)):
                    low, high = rng[0], rng[1]
                    mid = (low + high) / 2
                    if val < mid:
                        direction = "↓"
                    elif val > mid:
                        direction = "↑"
                    # 恰好等于中值时保持默认 ↑
            parts.append(f"{t}{direction}")

        return " AND ".join(parts)

    # ==================================================================
    # 维修建议报告生成（§4.4 输出格式）
    # ==================================================================

    def generate_advice_report(self, alert: dict) -> dict:
        """
        根据预警事件生成结构化维修建议报告。

        Args:
            alert: 预警事件字典，应包含 fault_type / risk_level / device_name /
                   trigger_time / confidence / abnormal_params / param_values

        Returns:
            {
                "alert_id": ...,
                "report_text": "📋 维修建议报告 markdown 文本",
                "top_cases": [...],   # 检索到的 Top-3 案例
                "primary_case": {...}, # 最相似案例
                "work_order": {...},   # 工单草稿
            }
        """
        fault_type = alert.get("fault_type", "")
        symptom = self._build_symptom_from_alert(alert)
        param_values = _parse_param_values(alert.get("param_values", {}))

        logger.info(
            f"生成维修建议: alert_id={alert.get('alert_id')} "
            f"fault_type='{fault_type}' symptom='{symptom}'"
        )

        top_cases = self.retriever.retrieve(
            fault_type=fault_type,
            symptom_pattern=symptom,
            param_values=param_values,
            top_k=3,
        )

        primary_case = top_cases[0] if top_cases else None
        report_text = self._format_report_text(alert, top_cases, primary_case)
        work_order = self._build_work_order(alert, primary_case)

        return {
            "alert_id": alert.get("alert_id", ""),
            "device_id": alert.get("device_id", ""),
            "device_name": alert.get("device_name", ""),
            "fault_type": fault_type,
            "risk_level": alert.get("risk_level", ""),
            "top_cases": top_cases,
            "primary_case": primary_case,
            "report_text": report_text,
            "work_order": work_order,
        }

    def _format_report_text(self, alert: dict, top_cases: list,
                            primary_case: Optional[dict]) -> str:
        """格式化维修建议报告 Markdown 文本（遵循 §4.4 模板）"""
        lines = []
        lines.append("📋 维修建议报告")
        lines.append("━" * 30)
        lines.append(f"设备：{alert.get('device_name', '-')}")
        lines.append(f"设备编号：{alert.get('device_id', '-')}")
        lines.append(f"故障类型：{alert.get('fault_type', '-')}")
        lines.append(f"预警时间：{alert.get('trigger_time', '-')}")
        lines.append(f"风险等级：{alert.get('risk_level', '-')}")
        lines.append(f"置信度：{alert.get('confidence', 0)}")
        if alert.get("abnormal_params"):
            lines.append(f"异常参数：{alert['abnormal_params']}")
        pv = _parse_param_values(alert.get("param_values", {}))
        if pv:
            pv_str = ", ".join(f"{k}={v}" for k, v in pv.items())
            lines.append(f"当前值：{pv_str}")
        lines.append("")

        if not primary_case:
            lines.append("⚠️ 未检索到匹配的历史案例，建议人工分析并补充知识库")
            return "\n".join(lines)

        lines.append("🔍 最相似历史案例")
        lines.append(f"案例#{primary_case.get('case_id')}：{primary_case.get('fault_subtype', '')}")
        lines.append(f"匹配度：{primary_case.get('match_score', 0)}")
        lines.append(f"故障描述：{primary_case.get('description', '')}")
        lines.append("")

        lines.append("🛠 推荐维修方案")
        lines.append(primary_case.get("repair_plan", "无"))
        lines.append("")

        lines.append(f"🔧 所需工具：{primary_case.get('required_tools', '无')}")
        lines.append(f"📦 所需备件：{primary_case.get('required_parts', '无')}")
        lines.append(f"⏱ 预估处理时间：{primary_case.get('estimated_duration_min', '-')}分钟")
        lines.append(f"📌 严重程度：{primary_case.get('severity', '-')}")
        lines.append("")

        lines.append("📝 历史参考")
        freq = str(primary_case.get("historical_frequency", "未知")).rstrip("次")
        lines.append(f"近6个月同类故障发生{freq}次")

        # 备选案例
        if len(top_cases) > 1:
            lines.append("")
            lines.append("📚 其他备选案例")
            for c in top_cases[1:]:
                lines.append(
                    f"- {c.get('case_id')} {c.get('fault_subtype', '')} "
                    f"(匹配度 {c.get('match_score', 0)})"
                )
        return "\n".join(lines)

    # ==================================================================
    # 工单草稿生成（§4.4 工单草稿）
    # ==================================================================

    def _build_work_order(self, alert: dict,
                          primary_case: Optional[dict]) -> dict:
        """
        构造工单草稿字段，用于写入多维表格工单表。
        字段映射遵循技术框架 §4.4：
          - 优先级：红色→紧急/橙色→高/黄色→中
          - 建议处理时限：基于 estimated_duration_min
        """
        risk_level = str(alert.get("risk_level", "yellow")).lower()
        priority = RISK_TO_PRIORITY.get(risk_level, "中")

        # 生成工单号
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        wo_id = f"WO_{ts}"

        # 故障描述
        abnormal_params = alert.get("abnormal_params", "")
        param_values = _parse_param_values(alert.get("param_values", {}))
        if param_values:
            pv_str = ", ".join(f"{k}={v}" for k, v in param_values.items())
            fault_desc = f"异常参数：{abnormal_params}；当前值：{pv_str}"
        else:
            fault_desc = f"异常参数：{abnormal_params}" if abnormal_params else "异常工况"

        # 推荐维修方案
        repair_plan = primary_case.get("repair_plan", "") if primary_case else ""
        required_tools = primary_case.get("required_tools", "") if primary_case else ""
        required_parts = primary_case.get("required_parts", "") if primary_case else ""
        estimated_duration = (
            primary_case.get("estimated_duration_min", 0) if primary_case else 0
        )
        matched_case_id = primary_case.get("case_id", "") if primary_case else ""

        # 处理时限（小时）：取维修时长对应小时数与严重程度下限的较大值
        severity = primary_case.get("severity", "中") if primary_case else "中"
        deadline_hours = SEVERITY_TO_DEADLINE_HOURS.get(severity, 24)
        repair_hours = (estimated_duration or 0) / 60.0
        # 紧急工单至少给 4 小时，高优先级至少 8 小时
        min_hours = {"紧急": 4, "高": 8, "中": 24}.get(priority, 24)
        suggested_deadline_hours = max(deadline_hours, repair_hours, min_hours)
        suggested_deadline = (
            datetime.datetime.now()
            + datetime.timedelta(hours=suggested_deadline_hours)
        ).strftime("%Y-%m-%d %H:%M:%S")

        return {
            "work_order_id": wo_id,
            "alert_id": alert.get("alert_id", ""),
            "device_id": alert.get("device_id", ""),
            "device_name": alert.get("device_name", ""),
            "fault_type": alert.get("fault_type", ""),
            "fault_description": fault_desc,
            "recommended_repair_plan": repair_plan,
            "required_tools": required_tools,
            "required_parts": required_parts,
            "estimated_duration_min": estimated_duration,
            "priority": priority,
            "matched_case_id": matched_case_id,
            "suggested_deadline": suggested_deadline,
            "status": "待分配",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ==================================================================
    # 离线保存（demo 验收）
    # ==================================================================

    def save_advice_to_file(self, advice: dict, output_dir: str) -> str:
        """
        将维修建议报告与工单草稿保存为 JSON 文件（离线 demo 验收用）。

        Args:
            advice: generate_advice_report 返回的报告字典
            output_dir: 输出目录

        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        alert_id = advice.get("alert_id", "unknown").replace(":", "-").replace("/", "_")
        out_path = os.path.join(output_dir, f"advice_{alert_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(advice, f, ensure_ascii=False, indent=2)
        logger.info(f"维修建议已保存: {out_path}")
        return out_path

    def stats(self) -> dict:
        """返回知识库统计信息（委托给 retriever）"""
        return self.retriever.stats()
