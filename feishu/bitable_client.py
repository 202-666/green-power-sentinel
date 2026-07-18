"""
飞书多维表格API封装
提供记录的增删改查功能，以及针对「预警事件」表的高层封装。
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

import requests

from .auth import FeishuAuth

logger = logging.getLogger(__name__)


# 风险等级英文 → 中文（多维表格「预警事件」表 risk_level 单选字段值）
RISK_LEVEL_CN = {"red": "红", "orange": "橙", "yellow": "黄"}

# 预警状态英文 → 中文（多维表格「预警事件」表 alert_status 单选字段值）
ALERT_STATUS_CN = {
    "pending": "待推送",
    "pushed": "已推送",
    "confirmed": "已确认",
    "handled": "已处理",
    "false_alarm": "误报",
}


def _to_multi_select(value) -> list:
    """
    将输入值转换为飞书多维表格多选字段所需的列表格式。
    支持逗号分隔的字符串或列表，其它类型转为空列表。
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _to_millis(ts_str: str) -> int:
    """
    将 'YYYY-MM-DD HH:MM:SS' 字符串转为飞书多维表格日期字段所需的毫秒时间戳。
    解析失败时返回当前时间。
    """
    if not ts_str:
        return int(time.time() * 1000)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return int(datetime.strptime(str(ts_str), fmt).timestamp() * 1000)
        except ValueError:
            continue
    return int(time.time() * 1000)


class BitableClient:
    """飞书多维表格客户端"""

    def __init__(self, app_id: str, app_secret: str, app_token: str):
        """
        初始化多维表格客户端

        Args:
            app_id: 飞书应用ID
            app_secret: 飞书应用密钥
            app_token: 多维表格的app_token（从URL中获取）
        """
        self.auth = FeishuAuth(app_id, app_secret)
        self.app_token = app_token
        self.base_url = "https://open.feishu.cn/open-apis/bitable/v1"

    def _get_headers(self) -> dict:
        """获取带认证的请求头"""
        token = self.auth.get_tenant_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def get_records(
        self,
        table_id: str,
        view_id: Optional[str] = None,
        filter_expr: Optional[str] = None,
        page_size: int = 500,
        page_token: Optional[str] = None,
    ) -> dict:
        """
        读取多维表格记录

        Args:
            table_id: 数据表ID（如 tbl8TerzZHfvFbFa）
            view_id: 视图ID（可选）
            filter_expr: 筛选条件（可选）
            page_size: 每页记录数，最大500
            page_token: 分页token

        Returns:
            API响应字典
        """
        url = f"{self.base_url}/apps/{self.app_token}/tables/{table_id}/records"
        params = {"page_size": page_size}
        if view_id:
            params["view_id"] = view_id
        if filter_expr:
            params["filter"] = filter_expr
        if page_token:
            params["page_token"] = page_token

        try:
            resp = requests.get(url, headers=self._get_headers(), params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"code": -1, "msg": str(e), "data": {}}

    def append_records(self, table_id: str, records: list) -> dict:
        """
        批量写入记录

        Args:
            table_id: 数据表ID
            records: 记录列表，每条记录为 {"fields": {字段名: 值, ...}}

        Returns:
            API响应字典
        """
        url = f"{self.base_url}/apps/{self.app_token}/tables/{table_id}/records/batch_create"
        payload = {"records": records}

        try:
            resp = requests.post(url, headers=self._get_headers(), json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"code": -1, "msg": str(e), "data": {}}

    def update_record(self, table_id: str, record_id: str, fields: dict) -> dict:
        """
        更新单条记录

        Args:
            table_id: 数据表ID
            record_id: 记录ID
            fields: 要更新的字段字典

        Returns:
            API响应字典
        """
        url = f"{self.base_url}/apps/{self.app_token}/tables/{table_id}/records/{record_id}"
        payload = {"fields": fields}

        try:
            resp = requests.put(url, headers=self._get_headers(), json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"code": -1, "msg": str(e), "data": {}}

    def get_fields(self, table_id: str) -> dict:
        """
        获取数据表字段列表

        Args:
            table_id: 数据表ID

        Returns:
            API响应字典
        """
        url = f"{self.base_url}/apps/{self.app_token}/tables/{table_id}/fields"

        try:
            resp = requests.get(url, headers=self._get_headers())
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"code": -1, "msg": str(e), "data": {}}

    # ====================================================================
    # 「预警事件」表高层封装（Agent 3 使用）
    # ====================================================================

    @staticmethod
    def alert_to_fields(alert: dict) -> dict:
        """
        将流水线产生的预警事件字典转换为飞书多维表格「预警事件」表的字段格式。

        字段映射遵循技术框架 §3.2：
        - risk_level: red/orange/yellow → 红/橙/黄（单选）
        - param_values: dict → JSON 字符串（文本字段）
        - trigger_time / push_time: 字符串 → 毫秒时间戳（日期字段）
        - alert_status: 若缺省则置为「待推送」

        Args:
            alert: 预警事件字典（来自 SentinelPipeline.run_anomaly_detection）

        Returns:
            飞书多维表格 fields 字典
        """
        risk_level_raw = str(alert.get("risk_level", "yellow")).lower()
        risk_level_cn = RISK_LEVEL_CN.get(risk_level_raw, "黄")

        status_raw = alert.get("alert_status", "待推送")
        # 兼容中英文输入
        alert_status = ALERT_STATUS_CN.get(status_raw, status_raw) if status_raw else "待推送"

        param_values = alert.get("param_values", {})
        if isinstance(param_values, (dict, list)):
            param_values_str = json.dumps(param_values, ensure_ascii=False)
        else:
            param_values_str = str(param_values)

        fields = {
            "alert_id": alert.get("alert_id", ""),
            "trigger_time": _to_millis(alert.get("trigger_time", "")),
            "device_id": alert.get("device_id", ""),
            "fault_type": alert.get("fault_type", ""),
            "risk_level": risk_level_cn,
            "confidence": float(alert.get("confidence", 0.0)),
            "detection_method": alert.get("detection_method", ""),
            "abnormal_params": _to_multi_select(alert.get("abnormal_params", "")),
            "param_values": param_values_str,
            "alert_status": alert_status,
        }

        # 可选字段：仅在原数据存在时填充，避免覆盖多维表格默认值
        if "source_record_id" in alert and alert["source_record_id"]:
            fields["source_record_id"] = alert["source_record_id"]
        if "predicted_advance_min" in alert and alert["predicted_advance_min"] is not None:
            fields["duration"] = int(alert["predicted_advance_min"])
        if "recommended_action" in alert and alert["recommended_action"]:
            fields["recommended_action"] = alert["recommended_action"]
        if "matched_case_id" in alert and alert["matched_case_id"]:
            fields["matched_case_id"] = alert["matched_case_id"]

        # 附加根因信息（写入 detection_method 的补充说明，便于运维人员快速理解）
        primary_cause = alert.get("primary_cause")
        if primary_cause and primary_cause not in fields["detection_method"]:
            fields["detection_method"] = (
                f"{fields['detection_method']}（根因: {primary_cause}）"
            )

        return fields

    def append_alert_events(self, table_id: str, alerts: list) -> dict:
        """
        批量将预警事件写入「预警事件」表。

        Args:
            table_id: 「预警事件」表 ID（config.feishu.tables.alert_events）
            alerts: 预警事件字典列表

        Returns:
            API 响应字典。成功时 code=0，data.record_ids 为新增记录ID列表。
            离线降级场景 code=-1，msg 描述失败原因。
        """
        if not alerts:
            return {"code": 0, "msg": "empty", "data": {"record_ids": []}}

        records = [{"fields": self.alert_to_fields(a)} for a in alerts]
        logger.info(f"向多维表格写入 {len(records)} 条预警事件 (table={table_id})")
        resp = self.append_records(table_id, records)

        if resp.get("code") == 0:
            record_ids = [r.get("record_id") for r in resp.get("data", {}).get("records", [])]
            logger.info(f"写入成功，返回 {len(record_ids)} 个 record_id")
            return {"code": 0, "msg": "ok", "data": {"record_ids": record_ids}}
        logger.error(f"写入多维表格失败: {resp.get('msg')}")
        return resp

    def update_alert_status(
        self,
        table_id: str,
        record_id: str,
        status: str,
        push_time: Optional[str] = None,
    ) -> dict:
        """
        更新预警事件状态（如「待推送」→「已推送」）。

        Args:
            table_id: 「预警事件」表 ID
            record_id: 飞书多维表格记录 ID
            status: 目标状态，支持中文（已推送）或英文（pushed）
            push_time: 推送时间字符串，仅状态为「已推送」时填充

        Returns:
            API 响应字典
        """
        status_cn = ALERT_STATUS_CN.get(status, status) if status else "已推送"
        fields = {"alert_status": status_cn}
        if push_time:
            fields["push_time"] = _to_millis(push_time)

        logger.info(f"更新预警状态 record_id={record_id} → {status_cn}")
        return self.update_record(table_id, record_id, fields)

    # ====================================================================
    # 「故障知识库」表高层封装（Agent 4 使用）
    # ====================================================================

    def get_knowledge_cases(self, table_id: str) -> list:
        """
        读取多维表格「故障知识库」表的全部案例。

        用于在线模式下将飞书多维表格作为知识库源（与本地 knowledge_base.yaml 互为补充）。
        字段结构遵循技术框架 §3.3。

        Args:
            table_id: 「故障知识库」表 ID（config.feishu.tables.knowledge_base）

        Returns:
            案例字典列表，每条为展平后的字段字典（去除多维表格 record_id 包装）
        """
        cases = []
        page_token = None
        while True:
            resp = self.get_records(table_id, page_token=page_token, page_size=500)
            if resp.get("code") != 0:
                logger.error(f"读取知识库失败: {resp.get('msg')}")
                break
            data = resp.get("data", {}) or {}
            items = data.get("items", []) or []
            for rec in items:
                fields = rec.get("fields", {}) or {}
                cases.append(self._fields_to_case(fields))
            page_token = data.get("page_token")
            if not data.get("has_more") or not page_token:
                break
        logger.info(f"从多维表格读取 {len(cases)} 条知识库案例 (table={table_id})")
        return cases

    @staticmethod
    def _fields_to_case(fields: dict) -> dict:
        """将多维表格字段（含单选/多选对象）展平为知识库案例字典"""
        def _flat(v):
            # 飞书单选/多选字段返回 {"text": "..."} 或 [{"text": "..."}]
            if isinstance(v, dict):
                return v.get("text", v.get("name", ""))
            if isinstance(v, list):
                return ", ".join(_flat(x) for x in v if x)
            return v
        return {k: _flat(v) for k, v in fields.items()}

    # ====================================================================
    # 「工单」表高层封装（Agent 4 使用）
    # ====================================================================

    def append_work_orders(self, table_id: str, work_orders: list) -> dict:
        """
        批量将工单草稿写入「工单」表。

        字段映射遵循技术框架 §4.4 工单草稿。

        Args:
            table_id: 「工单」表 ID（config.feishu.tables.work_orders）
            work_orders: 工单字典列表（来自 MaintenanceAdvisor._build_work_order）

        Returns:
            API 响应字典。成功时 code=0，data.record_ids 为新增记录ID列表。
        """
        if not work_orders:
            return {"code": 0, "msg": "empty", "data": {"record_ids": []}}

        records = [{"fields": self.work_order_to_fields(w)} for w in work_orders]
        logger.info(f"向多维表格写入 {len(records)} 条工单 (table={table_id})")
        resp = self.append_records(table_id, records)

        if resp.get("code") == 0:
            record_ids = [r.get("record_id") for r in resp.get("data", {}).get("records", [])]
            logger.info(f"工单写入成功，返回 {len(record_ids)} 个 record_id")
            return {"code": 0, "msg": "ok", "data": {"record_ids": record_ids}}
        logger.error(f"写入工单失败: {resp.get('msg')}")
        return resp

    @staticmethod
    def work_order_to_fields(work_order: dict) -> dict:
        """
        将工单字典转换为飞书多维表格「工单」表的字段格式。

        处理：日期字段转毫秒时间戳；优先级映射为单选值。
        """
        priority = str(work_order.get("priority", "中"))
        status = str(work_order.get("status", "待分配"))

        fields = {
            "work_order_id": work_order.get("work_order_id", ""),
            "alert_id": work_order.get("alert_id", ""),
            "device_id": work_order.get("device_id", ""),
            "device_name": work_order.get("device_name", ""),
            "fault_type": work_order.get("fault_type", ""),
            "fault_description": work_order.get("fault_description", ""),
            "recommended_repair_plan": work_order.get("recommended_repair_plan", ""),
            "required_tools": work_order.get("required_tools", ""),
            "required_parts": work_order.get("required_parts", ""),
            "estimated_duration_min": int(work_order.get("estimated_duration_min", 0) or 0),
            "priority": priority,
            "matched_case_id": work_order.get("matched_case_id", ""),
            "status": status,
        }
        if work_order.get("suggested_deadline"):
            fields["suggested_deadline"] = _to_millis(work_order["suggested_deadline"])
        if work_order.get("created_at"):
            fields["created_at"] = _to_millis(work_order["created_at"])
        return fields