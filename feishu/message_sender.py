"""
飞书消息推送模块
支持发送消息卡片和文本消息，并实现「绿电哨兵」Agent 3 的风险分级推送策略。

推送策略（依据技术框架 §4.3）：
- 红色预警（score >= 0.8）：即时卡片 + @责任人+主管+厂长
- 橙色预警（0.5 <= score < 0.8）：即时卡片 + @责任人
- 黄色预警（0.3 <= score < 0.5）：不即时推送，汇总至每日运维日报
"""

import json
import logging
from typing import Optional

import requests

from .auth import FeishuAuth

logger = logging.getLogger(__name__)

# L4：所有飞书 API 请求统一超时，避免上游挂起卡死流水线
REQUEST_TIMEOUT = 10


# 风险等级 → 飞书卡片 header template（颜色）
# 飞书官方支持：red / orange / yellow / green / blue / turquoise / purple / grey
RISK_CARD_TEMPLATE = {"red": "red", "orange": "orange", "yellow": "yellow"}

# 风险等级 → 卡片标题前缀 emoji
RISK_EMOJI = {"red": "🚨", "orange": "🟠", "yellow": "🟡"}

# 风险等级 → 中文标签
RISK_LABEL_CN = {"red": "红色", "orange": "橙色", "yellow": "黄色"}

# 风险等级 → 是否即时推送
INSTANT_PUSH_LEVELS = {"red", "orange"}


def parse_open_ids(value) -> list:
    """
    将 FEISHU_MENTION_OPEN_IDS 配置值解析为 open_id 列表（M1 修复）。

    兼容三种形态：
    - 列表（YAML 原生列表或 env 中 JSON 数组）→ 直接使用
    - JSON 数组字符串（如 '["ou_1","ou_2"]'）→ json.loads 解析
    - 逗号分隔字符串（如 'ou_1,ou_2'）→ 按逗号拆分

    空值 / 空字符串 → []。解析失败时按逗号拆分兜底，杜绝把 "[", "]" 当 open_id。
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except (ValueError, TypeError):
                pass
        return [v.strip() for v in s.split(",") if v.strip()]
    return []


class MessageSender:
    """飞书消息推送客户端"""

    def __init__(self, app_id: str, app_secret: str, receive_id_type: str = "chat_id"):
        """
        初始化消息推送客户端

        Args:
            app_id: 飞书应用ID
            app_secret: 飞书应用密钥
            receive_id_type: 接收者ID类型 (open_id/user_id/email/chat_id)
                             默认 chat_id，即推送到运维群
        """
        self.auth = FeishuAuth(app_id, app_secret)
        self.receive_id_type = receive_id_type
        self.base_url = "https://open.feishu.cn/open-apis/im/v1"

    def _get_headers(self) -> dict:
        """获取带认证的请求头"""
        token = self.auth.get_tenant_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def send_text_message(self, receive_id: str, text: str) -> dict:
        """
        发送文本消息

        Args:
            receive_id: 接收者ID（群 chat_id 或用户 open_id）
            text: 文本内容

        Returns:
            API响应字典
        """
        url = f"{self.base_url}/messages"
        params = {"receive_id_type": self.receive_id_type}
        payload = {
            "receive_id": receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}),
        }

        try:
            resp = requests.post(
                url,
                headers=self._get_headers(),
                params=params,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"发送文本消息失败: {e}")
            return {"code": -1, "msg": str(e), "data": {}}

    def send_card_message(self, receive_id: str, card: dict) -> dict:
        """
        发送消息卡片（标准 interactive 卡片）

        Args:
            receive_id: 接收者ID
            card: 飞书消息卡片内容字典（config/header/elements 结构）

        Returns:
            API响应字典
        """
        url = f"{self.base_url}/messages"
        params = {"receive_id_type": self.receive_id_type}
        payload = {
            "receive_id": receive_id,
            "msg_type": "interactive",
            "content": json.dumps(card),
        }

        try:
            resp = requests.post(
                url,
                headers=self._get_headers(),
                params=params,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"发送卡片消息失败: {e}")
            return {"code": -1, "msg": str(e), "data": {}}

    # ====================================================================
    # 预警卡片生成（飞书标准 interactive 卡片 JSON）
    # ====================================================================

    def create_alert_card(self, alert: dict) -> dict:
        """
        根据预警事件生成飞书标准 interactive 消息卡片。

        卡片结构遵循飞书消息卡片 JSON 规范：
        - header.template 控制卡片颜色（red/orange/yellow）
        - elements 使用 lark_md 富文本 + action 按钮区
        - 红色/橙色卡片附带「确认预警」「标记误报」操作按钮

        Args:
            alert: 预警事件字典，字段同 SentinelPipeline.run_anomaly_detection 输出

        Returns:
            飞书 interactive 卡片字典
        """
        level = str(alert.get("risk_level", "yellow")).lower()
        template = RISK_CARD_TEMPLATE.get(level, "yellow")
        emoji = RISK_EMOJI.get(level, "🟡")
        label_cn = RISK_LABEL_CN.get(level, "黄色")

        device_name = alert.get("device_name", "未知设备")
        fault_type = alert.get("fault_type", "未知故障")
        alert_id = alert.get("alert_id", "")
        trigger_time = alert.get("trigger_time", "-")
        confidence = alert.get("confidence", 0.0)
        abnormal_params = alert.get("abnormal_params", "-")
        detection_method = alert.get("detection_method", "-")
        primary_cause = alert.get("primary_cause", "")

        # 参数当前值格式化
        param_values = alert.get("param_values", {})
        if isinstance(param_values, dict):
            if param_values:
                values_lines = "\n".join(
                    f"    • {k}: **{v}**" for k, v in param_values.items()
                )
            else:
                values_lines = "    • （无）"
        else:
            values_lines = f"    • {param_values}"

        # 风险等级对应的紧急程度提示
        urgency_hint = {
            "red": "⚠️ **建议立即现场确认！**",
            "orange": "⏰ **请于2小时内关注处理。**",
            "yellow": "📋 已汇总至每日运维日报。",
        }.get(level, "")

        elements = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        f"**设备编号**: {alert.get('device_id', '-')}\n"
                        f"**故障类型**: {fault_type}\n"
                        f"**置信度**: {confidence * 100:.0f}%\n"
                        f"**预警时间**: {trigger_time}"
                    ),
                },
            },
            {
                "tag": "hr",
            },
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        f"**异常参数**:\n    • {abnormal_params}\n"
                        f"**当前值**:\n{values_lines}\n"
                        f"**检测方式**: {detection_method}"
                    ),
                },
            },
        ]

        # 根因分析（如有）
        if primary_cause:
            elements.append(
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**根因分析**: {primary_cause}",
                    },
                }
            )

        # 紧急程度提示
        if urgency_hint:
            elements.append({"tag": "hr"})
            elements.append(
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": urgency_hint},
                }
            )

        # 红色/橙色卡片附带操作按钮（黄色仅汇总，不即时推送，故无需按钮）
        if level in ("red", "orange"):
            elements.append(
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "✅ 确认预警"},
                            "type": "primary",
                            "value": {
                                "action": "confirm_alert",
                                "alert_id": alert_id,
                            },
                        },
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "🚫 标记误报"},
                            "type": "danger",
                            "value": {
                                "action": "mark_false_alarm",
                                "alert_id": alert_id,
                            },
                        },
                    ],
                }
            )

        card = {
            "config": {"wide_screen_mode": True, "enable_forward": False},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"{emoji} [{label_cn}预警] {device_name} - {fault_type}",
                },
                "template": template,
            },
            "elements": elements,
        }
        return card

    # ====================================================================
    # Agent 3 风险分级推送策略
    # ====================================================================

    def send_alert(
        self,
        alert: dict,
        receive_id: str,
        mention_open_ids: Optional[list] = None,
    ) -> dict:
        """
        根据风险等级执行推送策略（Agent 3 入口方法）。

        策略：
        - red/orange：即时推送消息卡片；如提供 mention_open_ids 则在卡片后追加 @提醒
        - yellow：不即时推送，返回跳过标记（由调用方汇总至日报）

        Args:
            alert: 预警事件字典
            receive_id: 接收群/用户ID
            mention_open_ids: 需要 @提醒 的用户 open_id 列表

        Returns:
            推送结果字典：
            - code=0 且 pushed=True：已推送
            - code=0 且 pushed=False：黄色预警已跳过即时推送
            - code=-1：推送失败
        """
        level = str(alert.get("risk_level", "yellow")).lower()
        mention_open_ids = parse_open_ids(mention_open_ids)

        if level not in INSTANT_PUSH_LEVELS:
            logger.info(
                f"黄色预警 {alert.get('alert_id')} 跳过即时推送，将汇总至日报"
            )
            return {
                "code": 0,
                "msg": "yellow_skipped",
                "pushed": False,
                "data": {},
            }

        card = self.create_alert_card(alert)

        # 在卡片末尾追加 @提醒（飞书卡片通过 mention 元素实现）
        if mention_open_ids:
            mention_elements = [
                {
                    "tag": "mention",
                    "user_id": open_id,
                }
                for open_id in mention_open_ids
            ]
            # @提醒 需放在一个 div 内，且需要文本节点承载
            card["elements"].append(
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": "请相关责任人及时处理："},
                    "elements": mention_elements,
                }
            )

        logger.info(
            f"推送 {level} 预警卡片 alert_id={alert.get('alert_id')} → {receive_id}"
        )
        resp = self.send_card_message(receive_id, card)

        if resp.get("code") == 0:
            return {
                "code": 0,
                "msg": "ok",
                "pushed": True,
                "data": resp.get("data", {}),
            }
        return {
            "code": -1,
            "msg": resp.get("msg", "send_failed"),
            "pushed": False,
            "data": {},
        }

    def send_daily_digest(
        self,
        receive_id: str,
        yellow_alerts: list,
        date_str: Optional[str] = None,
    ) -> dict:
        """
        发送每日运维日报（汇总黄色预警）。

        Args:
            receive_id: 接收群ID
            yellow_alerts: 黄色预警事件列表
            date_str: 日报日期字符串，默认当天

        Returns:
            API 响应字典
        """
        from datetime import datetime

        date_str = date_str or datetime.now().strftime("%Y-%m-%d")

        if not yellow_alerts:
            content = f"📋 **{date_str} 运维日报**\n\n今日无黄色预警事件。"
        else:
            lines = [f"📋 **{date_str} 运维日报**", f"黄色预警共 {len(yellow_alerts)} 条：\n"]
            for i, a in enumerate(yellow_alerts[:50], 1):  # 最多列 50 条
                lines.append(
                    f"{i}. {a.get('trigger_time', '-')} | "
                    f"{a.get('device_name', '-')} | "
                    f"{a.get('fault_type', '-')} | "
                    f"置信度 {a.get('confidence', 0) * 100:.0f}%"
                )
            if len(yellow_alerts) > 50:
                lines.append(f"\n...（另有 {len(yellow_alerts) - 50} 条，已省略）")

            content = "\n".join(lines)

        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"🟡 {date_str} 运维日报",
                },
                "template": "yellow",
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": content}}
            ],
        }
        return self.send_card_message(receive_id, card)
