"""
绿电哨兵 — 最小调度器示例（M4 修复）

状态说明：本模块提供 config/config.yaml 中 scheduling 配置语义的最小可运行示例
（`while True + sleep` 循环），满足「调度器有配置也有实现」；
生产级调度（持久化游标、断点续跑、并发任务、失败重试队列）列为后续工作，
详见 docs/W8_技术文档.md §5.3。

配置语义（config.yaml scheduling）：
- agent1_interval_min: Agent1 数据采集周期（分钟）
- agent2_interval_min: Agent2 异常检测周期（分钟）
- agent3_mode: "realtime" → 检出即推送（当前 run_alert_push 行为）
- agent4_mode: "on_confirm" → 预警确认后生成运维建议（外部调用 run_maintenance_advice）
"""

import logging
import os
import time

logger = logging.getLogger(__name__)


def _resolve_data_source(pipeline) -> str:
    """按 config.data_source 解析调度器数据源（bitable 或 CSV 文件路径）。"""
    ds_cfg = pipeline.config.get("data_source", {}) or {}
    if ds_cfg.get("type") == "bitable":
        return "bitable"
    csv_dir = str(ds_cfg.get("csv_path", "data/sample_data"))
    if not os.path.isabs(csv_dir):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_dir = os.path.join(base, csv_dir)
    return os.path.join(csv_dir, "normal_30days.csv")


def run_scheduler(pipeline, data_source=None, device_id=None, once=False) -> dict:
    """
    最小调度循环：按 scheduling 配置周期性执行数据采集与异常检测。

    Args:
        pipeline: SentinelPipeline 实例（已加载 config）
        data_source: 数据源（默认按 config.data_source 解析）
        device_id: 设备编号（默认取 config.data_source.device_id）
        once: True 时只执行一轮并返回摘要（用于测试/演示）；False 时持续循环

    Returns:
        once=True 时返回本轮摘要 dict；持续循环模式不返回。
    """
    scheduling = pipeline.config.get("scheduling", {}) or {}
    ds_cfg = pipeline.config.get("data_source", {}) or {}
    agent1_interval_min = max(1, int(scheduling.get("agent1_interval_min", 1) or 1))
    agent2_interval_min = max(1, int(scheduling.get("agent2_interval_min", 5) or 5))
    agent3_mode = str(scheduling.get("agent3_mode", "realtime"))
    agent4_mode = str(scheduling.get("agent4_mode", "on_confirm"))

    if data_source is None:
        data_source = _resolve_data_source(pipeline)
    if device_id is None:
        device_id = str(ds_cfg.get("device_id", "INC-01-BRG-01"))

    logger.info(
        f"调度器启动: agent1={agent1_interval_min}min, agent2={agent2_interval_min}min, "
        f"agent3={agent3_mode}, agent4={agent4_mode}, data_source={data_source}"
    )

    cycle = 0
    last_collect = 0.0
    last_detect = 0.0
    summary = {
        "cycles": 0,
        "records_collected": 0,
        "anomalies_detected": 0,
        "alerts_pushed": 0,
    }

    while True:
        cycle += 1
        now = time.monotonic()
        if cycle == 1 or now - last_collect >= agent1_interval_min * 60:
            logger.info(f"调度[第{cycle}轮]: Agent1 数据采集")
            try:
                n = pipeline.run_data_collection(data_source)
                summary["records_collected"] = n
                logger.info(f"调度[第{cycle}轮]: Agent1 采集 {n} 行")
            except Exception as e:
                logger.error(f"调度[第{cycle}轮]: Agent1 采集失败，下轮重试: {e}")
            last_collect = time.monotonic()

        if cycle == 1 or now - last_detect >= agent2_interval_min * 60:
            logger.info(f"调度[第{cycle}轮]: Agent2 异常检测")
            try:
                anomalies = pipeline.run_anomaly_detection(device_id)
                summary["anomalies_detected"] = len(anomalies)
                pushed = 0
                if agent3_mode == "realtime":
                    for alert in anomalies:
                        if pipeline.run_alert_push(alert):
                            pushed += 1
                summary["alerts_pushed"] = pushed
                logger.info(
                    f"调度[第{cycle}轮]: Agent2 检出 {len(anomalies)} 条, "
                    f"Agent3 推送 {pushed} 条"
                )
                # agent4_mode="on_confirm"：运维建议在预警确认后由外部调用
                # run_maintenance_advice(alert_id) 触发，调度器不主动生成
            except Exception as e:
                logger.error(f"调度[第{cycle}轮]: Agent2 检测失败，下轮重试: {e}")
            last_detect = time.monotonic()

        summary["cycles"] = cycle
        if once:
            break
        time.sleep(min(agent1_interval_min, agent2_interval_min) * 60)

    return summary
