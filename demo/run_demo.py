"""
绿电哨兵 Demo 运行脚本
一键运行完整流水线演示
"""

import argparse
import datetime
import json
import os
import sys

# 确保项目根目录在 Python 路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.pipeline import SentinelPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Green Power Sentinel Demo")
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "online"],
        help="offline: 使用本地CSV数据; online: 连接飞书多维表格"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sample_data/normal_30days.csv",
        help="数据源路径（CSV文件）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="INC-01-BRG-01",
        help="设备编号"
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"绿电哨兵 Demo")
    print(f"{'='*60}")
    print(f"模式: {args.mode}")
    print(f"数据源: {args.data}")
    print(f"设备: {args.device}")
    print(f"项目根目录: {project_root}")
    print(f"{'='*60}")

    # 初始化流水线，传入绝对路径配置，避免依赖当前工作目录
    config_path = os.path.join(project_root, "config", "config.yaml")
    pipeline = SentinelPipeline(config_path=config_path)

    if args.mode == "offline":
        # 离线模式：使用本地CSV（支持相对路径，基于项目根目录解析）
        data_path = args.data if os.path.isabs(args.data) else os.path.join(project_root, args.data)
        result = pipeline.run_full_pipeline(data_path, args.device)
    else:
        # 在线模式：从飞书多维表格「运行数据」表拉取（H1 修复）
        # 未配置 FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_APP_TOKEN /
        # FEISHU_TABLE_RUNTIME 时，pipeline 返回失败报告并给出明确提示，
        # 不再走 CSV 分支抛 FileNotFoundError。
        print("在线模式：从飞书多维表格读取运行数据（需配置飞书API凭据）")
        result = pipeline.run_full_pipeline("bitable", args.device)

    print(f"\n{'='*60}")
    print(f"执行结果:")
    print(f"{'='*60}")
    for key, value in result.items():
        print(f"  {key}: {value}")

    # 离线模式：输出检测结果JSON
    if args.mode == "offline":
        output_dir = os.path.join(script_dir, "demo_output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "detection_results.json")

        detection_detail = {
            "meta": {
                "mode": args.mode,
                "data_source": args.data,
                "device_id": args.device,
                "generated_at": datetime.datetime.now().isoformat(),
                "total_records": result.get("records_collected", 0),
                "anomalies_detected": result.get("anomalies_detected", 0),
            },
            "time_series": [],
            "anomalies": [],
        }

        summary = pipeline.get_detection_summary()
        if summary.get("scores"):
            scores = summary["scores"]
            df = pipeline._detection_df

            for i, score in enumerate(scores):
                ts = None
                if df is not None and "timestamp" in df.columns and i < len(df):
                    ts = str(df.iloc[i]["timestamp"])
                entry = {
                    "index": i,
                    "timestamp": ts,
                    "score": score["score"],
                    "level": score["level"],
                    "confidence": score["confidence"],
                    "primary_cause": score.get("primary_cause", ""),
                }
                detection_detail["time_series"].append(entry)

        if summary.get("anomalies"):
            detection_detail["anomalies"] = summary["anomalies"]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(detection_detail, f, ensure_ascii=False, indent=2)

        print(f"\n检测结果已保存: {output_path}")
        print(f"  - 总记录数: {detection_detail['meta']['total_records']}")
        print(f"  - 异常事件数: {detection_detail['meta']['anomalies_detected']}")
        print(f"  - 时间序列评分点数: {len(detection_detail['time_series'])}")

    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
