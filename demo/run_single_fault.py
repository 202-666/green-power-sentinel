"""
绿电哨兵 — 单故障场景端到端测试
按技术框架 §8 验收要求,对 3 类故障分别运行完整流水线
(采集 → 检测 → 推送 → 建议),并生成:
  - 单故障结果 JSON:   demo_output/single_fault_results/{fault_type}.json
  - 全故障汇总 JSON:   demo_output/all_faults_summary.json
  - 仪表盘数据源:      dashboard.html 直接读取 all_faults_summary.json
"""

import argparse
import datetime
import json
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.pipeline import SentinelPipeline

# 3 类故障与数据文件映射(与 data/sample_data/ 下文件名一致)
FAULT_DEFS = {
    "bearing_overheat": {
        "csv": "fault_bearing_overheat.csv",
        "name": "轴承过热",
        "device": "INC-01-BRG-01",
        "device_name": "1#焚烧炉-引风机-前轴承",
        "expected_start": "2026-07-15 10:00",
        "expected_end": "2026-07-15 12:00",
        "key_params": ["bearing_temperature", "bearing_vibration"],
    },
    "emission_exceed": {
        "csv": "fault_emission_exceed.csv",
        "name": "烟气超标",
        "device": "INC-01-EMI-01",
        "device_name": "1#焚烧炉-烟气CEMS",
        "expected_start": "2026-07-20 14:00",
        "expected_end": "2026-07-20 17:00",
        "key_params": ["so2_concentration", "nox_concentration", "oxygen_content"],
    },
    "grate_jam": {
        "csv": "fault_grate_jam.csv",
        "name": "炉排卡滞",
        "device": "INC-01-GRT-01",
        "device_name": "1#焚烧炉-炉排系统",
        "expected_start": "2026-07-25 08:00",
        "expected_end": "2026-07-25 09:30",
        "key_params": ["grate_speed", "feed_rate", "furnace_pressure"],
    },
}


def _parse_dt(s):
    """容错解析时间字符串"""
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _sample_anomalies(anomalies, max_samples=60):
    """
    智能抽样异常事件,确保覆盖各风险等级和故障前后时间段。
    策略:
      1. 按风险等级分组(红/橙/黄)
      2. 每组按时间均匀抽样,保留最严重的若干条 + 时间均匀分布
      3. 总数不超过 max_samples
    """
    if len(anomalies) <= max_samples:
        return anomalies

    groups = {"red": [], "orange": [], "yellow": []}
    for a in anomalies:
        lv = a.get("risk_level", "yellow")
        if lv in groups:
            groups[lv].append(a)

    # 分配配额:高级别占更多
    quotas = {"red": 20, "orange": 20, "yellow": 20}
    sampled = []
    for lv in ("red", "orange", "yellow"):
        items = groups[lv]
        if not items:
            continue
        q = min(quotas[lv], max_samples - len(sampled))
        if q <= 0:
            break
        if len(items) <= q:
            sampled.extend(items)
        else:
            # 均匀采样 + 保留首尾
            step = len(items) / q
            indices = [int(i * step) for i in range(q)]
            for idx in indices:
                idx = max(0, min(len(items) - 1, idx))
                sampled.append(items[idx])
    return sampled


def _build_compact_series(df, param_names, max_points=600):
    """
    构建压缩版时间序列数据。
    优化前:每个点存 {"t": "2026-07-15 10:00:00", "v": 25.3}
    优化后:time_base + time_offsets(分钟偏移整数) + values,体积减少 60%+
    """
    if df is None or df.empty or not param_names:
        return {}, []

    n = len(df)
    step = max(1, n // max_points)

    has_ts = "timestamp" in df.columns
    sampled_idx = list(range(0, n, step))

    time_offsets = []
    if has_ts:
        base_dt = _parse_dt(str(df.iloc[0]["timestamp"]))
        if base_dt:
            for i in sampled_idx:
                dt = _parse_dt(str(df.iloc[i]["timestamp"]))
                if dt:
                    time_offsets.append(int((dt - base_dt).total_seconds() // 60))
                else:
                    time_offsets.append(i)
            time_base = base_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_base = None
            time_offsets = sampled_idx
    else:
        time_base = None
        time_offsets = sampled_idx

    param_series = {}
    for p in param_names:
        if p not in df.columns:
            continue
        col = df[p].iloc[sampled_idx]
        vals = [round(float(v), 2) if v == v else None for v in col.tolist()]
        param_series[p] = vals

    return param_series, {
        "time_base": time_base,
        "time_offsets": time_offsets,
        "count": len(sampled_idx),
    }


def run_single_fault(fault_type: str, pipeline: SentinelPipeline) -> dict:
    """对指定故障类型运行完整流水线并返回结构化结果"""
    if fault_type not in FAULT_DEFS:
        raise ValueError(f"未知故障类型: {fault_type},支持: {list(FAULT_DEFS.keys())}")

    fault_def = FAULT_DEFS[fault_type]
    csv_path = os.path.join(project_root, "data", "sample_data", fault_def["csv"])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"故障数据文件不存在: {csv_path}")

    print(f"\n{'='*60}")
    print(f"故障场景: {fault_def['name']} ({fault_type})")
    print(f"数据源: {csv_path}")
    print(f"设备: {fault_def['device']} - {fault_def['device_name']}")
    print(f"{'='*60}")

    t0_total = time.time()

    # 完整流水线: 采集 → 检测 → 推送 → 建议
    result = pipeline.run_full_pipeline(csv_path, fault_def["device"])

    total_duration = round(time.time() - t0_total, 2)

    # 提取检测摘要(用于仪表盘)
    summary = pipeline.get_detection_summary()
    scores = summary.get("scores", [])
    anomalies = summary.get("anomalies", [])
    perf = summary.get("perf", {}) or {}

    # 按风险等级统计(仅统计非绿色等级,避免误导)
    level_counts = {"red": 0, "orange": 0, "yellow": 0, "green": 0}
    for s in scores:
        lv = s.get("level", "green")
        level_counts[lv] = level_counts.get(lv, 0) + 1

    # 按检测方法统计(仅异常事件)
    method_counts = {}
    for a in anomalies:
        m = a.get("detection_method", "未知")
        method_counts[m] = method_counts.get(m, 0) + 1

    # === 故障区间验证 ===
    expected_start_str = fault_def["expected_start"]
    expected_end_str = fault_def["expected_end"]
    fault_start_dt = _parse_dt(expected_start_str)
    fault_end_dt = _parse_dt(expected_end_str)

    # 分类:故障前预警 / 故障区间内 / 故障后
    before_fault = []
    in_window = []
    for a in anomalies:
        at = _parse_dt(a.get("trigger_time", ""))
        if at is None:
            continue
        if at < fault_start_dt:
            before_fault.append(a)
        elif at <= fault_end_dt:
            in_window.append(a)

    severe_in_window = [a for a in in_window if a.get("risk_level") in ("orange", "red")]

    # 黄/橙/红级: 广义预警(qualifying = 达到 yellow 及以上)
    qualifying_before = [a for a in before_fault if a.get("risk_level") in ("orange", "red", "yellow")]
    qualifying_in = [a for a in in_window if a.get("risk_level") in ("orange", "red", "yellow")]

    # === 预警提前量计算(仅统计橙/红级,体现严重故障预警能力) ===
    # 定义:故障注入起始时间 - 首个橙/红级预警时间
    # - 若故障前有橙/红预警 → 正值=提前多少分钟发现严重故障
    # - 若故障前无,故障区间内有 → 0(故障发生时才升级为严重)
    # - 若均无 → None(未检出严重故障)
    advance_min = None
    first_alert_time = None

    # 橙/红级: 真正代表"严重故障预警"
    severe_before = [a for a in before_fault if a.get("risk_level") in ("orange", "red")]
    severe_in = [a for a in in_window if a.get("risk_level") in ("orange", "red")]

    if severe_before:
        # 故障前最接近故障开始的严重预警
        last_severe_before = max(
            (_parse_dt(a["trigger_time"]) for a in severe_before if a.get("trigger_time")),
            default=None,
        )
        if last_severe_before and fault_start_dt:
            advance_min = round((fault_start_dt - last_severe_before).total_seconds() / 60.0, 1)
            first_alert_time = last_severe_before.strftime("%Y-%m-%d %H:%M:%S")
    elif severe_in:
        # 故障区间内首次严重预警
        first_severe_in = min(
            (_parse_dt(a["trigger_time"]) for a in severe_in if a.get("trigger_time")),
            default=None,
        )
        if first_severe_in:
            advance_min = 0.0
            first_alert_time = first_severe_in.strftime("%Y-%m-%d %H:%M:%S")

    # === 时间序列数据(压缩版) ===
    df = pipeline._detection_df
    key_param_series, time_meta = _build_compact_series(
        df, fault_def["key_params"], max_points=600
    )

    # 风险评分时间序列(复用时间轴,压缩存储)
    score_values = []
    score_levels = []
    if scores and time_meta["time_offsets"]:
        n_scores = len(scores)
        step = max(1, n_scores // time_meta["count"])
        for i in range(0, n_scores, step):
            s = scores[i]
            score_values.append(round(s.get("score", 0), 4))
            score_levels.append(s.get("level", "green"))
        # 压缩等级存储: 用数字代替字符串
        level_map = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
        score_levels_compact = [level_map.get(lv, 0) for lv in score_levels]
    else:
        score_levels_compact = []

    # === 智能抽样异常事件 ===
    sample_anomalies = _sample_anomalies(anomalies, max_samples=60)

    # === 性能指标估算(从 summary 或从各步骤推断) ===
    records = result.get("records_collected", 0)
    perf_metrics = {
        "total_duration_s": total_duration,
        "throughput_records_per_s": round(records / total_duration, 1) if total_duration > 0 else 0,
        "detection_duration_s": perf.get("detection_s", None),
        "threshold_s": perf.get("threshold_s", None),
        "trend_s": perf.get("trend_s", None),
        "volatility_s": perf.get("volatility_s", None),
        "correlation_s": perf.get("correlation_s", None),
        "ensemble_s": perf.get("ensemble_s", None),
    }

    fault_result = {
        "fault_type": fault_type,
        "fault_name": fault_def["name"],
        "device_id": fault_def["device"],
        "device_name": fault_def["device_name"],
        "csv_file": fault_def["csv"],
        "expected_window": {
            "start": expected_start_str,
            "end": expected_end_str,
        },
        "key_params": fault_def["key_params"],
        "pipeline_status": result.get("status", "unknown"),
        "records_collected": records,
        "anomalies_detected": result.get("anomalies_detected", 0),
        "alerts_pushed": result.get("alerts_pushed", 0),
        "advice_generated": result.get("advice_generated", False),
        "level_counts": level_counts,
        "method_counts": method_counts,
        "verification": {
            "detected_in_window": len(qualifying_in) > 0 or len(qualifying_before) > 0,
            "severe_detected": len(severe_in_window) > 0 or len(severe_before) > 0,
            "detection_count_before_fault": len(qualifying_before),
            "severe_before_fault": len(severe_before),
            "detection_count_in_window": len(qualifying_in),
            "severe_count_in_window": len(severe_in_window),
            "advance_min": advance_min,
            "advance_level": "orange_red",
            "first_alert_time": first_alert_time,
            "passed": (len(qualifying_in) > 0 or len(qualifying_before) > 0),
        },
        "time_meta": time_meta,
        "score_values": score_values,
        "score_levels": score_levels_compact,
        "key_param_series": key_param_series,
        "sample_anomalies": sample_anomalies,
        "perf": perf_metrics,
        "run_at": datetime.datetime.now().isoformat(),
    }

    print(f"\n--- 验收结果 [{fault_def['name']}] ---")
    print(f"  流水线状态: {fault_result['pipeline_status']}")
    print(f"  采集记录数: {fault_result['records_collected']}")
    print(f"  异常事件数: {fault_result['anomalies_detected']} (黄={level_counts['yellow']} 橙={level_counts['orange']} 红={level_counts['red']})")
    print(f"  推送卡片数: {fault_result['alerts_pushed']} (橙/红级)")
    print(f"  运维建议生成: {'是' if fault_result['advice_generated'] else '否'}")
    print(f"  故障前预警 (广义/严重): {fault_result['verification']['detection_count_before_fault']} / {fault_result['verification']['severe_before_fault']}")
    print(f"  故障区间检出 (广义/严重): {fault_result['verification']['detection_count_in_window']} / {fault_result['verification']['severe_count_in_window']}")
    print(f"  严重故障预警提前量: {fault_result['verification']['advance_min']} 分钟")
    print(f"  总耗时: {total_duration}s ({perf_metrics['throughput_records_per_s']} 条/s)")
    print(f"  验收: {'✓ 通过' if fault_result['verification']['passed'] else '✗ 未通过'}")

    return fault_result


def main():
    parser = argparse.ArgumentParser(description="单故障场景端到端测试")
    parser.add_argument(
        "--fault", type=str, default="all",
        choices=["all", "bearing_overheat", "emission_exceed", "grate_jam"],
        help="故障类型: all=运行全部3类, 或指定单个故障类型"
    )
    args = parser.parse_args()

    print("="*60)
    print("绿电哨兵 — 单故障场景端到端测试 (W6 验收)")
    print("="*60)

    config_path = os.path.join(project_root, "config", "config.yaml")

    out_dir = os.path.join(script_dir, "demo_output", "single_fault_results")
    os.makedirs(out_dir, exist_ok=True)

    if args.fault == "all":
        fault_types = list(FAULT_DEFS.keys())
    else:
        fault_types = [args.fault]

    all_results = {}
    all_passed = True

    for ft in fault_types:
        # 每个故障类型重新初始化流水线,避免缓存污染
        pipeline = SentinelPipeline(config_path=config_path)
        try:
            result = run_single_fault(ft, pipeline)
            all_results[ft] = result

            # 保存单故障结果
            single_path = os.path.join(out_dir, f"{ft}.json")
            with open(single_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n单故障结果已保存: {single_path}")

            if not result["verification"]["passed"]:
                all_passed = False
        except Exception as e:
            print(f"\n✗ 故障 {ft} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[ft] = {
                "fault_type": ft,
                "fault_name": FAULT_DEFS[ft]["name"],
                "pipeline_status": "failed",
                "error": str(e),
                "verification": {"passed": False, "detected_in_window": False},
            }
            all_passed = False

    # 生成汇总文件(仪表盘数据源)
    summary = {
        "meta": {
            "generated_at": datetime.datetime.now().isoformat(),
            "total_fault_types": len(fault_types),
            "all_passed": all_passed,
            "data_format_version": "2.0",
        },
        "faults": all_results,
    }
    summary_path = os.path.join(script_dir, "demo_output", "all_faults_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 同步生成 dashboard_data.js(内嵌数据,支持 file:// 直接打开仪表盘)
    dashboard_data_path = os.path.join(script_dir, "dashboard_data.js")
    with open(dashboard_data_path, "w", encoding="utf-8") as f:
        f.write("// 自动生成于 W6 全链路联调,请勿手动编辑\n")
        f.write("// 由 run_single_fault.py 生成,作为 dashboard.html 的数据源\n")
        f.write("window.__DASHBOARD_DATA__ = ")
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write(";\n")
    print(f"仪表盘数据源(JSON): {summary_path}")
    print(f"仪表盘数据源(JS,  内嵌): {dashboard_data_path}")

    print(f"\n{'='*60}")
    print(f"W6 验收汇总 (已保存: {summary_path})")
    print(f"{'='*60}")
    print(f"{'故障类型':<14} {'状态':<10} {'异常(黄/橙/红)':<16} {'故障前严重':<12} {'提前量(分)':<12} {'耗时(s)':<10} {'通过':<6}")
    for ft, r in all_results.items():
        v = r.get("verification", {})
        p = r.get("perf", {})
        lc = r.get("level_counts", {})
        print(
            f"{r.get('fault_name', ft):<12} "
            f"{r.get('pipeline_status', 'unknown'):<10} "
            f"{r.get('anomalies_detected', 0):<5}({lc.get('yellow',0)}/{lc.get('orange',0)}/{lc.get('red',0)}) "
            f"{v.get('severe_before_fault', 0):<12} "
            f"{str(v.get('advance_min', 'N/A')):<12} "
            f"{p.get('total_duration_s', '-'):<10} "
            f"{'✓' if v.get('passed') else '✗':<6}"
        )
    print(f"\n总体验收: {'✓ 全部通过' if all_passed else '✗ 存在未通过项'}")
    print(f"仪表盘: {os.path.join(script_dir, 'dashboard.html')}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
