"""
绿电哨兵 — 四Agent流水线调度器
负责协调数据采集、异常检测、预警推送、运维建议四个Agent
"""

import yaml
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentinelPipeline:
    """四Agent流水线调度器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._cached_df = None
        self._data_source_path = None
        self._detection_scores = None
        self._detection_df = None
        self._last_anomalies = []
        self._yellow_tracker = {}
        self._init_modules()
        logger.info(f"SentinelPipeline initialized, mode={self.config.get('app', {}).get('mode', 'unknown')}")

    def _load_config(self) -> dict:
        """加载配置文件，支持环境变量解析"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config = self._resolve_env_vars(config)
        logger.debug(f"Config loaded from {self.config_path}")
        return config

    def _resolve_env_vars(self, obj):
        """递归解析配置中的环境变量，格式: ${ENV_VAR:-default}"""
        import re
        pattern = re.compile(r'\$\{([^}]+)\}')

        def _resolve(value):
            if isinstance(value, str):
                matches = pattern.findall(value)
                for match in matches:
                    parts = match.split(':-', 1)
                    env_name = parts[0].strip()
                    default = parts[1].strip() if len(parts) > 1 else ''
                    env_value = os.environ.get(env_name, default)
                    value = value.replace(f'${{{match}}}', env_value)
                return value
            elif isinstance(value, dict):
                return {k: _resolve(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_resolve(item) for item in value]
            else:
                return value

        return _resolve(obj)

    def _init_modules(self):
        """初始化各Agent模块"""
        from core.data_cleaner import clean_data, check_sensor_health
        from models.threshold_detector import detect_threshold, detect_threshold_batch, load_thresholds_from_yaml
        from models.trend_detector import detect_trend, detect_trend_batch, detect_trend_multi_params
        from core.normal_baseline import compute_baseline, update_baseline

        self._clean_data_func = clean_data
        self._check_sensor_health_func = check_sensor_health
        self._detect_threshold_func = detect_threshold
        self._detect_threshold_batch_func = detect_threshold_batch
        self._detect_trend_func = detect_trend
        self._detect_trend_batch_func = detect_trend_batch
        self._detect_trend_multi_func = detect_trend_multi_params
        self._compute_baseline_func = compute_baseline
        self._update_baseline_func = update_baseline

        self._thresholds = None
        self._trend_thresholds = None  # 动态阈值（来自 normal_baseline）
        self._baseline_stats = None
        detection_cfg = self.config.get("detection", {})
        if detection_cfg and self.config.get("data_source", {}).get("type") != "bitable":
            try:
                import os
                thresholds_path = os.path.join(
                    os.path.dirname(self.config_path), "thresholds.yaml"
                )
                if os.path.exists(thresholds_path):
                    self._thresholds = load_thresholds_from_yaml(thresholds_path)
                    logger.debug(f"阈值配置已加载，包含 {len(self._thresholds)} 个参数")
            except Exception as e:
                logger.warning(f"加载阈值配置失败: {e}")

        self._bitable_client = None
        self._message_sender = None
        self._maintenance_advisor = None

    def run_data_collection(self, data_source: str) -> int:
        """
        Agent 1: 数据采集哨兵
        从数据源获取设备运行数据，清洗后缓存到内存（离线模式）

        Args:
            data_source: 数据源路径（CSV文件或多维表格ID）

        Returns:
            清洗后的记录数
        """
        import pandas as pd
        logger.info(f"Agent 1: Starting data collection from {data_source}")

        if not os.path.exists(data_source):
            ds_cfg = self.config.get("data_source", {})
            if ds_cfg.get("type") == "csv":
                csv_dir = ds_cfg.get("csv_path", "data/sample_data")
                if not os.path.isabs(csv_dir):
                    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    csv_dir = os.path.join(base, csv_dir)
                alt_path = os.path.join(csv_dir, os.path.basename(data_source))
                if os.path.exists(alt_path):
                    data_source = alt_path
                else:
                    alt_path2 = os.path.join(csv_dir, data_source)
                    if os.path.exists(alt_path2):
                        data_source = alt_path2

        if not os.path.exists(data_source):
            raise FileNotFoundError(f"Data source not found: {data_source}")

        raw_df = pd.read_csv(data_source, low_memory=False)
        logger.info(f"Raw data loaded: {len(raw_df)} rows from {data_source}")

        cleaned_df = self._clean_data_func(raw_df)
        self._cached_df = cleaned_df
        self._data_source_path = data_source
        logger.info(f"Data cleaned: {len(cleaned_df)} rows remaining")
        return len(cleaned_df)

    def run_anomaly_detection(self, device_id: str) -> list:
        """
        Agent 2: 异常检测哨兵
        从缓存数据读取，执行阈值/趋势/波动率/关联检测及综合评分

        Args:
            device_id: 设备编号

        Returns:
            检测结果列表（仅包含 yellow/orange/red 级别的时间点）
        """
        import time
        import pandas as pd
        import numpy as np
        from models.volatility_detector import detect_volatility_multi_params
        from models.correlation_detector import detect_correlation_batch
        from models.ensemble_scorer import compute_risk_score_batch
        from core.data_cleaner import PARAM_COLUMNS

        logger.info(f"Agent 2: Starting anomaly detection for {device_id}")

        if self._cached_df is None or self._cached_df.empty:
            logger.warning("No cached data available for anomaly detection")
            return []

        df = self._cached_df.reset_index(drop=True)
        param_columns = [c for c in PARAM_COLUMNS if c in df.columns]
        n = len(df)

        logger.info(f"Running detection on {n} rows, {len(param_columns)} params")

        perf = {}
        t_detect_start = time.time()
        detection_cfg = self.config.get("detection", {})

        # 1. 阈值检测 (batch) - 带异常恢复
        t0 = time.time()
        try:
            threshold_hits = self._detect_threshold_batch_func(df, self._thresholds, param_columns)
        except Exception as e:
            logger.error(f"阈值检测失败，跳过该模块: {e}")
            threshold_hits = []
        th_per_time = [[] for _ in range(n)]
        for hit in threshold_hits:
            row_idx = hit.get("row_idx", 0)
            if isinstance(row_idx, (int, np.integer)) and 0 <= row_idx < n:
                th_per_time[row_idx].append(hit)
        perf["threshold_s"] = round(time.time() - t0, 3)
        logger.info(f"  Threshold detection: {len(threshold_hits)} hits in {perf['threshold_s']:.2f}s")

        # 2. 趋势检测 (batch) - 带异常恢复
        t0 = time.time()
        trend_thresholds = None
        trend_multi = {}
        try:
            if detection_cfg.get("trend_dynamic_threshold", False):
                normal_mask = df.get("data_quality_flag", pd.Series(["正常"] * len(df))) != "故障注入"
                baseline_df = df[normal_mask] if normal_mask.any() else df
                baseline = self._compute_baseline_func(
                    baseline_df, param_columns,
                    window_size=detection_cfg.get("trend_baseline_window", 480),
                    trend_windows=detection_cfg.get("trend_windows", [10, 30, 60]),
                    sensitivity_k=detection_cfg.get("trend_sensitivity_k", 3.0),
                )
                self._trend_thresholds = baseline["slope_thresholds"]
                self._baseline_stats = baseline["stats"]
                trend_thresholds = self._trend_thresholds
                logger.info(f"  动态基线阈值已计算: {len(trend_thresholds)} 参数")
            trend_multi = self._detect_trend_multi_func(df, param_columns, thresholds=trend_thresholds)
        except Exception as e:
            logger.error(f"趋势检测失败，跳过该模块: {e}")
            trend_multi = {}
        tr_per_time = []
        for i in range(n):
            tdict = {}
            for col in param_columns:
                tdf = trend_multi.get(col)
                if tdf is None or i >= len(tdf):
                    continue
                row = tdf.iloc[i]
                tdict[col] = {
                    "param": col,
                    "any_detected": bool(row.get("any_detected", False)),
                    "max_level": row.get("max_level"),
                }
                for w in [10, 30, 60]:
                    sk = f"slope_{w}"
                    dk = f"detected_{w}"
                    lk = f"level_{w}"
                    if sk in row:
                        tdict[col][f"window_{w}"] = {
                            "slope": float(row[sk]),
                            "detected": bool(row.get(dk, False)),
                            "level": row.get(lk),
                        }
            tr_per_time.append(tdict)
        perf["trend_s"] = round(time.time() - t0, 3)
        logger.info(f"  Trend detection: completed in {perf['trend_s']:.2f}s")

        # 3. 波动率检测 (batch) - 带异常恢复
        t0 = time.time()
        vol_multi = {}
        try:
            vol_current = detection_cfg.get("volatility_current_window", 30)
            vol_baseline = detection_cfg.get("volatility_baseline_window", 1440)
            vol_multi = detect_volatility_multi_params(
                df, param_columns, current_window=vol_current, baseline_window=vol_baseline
            )
        except Exception as e:
            logger.error(f"波动率检测失败，跳过该模块: {e}")
            vol_multi = {}
        vol_per_time = []
        for i in range(n):
            vdict = {}
            for col in param_columns:
                vdf = vol_multi.get(col)
                if vdf is None or i >= len(vdf):
                    continue
                row = vdf.iloc[i]
                vdict[col] = {
                    "param": col,
                    "current_std": float(row.get("current_std", 0.0)),
                    "baseline_std": float(row.get("baseline_std", 0.0)),
                    "ratio": float(row.get("ratio", 1.0)),
                    "level": row.get("level"),
                    "detected": bool(row.get("detected", False)),
                }
            vol_per_time.append(vdict)
        perf["volatility_s"] = round(time.time() - t0, 3)
        logger.info(f"  Volatility detection: completed in {perf['volatility_s']:.2f}s")

        # 4. 关联检测 (batch) - 带异常恢复
        t0 = time.time()
        rules = []
        corr_batch = []
        try:
            rules_path = os.path.join(os.path.dirname(self.config_path), "rules.yaml")
            if os.path.exists(rules_path):
                with open(rules_path, "r", encoding="utf-8") as f:
                    rules = yaml.safe_load(f).get("rules", [])
            corr_batch = detect_correlation_batch(df, rules, trend_results=trend_multi, volatility_results=vol_multi)
        except Exception as e:
            logger.error(f"关联检测失败，跳过该模块: {e}")
            corr_batch = [{"matched_rules": []} for _ in range(n)]
        corr_per_time = [item["matched_rules"] for item in corr_batch]
        perf["correlation_s"] = round(time.time() - t0, 3)
        logger.info(f"  Correlation detection: completed in {perf['correlation_s']:.2f}s")

        # 5. 综合评分 - 带异常恢复
        t0 = time.time()
        risk_scores = []
        try:
            weights = detection_cfg.get("weights", None)
            risk_levels_cfg = detection_cfg.get("risk_levels", None)
            ens_cfg = detection_cfg.get("ensemble", {}) or {}
            risk_scores = compute_risk_score_batch(
                th_per_time, tr_per_time, vol_per_time, corr_per_time,
                weights=weights, risk_levels=risk_levels_cfg,
                allow_single_module_alert=ens_cfg.get("allow_single_module_alert", False),
                single_module_threshold=float(ens_cfg.get("single_module_threshold", 0.8)),
                single_module_score_ratio=float(ens_cfg.get("single_module_score_ratio", 0.5)),
                persistence_filter=ens_cfg.get("persistence_filter", False),
                persistence_n=int(ens_cfg.get("persistence_n", 3)),
                persistence_mode=str(ens_cfg.get("persistence_mode", "consecutive")),
                persistence_window=int(ens_cfg.get("persistence_window", 30)),
                persistence_min_count=int(ens_cfg.get("persistence_min_count", 8)),
                persistence_backfill=int(ens_cfg.get("persistence_backfill", 0)),
            )
        except Exception as e:
            logger.error(f"综合评分失败，返回空结果: {e}")
            risk_scores = []
        perf["ensemble_s"] = round(time.time() - t0, 3)
        perf["detection_s"] = round(time.time() - t_detect_start, 3)
        logger.info(f"  Ensemble scoring: completed in {perf['ensemble_s']:.2f}s")
        logger.info(f"  Total detection: {perf['detection_s']:.2f}s ({n / perf['detection_s']:.0f} records/s)")

        self._detection_scores = risk_scores
        self._detection_df = df
        self._detection_perf = perf

        # 6. 构建异常事件列表
        anomalies = []
        for i, score in enumerate(risk_scores):
            level = score.get("level")
            if level not in ("yellow", "orange", "red"):
                continue

            row = df.iloc[i]
            ts = row.get("timestamp")
            ts_str = str(ts) if pd.notna(ts) else ""

            corr_faults = score.get("details", {}).get("correlation_faults", [])
            fault_type = corr_faults[0] if corr_faults else "异常工况"

            methods = []
            ms = score.get("module_scores", {})
            if ms.get("threshold", {}).get("score", 0) > 0:
                methods.append("阈值检测")
            if ms.get("trend", {}).get("score", 0) > 0:
                methods.append("趋势检测")
            if ms.get("volatility", {}).get("score", 0) > 0:
                methods.append("波动率检测")
            if ms.get("correlation", {}).get("score", 0) > 0:
                methods.append("多参数关联")
            detection_method = "+".join(methods) if methods else "综合评分"

            # 推断主要根因（cause_map 从 config.yaml 读取，便于配置化）
            module_scores_flat = {k: v.get("score", 0.0) for k, v in ms.items()}
            if module_scores_flat:
                primary_module = max(module_scores_flat, key=module_scores_flat.get)
                cause_map = detection_cfg.get("cause_map", {
                    "threshold": "参数超限",
                    "trend": "趋势异常",
                    "volatility": "波动异常",
                    "correlation": "多参数关联异常",
                })
                primary_cause = cause_map.get(primary_module, "综合异常")
            else:
                primary_cause = "综合异常"

            triggered_params = score.get("details", {}).get("triggered_params", [])
            param_values = {}
            for p in triggered_params:
                if p in row and pd.notna(row[p]):
                    try:
                        param_values[p] = round(float(row[p]), 2)
                    except (TypeError, ValueError):
                        param_values[p] = str(row[p])

            alert_id = f"ALT_{pd.to_datetime(ts).strftime('%Y%m%d_%H%M%S')}_{i}" if pd.notna(ts) else f"ALT_{i}"
            anomalies.append({
                "alert_id": alert_id,
                "trigger_time": ts_str,
                "device_id": device_id,
                "device_name": row.get("device_name", device_id) if "device_name" in row else device_id,
                "fault_type": fault_type,
                "risk_level": level,
                "confidence": score["confidence"],
                "score": score["score"],
                "detection_method": detection_method,
                "abnormal_params": ", ".join(triggered_params),
                "param_values": param_values,
                "alert_status": "待推送",
                "primary_cause": primary_cause,
                "row_idx": int(i),
            })

        self._last_anomalies = anomalies
        logger.info(f"Anomaly detection completed: {len(anomalies)} anomalies found")
        return anomalies

    def run_alert_push(self, alert: dict) -> bool:
        """
        Agent 3: 预警推送哨兵
        将预警事件以飞书消息卡片推送。

        工作流程（依据技术框架 §4.3）：
        1. 根据风险等级选择推送策略（红/橙即时推送，黄色汇总至日报）
        2. 黄色预警自动升级：连续 N 次 yellow 自动升级为 orange 推送
        3. 生成飞书消息卡片
        4. 在线模式：预警事件写入多维表格 → 推送卡片 → 更新状态为「已推送」
        5. 离线模式（demo 或未配置 chat_id）：卡片 JSON 保存至本地 demo_output/alert_cards/

        Args:
            alert: 预警事件字典

        Returns:
            是否推送成功（黄色预警跳过即时推送时返回 False，不计入 alerts_pushed）
        """
        logger.info(f"Agent 3: Pushing alert {alert.get('alert_id', 'unknown')}")

        app_mode = self.config.get("app", {}).get("mode", "demo")
        feishu_cfg = self.config.get("feishu", {}) or {}
        push_cfg = feishu_cfg.get("push", {}) or {}
        chat_id = push_cfg.get("chat_id", "")

        level = str(alert.get("risk_level", "yellow")).lower()

        # 黄色预警自动升级逻辑
        if level == "yellow":
            escalation_cfg = self.config.get("alert_escalation", {}) or {}
            if escalation_cfg.get("yellow_auto_escalate", False):
                device_id = alert.get("device_id", "")
                fault_type = alert.get("fault_type", "")
                key = f"{device_id}_{fault_type}"

                current_count = self._yellow_tracker.get(key, 0) + 1
                self._yellow_tracker[key] = current_count
                max_count = escalation_cfg.get("yellow_consecutive_count", 5)

                if current_count >= max_count:
                    escalate_to = escalation_cfg.get("yellow_escalate_to", "orange")
                    alert = alert.copy()
                    alert["risk_level"] = escalate_to
                    alert["original_level"] = "yellow"
                    level = escalate_to
                    logger.info(f"黄色预警自动升级: {key} count={current_count} -> {escalate_to}")
                    self._yellow_tracker[key] = 0
                else:
                    logger.info(f"黄色预警累计: {key} count={current_count}/{max_count}")
                    return False
            else:
                logger.info(f"黄色预警 {alert.get('alert_id')} 跳过即时推送，将汇总至日报")
                return False

        # 非黄色预警时，重置对应设备/故障类型的黄色计数器
        if level != "yellow":
            device_id = alert.get("device_id", "")
            fault_type = alert.get("fault_type", "")
            key = f"{device_id}_{fault_type}"
            self._yellow_tracker.pop(key, None)

        # 离线模式或未配置 chat_id：卡片写入本地
        if app_mode == "demo" or not chat_id:
            return self._push_alert_offline(alert)

        # 在线模式：写入多维表格 → 推送卡片 → 更新状态
        return self._push_alert_online(alert, feishu_cfg, push_cfg)

    def _push_alert_offline(self, alert: dict) -> bool:
        """
        离线模式推送：将消息卡片 JSON 保存至本地 demo_output/alert_cards/，
        用于 Demo 展示与验收（无需飞书 API 凭据）。

        Args:
            alert: 预警事件字典

        Returns:
            是否保存成功
        """
        import json
        import datetime
        from feishu.message_sender import MessageSender

        # 离线模式下 MessageSender 不需要真实凭据，仅用其卡片生成能力
        sender = MessageSender.__new__(MessageSender)
        card = sender.create_alert_card(alert)

        # 输出目录：demo/demo_output/alert_cards/
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(base, "demo", "demo_output", "alert_cards")
        os.makedirs(out_dir, exist_ok=True)

        alert_id = alert.get("alert_id", "unknown").replace(":", "-").replace("/", "_")
        out_path = os.path.join(out_dir, f"{alert_id}.json")

        payload = {
            "alert_id": alert.get("alert_id"),
            "risk_level": alert.get("risk_level"),
            "device_name": alert.get("device_name"),
            "fault_type": alert.get("fault_type"),
            "card": card,
            "saved_at": datetime.datetime.now().isoformat(),
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"离线卡片已保存: {out_path}")
            return True
        except Exception as e:
            logger.error(f"离线卡片保存失败: {e}")
            return False

    def _push_alert_online(
        self, alert: dict, feishu_cfg: dict, push_cfg: dict
    ) -> bool:
        """
        在线模式推送：调用飞书 API 完成多维表格写入 + 卡片推送 + 状态更新。

        Args:
            alert: 预警事件字典
            feishu_cfg: config.feishu 配置字典
            push_cfg: config.feishu.push 配置字典

        Returns:
            是否推送成功
        """
        import datetime
        from feishu.bitable_client import BitableClient
        from feishu.message_sender import MessageSender

        app_id = feishu_cfg.get("app_id", "")
        app_secret = feishu_cfg.get("app_secret", "")
        app_token = feishu_cfg.get("app_token", "")
        tables = feishu_cfg.get("tables", {}) or {}
        alert_table_id = tables.get("alert_events", "")
        chat_id = push_cfg.get("chat_id", "")
        mention_open_ids = push_cfg.get("mention_open_ids", []) or []

        if not (app_id and app_secret and app_token and alert_table_id and chat_id):
            logger.error("在线推送缺少必要配置，降级为离线模式")
            return self._push_alert_offline(alert)

        # 按风险等级选择 @提醒范围
        level = str(alert.get("risk_level", "yellow")).lower()
        if level == "red":
            mentions = list(mention_open_ids)
        else:  # orange
            mentions = mention_open_ids[:1] if mention_open_ids else []

        # 1. 预警事件写入多维表格
        try:
            bitable = BitableClient(app_id, app_secret, app_token)
            write_resp = bitable.append_alert_events(alert_table_id, [alert])
            if write_resp.get("code") != 0:
                logger.error(f"多维表格写入失败: {write_resp.get('msg')}")
                return False
            record_ids = write_resp.get("data", {}).get("record_ids", [])
            record_id = record_ids[0] if record_ids else ""
        except Exception as e:
            logger.exception(f"写入多维表格异常: {e}")
            return False

        # 2. 推送消息卡片
        try:
            sender = MessageSender(app_id, app_secret, receive_id_type="chat_id")
            push_resp = sender.send_alert(alert, chat_id, mention_open_ids=mentions)
            if not push_resp.get("pushed"):
                logger.error(f"卡片推送失败: {push_resp.get('msg')}")
                return False
        except Exception as e:
            logger.exception(f"卡片推送异常: {e}")
            return False

        # 3. 更新多维表格预警状态为「已推送」
        if record_id:
            try:
                push_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                bitable.update_alert_status(
                    alert_table_id, record_id, "pushed", push_time=push_time
                )
            except Exception as e:
                logger.warning(f"更新预警状态失败（不影响推送结果）: {e}")

        logger.info(f"在线推送成功: alert_id={alert.get('alert_id')}")
        return True

    def get_detection_summary(self) -> dict:
        """
        获取最近一次异常检测的完整结果摘要
        （替代直接访问私有属性的公共接口）

        Returns:
            包含评分序列、异常事件和元数据的字典
        """
        return {
            "scores": self._detection_scores if self._detection_scores else [],
            "anomalies": self._last_anomalies if self._last_anomalies else [],
            "total_rows": len(self._detection_df) if self._detection_df is not None else 0,
            "columns": list(self._detection_df.columns) if self._detection_df is not None else [],
            "perf": getattr(self, "_detection_perf", {}),
        }

    def save_detection_results(self, out_path: str, compact: bool = False) -> str:
        """
        将最近一次检测结果保存为 JSON 文件

        Args:
            out_path: 输出文件路径
            compact: 紧凑模式
                - False（默认）: 输出全量 scores + time_series（8MB 级，便于离线分析）
                - True: 仅输出 anomalies（KB 级，便于传输/展示），scores 字段置空

        Returns:
            实际写入的文件路径
        """
        import json
        summary = self.get_detection_summary()
        if compact:
            payload = {
                "compact": True,
                "total_rows": summary["total_rows"],
                "anomalies_count": len(summary["anomalies"]),
                "anomalies": summary["anomalies"],
                "perf": summary["perf"],
            }
        else:
            payload = {
                "compact": False,
                "total_rows": summary["total_rows"],
                "columns": summary["columns"],
                "scores": summary["scores"],
                "anomalies": summary["anomalies"],
                "perf": summary["perf"],
            }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"检测结果已保存: {out_path} (compact={compact})")
        return out_path

    def run_incremental_detection(
        self, data_source: str, device_id: str, last_row_idx: int = 0
    ) -> dict:
        """
        增量检测接口：仅对 last_row_idx 之后的新增数据执行检测

        适用场景：实时流接入（W5/W6），避免每次全量重算 43200 行。
        策略：追加读取新数据 → 仅对新增行跑检测 → 与历史结果拼接。

        限制说明：
        - 趋势/波动率检测依赖滚动窗口，新增行需包含前 max(window)+buffer 行作为上下文
        - 增量结果与全量结果在边界点可能存在微小差异（窗口预热）
        - 首次调用（last_row_idx=0）等价于全量检测

        Args:
            data_source: 数据源 CSV 路径
            device_id: 设备编号
            last_row_idx: 上次检测到的行索引（新增数据从该索引之后开始）

        Returns:
            {
                "new_rows": int,           # 新增行数
                "new_anomalies": list,     # 新增异常事件
                "last_row_idx": int,       # 本次检测到的最后行索引（下次调用传入）
                "context_rows": int,       # 用于窗口预热的上下文行数
            }
        """
        import time
        import pandas as pd
        import numpy as np
        from core.data_cleaner import PARAM_COLUMNS
        from models.volatility_detector import detect_volatility_multi_params
        from models.correlation_detector import detect_correlation_batch
        from models.ensemble_scorer import compute_risk_score_batch

        logger.info(f"Incremental detection: data_source={data_source}, last_row_idx={last_row_idx}")

        # 读取完整数据（实时场景可改为追加读取）
        raw_df = pd.read_csv(data_source, low_memory=False)
        full_df = self._clean_data_func(raw_df).reset_index(drop=True)
        total = len(full_df)

        if last_row_idx >= total:
            logger.info(f"无新增数据 (last_row_idx={last_row_idx} >= total={total})")
            return {
                "new_rows": 0, "new_anomalies": [],
                "last_row_idx": last_row_idx, "context_rows": 0,
            }

        # 窗口预热：取 last_row_idx 前 max_window+buffer 行作为上下文
        detection_cfg = self.config.get("detection", {})
        max_window = max(detection_cfg.get("trend_windows", [10, 30, 60]))
        vol_baseline = detection_cfg.get("volatility_baseline_window", 480)
        context_needed = max(max_window, vol_baseline) + 10
        ctx_start = max(0, last_row_idx - context_needed)
        # 检测区间：上下文 + 新增行
        detect_df = full_df.iloc[ctx_start:].reset_index(drop=True)
        new_start = last_row_idx - ctx_start  # 新增行在 detect_df 中的起始位置
        new_rows_count = total - last_row_idx
        logger.info(
            f"增量检测: total={total}, new={new_rows_count}, "
            f"context={new_start}, detect_len={len(detect_df)}"
        )

        param_columns = [c for c in PARAM_COLUMNS if c in detect_df.columns]
        n_detect = len(detect_df)

        # 阈值检测
        threshold_hits = self._detect_threshold_batch_func(
            detect_df, self._thresholds, param_columns
        )
        th_per_time = [[] for _ in range(n_detect)]
        for hit in threshold_hits:
            ri = hit.get("row_idx", 0)
            if isinstance(ri, (int, np.integer)) and 0 <= ri < n_detect:
                th_per_time[ri].append(hit)

        # 趋势/波动率/关联检测（含上下文，保证窗口完整）
        trend_multi = self._detect_trend_multi_func(
            detect_df, param_columns, thresholds=self._trend_thresholds
        )
        vol_current = detection_cfg.get("volatility_current_window", 30)
        vol_multi = detect_volatility_multi_params(
            detect_df, param_columns,
            current_window=vol_current, baseline_window=vol_baseline,
        )
        rules = []
        rules_path = os.path.join(os.path.dirname(self.config_path), "rules.yaml")
        if os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as f:
                rules = yaml.safe_load(f).get("rules", [])
        corr_batch = detect_correlation_batch(detect_df, rules, trend_multi, vol_multi)

        # 构造 per-time 状态
        tr_per_time, vol_per_time, corr_per_time = [], [], []
        for i in range(n_detect):
            tdict, vdict = {}, {}
            for col in param_columns:
                tdf = trend_multi.get(col)
                if tdf is not None and i < len(tdf):
                    row = tdf.iloc[i]
                    tdict[col] = {
                        "param": col,
                        "any_detected": bool(row.get("any_detected", False)),
                        "max_level": row.get("max_level"),
                    }
                    for w in [10, 30, 60]:
                        if f"slope_{w}" in row:
                            tdict[col][f"window_{w}"] = {
                                "slope": float(row[f"slope_{w}"]),
                                "detected": bool(row.get(f"detected_{w}", False)),
                            }
                vdf = vol_multi.get(col)
                if vdf is not None and i < len(vdf):
                    row = vdf.iloc[i]
                    vdict[col] = {
                        "param": col, "ratio": float(row.get("ratio", 1.0)),
                        "level": row.get("level"),
                        "detected": bool(row.get("detected", False)),
                    }
            tr_per_time.append(tdict)
            vol_per_time.append(vdict)
            corr_per_time.append(corr_batch[i]["matched_rules"] if i < len(corr_batch) else [])

        # 综合评分（全区间，含上下文，保证持久性过滤正确）
        ens_cfg = detection_cfg.get("ensemble", {}) or {}
        all_scores = compute_risk_score_batch(
            th_per_time, tr_per_time, vol_per_time, corr_per_time,
            weights=detection_cfg.get("weights"),
            risk_levels=detection_cfg.get("risk_levels"),
            allow_single_module_alert=ens_cfg.get("allow_single_module_alert", False),
            single_module_threshold=float(ens_cfg.get("single_module_threshold", 0.8)),
            single_module_score_ratio=float(ens_cfg.get("single_module_score_ratio", 0.5)),
            persistence_filter=ens_cfg.get("persistence_filter", False),
            persistence_n=int(ens_cfg.get("persistence_n", 3)),
            persistence_mode=str(ens_cfg.get("persistence_mode", "consecutive")),
            persistence_window=int(ens_cfg.get("persistence_window", 30)),
            persistence_min_count=int(ens_cfg.get("persistence_min_count", 8)),
            persistence_backfill=int(ens_cfg.get("persistence_backfill", 0)),
        )

        # 仅提取新增行对应的评分与异常
        new_scores = all_scores[new_start:]
        new_anomalies = []
        for i, score in enumerate(new_scores):
            level = score.get("level")
            if level not in ("yellow", "orange", "red"):
                continue
            global_idx = last_row_idx + i
            row = full_df.iloc[global_idx]
            ts = row.get("timestamp")
            ts_str = str(ts) if pd.notna(ts) else ""
            corr_faults = score.get("details", {}).get("correlation_faults", [])
            fault_type = corr_faults[0] if corr_faults else "异常工况"
            alert_id = (
                f"ALT_{pd.to_datetime(ts).strftime('%Y%m%d_%H%M%S')}_{global_idx}"
                if pd.notna(ts) else f"ALT_{global_idx}"
            )
            new_anomalies.append({
                "alert_id": alert_id,
                "trigger_time": ts_str,
                "device_id": device_id,
                "device_name": row.get("device_name", device_id) if "device_name" in row else device_id,
                "fault_type": fault_type,
                "risk_level": level,
                "confidence": score["confidence"],
                "score": score["score"],
                "row_idx": int(global_idx),
            })

        # 更新缓存（保留全量数据，供下次增量调用）
        self._cached_df = full_df
        self._detection_df = full_df
        self._detection_scores = all_scores
        self._last_anomalies = (self._last_anomalies or []) + new_anomalies

        logger.info(
            f"增量检测完成: new_rows={new_rows_count}, new_anomalies={len(new_anomalies)}"
        )
        return {
            "new_rows": new_rows_count,
            "new_anomalies": new_anomalies,
            "last_row_idx": total,
            "context_rows": new_start,
        }

    def run_maintenance_advice(self, alert_id: str) -> dict:
        """
        Agent 4: 运维建议哨兵
        根据预警事件信息，从故障知识库检索相似案例，生成结构化维修建议报告与工单草稿。

        触发条件（技术框架 §4.4）：预警事件被责任人确认后（alert_status=已确认）自动触发。
        离线 demo 模式下可由 run_full_pipeline 或外部直接调用验收。

        工作流程：
        1. 根据 alert_id 在最近一次检测结果中定位预警事件
        2. 调用 MaintenanceAdvisor 检索 Top-3 相似案例
        3. 生成维修建议报告 + 工单草稿
        4. 在线模式：工单草稿写入多维表格「工单」表
           离线模式：报告与工单 JSON 保存至 demo_output/advice/

        Args:
            alert_id: 预警事件ID

        Returns:
            维修建议报告字典（含 report_text / top_cases / work_order）
        """
        logger.info(f"Agent 4: Generating maintenance advice for {alert_id}")

        from core.maintenance_advisor import MaintenanceAdvisor

        # 1. 定位预警事件
        alert = self._find_alert(alert_id)
        if not alert:
            logger.warning(f"未找到预警事件 {alert_id}，Agent 4 跳过")
            return {}

        # 2. 懒加载 MaintenanceAdvisor
        if self._maintenance_advisor is None:
            kb_path = os.path.join(
                os.path.dirname(self.config_path), "knowledge_base.yaml"
            )
            if not os.path.exists(kb_path):
                base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                kb_path = os.path.join(base, "config", "knowledge_base.yaml")

            # 从 thresholds.yaml 构建 param_ranges 用于方向推断
            param_ranges = self._build_param_ranges()
            self._maintenance_advisor = MaintenanceAdvisor(
                kb_path=kb_path, param_ranges=param_ranges
            )

        # 3. 生成维修建议报告与工单草稿
        advice = self._maintenance_advisor.generate_advice_report(alert)

        # 4. 工单落地
        self._persist_advice(advice)

        logger.info(
            f"Agent 4 完成: alert_id={alert_id} matched_case="
            f"{(advice.get('primary_case') or {}).get('case_id')} "
            f"work_order={advice.get('work_order', {}).get('work_order_id')}"
        )
        return advice

    def _build_param_ranges(self) -> dict:
        """
        从 thresholds.yaml 构建参数正常范围字典，用于 Agent 4 方向推断。
        输出格式：{param_name: [low, high]}
        """
        param_ranges = {}
        if self._thresholds and isinstance(self._thresholds, dict):
            params = self._thresholds.get("parameters", [])
            if isinstance(params, list):
                for p in params:
                    name = p.get("name")
                    rng = p.get("normal_range")
                    if name and isinstance(rng, (list, tuple)) and len(rng) >= 2:
                        param_ranges[name] = [float(rng[0]), float(rng[1])]
        return param_ranges

    def _find_alert(self, alert_id: str) -> dict:
        """
        根据 alert_id 在最近一次检测结果（_last_anomalies）中查找预警事件。
        若未找到则返回空字典（调用方应处理空值）。
        """
        if not alert_id:
            return {}
        for a in self._last_anomalies or []:
            if a.get("alert_id") == alert_id:
                return a
        # 兼容：若 _last_anomalies 为空但调用方传入完整 alert dict 的场景
        return {}

    def _persist_advice(self, advice: dict) -> None:
        """
        维修建议落地：
        - 在线模式（配置了 work_orders 表 token 且非 demo）：工单写入多维表格
        - 离线模式：报告与工单 JSON 保存至 demo_output/advice/
        """
        if not advice:
            return

        app_mode = self.config.get("app", {}).get("mode", "demo")
        feishu_cfg = self.config.get("feishu", {}) or {}
        tables = feishu_cfg.get("tables", {}) or {}
        wo_table_id = tables.get("work_orders", "")

        # 在线模式：工单写入多维表格
        if app_mode != "demo" and wo_table_id:
            try:
                app_id = feishu_cfg.get("app_id", "")
                app_secret = feishu_cfg.get("app_secret", "")
                app_token = feishu_cfg.get("app_token", "")
                if app_id and app_secret and app_token:
                    from feishu.bitable_client import BitableClient
                    bitable = BitableClient(app_id, app_secret, app_token)
                    work_order = advice.get("work_order")
                    if work_order:
                        resp = bitable.append_work_orders(wo_table_id, [work_order])
                        if resp.get("code") == 0:
                            record_ids = resp.get("data", {}).get("record_ids", [])
                            if record_ids:
                                work_order["feishu_record_id"] = record_ids[0]
                            logger.info("工单已写入多维表格")
                            return
                        logger.error(f"工单写入多维表格失败: {resp.get('msg')}")
            except Exception as e:
                logger.warning(f"在线工单写入异常，降级为离线保存: {e}")

        # 离线模式：保存到本地
        try:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            out_dir = os.path.join(base, "demo", "demo_output", "advice")
            self._maintenance_advisor.save_advice_to_file(advice, out_dir)
        except Exception as e:
            logger.error(f"离线保存维修建议失败: {e}")

    def run_full_pipeline(self, data_source: str, device_id: str) -> dict:
        """
        执行完整流水线：采集 → 检测 → 推送 → 建议

        Args:
            data_source: 数据源
            device_id: 设备编号

        Returns:
            完整执行报告
        """
        logger.info(f"Full pipeline started: data_source={data_source}, device_id={device_id}")

        report = {
            "status": "success",
            "data_source": data_source,
            "device_id": device_id,
            "records_collected": 0,
            "anomalies_detected": 0,
            "alerts_pushed": 0,
            "advice_generated": False,
        }

        try:
            # Step 1: 数据采集
            report["records_collected"] = self.run_data_collection(data_source)

            # Step 2: 异常检测
            anomalies = self.run_anomaly_detection(device_id)
            report["anomalies_detected"] = len(anomalies)

            # Step 3: 预警推送
            for alert in anomalies:
                if self.run_alert_push(alert):
                    report["alerts_pushed"] += 1

            # Step 4: 运维建议
            if anomalies:
                advice = self.run_maintenance_advice(anomalies[0].get("alert_id", ""))
                report["advice_generated"] = bool(advice)
        except Exception as e:
            logger.exception("Pipeline execution failed")
            report["status"] = "failed"
            report["error"] = str(e)

        logger.info(f"Full pipeline completed: {report}")
        return report