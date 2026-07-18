"""
绿电哨兵 — 阈值检测模块（Agent 2 方法1）

对 13 个监测参数执行阈值检测，覆盖以下阈值类型：
- danger_threshold      : 上限危险阈值（如炉膛温度 1050°C）
- warning_threshold     : 上限警告阈值（如轴承振动 4.5 mm/s）
- danger_low/danger_high              : 双向危险阈值（如蒸汽压力 3.0~4.5 MPa）
- danger_threshold_low/danger_threshold_high : 双向危险阈值（如炉排速度 10~95%）
- danger_threshold_low : 下限危险阈值（如蒸汽流量 30 t/h）
- standard_limit        : 国标限值（如 SO2 200 mg/Nm³，GB18485-2020）

验收标准：100% 检出超限值
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 阈值优先级：danger > warning > standard_limit
# (type_key, direction, level)
THRESHOLD_TYPES = [
    ("danger_threshold", "upper", "danger"),
    ("danger_threshold_high", "upper", "danger"),
    ("danger_threshold_low", "lower", "danger"),
    ("danger_high", "upper", "danger"),
    ("danger_low", "lower", "danger"),
    ("warning_threshold", "upper", "warning"),
    ("standard_limit", "upper", "limit"),
]

# 单条记录同时触发多个阈值时，仅保留最高级别
LEVEL_PRIORITY = {"danger": 3, "limit": 2, "warning": 1}


def _normalize_thresholds(thresholds: Any) -> dict:
    """
    将 thresholds.yaml 的多种结构归一化为 {param_name: {type_key: value}} 形式

    支持：
    1. 完整 yaml: {"parameters": [{"name": "x", "danger_threshold": 1050}, ...]}
    2. 参数列表: [{"name": "x", "danger_threshold": 1050}, ...]
    3. 扁平 dict: {"x": {"danger_threshold": 1050}}
    """
    normalized = {}

    if isinstance(thresholds, dict) and "parameters" in thresholds:
        params = thresholds["parameters"]
    elif isinstance(thresholds, list):
        params = thresholds
    elif isinstance(thresholds, dict):
        # 扁平 dict，已是 {param_name: {type_key: value}} 形式
        return thresholds
    else:
        return normalized

    for p in params:
        if not isinstance(p, dict) or "name" not in p:
            continue
        name = p["name"]
        normalized[name] = {k: v for k, v in p.items() if k != "name"}

    return normalized


def detect_threshold(current_values: dict, thresholds: dict) -> list:
    """
    阈值检测（单时刻）

    Args:
        current_values: {param_name: value}
        thresholds: thresholds.yaml 加载结果（支持完整 yaml / 列表 / 扁平 dict）

    Returns:
        [{"param": "furnace_temperature", "value": 1060.5,
          "threshold": 1050, "threshold_type": "danger_threshold",
          "direction": "upper", "level": "danger",
          "exceed_amount": 10.5, "exceed_ratio": 0.01}, ...]
    """
    if not current_values:
        return []

    norm_thresholds = _normalize_thresholds(thresholds)
    results = []

    for param, value in current_values.items():
        # 跳过非参数键（如 device_id、timestamp）
        if param not in norm_thresholds:
            continue
        # 跳过无效值
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN
            continue

        param_th = norm_thresholds[param]
        # 对同一参数收集所有触发的阈值，按优先级取最高
        triggered = []
        for type_key, direction, level in THRESHOLD_TYPES:
            if type_key not in param_th:
                continue
            th = param_th[type_key]
            try:
                th = float(th)
            except (TypeError, ValueError):
                continue

            if direction == "upper" and v > th:
                triggered.append((type_key, direction, level, th, v - th))
            elif direction == "lower" and v < th:
                triggered.append((type_key, direction, level, th, th - v))

        if not triggered:
            continue

        # 选取最高级别；同级取 exceed_amount 最大者
        triggered.sort(
            key=lambda x: (LEVEL_PRIORITY.get(x[2], 0), x[4]),
            reverse=True,
        )
        type_key, direction, level, th, exceed = triggered[0]

        results.append({
            "param": param,
            "value": round(v, 4),
            "threshold": th,
            "threshold_type": type_key,
            "direction": direction,
            "level": level,
            "exceed_amount": round(exceed, 4),
            "exceed_ratio": round(exceed / abs(th) if th != 0 else 0.0, 6),
        })

    return results


def detect_threshold_batch(df, thresholds: dict, param_columns: list = None) -> list:
    """
    阈值检测（批量，对 DataFrame 每一行执行）

    Args:
        df: 含 timestamp + 参数列的 DataFrame
        thresholds: thresholds.yaml 加载结果
        param_columns: 待检测的参数列名；None 则自动从 thresholds 推导

    Returns:
        [{"timestamp": ..., "row_idx": i, "param": ..., "value": ..., ...}, ...]
    """
    if df is None or df.empty:
        return []

    if param_columns is None:
        norm = _normalize_thresholds(thresholds)
        param_columns = [c for c in norm.keys() if c in df.columns]

    all_results = []
    for idx, row in df.iterrows():
        current_values = {col: row[col] for col in param_columns if col in row}
        hits = detect_threshold(current_values, thresholds)
        for h in hits:
            h["timestamp"] = row.get("timestamp") if "timestamp" in row else None
            h["row_idx"] = idx
        all_results.extend(hits)

    return all_results


def load_thresholds_from_yaml(yaml_path: str) -> dict:
    """
    从 thresholds.yaml 加载并归一化阈值配置

    Args:
        yaml_path: thresholds.yaml 路径

    Returns:
        扁平 dict: {param_name: {threshold_type: value}}
    """
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _normalize_thresholds(raw)
