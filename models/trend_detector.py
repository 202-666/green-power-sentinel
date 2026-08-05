"""
绿电哨兵 — 趋势检测模块（Agent 2 方法2）

对每个参数计算 10/30/60 分钟窗口的线性回归斜率，判断是否存在持续性变化趋势。

算法：
1. 对原始序列做滚动均值平滑（默认 window=5，可配置）以降低高频噪声
2. 在每个时间点上，对其前 N 分钟的数据做最小二乘线性回归，得到斜率 (units/min)
3. 比较斜率绝对值与参数对应的窗口阈值，输出方向 (up/down) 与等级 (yellow/orange/red)

参考框架示例（炉膛温度）：
- 10分钟内上升>5°C  → 橙色 (slope≈0.5 °C/min)
- 30分钟内上升>10°C → 红色 (slope≈0.33 °C/min)
- 60分钟内持续单向变化>15°C → 红色 (slope≈0.25 °C/min)

验收标准（按故障类型分口径）：
- 轴承过热 / 炉排卡滞：原始趋势模块故障段检出率 > 80%（直接承担）
- 烟气超标：原始趋势模块按量程标定，属中后期检出（对照实测约 49%）；
  早期检出由 R2 关联规则 + ensemble 产品路径覆盖（产品路径故障段检出率 ≥ 75%、
  首次检出 ≤ 60min，W7 口径；W7 报告历史值 87.78%，当前实现实测约 79%，
  差异源于 W8 起 backfill 增加 correlation 约束及风险等级/权重按 W8 标定）
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 默认检测窗口（分钟）
DEFAULT_WINDOWS = [10, 30, 60]

# 默认平滑窗口（分钟），用于降低噪声对斜率的影响
# 取 15 是经验值：在炉膛温度 σ≈23 的数据上，可将 10min 回归斜率 std 从 0.78 降到 0.20，
# 使 30/60min 窗口的误报率 < 1%，10min 窗口保持对突变的灵敏度
DEFAULT_SMOOTHING_WINDOW = 15

# 各窗口的等级映射（参考框架示例）
# 10 分钟窗口：短期快速变化 → 黄色（短窗口噪声大，可靠性低，仅作辅助）
# 30 分钟窗口：中期持续变化 → 橙色（主趋势检测器）
# 60 分钟窗口：长期持续变化 → 红色（最可靠，几乎零误报）
# 调整：10min 从 orange→yellow，避免正常数据短窗口波动触发 trend 0.65 分导致 FP
WINDOW_LEVEL = {10: "yellow", 30: "orange", 60: "red"}

# 各参数的默认斜率阈值（units/min）
# 10min：取正常范围宽度的 3%/min（约 30% 范围/10min），仅检测真正的突变
#         例：furnace_temp 范围 200 → 0.03×200=6.0°C/min（10min 内变化 60°C 才触发）
# 30min：取正常范围宽度的 0.5%~1%/min（约 15%~30% 范围/30min）
#         例：furnace_temp → 0.33°C/min（30min 内变化 10°C 触发，符合框架示例）
# 60min：取正常范围宽度的 0.3%~0.5%/min（约 20%~30% 范围/60min）
#         例：furnace_temp → 0.25°C/min（60min 内变化 15°C 触发，符合框架示例）
# 小量级参数（如 steam_pressure）按物理意义单独标定
DEFAULT_SLOPE_THRESHOLDS = {
    "furnace_temperature":   {10: 15.0,  30: 0.50, 60: 0.30},   # range 200
    "flue_gas_temperature":  {10: 5.0,   30: 0.30, 60: 0.20},   # range 70
    "steam_pressure":         {10: 0.05,  30: 0.020, 60: 0.015}, # range 0.7
    "steam_flow":             {10: 4.0,   30: 0.30, 60: 0.20},   # range 25
    "bearing_vibration":     {10: 0.40,  30: 0.10, 60: 0.07},    # range 4.5
    "bearing_temperature":    {10: 3.0,   30: 0.20, 60: 0.15},    # range 40
    "so2_concentration":     {10: 8.0,   30: 0.50, 60: 0.30},    # range 100
    "nox_concentration":     {10: 12.0,  30: 0.50, 60: 0.30},    # range 150
    "grate_speed":           {10: 4.0,   30: 0.50, 60: 0.30},    # range 50
    "feed_rate":             {10: 0.50,  30: 0.10, 60: 0.07},    # range 7
    "oxygen_content":        {10: 0.50,  30: 0.10, 60: 0.07},    # range 6
    "furnace_pressure":      {10: 8.0,   30: 0.50, 60: 0.30},    # range 100
    "cooling_water_temp":    {10: 1.5,   30: 0.30, 60: 0.20},    # range 20
}


def _compute_slope(values: np.ndarray) -> float:
    """
    最小二乘线性回归求斜率

    Args:
        values: 一维数组，长度为窗口大小

    Returns:
        斜率 (units/min)
    """
    n = len(values)
    if n < 2:
        return 0.0
    # x 为时间索引 0,1,...,n-1（每点代表 1 分钟）
    x = np.arange(n, dtype=float)
    y = np.asarray(values, dtype=float)

    # 处理 NaN：用前一个有效值填充
    if np.isnan(y).any():
        y = pd.Series(y).ffill().bfill().to_numpy()
        if np.isnan(y).all():
            return 0.0

    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    return float(slope)


def detect_trend(
    series: pd.Series,
    windows: list = None,
    thresholds: dict = None,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> dict:
    """
    趋势检测（单参数单时刻，基于序列尾部）

    对序列末尾的每个窗口分别计算斜率。

    Args:
        series: 某参数的时间序列（按时间升序），长度 >= max(windows)
        windows: 检测窗口列表（分钟），默认 [10, 30, 60]
        thresholds: {window: slope_threshold} 或 {param_name: {window: threshold}}
                    None 则按 series.name 查 DEFAULT_SLOPE_THRESHOLDS
        smoothing_window: 平滑窗口大小，<=1 表示不平滑

    Returns:
        {
            "param": "furnace_temperature",
            "window_10": {"slope": 0.8, "direction": "up",
                          "threshold": 0.5, "level": "orange", "detected": True},
            "window_30": {...},
            "window_60": {...},
            "any_detected": True,
            "max_level": "red"
        }
    """
    if windows is None:
        windows = list(DEFAULT_WINDOWS)

    param_name = series.name if hasattr(series, "name") and series.name else "unknown"

    # 解析阈值
    if thresholds is None:
        thresholds_dict = DEFAULT_SLOPE_THRESHOLDS.get(param_name, {})
    elif isinstance(thresholds, dict) and param_name in thresholds and isinstance(
        thresholds[param_name], dict
    ):
        thresholds_dict = thresholds[param_name]
    elif isinstance(thresholds, dict) and all(
        isinstance(v, (int, float)) for v in thresholds.values()
    ):
        # 形如 {10: 0.5, 30: 0.33}
        thresholds_dict = thresholds
    else:
        thresholds_dict = {}

    # 平滑
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if smoothing_window and smoothing_window > 1:
        s = s.rolling(window=smoothing_window, min_periods=1).mean()

    n = len(s)
    results = {
        "param": param_name,
        "any_detected": False,
        "max_level": None,
    }

    level_priority = {"red": 3, "orange": 2, "yellow": 1}
    max_priority = 0

    for w in windows:
        if n < w:
            # 数据不足，跳过
            results[f"window_{w}"] = {
                "slope": 0.0,
                "direction": "none",
                "threshold": thresholds_dict.get(w, None),
                "level": None,
                "detected": False,
                "reason": "insufficient_data",
            }
            continue

        # 取末尾 w 个点
        window_values = s.iloc[-w:].to_numpy()
        slope = _compute_slope(window_values)
        th = thresholds_dict.get(w)
        direction = "up" if slope > 0 else ("down" if slope < 0 else "none")

        detected = False
        level = None
        if th is not None and abs(slope) >= th:
            detected = True
            level = WINDOW_LEVEL.get(w, "yellow")

        results[f"window_{w}"] = {
            "slope": round(slope, 6),
            "direction": direction,
            "threshold": th,
            "level": level,
            "detected": detected,
        }

        if detected:
            results["any_detected"] = True
            pri = level_priority.get(level, 0)
            if pri > max_priority:
                max_priority = pri
                results["max_level"] = level

    return results


def detect_trend_batch(
    series: pd.Series,
    windows: list = None,
    thresholds: dict = None,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> pd.DataFrame:
    """
    趋势检测（批量，对序列每个时间点 i 取 [i-w+1, i] 窗口计算斜率）

    Args:
        series: 某参数的完整时间序列
        windows: 检测窗口列表
        thresholds: 斜率阈值
        smoothing_window: 平滑窗口

    Returns:
        DataFrame，每行对应原序列一个时间点，列：
            slope_10, slope_30, slope_60,
            detected_10, detected_30, detected_60,
            any_detected, max_level
    """
    if windows is None:
        windows = list(DEFAULT_WINDOWS)

    param_name = series.name if hasattr(series, "name") and series.name else "unknown"

    if thresholds is None:
        thresholds_dict = DEFAULT_SLOPE_THRESHOLDS.get(param_name, {})
    elif isinstance(thresholds, dict) and param_name in thresholds and isinstance(
        thresholds[param_name], dict
    ):
        thresholds_dict = thresholds[param_name]
    elif isinstance(thresholds, dict) and all(
        isinstance(v, (int, float)) for v in thresholds.values()
    ):
        thresholds_dict = thresholds
    else:
        thresholds_dict = {}

    s = pd.to_numeric(series, errors="coerce").astype(float)
    if smoothing_window and smoothing_window > 1:
        s = s.rolling(window=smoothing_window, min_periods=1).mean()

    n = len(s)
    out = {
        "timestamp": series.index if hasattr(series, "index") else range(n),
    }

    # 向量化斜率计算（滚动均值差分近似，替代 rolling.apply 最小二乘）
    # 原理：对线性趋势 y=a+bx，OLS slope = b；
    # 近似：slope ≈ (mean(后半窗口) - mean(前半窗口)) / (窗口/2)
    # 对线性数据两者等价，对噪声数据均值差分更稳健，且为 O(n) 向量化
    for w in windows:
        slopes = np.zeros(n)
        detected = np.zeros(n, dtype=bool)
        levels = np.array([None] * n, dtype=object)

        if n >= w:
            half_w = max(w // 2, 1)
            # 后半窗口均值：mean(s[i-half_w+1 .. i])
            second_half = s.rolling(half_w, min_periods=half_w).mean()
            # 前半窗口均值：mean(s[i-w+1 .. i-half_w]) = second_half.shift(half_w)
            first_half = second_half.shift(half_w)
            # 斜率 ≈ (后半均值 - 前半均值) / half_w
            slope_series = (second_half - first_half) / half_w
            slopes = slope_series.to_numpy()

            th = thresholds_dict.get(w)
            if th is not None:
                detected_mask = np.abs(slopes) >= th
                detected = detected_mask
                for i in np.where(detected_mask)[0]:
                    levels[i] = WINDOW_LEVEL.get(w, "yellow")

        out[f"slope_{w}"] = slopes
        out[f"detected_{w}"] = detected
        out[f"level_{w}"] = levels

    # 综合判断（向量化：任一窗口检出即 any_detected，取最高等级）
    # 按等级优先级排序窗口，高优先级窗口的检出覆盖低优先级
    level_priority = {"red": 3, "orange": 2, "yellow": 1}
    # 按 priority 降序排列 windows（60→red, 30→red, 10→orange）
    sorted_windows = sorted(windows, key=lambda w: -level_priority.get(WINDOW_LEVEL.get(w), 0))
    any_detected = np.zeros(n, dtype=bool)
    max_level = np.array([None] * n, dtype=object)
    for w in sorted_windows:
        det = out[f"detected_{w}"]
        lv = WINDOW_LEVEL.get(w, "yellow")
        # 该窗口检出且尚未被更高优先级窗口覆盖的点
        is_none = np.array([x is None for x in max_level])
        mask = det & is_none
        max_level[mask] = lv
        any_detected = any_detected | det

    out["any_detected"] = any_detected
    out["max_level"] = max_level

    return pd.DataFrame(out, index=series.index)


def detect_trend_multi_params(
    df: pd.DataFrame,
    param_columns: list,
    windows: list = None,
    thresholds: dict = None,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    dynamic_threshold: bool = False,
    baseline_window: int = 480,
    sensitivity_k: float = 3.0,
) -> dict:
    """
    多参数趋势检测（批量）

    Args:
        df: 含 timestamp + 参数列的 DataFrame
        param_columns: 待检测的参数列名列表
        windows: 检测窗口列表
        thresholds: 各参数的阈值字典，None 则用默认
        smoothing_window: 平滑窗口
        dynamic_threshold: 是否启用动态阈值（基于基线窗口标准差）
        baseline_window: 基线窗口大小（分钟），用于计算正常波动标准差
        sensitivity_k: 动态阈值灵敏度系数（k * std / sqrt(window)）

    Returns:
        {param_name: trend_batch_df, ...}
        每个 DataFrame 包含每行的 slope/detected/level
    """
    if windows is None:
        windows = list(DEFAULT_WINDOWS)

    results = {}
    for col in param_columns:
        if col not in df.columns:
            continue
        series = df[col]
        
        col_thresholds = thresholds
        if dynamic_threshold:
            col_thresholds = _compute_dynamic_thresholds(
                series, windows, baseline_window, sensitivity_k
            )
        
        results[col] = detect_trend_batch(
            series, windows=windows, thresholds=col_thresholds,
            smoothing_window=smoothing_window,
        )

    return results


def _compute_dynamic_thresholds(
    series: pd.Series,
    windows: list,
    baseline_window: int,
    sensitivity_k: float,
) -> dict:
    """
    根据基线窗口计算动态斜率阈值

    公式：threshold = k * std(baseline) / sqrt(window_size)

    Args:
        series: 参数时间序列
        windows: 检测窗口列表
        baseline_window: 基线窗口大小
        sensitivity_k: 灵敏度系数

    Returns:
        {window: threshold, ...}
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    
    if len(s) < baseline_window:
        return None
    
    baseline = s.iloc[:baseline_window]
    baseline_std = baseline.std()
    
    if baseline_std == 0:
        return None
    
    thresholds = {}
    for w in windows:
        thresholds[w] = sensitivity_k * baseline_std / (w ** 0.5)
    
    return thresholds
