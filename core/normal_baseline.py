"""
绿电哨兵 — 正常运行基线计算模块

为趋势检测提供数据驱动的动态斜率阈值，替代 trend_detector 中的固定
DEFAULT_SLOPE_THRESHOLDS。物理依据：相同斜率在不同基线噪声水平的参数
上意义不同（如炉膛温度 σ≈23 vs 蒸汽压力 σ≈0.1），用基线 std 归一化后
的斜率阈值更自适应。

算法：
1. 对每个参数计算滚动基线窗口（默认 480 分钟 = 8 小时，匹配班组交接周期）
   内的均值 mean 与标准差 std
2. 动态斜率阈值 = k * std / sqrt(window)
   - std 反映该参数正常波动幅度
   - sqrt(window) 归一化窗口长度（长窗口斜率自然更稳定）
   - k 为灵敏度系数（默认 3.0，对应 3σ 显著性）
3. 输出与 trend_detector.DEFAULT_SLOPE_THRESHOLDS 同构的字典，可直接注入

集成方式（pipeline 层）：
    baseline = compute_baseline(df, window_size=480)
    trend_results = detect_trend_multi_params(df, params, thresholds=baseline["slope_thresholds"])

注意：基线应基于正常数据计算。若数据含故障段，建议先用 data_quality_flag
过滤，或使用前序正常窗口。本模块不做过滤，由调用方保证数据质量。
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 默认基线窗口（分钟），匹配 config.detection.volatility_baseline_window
DEFAULT_BASELINE_WINDOW = 480

# 默认灵敏度系数（3σ 显著性）
DEFAULT_SENSITIVITY_K = 3.0

# 各窗口的归一化系数：短窗口噪声大，需更严格阈值
# slope_threshold = k * std / sqrt(window) * window_factor
# window_factor < 1 表示该窗口阈值放宽（短窗口允许更大斜率）
WINDOW_FACTORS = {10: 1.5, 30: 1.0, 60: 0.8}


def compute_baseline(
    df: pd.DataFrame,
    param_columns: list = None,
    window_size: int = DEFAULT_BASELINE_WINDOW,
    trend_windows: list = None,
    sensitivity_k: float = DEFAULT_SENSITIVITY_K,
) -> dict:
    """
    计算正常运行基线，输出动态斜率阈值

    Args:
        df: 含参数列的 DataFrame（建议为正常数据，或已过滤故障段）
        param_columns: 待计算基线的参数列名列表，None 则自动推断
        window_size: 基线滚动窗口大小（分钟）
        trend_windows: 趋势检测窗口列表（与 trend_detector 对齐），默认 [10,30,60]
        sensitivity_k: 灵敏度系数，越大越严格（3.0 = 3σ 显著性）

    Returns:
        {
            "slope_thresholds": {param: {window: threshold}},
            "stats": {param: {"mean": float, "std": float, "baseline_window": int}},
        }
    """
    if trend_windows is None:
        trend_windows = [10, 30, 60]

    SKIP_COLS = {
        "timestamp", "device_id", "device_name", "device_type",
        "data_quality_flag", "qc_note",
    }
    if param_columns is None:
        param_columns = [c for c in df.columns if c not in SKIP_COLS]

    slope_thresholds = {}
    stats = {}

    for col in param_columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").astype(float).dropna()
        if len(series) < window_size:
            # 数据不足，用全序列统计
            rolling_std = series.std()
            rolling_mean = series.mean()
        else:
            # 滚动基线统计（取最近 window_size 点的中位数，稳健）
            rolling = series.rolling(window_size, min_periods=max(window_size // 4, 10))
            rolling_std = rolling.std().median()
            rolling_mean = rolling.mean().median()

        if pd.isna(rolling_std) or rolling_std <= 0:
            # 退化：用全序列 std
            rolling_std = series.std()
            rolling_mean = series.mean()
            if pd.isna(rolling_std) or rolling_std <= 0:
                logger.warning(f"基线计算失败 {col}: std<=0，跳过")
                continue

        # 动态斜率阈值 = k * std / sqrt(window) * window_factor
        col_thresholds = {}
        for w in trend_windows:
            factor = WINDOW_FACTORS.get(w, 1.0)
            th = sensitivity_k * rolling_std / np.sqrt(w) * factor
            col_thresholds[w] = round(float(th), 6)
        slope_thresholds[col] = col_thresholds
        stats[col] = {
            "mean": round(float(rolling_mean), 4),
            "std": round(float(rolling_std), 4),
            "baseline_window": window_size,
        }

    logger.info(
        f"基线计算完成: {len(slope_thresholds)} 参数, window={window_size}, k={sensitivity_k}"
    )
    return {"slope_thresholds": slope_thresholds, "stats": stats}


def update_baseline(new_data: pd.DataFrame, existing_baseline: dict) -> dict:
    """
    增量更新基线（EWMA 平滑）

    用新数据更新已有基线的 mean/std，适用于在线场景。
    保留原有阈值结构，仅更新统计量。

    Args:
        new_data: 新到达的数据 DataFrame
        existing_baseline: compute_baseline 的输出

    Returns:
        更新后的 baseline 字典
    """
    if not existing_baseline or "stats" not in existing_baseline:
        return compute_baseline(new_data)

    alpha = 0.1  # EWMA 平滑系数，新数据权重 10%
    updated_stats = {}
    updated_thresholds = {}

    for col, st in existing_baseline["stats"].items():
        if col not in new_data.columns:
            updated_stats[col] = st
            updated_thresholds[col] = existing_baseline["slope_thresholds"].get(col, {})
            continue
        new_series = pd.to_numeric(new_data[col], errors="coerce").dropna()
        if len(new_series) == 0:
            updated_stats[col] = st
            updated_thresholds[col] = existing_baseline["slope_thresholds"].get(col, {})
            continue
        new_mean = new_series.mean()
        new_std = new_series.std()
        if pd.isna(new_std) or new_std <= 0:
            updated_stats[col] = st
            updated_thresholds[col] = existing_baseline["slope_thresholds"].get(col, {})
            continue
        # EWMA 更新
        ema_mean = (1 - alpha) * st["mean"] + alpha * new_mean
        ema_std = (1 - alpha) * st["std"] + alpha * new_std
        updated_stats[col] = {
            "mean": round(float(ema_mean), 4),
            "std": round(float(ema_std), 4),
            "baseline_window": st["baseline_window"],
        }
        # 重算阈值
        k = DEFAULT_SENSITIVITY_K
        col_thresholds = {}
        for w, factor in WINDOW_FACTORS.items():
            th = k * ema_std / np.sqrt(w) * factor
            col_thresholds[w] = round(float(th), 6)
        updated_thresholds[col] = col_thresholds

    return {"slope_thresholds": updated_thresholds, "stats": updated_stats}
