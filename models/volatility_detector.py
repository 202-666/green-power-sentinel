"""
绿电哨兵 — 波动率检测模块（Agent 2 方法3）

检测参数波动率的异常增大，覆盖以下场景：
- 炉排卡滞前期：grate_speed 波动增大（σ从2→8）
- 轴承过热：bearing_vibration 噪声增大（较弱）

算法：
1. 预计算全序列的 current_window 滚动标准差
2. 对每个时间点，取 [i-baseline_window-gap, i-gap] 的滚动std中位数作为基线
   （gap = current_window × 2，避免基线被当前波动污染）
3. ratio = 当前滚动std / 基线滚动std中位数
4. 根据 multipliers 判定等级（yellow/orange/red）

关键设计：
- 基线与当前窗口使用相同大小的滚动std，比较公平
- gap 隔离带确保基线只包含历史正常波动
- multipliers 经标定可在正常数据上保持极低误报
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 经标定的倍数阈值（current_window=30, baseline_window=1440）
# 正常数据误报≈0次/30天，炉排卡滞检出率≈35%，轴承过热≈7%
DEFAULT_MULTIPLIERS = {"yellow": 1.7, "orange": 2.2, "red": 3.0}
DEFAULT_CURRENT_WINDOW = 30
DEFAULT_BASELINE_WINDOW = 1440  # 1天


def detect_volatility(
    series: pd.Series,
    current_window: int = DEFAULT_CURRENT_WINDOW,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
    multipliers: dict = None,
) -> dict:
    """
    单参数单时刻波动率检测

    Args:
        series: 时间序列（按时间升序），长度 >= baseline_window + current_window * 3
        current_window: 当前波动窗口（分钟）
        baseline_window: 基线历史窗口（分钟）
        multipliers: {level: ratio_threshold}

    Returns:
        {"param": ..., "current_std": ..., "baseline_std": ..., "ratio": ..., "level": ..., "detected": ...}
    """
    if multipliers is None:
        multipliers = DEFAULT_MULTIPLIERS.copy()

    param_name = getattr(series, "name", "unknown")
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)

    gap = current_window * 2
    min_required = baseline_window + gap + current_window
    if n < min_required:
        if n < current_window * 3:
            return {
                "param": param_name,
                "current_std": 0.0,
                "baseline_std": 0.0,
                "ratio": 1.0,
                "level": None,
                "detected": False,
                "reason": "insufficient_data",
            }
        baseline_window = n - gap - current_window

    rolling_std = s.rolling(window=current_window, min_periods=current_window).std()

    c_std = float(rolling_std.iloc[-1])
    if np.isnan(c_std):
        c_std = 0.0

    # 基线：带 gap 隔离，避免当前波动污染基线
    baseline_rolling_std = rolling_std.iloc[-baseline_window - gap : -gap].dropna()
    if len(baseline_rolling_std) == 0:
        return {
            "param": param_name,
            "current_std": round(c_std, 6),
            "baseline_std": 0.0,
            "ratio": 1.0,
            "level": None,
            "detected": False,
            "reason": "insufficient_baseline",
        }

    b_std = float(baseline_rolling_std.median())
    if b_std == 0 or np.isnan(b_std):
        b_std = float(baseline_rolling_std.mean())
    if b_std == 0 or np.isnan(b_std):
        b_std = 1e-6

    ratio = c_std / b_std if b_std > 0 else 1.0

    level = None
    detected = False
    for lv, th in sorted(multipliers.items(), key=lambda x: -x[1]):
        if ratio >= th:
            level = lv
            detected = True
            break

    return {
        "param": param_name,
        "current_std": round(float(c_std), 6),
        "baseline_std": round(float(b_std), 6),
        "ratio": round(float(ratio), 4),
        "level": level,
        "detected": detected,
    }


def detect_volatility_batch(
    series: pd.Series,
    current_window: int = DEFAULT_CURRENT_WINDOW,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
    multipliers: dict = None,
) -> pd.DataFrame:
    """
    批量波动率检测

    预计算全序列 rolling_std，然后对每个有效点取带 gap 的历史 median 作为基线。
    """
    if multipliers is None:
        multipliers = DEFAULT_MULTIPLIERS.copy()

    param_name = getattr(series, "name", "unknown")
    s = pd.to_numeric(series, errors="coerce")
    n = len(s)

    rolling_std = s.rolling(window=current_window, min_periods=current_window).std().to_numpy()

    ratios = np.ones(n)
    current_stds = np.zeros(n)
    baseline_stds = np.zeros(n)
    levels = np.empty(n, dtype=object)
    detected = np.zeros(n, dtype=bool)

    gap = current_window * 2
    min_i = baseline_window + gap + current_window
    sorted_multipliers = sorted(multipliers.items(), key=lambda x: -x[1])

    for i in range(min_i, n):
        c_std = rolling_std[i]
        if np.isnan(c_std):
            continue

        # 基线带 gap 隔离
        baseline_arr = rolling_std[i - baseline_window - gap : i - gap]
        baseline_arr = baseline_arr[~np.isnan(baseline_arr)]
        if len(baseline_arr) == 0:
            continue

        b_std = float(np.median(baseline_arr))
        if b_std == 0 or np.isnan(b_std):
            b_std = float(np.mean(baseline_arr))
        if b_std == 0 or np.isnan(b_std):
            b_std = 1e-6

        ratio = c_std / b_std

        current_stds[i] = c_std
        baseline_stds[i] = b_std
        ratios[i] = ratio

        for lv, th in sorted_multipliers:
            if ratio >= th:
                levels[i] = lv
                detected[i] = True
                break

    return pd.DataFrame(
        {
            "current_std": current_stds,
            "baseline_std": baseline_stds,
            "ratio": ratios,
            "level": levels,
            "detected": detected,
        },
        index=series.index,
    )


def detect_volatility_multi_params(
    df: pd.DataFrame,
    param_columns: list,
    current_window: int = DEFAULT_CURRENT_WINDOW,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
    multipliers: dict = None,
) -> dict:
    """多参数批量波动率检测"""
    results = {}
    for col in param_columns:
        if col not in df.columns:
            logger.warning("参数 %s 不在DataFrame中，跳过", col)
            continue
        results[col] = detect_volatility_batch(
            df[col],
            current_window=current_window,
            baseline_window=baseline_window,
            multipliers=multipliers,
        )
    return results
