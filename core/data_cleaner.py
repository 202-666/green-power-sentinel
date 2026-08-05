"""
绿电哨兵 — 数据清洗模块（Agent 1 核心组件）

功能：
1. 去重：同一 timestamp 仅保留最后一条
2. 时间对齐：将时间戳四舍五入到整分钟
3. 缺失值线性插值
4. 物理范围截断：超出物理合理范围的值标记为"异常"但不删除
5. 传感器健康检查：卡死/漂移检测

输入：raw_df（含 timestamp + 13个参数列 + 设备元数据）
输出：清洗后的 DataFrame，新增/更新 data_quality_flag、qc_note 列
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# 13个监测参数列名（与 thresholds.yaml 保持一致）
PARAM_COLUMNS = [
    "furnace_temperature",
    "flue_gas_temperature",
    "steam_pressure",
    "steam_flow",
    "bearing_vibration",
    "bearing_temperature",
    "so2_concentration",
    "nox_concentration",
    "grate_speed",
    "feed_rate",
    "oxygen_content",
    "furnace_pressure",
    "cooling_water_temp",
]

# 物理合理范围（远宽于正常运行范围，用于识别明显异常数据）
# 规则：温度类 -50~2000°C；压力类按量级；浓度类 0~上限；百分比 0~100
PHYSICAL_RANGES = {
    "furnace_temperature":   (-50, 2000),
    "flue_gas_temperature":  (-50, 500),
    "steam_pressure":        (-1, 20),
    "steam_flow":            (0, 500),
    "bearing_vibration":     (0, 50),
    "bearing_temperature":   (-50, 300),
    "so2_concentration":     (0, 5000),
    "nox_concentration":     (0, 5000),
    "grate_speed":           (0, 100),
    "feed_rate":             (0, 100),
    "oxygen_content":        (0, 25),
    "furnace_pressure":      (-1000, 1000),
    "cooling_water_temp":    (-50, 200),
}


def clean_data(raw_df: pd.DataFrame, param_defs: list = None) -> pd.DataFrame:
    """
    数据清洗流水线

    Args:
        raw_df: 原始数据，必须包含 timestamp 列和参数列
        param_defs: 参数定义列表（来自 thresholds.yaml），可选，目前仅用于校验

    Returns:
        清洗后的 DataFrame，包含 data_quality_flag 与 qc_note 列
    """
    if raw_df.empty:
        logger.warning("clean_data: 输入 DataFrame 为空")
        return raw_df.copy()

    df = raw_df.copy()

    # 确保 timestamp 列存在并转为 datetime
    if "timestamp" not in df.columns:
        raise ValueError("输入数据缺少 timestamp 列")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 0. 删除 timestamp 解析失败的记录
    n_before = len(df)
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    if len(df) < n_before:
        logger.warning(f"clean_data: 丢弃 {n_before - len(df)} 条 timestamp 无效的记录")

    # 1. 按原始时间排序（L6：去重语义不依赖输入顺序，
    #    同一取整分钟内的重复记录按实际时间取最后一条）
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2. 时间对齐：四舍五入到整分钟（避免 30 秒漂移）
    df["timestamp"] = df["timestamp"].dt.round("1min")

    # 3. 去重：同一 timestamp 保留最后一条（排序在取整前，keep="last" 按实际时间生效）
    n_before = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if len(df) < n_before:
        logger.info(f"clean_data: 去重 {n_before - len(df)} 条重复记录")

    # 3.5 严重缺失处理：超过50%参数缺失的记录跳过（写入日志）
    # 仅统计 PARAM_COLUMNS 中存在的列
    present_params = [c for c in PARAM_COLUMNS if c in df.columns]
    if present_params:
        n_params = len(present_params)
        threshold_missing = int(n_params * 0.5)
        missing_count = df[present_params].isna().sum(axis=1)
        dropped_mask = missing_count > threshold_missing
        n_dropped = int(dropped_mask.sum())
        if n_dropped > 0:
            dropped_indices = df.index[dropped_mask]
            logger.warning(f"clean_data: 跳过 {n_dropped} 条严重缺失记录（>50%参数缺失），索引: {list(dropped_indices[:5])}{'...' if n_dropped > 5 else ''}")
            df = df[~dropped_mask].reset_index(drop=True)

    # 4. 数值列强制为 float（避免字符串污染）
    for col in PARAM_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. 缺失值线性插值（仅在参数列上，不填充元数据列）
    for col in PARAM_COLUMNS:
        if col in df.columns:
            n_missing_before = df[col].isna().sum()
            if n_missing_before > 0:
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
                logger.info(f"clean_data: {col} 插值填充 {n_missing_before} 个缺失值")

    # 6. 物理范围检查（标记不删除）
    if "data_quality_flag" not in df.columns:
        df["data_quality_flag"] = "正常"
    if "qc_note" not in df.columns:
        df["qc_note"] = ""

    # 字符串列确保为 str 类型（避免 NaN）
    df["data_quality_flag"] = df["data_quality_flag"].astype(str).replace("nan", "正常")
    df["qc_note"] = df["qc_note"].astype(str).replace("nan", "")

    for col, (lo, hi) in PHYSICAL_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range_mask = (df[col] < lo) | (df[col] > hi)
        n_out = int(out_of_range_mask.sum())
        if n_out > 0:
            # 仅对原本是"正常"的记录升级为"异常"，保留"故障注入"标记
            upgrade_mask = out_of_range_mask & (df["data_quality_flag"] == "正常")
            df.loc[upgrade_mask, "data_quality_flag"] = "异常"
            for idx in df.index[upgrade_mask]:
                note = f"{col}={df.at[idx, col]:.2f} 超出物理范围[{lo},{hi}]"
                df.at[idx, "qc_note"] = (df.at[idx, "qc_note"] + "; " + note).strip("; ")
            logger.warning(f"clean_data: {col} 有 {n_out} 条记录超出物理范围 [{lo},{hi}]，已标记为异常")

    # 7. 传感器健康检查（卡死/漂移）
    qc_flags = check_sensor_health(df, window=10)
    # 合并 qc_flags 到 data_quality_flag
    for idx, flag in qc_flags.items():
        if flag != "正常":
            current = df.at[idx, "data_quality_flag"]
            # 不覆盖"故障注入"标记
            if current == "正常":
                df.at[idx, "data_quality_flag"] = flag
            note = f"传感器:{flag}"
            df.at[idx, "qc_note"] = (df.at[idx, "qc_note"] + "; " + note).strip("; ")

    return df


def check_sensor_health(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    传感器健康检查

    Args:
        df: 已清洗的 DataFrame（按 timestamp 排序）
        window: 滑动窗口大小（连续不变点数阈值）

    Returns:
        qc_flag Series，索引与 df 对齐，值为：正常/卡死/漂移
    """
    qc_flags = pd.Series(["正常"] * len(df), index=df.index)

    if df.empty or len(df) < window:
        return qc_flags

    for col in PARAM_COLUMNS:
        if col not in df.columns:
            continue

        series = df[col].astype(float)

        # --- 卡死检测：连续 window 个点值完全不变 ---
        # 用 diff 累积判断
        diff = series.diff().abs()
        # 第一个点 diff 为 NaN，置为 0 避免误判
        diff = diff.fillna(0)
        is_stuck = (diff == 0).astype(int)

        # 滑动窗口求和，>=window 表示窗口内全为 0（即全部不变）
        # 注意：window 内连续不变，需用前向窗口
        stuck_sum = is_stuck.rolling(window=window, min_periods=window).sum()
        # 当窗口内 window 个点全部"未变化"（包括起始点），认为卡死
        stuck_mask = stuck_sum >= window

        for idx in df.index[stuck_mask.fillna(False)]:
            if qc_flags[idx] == "正常":
                qc_flags[idx] = "卡死"

        # --- 漂移检测：window 内斜率绝对值超过 3σ/window ---
        # 使用滚动窗口标准差（基于历史数据），避免全序列包含趋势时 sigma 被拉高
        rolling_sigma = series.rolling(window=window * 2, min_periods=window).std()
        rolling_sigma = rolling_sigma.fillna(series.std())

        if not rolling_sigma.isna().all() and (rolling_sigma > 0).any():
            # 计算滚动斜率（最小二乘线性回归）
            x = np.arange(window, dtype=float)
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()

            def _slope(x):
                if len(x) < 2:
                    return 0.0
                y = np.asarray(x, dtype=float)
                if np.isnan(y).any():
                    y = pd.Series(y).ffill().bfill().to_numpy()
                    if np.isnan(y).all():
                        return 0.0
                y_mean = y.mean()
                return float(((x - x_mean) * (y - y_mean)).sum() / x_var)

            slopes = (
                series.rolling(window=window, min_periods=window)
                .apply(_slope, raw=True)
                .fillna(0.0)
            )

            # 斜率阈值 = 3 * sigma / window（转化为 per-point 变化率）
            slope_threshold = 3 * rolling_sigma / window
            drift_mask = np.abs(slopes) >= slope_threshold

            for idx in df.index[drift_mask.fillna(False)]:
                if qc_flags[idx] == "正常":
                    qc_flags[idx] = "漂移"

    return qc_flags
