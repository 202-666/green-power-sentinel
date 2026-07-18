"""
绿电哨兵 — 模拟数据生成器
生成30天正常工况数据 + 3类故障注入CSV
"""

import pandas as pd
import numpy as np
import os

# 13个监测参数定义（与 params_definition.yaml 保持一致）
# range: 正常运行范围，用于生成数据基准值
# cycle_amp: 日周期振幅（模拟昼夜温差等）
# danger_threshold: 危险/警告阈值，用于验证故障注入是否可被检测
PARAM_DEFS = {
    "furnace_temperature":    {"range": (850, 1050), "cycle_amp": 15,  "unit": "°C",      "danger_threshold": 1050},
    "flue_gas_temperature":   {"range": (180, 250),  "cycle_amp": 5,   "unit": "°C",      "danger_threshold": 280},
    "steam_pressure":         {"range": (3.5, 4.2),  "cycle_amp": 0.08,"unit": "MPa",     "danger_high": 4.5, "danger_low": 3.0},
    "steam_flow":             {"range": (40, 65),     "cycle_amp": 2,   "unit": "t/h",     "danger_threshold_low": 30},
    "bearing_vibration":      {"range": (0.5, 4.0),  "cycle_amp": 0.15,"unit": "mm/s",    "danger_threshold": 7.1},
    "bearing_temperature":    {"range": (35, 65),     "cycle_amp": 2,   "unit": "°C",      "danger_threshold": 85},
    "so2_concentration":      {"range": (20, 80),     "cycle_amp": 3,   "unit": "mg/Nm³",  "standard_limit": 200},
    "nox_concentration":      {"range": (30, 120),    "cycle_amp": 4,   "unit": "mg/Nm³",  "standard_limit": 200},
    "grate_speed":            {"range": (35, 75),     "cycle_amp": 2,   "unit": "%",       "danger_threshold_low": 10},
    "feed_rate":              {"range": (9, 14),      "cycle_amp": 0.4, "unit": "t/h",     "danger_threshold_low": 5},
    "oxygen_content":         {"range": (7, 11),      "cycle_amp": 0.3, "unit": "%",       "danger_low": 4},
    "furnace_pressure":       {"range": (-30, 30),    "cycle_amp": 3,   "unit": "Pa",      "danger_high": 200},
    "cooling_water_temp":     {"range": (28, 42),     "cycle_amp": 1.5, "unit": "°C",      "danger_threshold": 55},
}

PARAM_NAMES = list(PARAM_DEFS.keys())


def _generate_correlated_noise(n: int, rng: np.random.Generator) -> dict:
    """
    生成带参数间相关性的噪声
    相关性设计（基于焚烧炉物理耦合关系）：
      - furnace_temperature ↑ → steam_pressure ↑ (r≈0.6, 热力耦合)
      - furnace_temperature ↑ → so2_concentration ↑ (r≈0.3, 燃烧影响排放)
      - grate_speed ↑ → feed_rate ↑ (r≈0.7, 机械联动)
      - oxygen_content ↓ → nox_concentration ↑ (r≈-0.4, 燃烧化学)
    """
    z = rng.standard_normal((n, len(PARAM_NAMES)))

    idx = {name: i for i, name in enumerate(PARAM_NAMES)}
    corr = np.eye(len(PARAM_NAMES))

    corr[idx["furnace_temperature"], idx["steam_pressure"]] = 0.6
    corr[idx["steam_pressure"], idx["furnace_temperature"]] = 0.6
    corr[idx["furnace_temperature"], idx["so2_concentration"]] = 0.3
    corr[idx["so2_concentration"], idx["furnace_temperature"]] = 0.3
    corr[idx["grate_speed"], idx["feed_rate"]] = 0.7
    corr[idx["feed_rate"], idx["grate_speed"]] = 0.7
    corr[idx["oxygen_content"], idx["nox_concentration"]] = -0.4
    corr[idx["nox_concentration"], idx["oxygen_content"]] = -0.4

    L = np.linalg.cholesky(corr)
    correlated = z @ L.T

    return {name: correlated[:, i] for i, name in enumerate(PARAM_NAMES)}


def generate_normal_data(
    days: int = 30,
    sampling_interval_min: int = 1,
    output_path: str = "data/sample_data/normal_30days.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成正常工况的模拟数据
    - 各参数在normal_range内随机波动（高斯噪声）
    - 加入日周期性（14:00峰值，模拟午后最高负荷）
    - 加入微弱趋势（30天内总偏移≤0.5σ，模拟设备缓慢老化）
    - 参数间相关性（通过Cholesky分解）
    """
    rng = np.random.default_rng(seed)
    n_points = days * 1440 // sampling_interval_min
    timestamps = pd.date_range(
        start="2026-07-01", periods=n_points, freq=f"{sampling_interval_min}min"
    )

    noise = _generate_correlated_noise(n_points, rng)
    data = {"timestamp": timestamps}

    # 日周期：sin在hour=14时达到峰值(值=1)
    hours = timestamps.hour + timestamps.minute / 60.0
    day_cycle = np.sin(2 * np.pi * (hours - 8) / 24)

    t = np.arange(n_points, dtype=float)

    for name, pdef in PARAM_DEFS.items():
        lo, hi = pdef["range"]
        mid = (lo + hi) / 2
        sigma = (hi - lo) / 10
        amp = pdef["cycle_amp"]

        # 微弱老化趋势：30天内总偏移 ≤ 0.5σ
        aging_direction = rng.choice([-1, 1])
        aging_trend = 0.5 * sigma * (t / n_points) * aging_direction

        values = mid + amp * day_cycle + sigma * noise[name] + aging_trend

        # 截断到正常范围
        values = np.clip(values, lo, hi)
        data[name] = np.round(values, 2)

    df = pd.DataFrame(data)

    df["device_id"] = "INC-01-BRG-01"
    df["device_name"] = "1#焚烧炉-引风机-前轴承"
    df["device_type"] = "焚烧炉"
    df["data_quality_flag"] = "正常"
    df["qc_note"] = ""

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"正常工况数据已生成: {output_path} ({len(df)} 行)")

    return df


def inject_fault(
    normal_df: pd.DataFrame,
    fault_type: str,
    start_idx: int,
    duration_min: int = 120,
    severity: float = 1.0,
    seed: int = 123,
) -> pd.DataFrame:
    """
    在正常数据上注入指定类型的故障
    故障值从正常数据在故障起点的实际值开始渐变，避免跳变

    fault_type:
      - "bearing_overheat": 轴承温度线性上升 + 振动增大
      - "emission_exceed": SO2/NOx指数上升 + 含氧量下降
      - "grate_jam": 炉排速度波动增大后骤降 + 给料速度下降 + 炉膛压力升高

    severity: 0-1, 故障严重程度
    """
    df = normal_df.copy()
    rng = np.random.default_rng(seed)
    end_idx = min(start_idx + duration_min, len(df))
    fault_range = range(start_idx, end_idx)
    n_fault = len(fault_range)

    # 归一化进度 0→1
    progress = np.linspace(0, 1, n_fault)

    if fault_type == "bearing_overheat":
        # 故障注入规格：轴承温度 45→85°C，振动 2→7 mm/s
        # 从正常数据实际值开始渐变到目标峰值
        start_btemp = df.loc[start_idx, "bearing_temperature"]
        start_bvib = df.loc[start_idx, "bearing_vibration"]
        peak_btemp = 85.0 * severity
        peak_bvib = 7.0 * severity

        # 轴承温度：线性上升，从当前值到峰值
        df.loc[fault_range, "bearing_temperature"] = np.round(
            start_btemp + (peak_btemp - start_btemp) * progress, 2
        )

        # 轴承振动：增大 + 随机噪声
        vib_noise = rng.normal(0, 0.3 * severity, n_fault)
        df.loc[fault_range, "bearing_vibration"] = np.round(
            np.clip(
                start_bvib + (peak_bvib - start_bvib) * progress + vib_noise,
                0, 10
            ), 2
        )

        # 关联影响：炉膛温度微升、蒸汽压力微降
        df.loc[fault_range, "furnace_temperature"] = np.round(
            df.loc[fault_range, "furnace_temperature"] + 10 * severity * progress, 2
        )
        df.loc[fault_range, "steam_pressure"] = np.round(
            df.loc[fault_range, "steam_pressure"] - 0.1 * severity * progress, 2
        )

    elif fault_type == "emission_exceed":
        # 故障注入规格：SO2 60→180, NOx 100→190, O2 8.5→5.5
        start_so2 = df.loc[start_idx, "so2_concentration"]
        start_nox = df.loc[start_idx, "nox_concentration"]
        start_o2 = df.loc[start_idx, "oxygen_content"]
        peak_so2 = 180.0 * severity
        peak_nox = 190.0 * severity
        target_o2 = max(5.5, start_o2 - 3.0 * severity)

        # 指数上升曲线：初期缓慢、后期加速
        exp_curve = (np.exp(3 * progress) - 1) / (np.exp(3) - 1)

        df.loc[fault_range, "so2_concentration"] = np.round(
            start_so2 + (peak_so2 - start_so2) * exp_curve, 2
        )
        df.loc[fault_range, "nox_concentration"] = np.round(
            start_nox + (peak_nox - start_nox) * exp_curve, 2
        )
        df.loc[fault_range, "oxygen_content"] = np.round(
            start_o2 + (target_o2 - start_o2) * exp_curve, 2
        )

        # 关联影响：烟气温度微升
        df.loc[fault_range, "flue_gas_temperature"] = np.round(
            df.loc[fault_range, "flue_gas_temperature"] + 15 * severity * exp_curve, 2
        )

    elif fault_type == "grate_jam":
        # 故障注入规格：炉排速度波动增大后骤降，给料下降，炉膛压力升高
        start_grate = df.loc[start_idx, "grate_speed"]
        start_feed = df.loc[start_idx, "feed_rate"]
        start_fp = df.loc[start_idx, "furnace_pressure"]
        target_grate = 15.0
        target_feed = max(6.0, start_feed - 5.0 * severity)
        target_fp = start_fp + 100.0 * severity

        half = n_fault // 2

        # 阶段1：炉排速度波动增大（前半段）
        inc_sigmas = 2 + 6 * severity * np.linspace(0, 1, half)
        grate_phase1 = start_grate + rng.normal(0, inc_sigmas, half)

        # 阶段2：炉排速度骤降（后半段）
        grate_phase2 = np.linspace(start_grate, target_grate, n_fault - half)

        grate_values = np.concatenate([grate_phase1, grate_phase2])
        n_actual = min(len(grate_values), n_fault)
        df.loc[start_idx:start_idx + n_actual - 1, "grate_speed"] = np.round(
            np.clip(grate_values[:n_actual], 0, 95), 2
        )

        # 给料速度：线性下降
        df.loc[fault_range, "feed_rate"] = np.round(
            start_feed + (target_feed - start_feed) * progress, 2
        )

        # 炉膛压力：线性上升
        df.loc[fault_range, "furnace_pressure"] = np.round(
            start_fp + (target_fp - start_fp) * progress, 2
        )

        # 关联影响：炉膛温度微升
        df.loc[fault_range, "furnace_temperature"] = np.round(
            df.loc[fault_range, "furnace_temperature"] + 20 * severity * progress, 2
        )

    else:
        raise ValueError(f"未知故障类型: {fault_type}，支持: bearing_overheat, emission_exceed, grate_jam")

    # 标记故障区间
    df.loc[fault_range, "data_quality_flag"] = "故障注入"
    df.loc[fault_range, "qc_note"] = f"注入故障: {fault_type}, 严重度={severity}"

    return df


def generate_all_fault_csvs(
    normal_df: pd.DataFrame,
    output_dir: str = "data/sample_data",
) -> dict:
    """
    根据框架规格生成3类故障CSV：
    - 轴承过热: 第15天 10:00, 持续120分钟
    - 烟气超标: 第20天 14:00, 持续180分钟
    - 炉排卡滞: 第25天 08:00, 持续90分钟
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = normal_df["timestamp"].iloc[0]

    fault_configs = [
        {
            "fault_type": "bearing_overheat",
            "fault_time": "2026-07-15 10:00",
            "duration_min": 120,
            "filename": "fault_bearing_overheat.csv",
        },
        {
            "fault_type": "emission_exceed",
            "fault_time": "2026-07-20 14:00",
            "duration_min": 180,
            "filename": "fault_emission_exceed.csv",
        },
        {
            "fault_type": "grate_jam",
            "fault_time": "2026-07-25 08:00",
            "duration_min": 90,
            "filename": "fault_grate_jam.csv",
        },
    ]

    results = {}
    for fc in fault_configs:
        fault_time = pd.Timestamp(fc["fault_time"])
        start_idx = int((fault_time - start_time).total_seconds() / 60)

        fault_df = inject_fault(
            normal_df.copy(),
            fault_type=fc["fault_type"],
            start_idx=start_idx,
            duration_min=fc["duration_min"],
            severity=1.0,
        )

        filepath = os.path.join(output_dir, fc["filename"])
        fault_df.to_csv(filepath, index=False)
        print(f"故障数据已生成: {filepath} (故障类型={fc['fault_type']}, "
              f"起始={fc['fault_time']}, 持续={fc['duration_min']}分钟)")
        results[fc["fault_type"]] = fault_df

    return results


def validate_data(df: pd.DataFrame, label: str = "") -> None:
    """验证生成数据的统计分布和故障特征"""
    print(f"\n{'='*60}")
    print(f"数据验证: {label}")
    print(f"{'='*60}")
    print(f"总行数: {len(df)}")

    # 正常/故障数据分组
    fault_mask = df["data_quality_flag"] == "故障注入"

    for name in PARAM_NAMES:
        if name not in df.columns:
            continue
        pdef = PARAM_DEFS[name]
        lo, hi = pdef["range"]
        vals = df[name]
        normal_vals = df.loc[~fault_mask, name]
        fault_vals = df.loc[fault_mask, name]

        # 检查正常数据是否在范围内（允许5%的clip容差）
        in_range_ratio = ((normal_vals >= lo) & (normal_vals <= hi)).mean()

        print(f"\n  {name} ({pdef['unit']}):")
        print(f"    正常区间: [{lo}, {hi}]")
        print(f"    全量统计: mean={vals.mean():.2f}, std={vals.std():.2f}, "
              f"min={vals.min():.2f}, max={vals.max():.2f}")
        print(f"    正常数据: mean={normal_vals.mean():.2f}, std={normal_vals.std():.2f}, "
              f"范围内占比={in_range_ratio:.1%}")
        if len(fault_vals) > 0:
            out_of_range = ((fault_vals < lo) | (fault_vals > hi)).sum()
            print(f"    故障数据: mean={fault_vals.mean():.2f}, std={fault_vals.std():.2f}, "
                  f"超正常范围={out_of_range}/{len(fault_vals)}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "data", "sample_data")

    # 1. 生成30天正常数据
    normal_path = os.path.join(output_dir, "normal_30days.csv")
    normal_df = generate_normal_data(days=30, output_path=normal_path)
    validate_data(normal_df, "30天正常工况数据")

    # 2. 生成3类故障数据
    fault_results = generate_all_fault_csvs(normal_df, output_dir)

    # 3. 验证故障数据
    for fault_type, fault_df in fault_results.items():
        validate_data(fault_df, f"故障注入: {fault_type}")

    print(f"\n{'='*60}")
    print("所有数据生成完成！")
    print(f"输出目录: {output_dir}")
    print(f"文件列表:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {f} ({size_mb:.1f} MB)")
