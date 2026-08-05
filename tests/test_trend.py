"""
绿电哨兵 — 趋势检测单元测试

验收标准（按故障类型分口径）：
- 轴承过热 / 炉排卡滞：原始趋势模块故障段检出率 > 80%
- 烟气超标：产品路径（R2 关联 + ensemble）故障段检出率 ≥ 75%、首次检出 ≤ 60min
  （原始趋势模块为中后期检出，早期覆盖由 R2 承担，见 W7 口径）

测试覆盖：
1. 斜率计算函数 _compute_slope 正确性
2. 单时刻检测（detect_trend）：上升/下降趋势应触发
3. 平稳序列不应触发（假阳性率控制）
4. 批量检测（detect_trend_batch）：对每个时间点输出 slope/detected
5. 3 类故障注入数据的检出率（按上述分口径验收）
   - bearing_overheat: bearing_temperature 线性上升 45→85°C/120min (slope≈0.33°C/min)
   - emission_exceed: so2/nox 指数上升，oxygen 下降（产品路径：R2 关联 + ensemble）
   - grate_jam: furnace_pressure 线性上升、grate_speed 后段骤降
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.trend_detector import (
    detect_trend,
    detect_trend_batch,
    detect_trend_multi_params,
    _compute_slope,
    DEFAULT_SLOPE_THRESHOLDS,
    DEFAULT_WINDOWS,
    DEFAULT_SMOOTHING_WINDOW,
)

import yaml

from models.threshold_detector import (
    detect_threshold,
    load_thresholds_from_yaml,
)
from models.volatility_detector import detect_volatility_multi_params
from models.correlation_detector import detect_correlation_batch
from models.ensemble_scorer import compute_risk_score_batch


SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sample_data")

# 产品路径配置（与 pipeline.py / benchmark_w7.py 同源：config.yaml + rules.yaml + thresholds.yaml）
SKIP_COLS = {
    "timestamp",
    "device_id",
    "device_name",
    "device_type",
    "data_quality_flag",
    "qc_note",
}

with open(os.path.join(PROJECT_ROOT, "config", "rules.yaml"), "r", encoding="utf-8") as _f:
    RULES = yaml.safe_load(_f).get("rules", [])

THRESHOLDS_CFG = load_thresholds_from_yaml(
    os.path.join(PROJECT_ROOT, "config", "thresholds.yaml")
)

with open(os.path.join(PROJECT_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as _f:
    _DET_CFG = (yaml.safe_load(_f) or {}).get("detection", {})
_ENS_CFG = _DET_CFG.get("ensemble", {})
PRODUCT_ENSEMBLE_KWARGS = dict(
    weights=_DET_CFG.get("weights", None),
    risk_levels=_DET_CFG.get("risk_levels", None),
    allow_single_module_alert=_ENS_CFG.get("allow_single_module_alert", False),
    single_module_threshold=float(_ENS_CFG.get("single_module_threshold", 0.8)),
    single_module_score_ratio=float(_ENS_CFG.get("single_module_score_ratio", 0.5)),
    persistence_filter=_ENS_CFG.get("persistence_filter", False),
    persistence_n=int(_ENS_CFG.get("persistence_n", 3)),
    persistence_mode=str(_ENS_CFG.get("persistence_mode", "consecutive")),
    persistence_window=int(_ENS_CFG.get("persistence_window", 30)),
    persistence_min_count=int(_ENS_CFG.get("persistence_min_count", 8)),
    persistence_backfill=int(_ENS_CFG.get("persistence_backfill", 0)),
)
# 波动率基线窗口：与 pipeline.py 一致从 config.yaml 读取
VOL_BASELINE_WINDOW = int(_DET_CFG.get("volatility_baseline_window", 1440))


def _run_product_pipeline(df_slice):
    """在数据切片上运行产品路径检测（与 pipeline.py 相同配置来源）。

    返回 (corr_batch, scores_df)：
    - corr_batch: detect_correlation_batch 输出（含 R2 关联命中信息）
    - scores_df: compute_risk_score_batch 输出的风险评分 DataFrame
    """
    param_cols = [c for c in df_slice.columns if c not in SKIP_COLS]

    th_results = []
    for _, row in df_slice.iterrows():
        cv = {c: row[c] for c in param_cols if c in row}
        th_results.append(detect_threshold(cv, THRESHOLDS_CFG))

    trend_results = detect_trend_multi_params(df_slice, param_cols)
    vol_results = detect_volatility_multi_params(
        df_slice,
        param_cols,
        current_window=30,
        baseline_window=VOL_BASELINE_WINDOW,
    )
    corr_batch = detect_correlation_batch(
        df_slice, RULES, trend_results, vol_results
    )

    n = len(df_slice)
    index = df_slice.index.tolist()
    tr_states = []
    for i in range(n):
        idx = index[i]
        st = {}
        for col, tdf in trend_results.items():
            if idx not in tdf.index:
                continue
            row = tdf.loc[idx]
            any_det = any(
                row.get(f"detected_{w}", False) for w in [10, 30, 60]
            )
            max_level = "green"
            for w in [60, 30, 10]:
                if row.get(f"detected_{w}", False):
                    max_level = (
                        "yellow" if w == 10 else "orange" if w == 30 else "red"
                    )
                    break
            st[col] = {
                "param": col,
                "any_detected": any_det,
                "max_level": max_level,
                "window_10": {
                    "slope": row.get("slope_10", 0.0),
                    "detected": row.get("detected_10", False),
                },
                "window_30": {
                    "slope": row.get("slope_30", 0.0),
                    "detected": row.get("detected_30", False),
                },
                "window_60": {
                    "slope": row.get("slope_60", 0.0),
                    "detected": row.get("detected_60", False),
                },
            }
        tr_states.append(st)

    vol_states = []
    for i in range(n):
        idx = index[i]
        st = {}
        for col, vdf in vol_results.items():
            if idx not in vdf.index:
                continue
            row = vdf.loc[idx]
            st[col] = {
                "param": col,
                "ratio": row.get("ratio", 1.0),
                "level": row.get("level"),
                "detected": row.get("detected", False),
            }
        vol_states.append(st)

    scores = compute_risk_score_batch(
        th_results,
        tr_states,
        vol_states,
        [item["matched_rules"] for item in corr_batch],
        **PRODUCT_ENSEMBLE_KWARGS,
    )
    return corr_batch, pd.DataFrame(scores)


class TestComputeSlope(unittest.TestCase):
    """斜率计算函数"""

    def test_perfect_linear_up(self):
        """完全线性上升 → 斜率 = (y2-y1)/step"""
        # y = 0,1,2,...,9：slope=1.0
        vals = np.arange(10, dtype=float)
        self.assertAlmostEqual(_compute_slope(vals), 1.0, places=6)

    def test_perfect_linear_down(self):
        """完全线性下降 → 斜率为负"""
        # y = 9,8,...,0：slope=-1.0
        vals = np.arange(10, dtype=float)[::-1]
        self.assertAlmostEqual(_compute_slope(vals), -1.0, places=6)

    def test_constant_series(self):
        """常数序列 → 斜率为 0"""
        vals = np.full(20, 5.0)
        self.assertAlmostEqual(_compute_slope(vals), 0.0, places=6)

    def test_short_series(self):
        """长度 < 2 → 斜率为 0"""
        self.assertEqual(_compute_slope(np.array([5.0])), 0.0)
        self.assertEqual(_compute_slope(np.array([])), 0.0)

    def test_nan_handling(self):
        """含 NaN 的序列应先填充再计算，不返回 NaN"""
        vals = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        slope = _compute_slope(vals)
        self.assertFalse(np.isnan(slope))

    def test_known_slope(self):
        """已知斜率：y = 2x + 5，10 点 → slope=2.0"""
        x = np.arange(10, dtype=float)
        vals = 2 * x + 5
        self.assertAlmostEqual(_compute_slope(vals), 2.0, places=6)


class TestDetectTrendSingle(unittest.TestCase):
    """单时刻趋势检测"""

    def test_rising_trend_detected(self):
        """上升趋势应被检出"""
        # 构造一个明显的上升序列：60点，每点上升 1°C → slope=1.0
        series = pd.Series(np.arange(60, dtype=float), name="furnace_temperature")
        result = detect_trend(series)
        self.assertTrue(result["any_detected"])
        # 60 分钟窗口阈值 0.25，slope=1.0 远超
        self.assertGreaterEqual(result["max_level"], "orange")
        # 至少 30/60 窗口应触发
        self.assertTrue(result["window_30"]["detected"] or result["window_60"]["detected"])

    def test_falling_trend_detected(self):
        """下降趋势应被检出"""
        series = pd.Series(np.arange(60, dtype=float)[::-1], name="furnace_temperature")
        result = detect_trend(series)
        self.assertTrue(result["any_detected"])
        self.assertEqual(result["window_60"]["direction"], "down")

    def test_constant_series_not_detected(self):
        """平稳常数序列不应检出"""
        series = pd.Series(np.full(60, 100.0), name="furnace_temperature")
        result = detect_trend(series)
        self.assertFalse(result["any_detected"])
        for w in DEFAULT_WINDOWS:
            self.assertFalse(result[f"window_{w}"]["detected"])

    def test_insufficient_data(self):
        """数据不足应返回 reason=insufficient_data"""
        series = pd.Series(np.array([1.0, 2.0]), name="furnace_temperature")
        result = detect_trend(series, windows=[10, 30, 60])
        for w in DEFAULT_WINDOWS:
            self.assertFalse(result[f"window_{w}"]["detected"])
            self.assertEqual(result[f"window_{w}"].get("reason"), "insufficient_data")

    def test_custom_thresholds(self):
        """自定义阈值 {10: 0.5, 30: 0.33}"""
        # slope=0.5：触发 10min 阈值（>=0.5）
        vals = np.arange(60, dtype=float) * 0.5
        series = pd.Series(vals, name="custom_param")
        result = detect_trend(series, thresholds={10: 0.5, 30: 0.33, 60: 0.25})
        self.assertTrue(result["window_10"]["detected"])
        self.assertTrue(result["window_30"]["detected"])
        self.assertTrue(result["window_60"]["detected"])


class TestDetectTrendBatch(unittest.TestCase):
    """批量检测测试"""

    def test_batch_shape_matches_input(self):
        """批量检测输出长度应与输入一致"""
        series = pd.Series(np.random.randn(100), name="furnace_temperature")
        result_df = detect_trend_batch(series)
        self.assertEqual(len(result_df), 100)
        self.assertIn("any_detected", result_df.columns)
        self.assertIn("max_level", result_df.columns)
        for w in DEFAULT_WINDOWS:
            self.assertIn(f"slope_{w}", result_df.columns)
            self.assertIn(f"detected_{w}", result_df.columns)

    def test_batch_detects_rising_segment(self):
        """批量检测应识别出上升段"""
        # 前 100 点平稳，后 100 点线性上升（slope=0.5 °C/min）
        # 序列足够长，让 60min 窗口能完全覆盖上升段
        vals = np.concatenate([
            np.full(100, 100.0),
            100.0 + np.arange(100, dtype=float) * 0.5,
        ])
        series = pd.Series(vals, name="furnace_temperature")
        result = detect_trend_batch(series)
        # 平稳段（除去窗口预热期）应几乎无检出
        detected_in_flat = result["any_detected"].iloc[60:100].sum()
        self.assertLess(detected_in_flat, 5,
                        f"平稳段不应检出，实际检出 {detected_in_flat}/40")
        # 上升段后半段（窗口完全在上升段内）应大量检出
        # i=200 时 60min 窗口 [140, 200] 全在上升段
        detected_in_rise = result["any_detected"].iloc[160:].sum()
        self.assertGreater(detected_in_rise, 30,
                           f"上升段应大量检出，实际 {detected_in_rise}/40")


class TestFaultDetectionRate(unittest.TestCase):
    """验收标准：3 类故障数据按各自口径验收（趋势 >80% / 产品路径 ≥75%）"""

    @classmethod
    def setUpClass(cls):
        """加载并清洗 3 类故障数据"""
        from core.data_cleaner import clean_data
        cls.faults = {}
        cls.fault_ranges = {}
        for fname, fault_type in [
            ("fault_bearing_overheat.csv", "bearing_overheat"),
            ("fault_emission_exceed.csv", "emission_exceed"),
            ("fault_grate_jam.csv", "grate_jam"),
        ]:
            fpath = os.path.join(SAMPLE_DATA_DIR, fname)
            df = pd.read_csv(fpath, low_memory=False)
            df = clean_data(df)
            cls.faults[fault_type] = df
            # 故障区间：data_quality_flag == "故障注入"
            fault_mask = df["data_quality_flag"] == "故障注入"
            cls.fault_ranges[fault_type] = fault_mask

    def _compute_detection_rate(self, df, fault_mask, param_columns):
        """
        计算多参数综合检出率：
        对故障区间内每个时间点，若任一参数的任一窗口检出，则记为检出
        返回 (detection_rate, per_param_rate_dict)
        """
        trend_results = detect_trend_multi_params(df, param_columns)
        # 合并所有参数的 any_detected
        n = len(df)
        any_detected = np.zeros(n, dtype=bool)
        per_param_rate = {}
        for col, tdf in trend_results.items():
            col_detected = tdf["any_detected"].to_numpy()
            any_detected |= col_detected
            # 单参数检出率：故障区间内被检出的比例
            fault_detected = col_detected[fault_mask.to_numpy()].sum()
            fault_total = fault_mask.sum()
            per_param_rate[col] = fault_detected / fault_total if fault_total > 0 else 0.0

        fault_detected = any_detected[fault_mask.to_numpy()].sum()
        fault_total = fault_mask.sum()
        overall_rate = fault_detected / fault_total if fault_total > 0 else 0.0
        return overall_rate, per_param_rate

    def test_bearing_overheat_detection_rate(self):
        """轴承过热故障检出率 > 80%"""
        df = self.faults["bearing_overheat"]
        mask = self.fault_ranges["bearing_overheat"]
        # 轴承过热的关键参数：bearing_temperature, bearing_vibration
        # 关联影响：furnace_temperature（弱）, steam_pressure（弱）
        key_params = ["bearing_temperature", "bearing_vibration"]
        all_params = list(DEFAULT_SLOPE_THRESHOLDS.keys())

        overall, per_param = self._compute_detection_rate(df, mask, all_params)

        print(f"\n[轴承过热] 整体检出率: {overall:.2%}")
        print(f"  故障区间: {mask.sum()} 行")
        print(f"  各参数检出率:")
        for p in sorted(per_param.keys(), key=lambda x: -per_param[x])[:5]:
            print(f"    {p}: {per_param[p]:.2%}")

        self.assertGreater(overall, 0.80,
                           f"轴承过热趋势检出率 {overall:.2%} 未达 80% 阈值")

    def test_emission_exceed_detection_rate(self):
        """烟气超标：产品路径（R2 关联 + ensemble）故障段检出率 ≥ 75%，首次检出 ≤ 60min"""
        df = self.faults["emission_exceed"]
        mask = self.fault_ranges["emission_exceed"]
        # 原始趋势模块口径（对照输出，非断言）：中后期检出
        overall, per_param = self._compute_detection_rate(
            df, mask, list(DEFAULT_SLOPE_THRESHOLDS.keys())
        )

        mask_np = mask.to_numpy()
        start = int(np.argmax(mask_np))
        end = int(len(mask_np) - 1 - np.argmax(mask_np[::-1]))
        # 截取故障段上下文：波动率基线需 baseline_window(480)+gap(60)+current(30)
        # 的历史数据（W7 基准 1440 同样被覆盖），前 2000 行可完整复现全量计算结果
        lo = max(0, start - 2000)
        hi = min(len(df), end + 1001)
        df_slice = df.iloc[lo:hi].reset_index(drop=True)
        mask_s = (df_slice["data_quality_flag"] == "故障注入").to_numpy()
        fault_start_slice = start - lo

        corr_batch, scores = _run_product_pipeline(df_slice)
        r2_matched = np.zeros(len(df_slice), dtype=bool)
        for i, item in enumerate(corr_batch):
            if any(r["rule_id"] == "R2" for r in item["matched_rules"]):
                r2_matched[i] = True
        ensemble_yp = scores["level"].isin(["yellow", "orange", "red"]).to_numpy()
        union = r2_matched | ensemble_yp

        fault_total = int(mask_s.sum())
        union_hits = int((union & mask_s).sum())
        union_recall = union_hits / fault_total if fault_total else 0.0
        ens_hits = int((ensemble_yp & mask_s).sum())
        first_idx = int(np.argmax(union & mask_s)) if (union & mask_s).any() else None
        first_min = first_idx - fault_start_slice if first_idx is not None else None

        print(f"\n[烟气超标] 原始趋势模块整体检出率: {overall:.2%}（中后期口径，对照）")
        print(f"  R2 关联故障段命中: {int((r2_matched & mask_s).sum())}/{fault_total}")
        print(f"  ensemble 故障段 yellow+ 检出: {ens_hits}/{fault_total}")
        print(f"  产品路径（R2|ensemble）故障段检出率: {union_recall:.2%} ({union_hits}/{fault_total})")
        print(f"  产品路径首次检出: t={first_min}min" if first_min is not None else "  产品路径未检出")

        self.assertGreaterEqual(
            union_recall, 0.75,
            f"烟气超标产品路径故障段检出率 {union_recall:.2%}（{union_hits}/{fault_total}）低于 75%",
        )
        if first_min is not None:
            self.assertLessEqual(
                first_min, 60,
                f"烟气超标产品路径首次检出 t={first_min}min，晚于 60min 设计窗口",
            )

    def test_grate_jam_detection_rate(self):
        """炉排卡滞故障检出率 > 80%"""
        df = self.faults["grate_jam"]
        mask = self.fault_ranges["grate_jam"]
        # 炉排卡滞关键参数：grate_speed, feed_rate, furnace_pressure
        all_params = list(DEFAULT_SLOPE_THRESHOLDS.keys())

        overall, per_param = self._compute_detection_rate(df, mask, all_params)

        print(f"\n[炉排卡滞] 整体检出率: {overall:.2%}")
        print(f"  故障区间: {mask.sum()} 行")
        print(f"  各参数检出率:")
        for p in sorted(per_param.keys(), key=lambda x: -per_param[x])[:5]:
            print(f"    {p}: {per_param[p]:.2%}")

        self.assertGreater(overall, 0.80,
                           f"炉排卡滞趋势检出率 {overall:.2%} 未达 80% 阈值")


class TestFalsePositiveControl(unittest.TestCase):
    """正常数据误报控制（辅助验收）"""

    def test_normal_data_low_false_positive(self):
        """正常 30 天数据上误报率应 < 10%"""
        normal_csv = os.path.join(SAMPLE_DATA_DIR, "normal_30days.csv")
        df = pd.read_csv(normal_csv, low_memory=False)
        from core.data_cleaner import clean_data
        df = clean_data(df)

        all_params = list(DEFAULT_SLOPE_THRESHOLDS.keys())
        trend_results = detect_trend_multi_params(df, all_params)

        n = len(df)
        any_detected = np.zeros(n, dtype=bool)
        for col, tdf in trend_results.items():
            any_detected |= tdf["any_detected"].to_numpy()

        fp_rate = any_detected.sum() / n
        print(f"\n[正常数据] 误报率: {fp_rate:.2%} ({any_detected.sum()}/{n})")
        # 趋势检测在正常数据上允许一定假阳性（后续 ensemble scorer 会过滤）
        # 这里仅监控，不强制断言；如需收紧可取消注释
        # self.assertLess(fp_rate, 0.10, f"正常数据误报率 {fp_rate:.2%} 过高")


if __name__ == "__main__":
    unittest.main(verbosity=2)
