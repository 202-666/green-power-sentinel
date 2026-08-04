"""
绿电哨兵 — 综合评分模块（Agent 2 融合层）

融合阈值检测、趋势检测、波动率检测、关联检测四个模块的结果，
计算综合风险评分、等级、置信度和根因分析。

评分策略：
1. 各子模块输出归一化为 [0, 1] 的分数
2. 加权求和得到综合评分
3. 根据评分映射风险等级
4. 置信度基于证据一致性（多个模块同时触发则置信度高）

性能说明：
- 加权求和、等级映射、置信度计算已用 NumPy 向量化；
- 逐点预提取（分数组装）仍为 Python 循环，是当前主要耗时项
  （W7 性能报告实测：43200 点全量综合评分约 29s）；
- 未启用 Numba JIT（该加速仅为 W7 报告的后续优化建议）。

验收标准：3类故障全部检出，误报<5次/30天
"""

import logging

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "threshold": 0.2,
    "trend": 0.3,
    "volatility": 0.2,
    "correlation": 0.3,
}

# 风险等级阈值（综合评分）
# 设计为：单一模块触发难以达到 yellow，需多模块协同才报警
RISK_LEVEL_THRESHOLDS = {"red": 0.75, "orange": 0.50, "yellow": 0.35}


def _score_threshold(threshold_result: list) -> tuple:
    """阈值检测结果 → [0,1] 分数"""
    if not threshold_result:
        return 0.0, None, []

    level_scores = {"danger": 1.0, "limit": 0.85, "warning": 0.45}
    max_score = 0.0
    max_level = None
    params = []

    for hit in threshold_result:
        level = hit.get("level", "warning")
        score = level_scores.get(level, 0.3)
        exceed_ratio = abs(hit.get("exceed_ratio", 0.0))
        score = min(1.0, score + exceed_ratio * 0.5)
        if score > max_score:
            max_score = score
            max_level = level
        params.append(hit.get("param"))

    return max_score, max_level, list(set(params))


def _score_trend(trend_result: dict) -> tuple:
    """趋势检测结果 → (分数, 等级)"""
    if not trend_result or not trend_result.get("any_detected"):
        return 0.0, None

    level_scores = {"red": 1.0, "orange": 0.65, "yellow": 0.35}
    max_level = trend_result.get("max_level")
    score = level_scores.get(max_level, 0.0)

    # 多个窗口触发加分
    triggered_windows = sum(
        1
        for k, v in trend_result.items()
        if k.startswith("window_") and isinstance(v, dict) and v.get("detected")
    )
    score = min(1.0, score + triggered_windows * 0.05)

    return score, max_level


def _score_volatility(volatility_result: dict) -> tuple:
    """波动率检测结果 → (分数, 等级)"""
    if not volatility_result or not volatility_result.get("detected"):
        return 0.0, None

    level_scores = {"red": 1.0, "orange": 0.65, "yellow": 0.35}
    level = volatility_result.get("level")
    score = level_scores.get(level, 0.0)

    ratio = volatility_result.get("ratio", 1.0)
    score = min(1.0, score + (ratio - 1.0) * 0.08)

    return score, level


def _score_correlation(correlation_result: list) -> tuple:
    """关联检测结果 → [0,1] 分数"""
    if not correlation_result:
        return 0.0, None, [], None

    max_score = 0.0
    best_rule = None
    faults = []

    for rule in correlation_result:
        if rule.get("matched"):
            conf = rule.get("confidence", 0.0)
            if conf > max_score:
                max_score = conf
                best_rule = rule
            faults.append(rule.get("fault_type"))

    level = None
    if max_score >= 0.6:
        level = "red"
    elif max_score >= 0.35:
        level = "orange"
    elif max_score >= 0.15:
        level = "yellow"

    return max_score, level, list(set(faults)), best_rule


def compute_risk_score(
    threshold_result: list,
    trend_result: dict,
    volatility_result: dict,
    correlation_result: list,
    weights: dict = None,
    risk_levels: dict = None,
) -> dict:
    """
    计算综合风险评分

    Args:
        threshold_result: 阈值检测结果列表（detect_threshold 输出）
        trend_result: 趋势检测结果字典 {param: detect_trend 输出}
        volatility_result: 波动率检测结果字典 {param: detect_volatility 输出}
        correlation_result: 关联检测结果列表（detect_correlation 输出）
        weights: 各模块权重，默认 threshold=0.2, trend=0.3, volatility=0.2, correlation=0.3
        risk_levels: 风险等级阈值 {red, orange, yellow}，默认使用 RISK_LEVEL_THRESHOLDS

    Returns:
        {
            "score": float,      # 0-1
            "level": str,        # green/yellow/orange/red
            "confidence": float, # 0-1
            "primary_cause": str,
            "module_scores": {...},
            "details": {...}
        }
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    if risk_levels is None:
        risk_levels = RISK_LEVEL_THRESHOLDS.copy()

    if threshold_result is None:
        threshold_result = []
    if trend_result is None:
        trend_result = {}
    if volatility_result is None:
        volatility_result = {}
    if correlation_result is None:
        correlation_result = []

    # 各模块评分（取多参数中最高者）
    th_score, th_level, th_params = _score_threshold(threshold_result)

    tr_score, tr_level, tr_params = 0.0, None, []
    for param, res in trend_result.items():
        s, l = _score_trend(res)
        if s > tr_score:
            tr_score = s
            tr_level = l
            tr_params = [param]

    vol_score, vol_level, vol_params = 0.0, None, []
    for param, res in volatility_result.items():
        s, l = _score_volatility(res)
        if s > vol_score:
            vol_score = s
            vol_level = l
            vol_params = [param]

    corr_score, corr_level, corr_faults, corr_rule = _score_correlation(
        correlation_result
    )

    # 加权综合
    total_weight = sum(weights.values())
    if total_weight == 0:
        score = 0.0
    else:
        score = (
            weights.get("threshold", 0) * th_score
            + weights.get("trend", 0) * tr_score
            + weights.get("volatility", 0) * vol_score
            + weights.get("correlation", 0) * corr_score
        ) / total_weight

    score = round(min(1.0, max(0.0, score)), 4)

    # 风险等级映射（使用传入的阈值，确保与 config 一致）
    level = "green"
    for lv, th in sorted(risk_levels.items(), key=lambda x: -x[1]):
        if score >= th:
            level = lv
            break

    # 置信度 = 激活模块数 + 评分强度
    active_modules = sum(
        [th_score > 0, tr_score > 0, vol_score > 0, corr_score > 0]
    )
    confidence = min(1.0, 0.2 + active_modules * 0.18 + score * 0.4)
    confidence = round(confidence, 4)

    # 根因分析
    all_params = set(th_params + tr_params + vol_params)
    primary_cause = ""
    if corr_rule:
        primary_cause = (
            f"{corr_rule.get('fault_type', '')}: {corr_rule.get('rule_name', '')}"
        )
    elif all_params:
        primary_cause = f"参数异常: {', '.join(sorted(all_params))}"

    return {
        "score": score,
        "level": level,
        "confidence": confidence,
        "primary_cause": primary_cause,
        "module_scores": {
            "threshold": {
                "score": round(th_score, 4),
                "level": th_level or "green",
            },
            "trend": {"score": round(tr_score, 4), "level": tr_level or "green"},
            "volatility": {
                "score": round(vol_score, 4),
                "level": vol_level or "green",
            },
            "correlation": {
                "score": round(corr_score, 4),
                "level": corr_level or "green",
            },
        },
        "details": {
            "triggered_params": sorted(list(all_params)),
            "correlation_faults": corr_faults,
            "active_modules": active_modules,
        },
    }


def compute_risk_score_batch(
    threshold_results: list,
    trend_results: list,
    volatility_results: list,
    correlation_results: list,
    weights: dict = None,
    risk_levels: dict = None,
    allow_single_module_alert: bool = False,
    single_module_threshold: float = 0.8,
    single_module_score_ratio: float = 0.5,
    persistence_filter: bool = False,
    persistence_n: int = 3,
    persistence_mode: str = "consecutive",
    persistence_window: int = 30,
    persistence_min_count: int = 8,
    persistence_backfill: int = 0,
) -> list:
    """
    批量计算综合风险评分（向量化实现）

    预提取各模块分数为 NumPy 数组，向量化加权求和与等级映射，
    消除逐点 Python 循环。保持与 compute_risk_score 相同的评分逻辑。

    Args:
        threshold_results: list of list，每个时间点的阈值检测结果
        trend_results: list of dict，每个时间点的趋势检测结果
        volatility_results: list of dict，每个时间点的波动率检测结果
        correlation_results: list of list，每个时间点的关联检测结果
        weights: 权重字典
        risk_levels: 风险等级阈值字典
        allow_single_module_alert: 是否允许强单模块信号（>=single_module_threshold）单独达 yellow
        single_module_threshold: 单模块报警的分数门槛（模块 score 阈值）
        single_module_score_ratio: 单模块报警的综合 score 门槛系数（yellow_th * ratio）
        persistence_filter: 是否启用持久性确认过滤
        persistence_n: 持久性确认的最小连续点数（consecutive 模式）
        persistence_mode: 持久性过滤模式
            - "consecutive": 严格连续 N 个 yellow+ 才确认（原行为，最严格，FP 最低）
            - "sliding": 过去 W 窗口内至少 K 个 yellow+ 才确认（容许间歇，Recall 更高）
        persistence_window: sliding 模式的窗口大小（点数，1点=1分钟）
        persistence_min_count: sliding 模式窗口内最小 yellow+ 计数
        persistence_backfill: 确认后向回填长度（点数）。一旦某点被确认 yellow+，
            其前 backfill 个点内若有"弱信号"（任一模块 score>0）则提升为 yellow。
            物理依据：故障确认后，前序弱信号属同一故障发育期。0=禁用。

    Returns:
        list of dict，每个时间点的风险评分
    """
    import numpy as np

    n = len(threshold_results)
    if n == 0:
        return []

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    if risk_levels is None:
        risk_levels = RISK_LEVEL_THRESHOLDS.copy()

    # --- 预提取各模块分数为数组（优化版）---
    th_scores = np.zeros(n)
    th_levels = np.array(["green"] * n, dtype=object)
    th_params_list = [set() for _ in range(n)]

    tr_scores = np.zeros(n)
    tr_levels = np.array(["green"] * n, dtype=object)
    tr_params_list = [set() for _ in range(n)]

    vol_scores = np.zeros(n)
    vol_levels = np.array(["green"] * n, dtype=object)
    vol_params_list = [set() for _ in range(n)]

    corr_scores = np.zeros(n)
    corr_levels = np.array(["green"] * n, dtype=object)
    corr_faults_list = [[] for _ in range(n)]
    corr_rule_list = [None] * n

    level_scores_th = {"danger": 1.0, "limit": 0.85, "warning": 0.45}
    level_scores_tr = {"red": 1.0, "orange": 0.65, "yellow": 0.35}
    level_scores_vol = {"red": 1.0, "orange": 0.65, "yellow": 0.35}

    # 预提取各模块分数为数组
    for i in range(n):
        # 阈值
        th_res = threshold_results[i] if i < len(threshold_results) else []
        if th_res:
            max_s = 0.0
            max_lv = None
            for hit in th_res:
                lv = hit.get("level", "warning")
                s = level_scores_th.get(lv, 0.3)
                ex = abs(hit.get("exceed_ratio", 0.0))
                s = min(1.0, s + ex * 0.5)
                if s > max_s:
                    max_s = s
                    max_lv = lv
                th_params_list[i].add(hit.get("param"))
            th_scores[i] = max_s
            th_levels[i] = max_lv or "green"

        # 趋势
        tr_res = trend_results[i] if i < len(trend_results) else {}
        if tr_res:
            max_s = 0.0
            max_lv = None
            for param, res in tr_res.items():
                if not res.get("any_detected"):
                    continue
                lv = res.get("max_level")
                s = level_scores_tr.get(lv, 0.0)
                tw = sum(1 for k, v in res.items()
                         if k.startswith("window_") and isinstance(v, dict) and v.get("detected"))
                s = min(1.0, s + tw * 0.05)
                if s > max_s:
                    max_s = s
                    max_lv = lv
                    tr_params_list[i] = {param}
            tr_scores[i] = max_s
            tr_levels[i] = max_lv or "green"

        # 波动率
        vol_res = volatility_results[i] if i < len(volatility_results) else {}
        if vol_res:
            max_s = 0.0
            max_lv = None
            for param, res in vol_res.items():
                if not res.get("detected"):
                    continue
                lv = res.get("level")
                s = level_scores_vol.get(lv, 0.0)
                ratio = res.get("ratio", 1.0)
                s = min(1.0, s + (ratio - 1.0) * 0.08)
                if s > max_s:
                    max_s = s
                    max_lv = lv
                    vol_params_list[i] = {param}
            vol_scores[i] = max_s
            vol_levels[i] = max_lv or "green"

        # 关联
        corr_res = correlation_results[i] if i < len(correlation_results) else []
        if corr_res:
            max_s = 0.0
            best_rule = None
            faults = []
            for rule in corr_res:
                if rule.get("matched"):
                    conf = rule.get("confidence", 0.0)
                    if conf > max_s:
                        max_s = conf
                        best_rule = rule
                    faults.append(rule.get("fault_type"))
            corr_scores[i] = max_s
            corr_faults_list[i] = list(set(faults))
            corr_rule_list[i] = best_rule
            if max_s >= 0.6:
                corr_levels[i] = "red"
            elif max_s >= 0.35:
                corr_levels[i] = "orange"
            elif max_s >= 0.15:
                corr_levels[i] = "yellow"

    # --- 向量化加权求和 ---
    total_weight = sum(weights.values())
    if total_weight == 0:
        scores_arr = np.zeros(n)
    else:
        scores_arr = (
            weights.get("threshold", 0) * th_scores
            + weights.get("trend", 0) * tr_scores
            + weights.get("volatility", 0) * vol_scores
            + weights.get("correlation", 0) * corr_scores
        ) / total_weight
    scores_arr = np.round(np.clip(scores_arr, 0.0, 1.0), 4)

    # --- 向量化等级映射 ---
    sorted_levels = sorted(risk_levels.items(), key=lambda x: -x[1])
    levels_arr = np.array(["green"] * n, dtype=object)
    for lv, th in sorted_levels:
        mask = (scores_arr >= th) & (levels_arr == "green")
        levels_arr[mask] = lv

    # --- allow_single_module_alert：强单模块信号单独达 yellow ---
    # 场景：trend red（score=1.0）单独触发时，加权 score 可能未达 yellow 阈值，
    # 但该信号本身足够强，应允许报警。仅对 trend/volatility/threshold 模块启用，
    # correlation 单独触发已可通过自身权重达 yellow。
    if allow_single_module_alert:
        yellow_th = risk_levels.get("yellow", 0.25)
        green_mask = levels_arr == "green"
        # 任一强单模块触发（score >= single_module_threshold）且当前 green → 提升至 yellow
        strong_single = (
            (th_scores >= single_module_threshold)
            | (tr_scores >= single_module_threshold)
            | (vol_scores >= single_module_threshold)
        )
        promote_mask = green_mask & strong_single & (scores_arr >= yellow_th * single_module_score_ratio)
        levels_arr[promote_mask] = "yellow"

    # --- 持久性确认过滤 ---
    # 两种模式（向量化实现，避免逐点 Python 循环）：
    #   consecutive: 严格连续 N 个 yellow+ 才确认（原行为，最严格，FP 最低）
    #   sliding:     过去 W 窗口内至少 K 个 yellow+ 才确认（容许间歇，Recall 更高）
    # 物理依据：故障段早期（0-14min）信号发育不全，raw 检出常呈间歇性；
    #          严格连续过滤会把这类间歇信号整段滤掉，sliding 模式可保留。
    if persistence_filter:
        import pandas as pd
        yellow_mask_arr = levels_arr != "green"
        if yellow_mask_arr.any():
            if persistence_mode == "sliding" and persistence_window > 1:
                # sliding 模式：rolling count，窗口内 yellow+ 计数 >= K 才保留
                yellow_series = pd.Series(yellow_mask_arr)
                win_count = yellow_series.rolling(
                    persistence_window, min_periods=1
                ).sum()
                keep_mask = yellow_mask_arr & (
                    win_count.to_numpy() >= persistence_min_count
                )
                levels_arr = np.where(keep_mask, levels_arr, "green")
            elif persistence_n > 1:
                # consecutive 模式：groupby cumsum trick，段长度 >= N 才保留
                yellow_series = pd.Series(yellow_mask_arr)
                group = (yellow_series != yellow_series.shift()).cumsum()
                run_len = yellow_series.groupby(group).transform("size")
                keep_mask = yellow_mask_arr & (run_len.to_numpy() >= persistence_n)
                levels_arr = np.where(keep_mask, levels_arr, "green")

        # --- 确认后向回填（fault development backfill）---
        # 一旦某点被持久性过滤确认 yellow+，其前 backfill 个点内若有"弱信号"
        # （任一模块 score > 0）则提升为 yellow。
        # 物理依据：故障确认后，前序弱信号属同一故障发育期；
        #          仅回填弱信号点，避免把纯噪声段误判为故障早期。
        # 优化：增加 correlation 模块约束，仅回填包含关联检测信号（score >= 0.15）的弱信号点，
        #       降低炉排卡滞等故障类型的误报（FP）。
        if persistence_backfill > 0:
            confirmed_mask = levels_arr != "green"
            if confirmed_mask.any():
                weak_signal_mask = (
                    (th_scores > 0)
                    | (tr_scores > 0)
                    | (vol_scores > 0)
                    | (corr_scores > 0)
                )
                # 增加 correlation 约束：弱信号点必须包含关联检测信号（score >= 0.15），
                # 避免将纯阈值/趋势/波动率的正常波动误判为故障早期
                corr_required_mask = corr_scores >= 0.15
                weak_signal_mask = weak_signal_mask & corr_required_mask

                confirmed_arr = confirmed_mask.astype(int)
                confirmed_series = pd.Series(confirmed_arr)
                forward_count = (
                    confirmed_series[::-1]
                    .rolling(persistence_backfill + 1, min_periods=1)
                    .sum()[::-1]
                )
                has_confirmed_ahead = forward_count.to_numpy() > 0
                promote_mask = (
                    weak_signal_mask & ~confirmed_mask & has_confirmed_ahead
                )
                levels_arr[promote_mask] = "yellow"

    # --- 向量化置信度 ---
    active_modules = (
        (th_scores > 0).astype(int)
        + (tr_scores > 0).astype(int)
        + (vol_scores > 0).astype(int)
        + (corr_scores > 0).astype(int)
    )
    confidence_arr = np.round(
        np.minimum(1.0, 0.2 + active_modules * 0.18 + scores_arr * 0.4), 4
    )

    # --- 构建结果 ---
    results = []
    for i in range(n):
        all_params = th_params_list[i] | tr_params_list[i] | vol_params_list[i]
        corr_rule = corr_rule_list[i]
        if corr_rule:
            primary_cause = (
                f"{corr_rule.get('fault_type', '')}: {corr_rule.get('rule_name', '')}"
            )
        elif all_params:
            primary_cause = f"参数异常: {', '.join(sorted(all_params))}"
        else:
            primary_cause = ""

        results.append({
            "score": float(scores_arr[i]),
            "level": str(levels_arr[i]),
            "confidence": float(confidence_arr[i]),
            "primary_cause": primary_cause,
            "module_scores": {
                "threshold": {"score": round(float(th_scores[i]), 4), "level": str(th_levels[i])},
                "trend": {"score": round(float(tr_scores[i]), 4), "level": str(tr_levels[i])},
                "volatility": {"score": round(float(vol_scores[i]), 4), "level": str(vol_levels[i])},
                "correlation": {"score": round(float(corr_scores[i]), 4), "level": str(corr_levels[i])},
            },
            "details": {
                "triggered_params": sorted(list(all_params)),
                "correlation_faults": corr_faults_list[i],
                "active_modules": int(active_modules[i]),
            },
        })

    return results
