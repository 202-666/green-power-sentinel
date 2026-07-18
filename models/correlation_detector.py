"""
绿电哨兵 — 关联检测模块（Agent 2 方法4）

基于规则引擎的多参数关联异常检测。
解析 rules.yaml 中的关联规则，评估当前设备状态是否匹配故障模式。

支持的条件类型：
- threshold: 与固定阈值比较（current_values）
- trend: 与趋势斜率比较（需预计算 trend_states）
- volatility: 与波动率倍数比较（需预计算 volatility_states）

规则逻辑：
- AND: 所有条件同时满足
- OR: 任一条件满足

输出：匹配的规则列表，含置信度、条件详情
"""

import logging

logger = logging.getLogger(__name__)


def _eval_condition(
    condition: dict,
    current_values: dict,
    trend_states: dict = None,
    volatility_states: dict = None,
) -> tuple:
    """
    评估单个条件是否满足

    Args:
        condition: 规则条件字典
        current_values: {param: value} 当前原始值
        trend_states: {param: {"window_30": {"slope": x}, ...}} 趋势状态
        volatility_states: {param: {"ratio": x, "detected": True}} 波动率状态

    Returns:
        (satisfied: bool, confidence: float)
    """
    param = condition.get("param")
    direction = condition.get("direction", "up")
    compare_to = condition.get("compare_to", "threshold")

    if param not in current_values:
        return False, 0.0

    raw_value = current_values.get(param)
    if raw_value is None:
        return False, 0.0

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return False, 0.0

    satisfied = False
    confidence = 0.0

    if compare_to == "threshold":
        threshold = condition.get("threshold")
        if threshold is None:
            return False, 0.0
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            return False, 0.0

        if direction == "up":
            satisfied = value > threshold
            if satisfied:
                # 置信度：超出量 / (阈值 × 20%)，10%超出≈0.5置信，20%超出≈1.0
                confidence = min(
                    1.0, (value - threshold) / max(abs(threshold) * 0.2, 1e-6)
                )
        elif direction == "down":
            satisfied = value < threshold
            if satisfied:
                confidence = min(
                    1.0, (threshold - value) / max(abs(threshold) * 0.2, 1e-6)
                )

    elif compare_to == "trend":
        if trend_states is None or param not in trend_states:
            return False, 0.0

        t_state = trend_states[param]
        trend_window = condition.get("trend_window", 30)
        window_key = f"window_{trend_window}"

        if isinstance(t_state, dict) and window_key in t_state:
            slope = t_state[window_key].get("slope", 0.0)
        elif isinstance(t_state, dict) and "slope" in t_state:
            slope = t_state.get("slope", 0.0)
        else:
            return False, 0.0

        slope_threshold = condition.get("slope_threshold", 0.0)

        if direction == "up":
            satisfied = slope > slope_threshold
            if satisfied and slope_threshold > 0:
                confidence = min(1.0, slope / slope_threshold)
        elif direction == "down":
            satisfied = slope < slope_threshold
            if satisfied and slope_threshold < 0:
                confidence = min(1.0, abs(slope) / abs(slope_threshold))

    elif compare_to == "volatility":
        if volatility_states is None or param not in volatility_states:
            return False, 0.0

        v_state = volatility_states[param]
        ratio = v_state.get("ratio", 1.0) if isinstance(v_state, dict) else 1.0
        vol_threshold = condition.get("volatility_ratio", 2.0)

        if direction == "volatile":
            satisfied = ratio > vol_threshold
            if satisfied and vol_threshold > 1.0:
                confidence = min(
                    1.0, (ratio - 1.0) / max(vol_threshold - 1.0, 1e-6)
                )

    return satisfied, confidence


def detect_correlation(
    current_values: dict,
    rules: list,
    trend_states: dict = None,
    volatility_states: dict = None,
) -> list:
    """
    关联检测（单时刻）

    Args:
        current_values: {param_name: value} 当前各参数原始值
        rules: 规则列表（如 rules.yaml 的 rules）
        trend_states: 各参数当前趋势状态（detect_trend 输出格式）
        volatility_states: 各参数当前波动率状态（detect_volatility 输出格式）

    Returns:
        [{
            "rule_id": str,
            "rule_name": str,
            "fault_type": str,
            "matched": bool,
            "confidence": float,
            "matched_conditions": int,
            "total_conditions": int,
            "condition_details": [...],
            "weight": float,
        }]
    """
    results = []

    for rule in rules:
        rule_id = rule.get("rule_id", "")
        rule_name = rule.get("rule_name", "")
        fault_type = rule.get("fault_type", "")
        conditions = rule.get("conditions", [])
        logic = rule.get("logic", "AND")
        weight = rule.get("weight", 0.5)

        condition_details = []
        all_satisfied = True
        any_satisfied = False
        total_confidence = 0.0

        for cond in conditions:
            satisfied, conf = _eval_condition(
                cond, current_values, trend_states, volatility_states
            )
            condition_details.append(
                {
                    "param": cond.get("param"),
                    "compare_to": cond.get("compare_to"),
                    "satisfied": satisfied,
                    "confidence": round(conf, 4),
                }
            )
            if satisfied:
                any_satisfied = True
                total_confidence += conf
            else:
                all_satisfied = False

        matched_conditions = sum(1 for d in condition_details if d["satisfied"])
        total_conditions = len(conditions)

        if logic == "AND":
            matched = all_satisfied and total_conditions > 0
        elif logic == "OR":
            matched = any_satisfied and total_conditions > 0
        elif logic == "MAJORITY":
            # 半数及以上条件满足即触发（matched*2 >= total，含等于）
            # total=3 → matched>=2；total=4 → matched>=2；total=5 → matched>=3
            matched = (matched_conditions * 2 >= total_conditions) and total_conditions > 0
        else:
            matched = all_satisfied

        if matched and matched_conditions > 0:
            avg_confidence = total_confidence / matched_conditions
            # 综合置信度 = 规则权重 × 平均条件置信度
            # 注：matched/total 比例已由 logic 门（AND/OR/MAJORITY）体现，不再重复惩罚
            confidence = weight * avg_confidence
            confidence = min(1.0, max(0.0, confidence))
        else:
            confidence = 0.0

        results.append(
            {
                "rule_id": rule_id,
                "rule_name": rule_name,
                "fault_type": fault_type,
                "matched": matched,
                "confidence": round(confidence, 4),
                "matched_conditions": matched_conditions,
                "total_conditions": total_conditions,
                "condition_details": condition_details,
                "weight": weight,
            }
        )

    return results


def detect_correlation_batch(
    df,
    rules: list,
    trend_results: dict = None,
    volatility_results: dict = None,
) -> list:
    """
    关联检测（批量，向量化实现）

    使用 NumPy 布尔掩码批量计算所有时间点的规则匹配，消除逐行 iterrows 开销。

    Args:
        df: DataFrame，含参数列
        rules: 规则列表
        trend_results: {param: trend_batch_df} 批量趋势结果
        volatility_results: {param: volatility_batch_df} 批量波动率结果

    Returns:
        list of {"timestamp": ..., "row_idx": ..., "matched_rules": [...], "all_rules": [...]}
    """
    import numpy as np
    import pandas as pd

    if trend_results is None:
        trend_results = {}
    if volatility_results is None:
        volatility_results = {}

    n = len(df)
    if n == 0 or not rules:
        return [{"timestamp": None, "row_idx": i, "matched_rules": [], "all_rules": []}
                for i in range(n)]

    # 预提取参数值数组（避免反复索引 DataFrame）
    df_index = df.index
    ts_col = df["timestamp"].to_numpy() if "timestamp" in df.columns else None
    param_arrays = {}
    for col in df.columns:
        if col in {"timestamp", "device_id", "device_name", "device_type",
                   "data_quality_flag", "qc_note"}:
            continue
        param_arrays[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    # 预提取趋势斜率数组 {param: {window: slope_array}}
    slope_arrays = {}
    for param, tdf in trend_results.items():
        slope_arrays[param] = {}
        for w in [10, 30, 60]:
            sk = f"slope_{w}"
            if sk in tdf.columns:
                slope_arrays[param][w] = pd.to_numeric(tdf[sk], errors="coerce").to_numpy(dtype=float)

    # 预提取波动率 ratio 数组 {param: ratio_array}
    vol_ratio_arrays = {}
    for param, vdf in volatility_results.items():
        if "ratio" in vdf.columns:
            vol_ratio_arrays[param] = pd.to_numeric(vdf["ratio"], errors="coerce").to_numpy(dtype=float)

    # 对每条规则向量化计算
    # rule_matched[ri] = bool array, rule_conf[ri] = float array
    rule_matched = []
    rule_conf = []
    rule_meta = []

    for rule in rules:
        rule_id = rule.get("rule_id", "")
        rule_name = rule.get("rule_name", "")
        fault_type = rule.get("fault_type", "")
        conditions = rule.get("conditions", [])
        logic = rule.get("logic", "AND")
        weight = rule.get("weight", 0.5)
        total_conditions = len(conditions)

        rule_meta.append({
            "rule_id": rule_id, "rule_name": rule_name, "fault_type": fault_type,
            "weight": weight, "logic": logic,
        })

        if total_conditions == 0:
            rule_matched.append(np.zeros(n, dtype=bool))
            rule_conf.append(np.zeros(n))
            continue

        # 每个条件的 satisfied mask 和 confidence array
        cond_satisfied = []
        cond_conf = []

        for cond in conditions:
            param = cond.get("param")
            direction = cond.get("direction", "up")
            compare_to = cond.get("compare_to", "threshold")

            sat = np.zeros(n, dtype=bool)
            conf = np.zeros(n)

            if param is None:
                cond_satisfied.append(sat)
                cond_conf.append(conf)
                continue

            if compare_to == "threshold":
                arr = param_arrays.get(param)
                threshold = cond.get("threshold")
                if arr is not None and threshold is not None:
                    try:
                        threshold = float(threshold)
                        denom = max(abs(threshold) * 0.2, 1e-6)
                        if direction == "up":
                            sat = arr > threshold
                            conf = np.where(sat, np.minimum(1.0, (arr - threshold) / denom), 0.0)
                        elif direction == "down":
                            sat = arr < threshold
                            conf = np.where(sat, np.minimum(1.0, (threshold - arr) / denom), 0.0)
                        # NaN 处理
                        sat = sat & ~np.isnan(arr)
                        conf = np.where(np.isnan(arr), 0.0, conf)
                    except (TypeError, ValueError):
                        pass

            elif compare_to == "trend":
                slopes_map = slope_arrays.get(param, {})
                trend_window = cond.get("trend_window", 30)
                slope_arr = slopes_map.get(trend_window)
                slope_threshold = cond.get("slope_threshold", 0.0)
                if slope_arr is not None:
                    try:
                        slope_threshold = float(slope_threshold)
                        if direction == "up":
                            sat = slope_arr > slope_threshold
                            if slope_threshold > 0:
                                conf = np.where(sat, np.minimum(1.0, slope_arr / slope_threshold), 0.0)
                        elif direction == "down":
                            sat = slope_arr < slope_threshold
                            if slope_threshold < 0:
                                conf = np.where(sat, np.minimum(1.0, np.abs(slope_arr) / np.abs(slope_threshold)), 0.0)
                        sat = sat & ~np.isnan(slope_arr)
                        conf = np.where(np.isnan(slope_arr), 0.0, conf)
                    except (TypeError, ValueError):
                        pass

            elif compare_to == "volatility":
                ratio_arr = vol_ratio_arrays.get(param)
                vol_threshold = cond.get("volatility_ratio", 2.0)
                if ratio_arr is not None:
                    try:
                        vol_threshold = float(vol_threshold)
                        if direction == "volatile":
                            sat = ratio_arr > vol_threshold
                            if vol_threshold > 1.0:
                                conf = np.where(sat, np.minimum(1.0, (ratio_arr - 1.0) / max(vol_threshold - 1.0, 1e-6)), 0.0)
                        sat = sat & ~np.isnan(ratio_arr)
                        conf = np.where(np.isnan(ratio_arr), 0.0, conf)
                    except (TypeError, ValueError):
                        pass

            cond_satisfied.append(sat)
            cond_conf.append(conf)

        # 合并条件（向量化）
        cond_sat_stack = np.stack(cond_satisfied, axis=0)  # (n_cond, n)
        cond_conf_stack = np.stack(cond_conf, axis=0)       # (n_cond, n)
        matched_count = cond_sat_stack.sum(axis=0)          # (n,)
        total_confidence = np.where(cond_sat_stack, cond_conf_stack, 0.0).sum(axis=0)
        avg_confidence = np.divide(total_confidence, matched_count,
                                   out=np.zeros(n), where=matched_count > 0)

        if logic == "AND":
            matched = (matched_count == total_conditions) & (total_conditions > 0)
        elif logic == "OR":
            matched = (matched_count > 0) & (total_conditions > 0)
        elif logic == "MAJORITY":
            # 半数及以上（含等于）：matched*2 >= total
            matched = (matched_count * 2 >= total_conditions) & (total_conditions > 0)
        else:
            matched = (matched_count == total_conditions) & (total_conditions > 0)

        confidence = np.where(matched, np.minimum(1.0, weight * avg_confidence), 0.0)

        rule_matched.append(matched)
        rule_conf.append(confidence)

    # 构建结果列表（轻量，仅索引预计算数组）
    results = []
    for i in range(n):
        ts = ts_col[i] if ts_col is not None else df_index[i]
        all_rules = []
        matched_rules = []
        for ri, meta in enumerate(rule_meta):
            is_matched = bool(rule_matched[ri][i])
            conf = float(rule_conf[ri][i])
            entry = {
                "rule_id": meta["rule_id"],
                "rule_name": meta["rule_name"],
                "fault_type": meta["fault_type"],
                "matched": is_matched,
                "confidence": round(conf, 4),
                "weight": meta["weight"],
            }
            all_rules.append(entry)
            if is_matched:
                matched_rules.append(entry)
        results.append({
            "timestamp": ts,
            "row_idx": i,
            "matched_rules": matched_rules,
            "all_rules": all_rules,
        })

    return results
