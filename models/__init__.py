"""
绿电哨兵 — 检测模型模块导出

包含：
- threshold_detector: 阈值检测
- trend_detector: 趋势检测（10/30/60min窗口线性回归）
- volatility_detector: 波动率检测
- correlation_detector: 多参数关联检测（规则引擎）
- ensemble_scorer: 综合评分与风险分级
"""

from .threshold_detector import (
    detect_threshold,
    detect_threshold_batch,
    load_thresholds_from_yaml,
    _normalize_thresholds,
)
from .trend_detector import (
    detect_trend,
    detect_trend_batch,
    detect_trend_multi_params,
    _compute_slope,
    DEFAULT_SLOPE_THRESHOLDS,
    DEFAULT_WINDOWS,
    DEFAULT_SMOOTHING_WINDOW,
)
from .volatility_detector import detect_volatility
from .correlation_detector import detect_correlation
from .ensemble_scorer import compute_risk_score

__all__ = [
    "detect_threshold",
    "detect_threshold_batch",
    "load_thresholds_from_yaml",
    "_normalize_thresholds",
    "detect_trend",
    "detect_trend_batch",
    "detect_trend_multi_params",
    "_compute_slope",
    "DEFAULT_SLOPE_THRESHOLDS",
    "DEFAULT_WINDOWS",
    "DEFAULT_SMOOTHING_WINDOW",
    "detect_volatility",
    "detect_correlation",
    "compute_risk_score",
]
