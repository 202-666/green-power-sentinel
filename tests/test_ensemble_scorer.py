"""
绿电哨兵 — 综合评分单点/批量一致性单元测试

覆盖 FEI-7 L3：
- 单点版 compute_risk_score 与批量版 compute_risk_score_batch
  的 triggered_params 行为一致（以结果字典的 key 为参数名）。
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ensemble_scorer import compute_risk_score, compute_risk_score_batch


def _trend_state(param: str) -> dict:
    """趋势结果状态（不携带冗余 param 字段，key 即参数名）"""
    return {
        param: {
            "any_detected": True,
            "max_level": "red",
            "window_10": {"detected": False},
            "window_30": {"detected": False},
            "window_60": {"detected": True},
        }
    }


def _vol_state(param: str) -> dict:
    """波动率结果状态（不携带冗余 param 字段，key 即参数名）"""
    return {
        param: {
            "detected": True,
            "level": "orange",
            "ratio": 2.5,
        }
    }


class TestTriggeredParamsConsistency(unittest.TestCase):
    """L3: 单点/批量版 triggered_params 行为一致"""

    def test_single_uses_dict_key_not_unknown(self):
        score = compute_risk_score(
            [],
            _trend_state("furnace_temperature"),
            _vol_state("furnace_pressure"),
            [],
        )
        self.assertEqual(
            score["details"]["triggered_params"],
            ["furnace_pressure", "furnace_temperature"],
        )
        self.assertNotIn("unknown", score["details"]["triggered_params"])

    def test_batch_matches_single(self):
        trend = _trend_state("furnace_temperature")
        vol = _vol_state("furnace_pressure")
        single = compute_risk_score([], trend, vol, [])
        batch = compute_risk_score_batch([[]], [trend], [vol], [[]])

        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0]["details"]["triggered_params"],
                         single["details"]["triggered_params"])
        self.assertEqual(batch[0]["score"], single["score"])
        self.assertEqual(batch[0]["level"], single["level"])
        self.assertEqual(batch[0]["module_scores"],
                         single["module_scores"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
