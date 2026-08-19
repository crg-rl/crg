from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "mllm-crl/evaluation/crl_metric/metrics.py"
SPEC = importlib.util.spec_from_file_location("crg_metrics", METRICS_PATH)
assert SPEC is not None and SPEC.loader is not None
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


class CrlMetricTests(unittest.TestCase):
    def test_core_metrics(self) -> None:
        matrix = [
            [0.4, 0.2, 0.1],
            [0.35, 0.6, 0.15],
            [0.3, 0.55, 0.8],
        ]
        summary = METRICS.summarize_continual_metrics(matrix)
        self.assertAlmostEqual(summary["R_ii"], 0.6)
        self.assertAlmostEqual(summary["FinalAvg"], 0.55)
        self.assertAlmostEqual(summary["BWT"], -0.075)

    def test_zero_shot_forward_transfer_skips_first_task(self) -> None:
        value = METRICS.zero_shot_forward_transfer(
            [0.1, 0.2, 0.3],
            [0.1, 0.25, 0.35],
        )
        self.assertAlmostEqual(value, 0.05)


if __name__ == "__main__":
    unittest.main()
