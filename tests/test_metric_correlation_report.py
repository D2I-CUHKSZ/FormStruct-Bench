from __future__ import annotations

import unittest

import numpy as np

from formtsr_exp.metric_correlation_report import METRICS, compute_correlations


class MetricCorrelationReportTest(unittest.TestCase):
    def test_perfect_monotonic_correlations(self) -> None:
        base = np.asarray([0.0, 1.0, 2.0, 3.0])
        values = {
            metric: base * (index + 1)
            for index, metric in enumerate(METRICS)
        }

        pearson, spearman, pairs = compute_correlations(values)

        self.assertEqual(pearson[METRICS[0]][METRICS[-1]], 1.0)
        self.assertEqual(spearman[METRICS[0]][METRICS[-1]], 1.0)
        self.assertEqual(len(pairs), len(METRICS) * (len(METRICS) - 1) // 2)


if __name__ == "__main__":
    unittest.main()
