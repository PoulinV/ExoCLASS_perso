"""Tests for direct integration of the CLASS exact-y history."""

from __future__ import division

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "scan_lowz_distortion_fraction",
    str(ROOT / "scripts" / "scan_lowz_distortion_fraction.py"),
)
SCAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCAN)


class ExactYHistoryIntegrationTest(unittest.TestCase):

    def test_total_uses_class_trapezoidal_weights(self):
        redshift = np.asarray([1.0, 2.0, 4.0])
        weights = np.asarray([0.5, 1.5, 1.0])
        branching = np.asarray([1.0, 0.5, 0.25])
        baseline = (
            redshift, np.asarray([4.0, 4.0, 4.0]), branching, weights
        )
        model = (
            redshift, np.asarray([8.0, 12.0, 20.0]), branching, weights
        )
        expected = 0.25 * np.dot(
            (model[1] - baseline[1]) * branching, weights
        )
        self.assertEqual(
            SCAN.integrate_exact_y_difference(model, baseline), expected
        )

    def test_partial_integral_inserts_cutoff_node(self):
        redshift = np.asarray([1.0, 2.0, 4.0])
        weights = np.asarray([0.5, 1.5, 1.0])
        branching = np.ones(3)
        baseline = (redshift, np.zeros(3), branching, weights)
        model = (redshift, 4.0 * redshift, branching, weights)
        # (1/4) integral_1^3 4z dz = 4.
        self.assertAlmostEqual(
            SCAN.integrate_exact_y_difference(
                model, baseline, scale=2.0, z_max=3.0
            ),
            8.0,
        )

    def test_mismatched_histories_are_rejected(self):
        weights = np.asarray([0.5, 0.5])
        model = (
            np.asarray([1.0, 2.0]), np.ones(2), np.ones(2), weights
        )
        baseline = (
            np.asarray([1.0, 2.1]), np.zeros(2), np.ones(2), weights
        )
        with self.assertRaises(RuntimeError):
            SCAN.integrate_exact_y_difference(model, baseline)


if __name__ == "__main__":
    unittest.main()
