import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cleaning import generate_sample_data
from eda_report import build_eda_pdf
from feature_engineering import engineer_features


class EdaReportTest(unittest.TestCase):
    @patch("eda_report.load_regression_model")
    def test_builds_nonempty_pdf(self, load_regression_model):
        data = engineer_features(generate_sample_data(n=500, seed=7))
        load_regression_model.return_value = {
            "mae": 12.5,
            "rmse": 18.75,
            "r2": 0.72,
            "n_train": 80,
            "n_test": 20,
            "y_test": np.array([100.0, 200.0, 300.0]),
            "y_pred": np.array([110.0, 190.0, 305.0]),
        }

        report = build_eda_pdf(data)

        load_regression_model.assert_called_once_with(data)
        self.assertTrue(report.startswith(b"%PDF-"))
        self.assertGreater(len(report), 12_000)
        self.assertIn(b"Report version 2.0", report)
        self.assertIn(b"Event Distribution", report)
        self.assertIn(b"Category Revenue Share", report)
        self.assertIn(b"Predictive Analysis", report)
        self.assertIn(b"Actual vs. Predicted Prices", report)
        self.assertIn(b"%%EOF", report[-1024:])

    def test_rejects_empty_dataset(self):
        data = engineer_features(generate_sample_data(n=10, seed=7)).iloc[0:0]

        with self.assertRaisesRegex(ValueError, "empty dataset"):
            build_eda_pdf(data)


if __name__ == "__main__":
    unittest.main()
