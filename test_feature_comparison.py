import tempfile
import unittest
from pathlib import Path

import pandas as pd

from crosswatch import FEATURES, PHASE1_FEATURES, engineer_features, generate_synthetic_levels
from crosswatch_phase3 import run_phase3_part1


class TestFeatureComparison(unittest.TestCase):
    def test_five_inputs_share_dates_and_outcomes_with_three_inputs(self):
        levels = generate_synthetic_levels(n=300, seed=2026)
        with tempfile.TemporaryDirectory() as temp:
            runs = [
                run_phase3_part1(
                    levels, Path(temp) / str(len(features)),
                    source_type='TEST', model_features=features,
                    comparison_features=PHASE1_FEATURES,
                )
                for features in (FEATURES, PHASE1_FEATURES)
            ]
        three, five = runs
        self.assertEqual(five['metadata']['features'], list(PHASE1_FEATURES))
        self.assertEqual(three['metadata']['features'], list(FEATURES))
        for split, frame in three['scored'].items():
            pd.testing.assert_series_equal(
                frame['forward_proxy'], five['scored'][split]['forward_proxy']
            )
            self.assertFalse(frame['anomaly_severity'].equals(five['scored'][split]['anomaly_severity']))
        self.assertEqual(len(five['scored']['train']), int(0.70 * 299) - 4)

    def test_trailing_volatility_uses_no_future_prices(self):
        levels = generate_synthetic_levels(n=60, seed=2026)
        original = engineer_features(levels)
        modified = levels.copy()
        modified.loc[levels.index[21]:, 'SP500'] *= 2
        revised = engineer_features(modified)
        column = 'trailing_5d_return_vol_pct'
        self.assertAlmostEqual(
            original.loc[levels.index[20], column],
            original.loc[levels.index[16]:levels.index[20], 'sp_return_pct'].std(ddof=1),
        )
        pd.testing.assert_series_equal(
            original.loc[:levels.index[20], column],
            revised.loc[:levels.index[20], column],
        )
