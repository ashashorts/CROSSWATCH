import unittest

import pandas as pd

from crosswatch import FEATURES
from zscore_baseline import ROLLING_WINDOW, zscore_baselines


class TestZscoreBaseline(unittest.TestCase):
    def test_scores_use_training_or_prior_rows(self):
        frame = pd.DataFrame({
            FEATURES[0]: [float(i % 2) for i in range(75)],
            FEATURES[1]: 0.0,
            FEATURES[2]: 0.0,
        })
        frame.loc[60, FEATURES[0]] = 3.0
        train = frame.iloc[:60]
        scores = zscore_baselines(train, frame)
        self.assertTrue(scores['rolling_zscore'].iloc[:ROLLING_WINDOW].isna().all())
        expected = (3.0 - train[FEATURES[0]].mean()) / train[FEATURES[0]].std(ddof=1)
        self.assertAlmostEqual(scores.loc[60, 'fixed_zscore'], expected)
        self.assertAlmostEqual(scores.loc[60, 'rolling_zscore'], expected)
        self.assertTrue(scores.loc[60, 'rolling_zscore_alert'])
        self.assertFalse(scores.loc[61, 'rolling_zscore_alert'])
        modified = frame.copy()
        modified.loc[62:, FEATURES[0]] = 1000
        pd.testing.assert_frame_equal(
            scores.loc[:61], zscore_baselines(train, modified).loc[:61]
        )
