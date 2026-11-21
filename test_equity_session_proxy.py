import tempfile
import unittest

import pandas as pd

from crosswatch import generate_synthetic_levels
from crosswatch_phase3 import future_five_day_decline_proxy, run_phase3_part1


class TestEquitySessionProxy(unittest.TestCase):
    def test_historical_run_uses_unjoined_equity_sessions_within_each_split(self):
        full = generate_synthetic_levels(n=300, seed=2026)
        closes = full['SP500'].copy()
        closes.iloc[15:22] = [100, 100, 100, 100, 100, 100, 97]
        closes.iloc[35:42] = [100, 100, 100, 100, 100, 97, 100]
        full['SP500'] = closes
        aligned = full.drop(full.index[[16, 40]])

        with tempfile.TemporaryDirectory() as output:
            result = run_phase3_part1(
                aligned, output, source_type='TEST', equity_closes=closes
            )

        train = result['scored']['train']
        legacy = future_five_day_decline_proxy(train)
        self.assertTrue(legacy.loc[full.index[15]])
        self.assertFalse(train.loc[full.index[15], 'forward_proxy'])
        self.assertFalse(legacy.loc[full.index[35]])
        self.assertTrue(train.loc[full.index[35], 'forward_proxy'])
        for split in result['scored'].values():
            self.assertTrue(split['forward_proxy'].iloc[-5:].isna().all())
