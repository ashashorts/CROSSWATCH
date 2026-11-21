import unittest

import pandas as pd

from crosswatch_phase3 import future_five_day_decline_proxy
from episode_evaluation import evaluate_episodes


class TestEpisodeEvaluation(unittest.TestCase):
    def test_multiple_alerts_count_once_and_lead_uses_equity_sessions(self):
        sessions = pd.bdate_range('2020-01-01', periods=12)
        closes = pd.Series(100.0, index=sessions)
        closes.iloc[5] = 97.0
        scored = pd.DataFrame(index=sessions.delete(2))
        scored['forward_proxy'] = future_five_day_decline_proxy(
            closes.to_frame('SP500')
        ).reindex(scored.index)
        scored['baseline_alert'] = scored.index.isin(sessions[[0, 1]])
        scored['IFOREST_ALERT'] = scored.index.isin(sessions[[1, 3]])
        scored['REGIME_ALERT'] = False
        scored['sp_return_pct'] = 0.0
        scored['anomaly_severity'] = 0.0
        scored['regime_excess_score'] = 0.0
        results = evaluate_episodes(
            {'three_features': {'scored': {'validation': scored, 'test': scored}}}, closes
        )
        row = results['episode_coverage'].query(
            "split == 'test' and mode == 'fixed_threshold' and method == 'baseline'"
        ).iloc[0]
        self.assertEqual(row['proxy_episodes'], 1)
        self.assertEqual(row['positive_days_alerted'], 2)
        self.assertEqual(row['episodes_alerted'], 1)
        self.assertEqual(row['median_equity_sessions_to_loss'], 5)
        pair = results['pairwise_alert_overlap'].query(
            "split == 'test' and mode == 'fixed_threshold' and method_a == 'baseline' "
            "and method_b == 'three_features_global_iforest'"
        ).iloc[0]
        self.assertEqual(pair['both_alert'], 1)
        self.assertAlmostEqual(pair['jaccard'], 1 / 3)
