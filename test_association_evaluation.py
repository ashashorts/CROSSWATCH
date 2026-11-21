import unittest

import pandas as pd

from association_evaluation import VOLATILITY_FLAG, evaluate_associations


class TestAssociationEvaluation(unittest.TestCase):
    def test_training_rules_frozen_and_contingency_counts(self):
        frame = pd.DataFrame({
            'EQUITY_TAIL_DOWN': [True] * 10 + [False] * 10,
            'VIX_SHARP_RISE': False,
            'LARGE_YIELD_MOVE': False,
            'trailing_5d_return_vol_pct': [3.0] * 5 + [1.0] * 15,
            'IFOREST_ALERT': [True] * 5 + [False] * 15,
            'REGIME_ALERT': [True] * 7 + [False] * 13,
        })
        splits = {name: frame.copy() for name in ('train', 'validation', 'test')}
        original, meta = evaluate_associations(splits)
        self.assertEqual(meta['training_volatility_95th_percentile'], 3.0)
        row = original.loc[
            original['split'].eq('train')
            & original['consequent'].eq('IFOREST_ALERT')
            & original['antecedent'].eq(VOLATILITY_FLAG)
        ].iloc[0]
        self.assertEqual(row['joint_count'], 5)
        self.assertAlmostEqual(row['support'], 0.25)
        self.assertAlmostEqual(row['confidence'], 1.0)
        self.assertAlmostEqual(row['lift'], 4.0)

        splits['test']['trailing_5d_return_vol_pct'] = 100.0
        revised, new_meta = evaluate_associations(splits)
        self.assertEqual(meta, new_meta)
        pd.testing.assert_frame_equal(
            original.query("split != 'test'"), revised.query("split != 'test'")
        )
        self.assertEqual(set(original['antecedent']), set(revised['antecedent']))
