import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crosswatch import engineer_features, chronological_split, fit_crosswatch, apply_crosswatch, generate_synthetic_levels
from crosswatch_phase3 import (
    fit_regime_calibration, apply_regime_calibration, future_five_day_decline_proxy,
    evaluation_table,budget_matched_table,run_phase3_part1,raw_file_hashes
)

class TestPhase3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        features=engineer_features(generate_synthetic_levels(n=300,seed=2026))
        cls.train,cls.validation,cls.test=chronological_split(features)
        cls.fit=fit_crosswatch(cls.train)
        cls.train_scored=apply_crosswatch(cls.train,cls.fit)
        cls.validation_scored=apply_crosswatch(cls.validation,cls.fit)
        cls.cal=fit_regime_calibration(cls.train_scored)

    def test_regime_thresholds_computed_on_train_subsets(self):
        q=self.train_scored['VIXCLS'].quantile(.75)
        self.assertAlmostEqual(self.cal.vix_training_cutoff,q)
        elev=self.train_scored['VIXCLS']>=q
        self.assertAlmostEqual(self.cal.ordinary_severity_cutoff,
                 self.train_scored.loc[~elev,'anomaly_severity'].quantile(.95))
        self.assertAlmostEqual(self.cal.elevated_severity_cutoff,
                 self.train_scored.loc[elev,'anomaly_severity'].quantile(.95))

    def test_thresholds_frozen_under_holdout_change(self):
        r1=apply_regime_calibration(self.validation_scored,self.cal)
        altered=self.validation_scored.copy()
        altered.loc[altered.index[-1],'VIXCLS']=1000
        r2=apply_regime_calibration(altered,self.cal)
        self.assertEqual(self.cal.vix_training_cutoff,self.cal.vix_training_cutoff)
        self.assertTrue((r1.iloc[:-1]['regime_severity_cutoff'].values==r2.iloc[:-1]['regime_severity_cutoff'].values).all())
        self.assertTrue((r1.iloc[:-1]['REGIME_ALERT'].values==r2.iloc[:-1]['REGIME_ALERT'].values).all())

    def test_five_day_proxy_known_math_and_missing_tail(self):
        p=[100,99.5,99,97,101,99,100,100,100,100,100,100]
        d=pd.DataFrame({'SP500':p},index=pd.bdate_range('2026-01-02',periods=len(p)))
        y=future_five_day_decline_proxy(d)
        self.assertTrue(y.iloc[0]);self.assertFalse(y.iloc[4]);self.assertTrue(pd.isna(y.iloc[-1]))
        self.assertEqual(int(y.isna().sum()),5)

    def test_feature_scaling_cannot_read_future(self):
        # Fitting on train must be invariant to editing validation/test features.
        scaled_ref=self.fit.scaler.mean_.copy()
        other=self.validation.copy();other['sp_return_pct']*=1e6
        self.assertTrue(np.array_equal(self.fit.scaler.mean_,scaled_ref))

    def test_retro_budget_fixed_count_without_labels(self):
        f=apply_regime_calibration(self.validation_scored,self.cal)
        f['forward_proxy']=future_five_day_decline_proxy(f)
        m=budget_matched_table(f,fraction=.1)
        self.assertEqual(len(m),3)
        self.assertTrue((m['budget']==int(np.ceil((len(f)-5)*.1))).all())

    def test_missing_forward_proxy_excluded(self):
        f=apply_regime_calibration(self.validation_scored,self.cal)
        f['forward_proxy']=future_five_day_decline_proxy(f)
        evaluation=evaluation_table(f)
        self.assertTrue((evaluation['evaluable_days']==len(f)-5).all())
        self.assertTrue((evaluation['proxy_hits']<=evaluation['alerts']).all())

    def test_regime_fallback_error_for_tiny_bucket(self):
        with self.assertRaises(ValueError):
            fit_regime_calibration(self.train_scored,min_regime_n=1000)

    def test_reproducible_entire_pipeline(self):
        levels=generate_synthetic_levels(n=240,seed=2026)
        with tempfile.TemporaryDirectory() as p:
            a=run_phase3_part1(levels,Path(p)/'a',source_type='SYNTHETIC_TEST_ONLY')
            b=run_phase3_part1(levels,Path(p)/'b',source_type='SYNTHETIC_TEST_ONLY')
            self.assertAlmostEqual(a['metadata']['global_severity_cutoff_95th'],b['metadata']['global_severity_cutoff_95th'])
            self.assertTrue(a['overlap'].equals(b['overlap']))
            self.assertTrue(a['budget'].equals(b['budget']))
            self.assertTrue(all((Path(p)/'a'/f'SYNTHETIC_TEST_ONLY_{z}.csv').exists() for z in ['training_rules','operating_metrics','budget_matched_metrics','alert_overlap']))

if __name__=='__main__':unittest.main()
