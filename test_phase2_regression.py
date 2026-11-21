import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crosswatch import (engineer_features, chronological_split,fit_crosswatch,apply_crosswatch,
                        apriori_frequent_itemsets,mine_alert_rules,generate_synthetic_levels,join_market_levels,read_fred_csv)

class TestCROSSWATCH(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels=generate_synthetic_levels(n=240)
        cls.data=engineer_features(cls.levels)

    def test_bp_convention(self):
        df=pd.DataFrame({'SP500':[100,102,101], 'VIXCLS':[20,21,20], 'DGS10':[4.00,4.08,4.03]},
                        index=pd.date_range('2026-01-01',periods=3))
        f=engineer_features(df)
        self.assertAlmostEqual(float(f.iloc[0]['sp_return_pct']),2.0)
        self.assertAlmostEqual(float(f.iloc[0]['vix_delta_points']),1.0)
        self.assertAlmostEqual(float(f.iloc[0]['yield_delta_bp']),8.0)
        self.assertAlmostEqual(float(f.iloc[1]['yield_delta_bp']),-5.0)

    def test_chronological_no_leakage(self):
        a,b,c=chronological_split(self.data)
        self.assertLess(a.index.max(),b.index.min())
        self.assertLess(b.index.max(),c.index.min())
        self.assertEqual(len(a)+len(b)+len(c),len(self.data))

    def test_alerts_and_score_range(self):
        train,valid,test=chronological_split(self.data)
        fit=fit_crosswatch(train)
        output=apply_crosswatch(test,fit)
        self.assertTrue(output['cross_asset_confirmation'].between(0,3).all())
        self.assertTrue(output['IFOREST_ALERT'].isin([True,False]).all())
        self.assertAlmostEqual(fit.train_cutoffs['equity_return_5th_pct'],np.quantile(train['sp_return_pct'],.05))

    def test_apriori_known_counts(self):
        toy=pd.DataFrame({'EQUITY_TAIL_DOWN':[1,1,0,1], 'VIX_SHARP_RISE':[1,1,1,0],
                          'LARGE_YIELD_MOVE':[0,1,0,0], 'IFOREST_ALERT':[1,1,0,0]},dtype=bool)
        fis=apriori_frequent_itemsets(toy,min_support=.25)
        mapping={frozenset(x['items']):x['support'] for x in fis}
        self.assertAlmostEqual(mapping[frozenset(('EQUITY_TAIL_DOWN','IFOREST_ALERT'))],.5)
        rules=mine_alert_rules(toy,min_support=.25,min_confidence=.25,min_lift=1)
        rule=rules.loc[rules.antecedent=='EQUITY_TAIL_DOWN'].iloc[0]
        self.assertAlmostEqual(rule['support'],.5)
        self.assertAlmostEqual(rule['confidence'],2/3)
        self.assertAlmostEqual(rule['lift'],(2/3)/.5)

    def test_realistic_fred_csv_headers(self):
        with tempfile.TemporaryDirectory() as dirname:
            pd.DataFrame({'observation_date':['2026-01-02','2026-01-05','2026-01-06'],
                          'VIXCLS':['20.5','.','21.1']}).to_csv(Path(dirname)/'VIXCLS.csv',index=False)
            series=read_fred_csv('VIXCLS',dirname,allow_download=False)
            self.assertEqual(len(series),3)
            self.assertTrue(np.isnan(series.iloc[1]))
            self.assertAlmostEqual(series.iloc[2],21.1)

    def test_join_does_not_forward_fill(self):
        idx=pd.date_range('2026-01-01',periods=33)
        raw={'SP500':pd.Series(100+np.arange(33),idx),
             'VIXCLS':pd.Series(20+np.arange(33)*.1,idx),
             'DGS10':pd.Series(4+np.arange(33)*.01,idx)}
        raw['VIXCLS'].iloc[2]=np.nan
        levels,meta=join_market_levels(raw,'2026-01-01','2026-02-10')
        self.assertNotIn(idx[2],levels.index)
        self.assertEqual(meta['excluded_due_to_incomplete_levels'],1)

if __name__=='__main__':unittest.main()
