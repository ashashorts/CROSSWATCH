"""Synthetic integration test for the WHOLE historical runner, never an authorization assertion.

The mock permission structures below are only for ephemeral pytest-generated SYNTHETIC
market values and are NEVER used with real market data or persisted in the release.
The genuine require_rights / require_provenance checks have separate negative tests.
"""
from pathlib import Path
import hashlib
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Sources'))


def test_full_historical_runner_writes_both_feature_variants_on_synthetic_fixture(tmp_path,monkeypatch):
    import run_fred_historical
    from crosswatch import SERIES
    raw=tmp_path/'SYNTHETIC_INPUTS';raw.mkdir()
    dates=pd.bdate_range('2017-01-03','2026-11-18')
    n=len(dates); rng=np.random.default_rng(2026)
    returns=rng.normal(0.00022, 0.008, n)
    returns[::127]=-0.052
    equity=2700*np.cumprod(1+returns)
    vix=np.maximum(8,18 + 75*np.maximum(0,-returns) + 2*np.sin(np.arange(n)/21))
    treasury=np.maximum(0.1,2.4+np.cumsum(rng.normal(0,.012,n)))
    for name,values in [('SP500',equity),('VIXCLS',vix),('DGS10',treasury)]:
        pd.DataFrame({'observation_date':dates.strftime('%Y-%m-%d'),name:values}).to_csv(raw/f'{name}.csv',index=False)
    monkeypatch.setattr(run_fred_historical,'require_rights',lambda *a,**kw:{
        'sources':{symbol:{'document':'SYNTHETIC_TEST_ONLY_NO_REAL_APPROVAL'} for symbol in SERIES}
    })
    monkeypatch.setattr(run_fred_historical,'require_provenance',lambda *a,**kw:{
        'sources':{symbol:{'official_source_url':f'https://example.invalid/SYNTHETIC_{symbol}',
                           'sha256':hashlib.sha256((raw/f'{symbol}.csv').read_bytes()).hexdigest()} for symbol in SERIES}
    })
    out=tmp_path/'generated_SYNTHETIC_TEST_ONLY'
    status=run_fred_historical.main(['--raw-dir',str(raw),'--out-dir',str(out),
                                      '--compare-features','--compare-baselines'])
    assert status==0
    assert (out/'feature_comparison_operating.csv').exists()
    assert (out/'feature_comparison_budget.csv').exists()
    assert (out/'episode_coverage.csv').exists()
    assert (out/'pairwise_alert_overlap.csv').exists()
    for name in ('three_features','five_features'):
        folder=out/name
        assert (folder/'RIGHTS_REVIEWED_MARKET_DATA_audit_manifest.json').exists()
        assert (folder/'association_rules.csv').exists()
        assert (folder/'RIGHTS_REVIEWED_MARKET_DATA_metadata.json').exists()
        assert (folder/'RIGHTS_REVIEWED_MARKET_DATA_test_derived_results.csv').exists()
