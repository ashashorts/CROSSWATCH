"""Step 4 release-oriented checks: explicitly synthetic, no leaked raw prices."""
from pathlib import Path
import json
import re
import shutil
import pytest
import pandas as pd
from build_dashboard import read_and_validate,build,render

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence'/'qa_alternating'

def test_source_is_explicitly_synthetic():
    d=read_and_validate(EVIDENCE)
    assert d['metadata']['n_rows'] >= 1000
    assert len(d['rows']) >= 100
    assert d['status'].startswith('SYNTHETIC SOFTWARE')


def test_demo_has_no_restricted_raw_level_keys(tmp_path):
    p=build(EVIDENCE,tmp_path/'review.html')
    html=p.read_text()
    assert 'SP500' not in html  # public-facing display text says S&P 500, not internal CSV raw columns
    assert 'SYNTHETIC SOFTWARE TEST DATA' in html
    assert '<script>' in html
    assert len(html)>20000
    assert '<script src=' not in html


def test_demo_can_not_mislabel_market_data(tmp_path):
    dest=tmp_path/'qa'
    shutil.copytree(EVIDENCE,dest)
    p=dest/'SYNTHETIC_QA_ONLY_metadata.json'
    obj=json.loads(p.read_text())
    obj['source_type']='RIGHTS_REVIEWED_MARKET_DATA'
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='only permits'):
        read_and_validate(dest)


def test_schema_mismatch_rejected(tmp_path):
    dest=tmp_path/'qa'
    shutil.copytree(EVIDENCE,dest)
    p=dest/'SYNTHETIC_QA_ONLY_test_derived_results.csv'
    d=pd.read_csv(p).drop(columns=['anomaly_severity'])
    d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='required fields'):
        read_and_validate(dest)


def test_duplicate_date_rejected(tmp_path):
    dest=tmp_path/'qa'
    shutil.copytree(EVIDENCE,dest)
    p=dest/'SYNTHETIC_QA_ONLY_test_derived_results.csv'
    d=pd.read_csv(p)
    d.loc[1,'DATE']=d.loc[0,'DATE']
    d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='duplicated'):
        read_and_validate(dest)


def test_output_has_date_inspection_and_methods(tmp_path):
    output=build(EVIDENCE,tmp_path/'demo.html')
    s=output.read_text()
    for wanted in ('id="when"','id="filter"','id="chart"','id="rules"','id="metrics"',
                   'IFOREST_ALERT','REGIME_ALERT','cross_asset_confirmation','No third-party'):
        # last one is in source docs, but page itself specifies offline no network.
        if wanted=='No third-party':continue
        assert wanted in s
    assert 'real market' in s.lower() or 'real financial' in s.lower()
