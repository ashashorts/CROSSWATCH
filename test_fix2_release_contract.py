"""Fix 2: one manifest, one source label, two models, all consumers, fail-closed UI.

These tests use GENERATED SYNTHETIC input only. No real permissions or market
performance are asserted by this fixture.
"""
import ast
import json
import shutil
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Sources'))
from release_integrity import (read_manifest, write_manifest, ReleaseIntegrityError,
                               MANIFEST_NAME, sha256)
from historical_cases import load_cases, case_figure, select_examples, matching_rules
from crosswatch import SERIES


@pytest.fixture(scope='module')
def synthetic_run(tmp_path_factory):
    from run_integrated_synthetic_qa import main
    base=tmp_path_factory.mktemp('fix2_integration')
    out,raw=base/'synthetic_results',base/'synthetic_csvs'
    assert main(['--out-dir',str(out),'--raw-dir',str(raw)])==0
    return out,raw


def test_manifest_complete_and_two_models(synthetic_run):
    out,raw=synthetic_run
    m=read_manifest(out,raw_dir=raw)
    assert m['schema_version']==3
    assert m['run_kind']=='synthetic_integration_qa'
    assert m['source_label']=='SYNTHETIC_INTEGRATION_ONLY'
    assert set(m['variants'])=={'three_features','five_features'}
    assert len(m['artifacts'])>=45
    assert m['includes_statistical_baselines'] is True
    assert all(len(m['input_sha256'][series])==64 for series in SERIES)
    assert (out/'Cases'/'cases.csv').is_file()
    assert not any('rights_evidence' in path for path in m['artifacts'])


@pytest.mark.parametrize('variant',['three_features','five_features'])
@pytest.mark.parametrize('split',['validation','test'])
def test_dashboard_case_loader_all_combinations(synthetic_run,variant,split):
    out,raw=synthetic_run
    frame,meta,rules,closes=load_cases(out,raw,variant,split)
    assert len(frame)==meta['splits'][split]['n_rows']
    assert len(frame)>=100
    assert frame['forward_proxy'].notna().sum()==len(frame)-5
    assert {'fixed_zscore_alert','rolling_zscore_alert','REGIME_ALERT'}.issubset(frame.columns)
    assert not frame.index.has_duplicates
    assert not any(name in frame.columns for name in SERIES)
    examples=select_examples(frame)
    assert examples
    date=next(iter(examples.values()))
    figure=case_figure(frame,meta,closes,date)
    assert len(figure.axes)==5
    import matplotlib.pyplot as plt
    plt.close(figure)
    selected=matching_rules(frame.loc[date],rules,split)
    assert 'alert_on_selected_date' in selected.columns


def test_source_hash_mismatch_fails_closed(synthetic_run,tmp_path):
    out,raw=synthetic_run
    forged=tmp_path/'altered-source';shutil.copytree(raw,forged)
    with (forged/'DGS10.csv').open('a') as f: f.write('\n2027-01-01,4.5\n')
    with pytest.raises(ReleaseIntegrityError,match='DGS10.csv'):
        read_manifest(out,raw_dir=forged)


def test_derived_file_tamper_detected(synthetic_run,tmp_path):
    out,raw=synthetic_run
    dest=tmp_path/'tampered';shutil.copytree(out,dest)
    p=dest/'five_features'/'SYNTHETIC_INTEGRATION_ONLY_test_derived_results.csv'
    with p.open('a') as f:f.write('FORGED,1\n')
    with pytest.raises(ReleaseIntegrityError,match='Artifact missing or changed'):
        read_manifest(dest,raw_dir=raw)


def test_cross_run_manifest_swap_detected(synthetic_run,tmp_path):
    out,raw=synthetic_run
    dest=tmp_path/'wrong-hash';shutil.copytree(out,dest)
    m=json.loads((dest/MANIFEST_NAME).read_text())
    m['input_sha256']['VIXCLS']='0'*64
    (dest/MANIFEST_NAME).write_text(json.dumps(m))
    with pytest.raises(ReleaseIntegrityError,match='VIXCLS.csv'):
        read_manifest(dest,raw_dir=raw)


def test_missing_model_output_blocks_loader(synthetic_run,tmp_path):
    out,raw=synthetic_run
    dest=tmp_path/'missing';shutil.copytree(out,dest)
    (dest/'three_features'/'association_rules.csv').unlink()
    with pytest.raises(ReleaseIntegrityError):
        load_cases(dest,raw,'three_features','test')


def test_unmanifested_archived_results_cannot_impersonate_release():
    with pytest.raises(ReleaseIntegrityError,match='manifest missing'):
        read_manifest(ROOT/'Results')


def test_labels_are_not_hardcoded_in_consumers():
    for name in ('historical_cases.py','streamlit_app.py','export_historical_cases.py'):
        text=(ROOT/'Sources'/name).read_text()
        assert 'FRED_HISTORICAL' not in text
        ast.parse(text)
    import nbformat
    notebook=nbformat.read(ROOT/'CROSSWATCH_Historical_Walkthrough.ipynb',as_version=4)
    nbformat.validate(notebook)
    code='\n'.join(''.join(c.source) for c in notebook.cells if c.cell_type=='code')
    assert 'read_manifest(' in code and 'load_cases(' in code
    assert 'FRED_HISTORICAL' not in code


def test_no_outdir_created_when_real_approval_missing(tmp_path):
    from run_fred_historical import main
    dest=tmp_path/'not-created'
    code=main(['--raw-dir',str(tmp_path/'missing-source'),
               '--out-dir',str(dest),'--rights-file',str(tmp_path/'no-rights.json'),
               '--origin-file',str(tmp_path/'no-origin.json')])
    assert code==2
    assert not dest.exists()


def test_exporter_cannot_modify_fingerprinted_run(synthetic_run,tmp_path):
    from export_historical_cases import main
    out,raw=synthetic_run
    with pytest.raises(ValueError,match='immutable'):
        main(['--results-dir',str(out),'--raw-dir',str(raw),'--output',str(out/'OtherCases')])
    dest=tmp_path/'exported-figures'
    assert main(['--results-dir',str(out),'--raw-dir',str(raw),'--output',str(dest)])==0
    assert (dest/'cases.csv').exists()
    assert len(list(dest.glob('*.png')))>=1
    read_manifest(out,raw_dir=raw)


def test_new_unregistered_artifact_detected(synthetic_run,tmp_path):
    out,raw=synthetic_run
    dest=tmp_path/'extra';shutil.copytree(out,dest)
    (dest/'new-unreviewed-output.csv').write_text('a,b\n1,2\n')
    with pytest.raises(ReleaseIntegrityError,match='Unregistered'):
        read_manifest(dest,raw_dir=raw)


def test_streamlit_source_executes_with_dry_run_ui_stub(synthetic_run,monkeypatch):
    """Tests UI data callbacks without claiming a real Streamlit browser session."""
    import runpy
    from types import ModuleType
    out,raw=synthetic_run
    monkeypatch.setenv('CROSSWATCH_RESULTS',str(out))
    monkeypatch.setenv('CROSSWATCH_RAW',str(raw))
    class Stub(ModuleType):
        def __init__(self):
            super().__init__('streamlit')
            self.visited=[]
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getattr__(self,name):
            if name=='stop':
                def stop():raise RuntimeError('UI stop was invoked: '+str(self.visited[-2:]))
                return stop
            def log(*args,**kwargs):
                self.visited.append(name)
                if name=='columns':return [self]*args[0]
                if name=='tabs':return [self]*len(args[0])
                if name=='selectbox':return args[1][kwargs.get('index',0)]
            return log
    stub=Stub()
    monkeypatch.setitem(sys.modules,'streamlit',stub)
    runpy.run_path(str(ROOT/'Sources'/'streamlit_app.py'),run_name='test_stub_execution')
    for op in ['pyplot','dataframe','line_chart','metric','selectbox','tabs']:
        assert op in stub.visited, op
    assert 'error' not in stub.visited


def test_readonly_release_verifier_recomputes_evaluation(synthetic_run,tmp_path):
    from verify_release import verify,main
    out,raw=synthetic_run
    status=verify(out,raw)
    assert status['pass'] and status['artifacts_verified']>=45
    assert all(status['models'][v]['test']['rows']>100 for v in ('three_features','five_features'))
    report=tmp_path/'test-qa-report.json'
    assert main(['--results-dir',str(out),'--raw-dir',str(raw),'--report',str(report)])==0
    assert json.loads(report.read_text())['raw_source_values_exported'] is False
