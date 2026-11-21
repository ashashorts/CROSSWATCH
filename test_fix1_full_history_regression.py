"""Fix 1 dataset continuity, source-label and fail-closed negative controls. Synthetic fixtures only."""
from pathlib import Path
import sys
import json

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Sources'))
from inspect_private_inputs import preflight, require_full_history_window, DatasetCoverageError
from data_quality_gate import audit, GateError
from experiment_audit import audit_all

SYMBOLS=('SP500','VIXCLS','DGS10')

def _fixtures(folder, start='2017-01-03', end='2026-11-18'):
    folder.mkdir(parents=True, exist_ok=True)
    dates=pd.bdate_range(start,end)
    for i,s in enumerate(SYMBOLS):
        values=(2500+pd.Series(range(len(dates)))/100 if s=='SP500' else
                15+pd.Series(range(len(dates)))/100 if s=='VIXCLS' else
                2+pd.Series(range(len(dates)))/1000)
        pd.DataFrame({'observation_date': dates.strftime('%Y-%m-%d'),s:values}).to_csv(folder/f'{s}.csv',index=False)

def test_full_year_coverage_is_reported(tmp_path):
    root=tmp_path/'raw';_fixtures(root)
    out=require_full_history_window(root)
    assert out['full_requested_window_covered']
    assert len(out['sources']['SP500']['year_coverage'])==10
    assert all(x['weekday_coverage_fraction']>=.70 for x in out['sources']['SP500']['year_coverage'].values())

def test_missing_middle_year_fails_even_when_boundary_dates_exist(tmp_path):
    root=tmp_path/'raw';_fixtures(root)
    for s in SYMBOLS:
        file=root/f'{s}.csv';df=pd.read_csv(file)
        df=df[~df.observation_date.str.startswith('2020-')]
        df.to_csv(file,index=False)
    out=preflight(root)
    assert not out['full_requested_window_covered']
    assert not out['sources']['SP500']['all_years_sufficient_coverage']
    with pytest.raises(DatasetCoverageError,match='HISTORICAL COVERAGE GATE'):
        require_full_history_window(root)

def test_shuffled_dates_rejected_not_silently_sorted(tmp_path):
    root=tmp_path/'raw';_fixtures(root)
    path=root/'SP500.csv';df=pd.read_csv(path)
    df.iloc[[3,4]] = df.iloc[[4,3]].to_numpy()
    df.to_csv(path,index=False)
    with pytest.raises(DatasetCoverageError,match='strictly increasing'):
        preflight(root)
    with pytest.raises(GateError,match='strictly increasing'):
        audit(root,tmp_path/'output')
    assert not (tmp_path/'output').exists()

def test_missing_months_detected_by_long_gap(tmp_path):
    root=tmp_path/'raw';_fixtures(root)
    file=root/'VIXCLS.csv';df=pd.read_csv(file)
    df=df[~df.observation_date.between('2022-05-01','2022-07-31')]
    df.to_csv(file,index=False)
    with pytest.raises(DatasetCoverageError,match='HISTORICAL COVERAGE GATE'):
        require_full_history_window(root)
    with pytest.raises(GateError,match='gap of more than 10'):
        audit(root,tmp_path/'output')

def test_researched_source_label_is_supported_by_auditor(tmp_path):
    # A real label must get past the label allowlist; input is intentionally empty,
    # so the normal missing-scored-data error follows.
    with pytest.raises(KeyError,match='scored'):
        audit_all({},tmp_path,'RIGHTS_REVIEWED_MARKET_DATA')
    with pytest.raises(ValueError,match='Explicit source type is mandatory'):
        audit_all({},tmp_path,'UNREVIEWED_FRED_DATA')

def test_unsorted_but_realistic_data_blocked_before_model_fit(tmp_path):
    root=tmp_path/'raw';_fixtures(root)
    file=root/'DGS10.csv';df=pd.read_csv(file)
    df=df.sort_values('observation_date',ascending=False)
    df.to_csv(file,index=False)
    with pytest.raises(DatasetCoverageError):
        require_full_history_window(root)


def test_legacy_main_is_fail_closed(tmp_path):
    from fetch_and_run_real import main as legacy_main
    output=tmp_path/'results'
    assert legacy_main(['--raw-dir',str(tmp_path),'--out-dir',str(output),'--offline'])==2
    assert not output.exists()


def test_stage_approved_real_refuses_missing_rights_before_output(tmp_path):
    from run_experiment_stage import main as stage_main
    output=tmp_path/'results'
    raw=tmp_path/'no-inputs'
    status=stage_main(['--mode','approved-real','--raw-dir',str(raw),
                       '--out-dir',str(output),'--rights-file',str(tmp_path/'absent.json'),
                       '--origin-file',str(tmp_path/'also_absent.json')])
    assert status==2
    assert not output.exists()
