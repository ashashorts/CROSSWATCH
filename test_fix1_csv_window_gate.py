"""Additional Fix 1 regression: preserve the full period of the actual report."""
from pathlib import Path
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'Sources'))
from inspect_private_inputs import preflight, require_full_history_window, DatasetCoverageError
from run_fred_historical import main as historical_main

SERIES=('SP500','VIXCLS','DGS10')

def make_csvs(path, first, last):
    path.mkdir(parents=True, exist_ok=True)
    dates=pd.bdate_range(first,last)
    for name in SERIES:
        values = ([2500.0+0.5*i for i in range(len(dates))] if name=='SP500' else
                  [15.0+(i%20)/10 for i in range(len(dates))] if name=='VIXCLS' else
                  [2.0+(i%20)/100 for i in range(len(dates))])
        pd.DataFrame({'observation_date':dates.strftime('%Y-%m-%d'),name:values}).to_csv(path/f'{name}.csv',index=False)

def test_full_history_window_passes(tmp_path):
    raw=tmp_path/'raw'
    make_csvs(raw,'2017-01-03','2026-11-18')
    rep=require_full_history_window(raw)
    assert rep['full_requested_window_covered']
    assert rep['complete_joined_dates']>2400

def test_recent_five_year_csvs_rejected_for_2017_report(tmp_path):
    raw=tmp_path/'raw'
    make_csvs(raw,'2021-11-19','2026-11-19')
    report=preflight(raw)
    assert not report['full_requested_window_covered']
    assert report['complete_joined_dates']>1000
    with pytest.raises(DatasetCoverageError,match='HISTORICAL COVERAGE GATE'):
        require_full_history_window(raw)

def test_changed_window_can_be_evaluated_but_not_silently_reused(tmp_path):
    raw=tmp_path/'raw'
    make_csvs(raw,'2021-11-19','2026-11-19')
    rep=require_full_history_window(raw,'2021-11-19','2026-11-18')
    assert rep['full_requested_window_covered']

def test_incomplete_csv_blocks_historical_runner_before_results(tmp_path,monkeypatch):
    import run_fred_historical
    raw=tmp_path/'raw'
    make_csvs(raw,'2021-11-19','2026-11-19')
    monkeypatch.setattr(run_fred_historical,'require_rights',lambda *a,**kw:{})
    monkeypatch.setattr(run_fred_historical,'require_provenance',lambda *a,**kw:{})
    out=tmp_path/'results'
    assert historical_main(['--raw-dir',str(raw),'--out-dir',str(out)])==3
    assert not out.exists()


def test_incomplete_csv_blocks_approved_runner_before_results(tmp_path,monkeypatch):
    import approved_real_run
    raw=tmp_path/'raw'
    make_csvs(raw,'2021-11-19','2026-11-19')
    monkeypatch.setattr(approved_real_run,'require_rights',lambda *a,**kw:{})
    monkeypatch.setattr(approved_real_run,'require_provenance',lambda *a,**kw:{})
    out=tmp_path/'results'
    with pytest.raises(DatasetCoverageError,match='HISTORICAL COVERAGE GATE'):
        approved_real_run.run(raw,out,tmp_path/'license.json',tmp_path/'origin.json','2017-01-03','2026-11-18')
    assert not out.exists()
