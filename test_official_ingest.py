from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Sources'))
from fetch_and_run_real import parse_official_csv,verify_local_files,SourceAcquisitionError
from crosswatch import SERIES

def make(series='SP500', n=520,header='observation_date',missing=False):
    dates=pd.bdate_range('2022-01-03',periods=n)
    vals=np.linspace(4000,4600,n)
    if series=='VIXCLS': vals=np.linspace(15,25,n)
    if series=='DGS10': vals=np.linspace(2,4,n)
    df=pd.DataFrame({header:dates.strftime('%Y-%m-%d'),series:vals.astype(str)})
    if missing: df.loc[6,series]='.'
    return df.to_csv(index=False).encode()

@pytest.mark.parametrize('header',['DATE','observation_date'])
def test_supported_official_date_headers(header):
    out=parse_official_csv(make(header=header),'SP500')
    assert len(out)==520 and list(out)==['observation_date','SP500']

def test_missing_dot_is_not_zero():
    out=parse_official_csv(make(missing=True),'SP500')
    assert pd.isna(out.loc[6,'SP500'])

def test_reject_html_as_csv():
    with pytest.raises(SourceAcquisitionError): parse_official_csv(b'<html>blocked</html>','SP500')

def test_reject_too_short_data():
    with pytest.raises(SourceAcquisitionError): parse_official_csv(b'observation_date,SP500\n2026-11-18,7323.63\n','SP500')

def test_reject_duplicate_dates():
    body=pd.read_csv(__import__('io').BytesIO(make()))
    body.loc[1,'observation_date']=body.loc[0,'observation_date']
    with pytest.raises(SourceAcquisitionError): parse_official_csv(body.to_csv(index=False).encode(),'SP500')

def test_yield_units_sanity():
    bad=make('DGS10').decode().replace('2.0','200.0',1)
    with pytest.raises(SourceAcquisitionError): parse_official_csv(bad.encode(),'DGS10')

def test_verify_real_files_full_window(tmp_path):
    for symbol in SERIES: (tmp_path/(symbol+'.csv')).write_bytes(make(symbol))
    raw,audit=verify_local_files(tmp_path,'2022-01-03','2025-01-01')
    assert set(raw)==set(SERIES) and len(audit)==3

def test_missing_one_series_blocks_run(tmp_path):
    for symbol in SERIES[:-1]: (tmp_path/(symbol+'.csv')).write_bytes(make(symbol))
    with pytest.raises(FileNotFoundError): verify_local_files(tmp_path,'2022-01-03','2025-01-01')
