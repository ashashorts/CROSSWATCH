"""CROSSWATCH legacy ingestion module: parser and local verification only.

Network download and model execution are blocked at all entry points.
Use run_fred_historical.py after documented approvals and provenance checks.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from crosswatch import SERIES, read_fred_csv, join_market_levels

START_DEFAULT='2016-11-21'
END_DEFAULT='2026-11-18'  # FRED pages verified through this date on 2026-11-20
MIN_TOTAL_OBS=500

class SourceAcquisitionError(RuntimeError): pass

def sha256(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_official_csv(content:bytes,series:str)->pd.DataFrame:
    if content[:2]==b'PK':
        raise SourceAcquisitionError('Received a ZIP archive instead of a single-series CSV; use one series per request.')
    try: txt=content.decode('utf-8-sig')
    except UnicodeDecodeError as e: raise SourceAcquisitionError('Response is not UTF-8 CSV') from e
    if txt.lstrip().lower().startswith(('<html','<!doctype','{')):
        raise SourceAcquisitionError('Response is HTML or JSON, not a FRED CSV; save the original CSV only')
    try: frame=pd.read_csv(io.StringIO(txt), dtype=str)
    except Exception as e: raise SourceAcquisitionError('Could not parse CSV') from e
    frame.columns=[str(c).strip().lstrip('\ufeff') for c in frame.columns]
    date_col=next((c for c in ('observation_date','DATE','Date','date') if c in frame),None)
    if not date_col or series not in frame:
        raise SourceAcquisitionError(f'Expected date and {series} columns; received {list(frame.columns)}')
    dates=pd.to_datetime(frame[date_col],errors='coerce')
    values=pd.to_numeric(frame[series].replace('.',np.nan),errors='coerce')
    if dates.isna().any(): raise SourceAcquisitionError('Invalid observation dates in CSV')
    if dates.duplicated().any(): raise SourceAcquisitionError('Duplicate date observations in CSV')
    if len(frame)<MIN_TOTAL_OBS: raise SourceAcquisitionError(f'Only {len(frame)} observations; at least {MIN_TOTAL_OBS} required')
    if values.notna().sum()<400: raise SourceAcquisitionError('Too few valid numeric observations')
    if series=='SP500' and (values.dropna()<=0).any(): raise SourceAcquisitionError('Invalid nonpositive equity index level')
    if series=='VIXCLS' and (values.dropna()<=0).any(): raise SourceAcquisitionError('Invalid nonpositive VIX level')
    if series=='DGS10' and (values.dropna().abs()>30).any(): raise SourceAcquisitionError('Implausible yield magnitude; expect units = percent')
    return pd.DataFrame({'observation_date':dates.dt.strftime('%Y-%m-%d'),series:values})

def get_fred_series(series: str, start: str, end: str, *, api_key: str | None = None):
    """Legacy network access is intentionally disabled; never contact FRED here."""
    raise SourceAcquisitionError('Automatic FRED acquisition disabled in Fix 1; use manually acquired authorized local CSVs.')

def verify_local_files(raw_dir:Path,start:str,end:str)->tuple[dict,dict]:
    sources={}
    audit={}
    for s in SERIES:
        p=raw_dir/f'{s}.csv'
        if not p.exists(): raise FileNotFoundError(f'Missing {p}')
        frame=parse_official_csv(p.read_bytes(),s)
        dates=pd.to_datetime(frame['observation_date'])
        window=(dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end))
        vals=frame.loc[window,s]
        if not len(vals) or vals.notna().sum()<350:
            raise ValueError(f'{s}: insufficient in-window observations ({int(vals.notna().sum())}); check date range.')
        audit[s]={
            'source':'FRED official series ID; file origin must be retained in acquisition manifest',
            'file_name':p.name,'sha256':sha256(p),
            'raw_observation_rows':int(len(frame)),
            'window_rows':int(window.sum()),'window_numeric_rows':int(vals.notna().sum()),
            'window_missing_rows':int(vals.isna().sum()),
            'first_observation':str(dates.min().date()),
            'last_observation':str(dates.max().date()),
            'window_first_valid':str(dates[window & frame[s].notna()].min().date()),
            'window_last_valid':str(dates[window & frame[s].notna()].max().date()),
        }
        sources[s]=read_fred_csv(s,raw_dir,allow_download=False)
    return sources,audit

def acquire(raw_dir: Path, start: str, end: str, *, refresh: bool = False):
    """Legacy downloader is intentionally disabled; no network or file mutation."""
    raise SourceAcquisitionError('Automatic FRED acquisition disabled in Fix 1; use manually acquired authorized local CSVs.')


def main(argv=None):
    """Never run historical modeling from this legacy, ungated entry point."""
    print('LEGACY HISTORICAL ENTRY POINT BLOCKED: use Sources/run_fred_historical.py '
          'with the private rights and provenance records.', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
