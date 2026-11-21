"""CROSSWATCH Phase 3: rigorously audit supplied CSVs before any market experiments.

A PASS here means *format and statistical quality* only. It does not prove origin,
permissions, or accuracy of every price. Source permissions are a separate gate.

Run: python Sources/data_quality_gate.py --raw-dir private_raw_data --out-dir evidence
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from crosswatch import SERIES, join_market_levels, engineer_features, chronological_split
from fetch_and_run_real import parse_official_csv

MIN_COMPLETE_DAYS = 1000
MIN_VALID_DAYS_PER_SOURCE = 1000
EXPECTED_SP500_START = '2016-11-21'  # earliest recent rolling window, informational only

class GateError(RuntimeError):
    pass

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _source_stats(path: Path, code: str, start: str, end: str):
    if not path.is_file():
        raise GateError(f'MISSING required CSV: {path}')
    raw = parse_official_csv(path.read_bytes(), code)
    d = pd.to_datetime(raw['observation_date'])
    if not d.is_monotonic_increasing:
        raise GateError(f'{code}: input file dates must be strictly increasing, not merely sortable')
    v = raw[code].astype(float)
    in_window = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
    s = pd.Series(v.to_numpy(dtype=float), index=pd.DatetimeIndex(d), name=code).sort_index()
    window = s.loc[start:end]
    if window.notna().sum() < MIN_VALID_DAYS_PER_SOURCE:
        raise GateError(f'{code}: only {window.notna().sum()} valid rows in window; minimum {MIN_VALID_DAYS_PER_SOURCE}')
    if not s.index.is_monotonic_increasing or s.index.has_duplicates:
        raise GateError(f'{code}: dates invalid / not strictly increasing after parse')
    if (window.dropna() <= 0).any() and code in ('SP500','VIXCLS'):
        raise GateError(f'{code}: index should be strictly positive')
    if code=='DGS10' and ((window.dropna()<-5).any() or (window.dropna()>30).any()):
        raise GateError('DGS10: yield outside conservative [-5%,30%] range; possible unit mismatch')
    unique_values=window.dropna().nunique()
    if unique_values < 50:
        raise GateError(f'{code}: suspiciously constant file: only {unique_values} unique values')
    observed_gaps = window.dropna().index.to_series().diff().dt.days
    gaps = observed_gaps > 7
    if (observed_gaps > 10).any():
        raise GateError(f'{code}: gap of more than 10 calendar days in observed values; inspect missing periods')
    report={
        'file_name':path.name,
        'sha256':sha256(path),
        'bytes':path.stat().st_size,
        'total_rows_in_source_csv':int(len(raw)),
        'raw_missing_numeric_rows':int(v.isna().sum()),
        'raw_missing_pct':round(100 * float(v.isna().mean()), 4),
        'first_csv_date':str(d.min().date()),
        'last_csv_date':str(d.max().date()),
        'source_date_order_increasing':bool(d.is_monotonic_increasing),
        'window_raw_rows':int(in_window.sum()),
        'window_valid_numeric_rows':int(window.notna().sum()),
        'window_missing_numeric_rows':int(window.isna().sum()),
        'window_first_numeric_date':str(window.dropna().index.min().date()),
        'window_last_numeric_date':str(window.dropna().index.max().date()),
        'min_value':float(window.min()),
        'max_value':float(window.max()),
        'unique_numeric_values':int(unique_values),
        'gap_longer_than_7_calendar_days_count':int(gaps.sum()),
        'gap_longer_than_7_days_examples':[str(x.date()) for x in gaps[gaps].index[:10]],
        'pass_numeric_quality':True,
        'origin_confirmed_by_this_audit':False,
        'licensing_confirmed_by_this_audit':False,
    }
    return s,report

def audit(raw_dir: str|Path, out_dir: str|Path, start='2017-01-03', end='2026-11-18'):
    raw_dir, out_dir=Path(raw_dir), Path(out_dir)
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise GateError('start date must precede end date')
    series, reports={},{}
    for k in SERIES:
        series[k], reports[k]=_source_stats(raw_dir/f'{k}.csv', k, start, end)
    complete, join_stats=join_market_levels(series,start,end)
    if len(complete)<MIN_COMPLETE_DAYS:
        raise GateError(f'Only {len(complete)} joined complete dates; at least {MIN_COMPLETE_DAYS} required')
    derived=engineer_features(complete)
    training, validation, testing=chronological_split(derived)
    if len(validation)<100 or len(testing)<100:
        raise GateError('Validation and test split size must each exceed 100')
    for name, frame in [('train', training),('validation',validation),('test',testing)]:
        if not np.isfinite(frame[['sp_return_pct','vix_delta_points','yield_delta_bp']].to_numpy()).all():
            raise GateError(f'{name} has nonfinite model features')
    if ((derived['sp_return_pct'].abs() > 40).any()):
        raise GateError('Suspicious one-day SP500 move >40%; investigate corporate action or bad data')
    # No data rows are exported by the gate. Index levels remain local/private.
    summary={
        'audit_time_utc':datetime.now(timezone.utc).isoformat(),
        'audit_scope':'file-format, values, common-date coverage; NOT license/source-authenticity validation',
        'analysis_window':{'start':start,'end':end},
        'raw_files':reports,'join':join_stats,
        'n_joined_level_days':int(len(complete)),
        'n_complete_feature_days':int(len(derived)),
        'split':{name:{'n':len(frame),'start':str(frame.index.min().date()),'end':str(frame.index.max().date())} for name,frame in [('train',training),('validation',validation),('test',testing)]},
        'feature_distributions':{k:{'train_mean':float(training[k].mean()),'train_std':float(training[k].std()),'validation_mean':float(validation[k].mean()),'test_mean':float(testing[k].mean())} for k in ['sp_return_pct','vix_delta_points','yield_delta_bp']},
        'data_quality_pass':True,'official_origin_verified':False,'usage_rights_verified':False,
        'analysis_authorized':False,
        'cautions':['A well-formed CSV can still be falsified or misattributed.','Official websites restrict data uses; permissions are a separate process.','This script does not fit or validate a machine-learning model.']
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    outfile=out_dir/'raw_csv_quality_report.json'
    outfile.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--raw-dir',type=Path,default=Path('private_raw_data'))
    p.add_argument('--out-dir',type=Path,default=Path('evidence'))
    p.add_argument('--start',default='2017-01-03')
    p.add_argument('--end',default='2026-11-18')
    a=p.parse_args(argv)
    try:
        r=audit(a.raw_dir,a.out_dir,a.start,a.end)
    except Exception as e:
        print('DATA QUALITY GATE: BLOCKED:',str(e))
        return 2
    print('CSV STRUCTURE/VALUES QUALITY GATE: PASS',r['n_joined_level_days'],'aligned days; full historical-window coverage must be checked separately with inspect_private_inputs.py; origin and rights NOT independently verified')
    print('OUTPUT:',a.out_dir/'raw_csv_quality_report.json')
    return 0
if __name__=='__main__':
    raise SystemExit(main())
