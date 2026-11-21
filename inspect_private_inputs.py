"""Safe local CSV preflight for CROSSWATCH Phase 3; no model fitting or data download.

Data-file hashes identify *these bytes*, not independent proof of source origin.
Report contains only aggregate diagnostics, no licensed daily observations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from crosswatch import SERIES
from fetch_and_run_real import parse_official_csv

PROJECT_START = '2017-01-03'
PROJECT_END = '2026-11-18'
WINDOW_TOLERANCE_DAYS = 7
MAX_VALID_GAP_DAYS = 10
MIN_BUSINESS_DAY_COVERAGE = 0.70

class DatasetCoverageError(ValueError):
    pass


def preflight(raw_dir: Path | str, start: str=PROJECT_START, end: str=PROJECT_END, *, tolerance_days: int=WINDOW_TOLERANCE_DAYS) -> dict:
    raw_dir = Path(raw_dir)
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if a >= b or tolerance_days < 0:
        raise ValueError('Invalid analysis date window or negative tolerance')
    by_source = {}
    date_frames=[]
    for symbol in SERIES:
        p=raw_dir/f'{symbol}.csv'
        if not p.is_file():
            raise FileNotFoundError(f'Missing required input: {p}')
        parsed=parse_official_csv(p.read_bytes(),symbol)
        date=pd.to_datetime(parsed['observation_date'],errors='raise')
        if not date.is_monotonic_increasing:
            raise DatasetCoverageError(f'{symbol}: source dates must be strictly increasing, without shuffled rows')
        values=pd.to_numeric(parsed[symbol],errors='coerce')
        in_window=(date>=a)&(date<=b)
        valid_in_window=in_window&values.notna()
        valid_dates=date[valid_in_window]
        if valid_dates.empty:
            raise DatasetCoverageError(f'{symbol}: no valid rows inside requested window')
        first,last=valid_dates.min(),valid_dates.max()
        valid_gaps = valid_dates.sort_values().diff().dt.days.dropna()
        max_gap = int(valid_gaps.max()) if not valid_gaps.empty else 0
        # For partial start/end years, compare against weekdays inside this actual window.
        year_coverage = {}
        for year in range(a.year, b.year + 1):
            year_start = max(a, pd.Timestamp(f'{year}-01-01'))
            year_end = min(b, pd.Timestamp(f'{year}-12-31'))
            expected_weekdays = len(pd.bdate_range(year_start, year_end))
            observed = int(((valid_dates >= year_start) & (valid_dates <= year_end)).sum())
            ratio = observed / expected_weekdays if expected_weekdays else 1.0
            year_coverage[str(year)] = {'valid_observations': observed, 'reference_weekdays': expected_weekdays, 'weekday_coverage_fraction': round(ratio, 6)}
        years_ok = all(x['weekday_coverage_fraction'] >= MIN_BUSINESS_DAY_COVERAGE for x in year_coverage.values() if x['reference_weekdays'] >= 20)
        gaps_ok = max_gap <= MAX_VALID_GAP_DAYS
        start_ok=bool(first <= a+pd.Timedelta(days=tolerance_days))
        end_ok=bool(last >= b-pd.Timedelta(days=tolerance_days))
        date_frames.append(pd.DataFrame({'date': date[in_window],symbol:values[in_window].to_numpy()}))
        by_source[symbol]={
            'file_name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'first_csv_date':str(date.min().date()), 'last_csv_date':str(date.max().date()),
            'first_valid_in_window':str(first.date()), 'last_valid_in_window':str(last.date()),
            'total_csv_rows':int(len(parsed)), 'valid_in_window':int(valid_in_window.sum()),
            'missing_numeric_in_window':int((in_window&values.isna()).sum()),
            'invalid_date_count':int(date.isna().sum()), 'duplicate_date_count':int(date.duplicated().sum()),
            'starts_near_requested_start':start_ok, 'ends_near_requested_end':end_ok,
            'max_consecutive_valid_date_gap_days':max_gap,
            'no_suspicious_long_valid_gaps':gaps_ok,
            'year_coverage':year_coverage, 'all_years_sufficient_coverage':years_ok,
            'requested_window_covered':start_ok and end_ok and gaps_ok and years_ok,
        }
    aligned=date_frames[0]
    for fr in date_frames[1:]:
        aligned=aligned.merge(fr,on='date',how='outer',validate='one_to_one')
    joined=aligned.dropna(subset=list(SERIES))
    within=joined[(joined['date']>=a)&(joined['date']<=b)]
    result={
        'assessment':'LOCAL_CSV_PREFLIGHT_ONLY',
        'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'requested_window':{'start':start,'end':end,'boundary_tolerance_days':tolerance_days,
                            'max_valid_date_gap_days':MAX_VALID_GAP_DAYS,
                            'min_weekday_coverage_per_year':MIN_BUSINESS_DAY_COVERAGE},
        'sources':by_source,
        'union_window_dates':int(len(aligned)),
        'complete_joined_dates':int(len(within)),
        'complete_feature_days_estimate':max(int(len(within))-5,0),
        'full_requested_window_covered':all(v['requested_window_covered'] for v in by_source.values()),
        'rights_evidence_independently_checked':False,
        'file_origin_independently_authenticated':False,
        'raw_data_exported':False,
        'notes':['Preserves original CSVs unchanged.',
                 'Format and local hash checks cannot prove genuine FRED origin or licensing.',
                 'Do not use an incomplete window with old 2017–2026 report figures.'],
    }
    return result


def require_full_history_window(raw_dir: Path | str, start: str=PROJECT_START, end: str=PROJECT_END) -> dict:
    result=preflight(raw_dir,start,end)
    if not result['full_requested_window_covered']:
        bad=[f"{name}: first valid {item['first_valid_in_window']}, last valid {item['last_valid_in_window']}, max gap {item['max_consecutive_valid_date_gap_days']}d, year coverage OK {item['all_years_sufficient_coverage']}" for name,item in result['sources'].items() if not item['requested_window_covered']]
        raise DatasetCoverageError(
            f'HISTORICAL COVERAGE GATE: requested {start} through {end}; input CSVs do not cover it: ' + '; '.join(bad)
            + '. Re-download each series for the full original window, or redesign the experiment and regenerate all results.'
        )
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--out-file',type=Path,required=True)
    p.add_argument('--start',default=PROJECT_START)
    p.add_argument('--end',default=PROJECT_END)
    args=p.parse_args(argv)
    result=preflight(args.raw_dir,args.start,args.end)
    args.out_file.parent.mkdir(parents=True,exist_ok=True)
    args.out_file.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('CSV PREFLIGHT:', 'FULL-WINDOW PASS' if result['full_requested_window_covered'] else 'INCOMPLETE HISTORICAL WINDOW')
    print('Joined complete days:',result['complete_joined_dates'])
    print('Non-sensitive JSON report:',args.out_file)
    for name,item in result['sources'].items():
        print(name,item['first_valid_in_window'],'to',item['last_valid_in_window'], 'valid',item['valid_in_window'])
    return 0 if result['full_requested_window_covered'] else 3

if __name__=='__main__':
    raise SystemExit(main())
