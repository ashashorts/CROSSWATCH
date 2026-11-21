"""CROSSWATCH Phase 2: transparent market anomaly analysis.

This module is an educational, reproducible research prototype. Actual market
observations MUST be downloaded from FRED or supplied as authentic FRED CSVs.
Synthetic examples are generated ONLY for software testing. No predictive
financial claims are made. Automated downloads are disabled by default in
this compliance-gated research package; check provider usage rights first.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable
import json
import urllib.request

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

SERIES = ('SP500', 'VIXCLS', 'DGS10')
FEATURES = ('sp_return_pct', 'vix_delta_points', 'yield_delta_bp')
PHASE1_FEATURES = (*FEATURES, 'trailing_5d_return_vol_pct', 'VIXCLS')
ITEMS = ('EQUITY_TAIL_DOWN', 'VIX_SHARP_RISE', 'LARGE_YIELD_MOVE', 'IFOREST_ALERT')


def read_fred_csv(series: str, raw_dir: str | Path, allow_download: bool = False) -> pd.Series:
    if series not in SERIES:
        raise ValueError(f'Unsupported FRED series: {series}')
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / f'{series}.csv'
    if allow_download:
        raise RuntimeError('Automatic FRED acquisition is disabled for this release. Use properly authorized local files and the rights-reviewed runner.')
    if not file_path.exists():
        if not allow_download:
            raise FileNotFoundError(f'Missing {file_path}. Download the official FRED CSV for {series} first.')
    data = pd.read_csv(file_path)
    data.columns = [str(col).strip() for col in data.columns]
    date_column = next((c for c in ('DATE','observation_date','Date','date') if c in data.columns), None)
    if date_column is None or series not in data.columns:
        raise ValueError(f'{file_path} must have DATE or observation_date and {series}; has {list(data.columns)}')
    date_index = pd.to_datetime(data[date_column], errors='coerce')
    values = pd.to_numeric(data[series].replace('.', np.nan), errors='coerce')
    series_obj = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(date_index), name=series)
    series_obj = series_obj.loc[~series_obj.index.isna()].sort_index()
    if series_obj.index.has_duplicates:
        raise ValueError(f'Duplicate dates in {file_path}')
    return series_obj


def join_market_levels(raw: dict[str, pd.Series], start: str, end: str) -> tuple[pd.DataFrame, dict]:
    for symbol in SERIES:
        if symbol not in raw:
            raise KeyError(f'Missing raw series {symbol}')
    # Use the union of source dates for diagnostics, then complete cases; no forward fill.
    master = pd.concat([raw[key].rename(key) for key in SERIES], axis=1, join='outer').sort_index()
    master = master.loc[(master.index >= pd.Timestamp(start)) & (master.index <= pd.Timestamp(end))]
    presence = master.notna().sum().to_dict()
    complete = master.dropna(subset=SERIES).copy()
    if len(complete) < 30:
        raise ValueError('Too few fully aligned observations; need at least 30 dates.')
    if (complete['SP500'] <= 0).any() or (complete['VIXCLS'] < 0).any():
        raise ValueError('Unexpected invalid market levels in source files.')
    meta = {
        'calendar_min': str(master.index.min().date()),
        'calendar_max': str(master.index.max().date()),
        'union_calendar_rows': int(len(master)),
        'complete_case_rows': int(len(complete)),
        'nonmissing_by_series': {k:int(v) for k,v in presence.items()},
        'missing_by_series_on_union': {k:int(master[k].isna().sum()) for k in SERIES},
        'excluded_due_to_incomplete_levels': int(len(master) - len(complete)),
        'note': 'No forward fill, no interpolation; changes represent differences between successive fully aligned observed dates.'
    }
    return complete, meta


def engineer_features(levels: pd.DataFrame) -> pd.DataFrame:
    data = levels.copy().sort_index()
    data['sp_return_pct'] = 100.0 * (data['SP500'] / data['SP500'].shift(1) - 1.0)
    data['vix_delta_points'] = data['VIXCLS'] - data['VIXCLS'].shift(1)
    # DGS10 is quoted in percentage points, so ×100 converts to bp.
    data['yield_delta_bp'] = 100.0 * (data['DGS10'] - data['DGS10'].shift(1))
    # Trailing volatility uses the current and previous four aligned returns.
    data['trailing_5d_return_vol_pct'] = data['sp_return_pct'].rolling(5, min_periods=5).std(ddof=1)
    data = data.dropna(subset=FEATURES)
    return data


def chronological_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError('Chronological split requires unique sorted dates.')
    n = len(data)
    t1, t2 = int(0.70*n), int(0.85*n)
    if min(t1, t2-t1, n-t2) < 5:
        raise ValueError('At least 5 observations needed per split.')
    return data.iloc[:t1].copy(), data.iloc[t1:t2].copy(), data.iloc[t2:].copy()


@dataclass
class FittedCrosswatch:
    scaler: StandardScaler
    detector: IsolationForest
    train_cutoffs: dict
    anomaly_score_threshold: float
    fitted_train_start: str
    fitted_train_end: str
    features: tuple[str, ...] = FEATURES


def fit_crosswatch(train: pd.DataFrame, random_state: int = 42, *,
                   features: tuple[str, ...] = FEATURES) -> FittedCrosswatch:
    matrix = train.loc[:, list(features)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite model features in training data.')
    scaler = StandardScaler().fit(matrix)
    scaled = scaler.transform(matrix)
    detector = IsolationForest(n_estimators=200, contamination='auto', max_samples='auto', random_state=random_state, n_jobs=1)
    detector.fit(scaled)
    # score_samples: lower means more anomalous, so invert for user-facing risk ordering.
    severity = -detector.score_samples(scaled)
    cutoff = float(np.quantile(severity, 0.95))
    thresholds = {
        'equity_return_5th_pct': float(np.quantile(train['sp_return_pct'], 0.05)),
        'vix_rise_95th_pct': float(np.quantile(train['vix_delta_points'], 0.95)),
        'abs_yield_95th_pct': float(np.quantile(train['yield_delta_bp'].abs(), 0.95)),
    }
    return FittedCrosswatch(scaler, detector, thresholds, cutoff,
                            str(train.index.min().date()), str(train.index.max().date()), tuple(features))


def apply_crosswatch(frame: pd.DataFrame, fitted: FittedCrosswatch) -> pd.DataFrame:
    out = frame.copy()
    z = fitted.scaler.transform(out.loc[:, list(fitted.features)].to_numpy(dtype=float))
    out['anomaly_severity'] = -fitted.detector.score_samples(z)
    out['IFOREST_ALERT'] = out['anomaly_severity'] >= fitted.anomaly_score_threshold
    out['EQUITY_TAIL_DOWN'] = out['sp_return_pct'] <= fitted.train_cutoffs['equity_return_5th_pct']
    out['VIX_SHARP_RISE'] = out['vix_delta_points'] >= fitted.train_cutoffs['vix_rise_95th_pct']
    out['LARGE_YIELD_MOVE'] = out['yield_delta_bp'].abs() >= fitted.train_cutoffs['abs_yield_95th_pct']
    out['baseline_alert'] = out['EQUITY_TAIL_DOWN']
    # Original, simple, auditable 0-3 interpretation layer proposed in Phase 1.
    out['cross_asset_confirmation'] = (
        out['EQUITY_TAIL_DOWN'].astype(int) +
        out['VIX_SHARP_RISE'].astype(int) +
        out['LARGE_YIELD_MOVE'].astype(int)
    )
    out['explanation'] = out.apply(
        lambda x: ', '.join(k for k in ITEMS[:3] if bool(x[k])) or 'No extreme component under training cutoffs', axis=1)
    return out


def apriori_frequent_itemsets(frame: pd.DataFrame, min_support: float=0.01, *,
                              items: tuple[str, ...]=ITEMS) -> list[dict]:
    """Small transparent Apriori implementation, avoiding a hidden library call.

    Candidate size-k itemsets are generated only from previously frequent
    subsets (Apriori pruning). This explicitly implements course Chapter 6.
    """
    if not (0 < min_support <= 1):
        raise ValueError('min_support must lie in (0,1]')
    if frame.empty:
        return []
    missing = set(items).difference(frame.columns)
    if missing:
        raise KeyError(missing)
    transactions = [set(k for k in items if bool(row[k])) for _,row in frame.iterrows()]
    n = len(transactions)
    results=[]
    L = set()
    for k in items:
        n_hits=sum(k in basket for basket in transactions)
        support=n_hits/n
        if support >= min_support:
            L.add(frozenset([k]))
            results.append({'items':(k,), 'count':n_hits, 'support':support})
    size=2
    while L:
        pool=set().union(*(set(item) for item in L))
        candidates=[frozenset(c) for c in combinations(sorted(pool),size) if all(frozenset(s) in L for s in combinations(c,size-1))]
        new_L=set()
        for candidate in candidates:
            hits=sum(candidate.issubset(basket) for basket in transactions)
            supp=hits/n
            if supp >= min_support:
                new_L.add(candidate)
                results.append({'items':tuple(sorted(candidate)), 'count':hits, 'support':supp})
        L=new_L
        size+=1
    return sorted(results,key=lambda a:(len(a['items']),-a['support'],a['items']))


def mine_alert_rules(frame: pd.DataFrame, min_support=0.01, min_confidence=0.10, min_lift=1.0, *,
                     items: tuple[str, ...]=ITEMS, consequent: str='IFOREST_ALERT') -> pd.DataFrame:
    """Rules require descriptive market antecedents and anomaly-alert consequent."""
    frequent=apriori_frequent_itemsets(frame, min_support, items=items)
    supports={frozenset(r['items']):r['support'] for r in frequent}
    result=[]
    alert=frozenset([consequent])
    for itemset,supp in supports.items():
        if consequent not in itemset or len(itemset)<2:
            continue
        antecedent=itemset - alert
        if not antecedent:
            continue
        ant_support=supports.get(antecedent,0)
        alert_support=supports.get(alert,0)
        if ant_support==0 or alert_support==0:
            continue
        confidence=supp/ant_support
        lift=confidence/alert_support
        if confidence >= min_confidence and lift >= min_lift:
            result.append({'antecedent':'+'.join(sorted(antecedent)), 'consequent':consequent,
                           'support':float(supp), 'confidence':float(confidence), 'lift':float(lift),
                           'joint_count':int(round(supp*len(frame)))})
    cols=['antecedent','consequent','support','confidence','lift','joint_count']
    if not result:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(result).sort_values(['lift','support'],ascending=[False,False]).reset_index(drop=True)[cols]


def evaluate_rules_on_holdout(holdout:pd.DataFrame, discovered:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    if holdout.empty:
        return pd.DataFrame()
    base=float(holdout['IFOREST_ALERT'].mean())
    for _,r in discovered.iterrows():
        ant=r['antecedent'].split('+')
        mask=holdout[ant].all(axis=1)
        sup_ant=float(mask.mean())
        joint=float((mask & holdout['IFOREST_ALERT']).mean())
        conf=joint/sup_ant if sup_ant>0 else float('nan')
        rows.append({'antecedent':r['antecedent'], 'test_support':joint,'test_confidence':conf,
                     'test_lift':conf/base if base>0 else float('nan'),
                     'antecedent_test_count':int(mask.sum())})
    return pd.DataFrame(rows)


def summarize_alerts(results:dict[str,pd.DataFrame]) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]
    for split, frame in results.items():
        a=frame['baseline_alert'].astype(bool)
        b=frame['IFOREST_ALERT'].astype(bool)
        union=int((a | b).sum())
        rows.append({'split':split,'n_days':len(frame),'baseline_alerts':int(a.sum()),
                     'iforest_alerts':int(b.sum()),'baseline_alert_rate':float(a.mean()),
                     'iforest_alert_rate':float(b.mean()),
                     'alert_overlap_days':int((a & b).sum()),
                     'jaccard_overlap':float((a & b).sum()/union) if union else float('nan'),
                     'mean_confirmation_among_iforest_alerts':float(frame.loc[b,'cross_asset_confirmation'].mean()) if b.any() else float('nan')})
    table=pd.DataFrame(rows)
    test=results['test']
    examples=test.loc[test['IFOREST_ALERT'],['sp_return_pct','vix_delta_points','yield_delta_bp','anomaly_severity','cross_asset_confirmation','explanation']].sort_values('anomaly_severity',ascending=False).head(10)
    return table,examples


def analyze_feature_frame(feature_data: pd.DataFrame, run_name: str, output_dir: str|Path, *, synthetic:bool=False) -> dict:
    outdir=Path(output_dir);outdir.mkdir(parents=True,exist_ok=True)
    train,valid,test=chronological_split(feature_data)
    fitted=fit_crosswatch(train)
    results={name:apply_crosswatch(frame,fitted) for name,frame in [('train',train),('validation',valid),('test',test)]}
    summary,examples=summarize_alerts(results)
    discovery=mine_alert_rules(results['train'],min_support=0.01,min_confidence=0.10,min_lift=1)
    holdout=evaluate_rules_on_holdout(results['test'],discovery)
    summary.to_csv(outdir/f'{run_name}_summary.csv',index=False)
    examples.to_csv(outdir/f'{run_name}_alert_examples.csv',index_label='date')
    discovery.to_csv(outdir/f'{run_name}_train_rules.csv',index=False)
    holdout.to_csv(outdir/f'{run_name}_holdout_rules.csv',index=False)
    metadata={
        'run_name':run_name,'synthetic':bool(synthetic),
        'data_start':str(feature_data.index.min().date()),'data_end':str(feature_data.index.max().date()),
        'n_feature_rows':int(len(feature_data)), 'split_sizes':{k:len(v) for k,v in results.items()},
        'training_cutoffs':fitted.train_cutoffs,
        'anomaly_severity_95th_train':fitted.anomaly_score_threshold,
        'model':{'algorithm':'IsolationForest','n_estimators':200,'random_state':42,'input_features':list(FEATURES)},
        'methodology_notes':['Fit scaler and IsolationForest on training dates only.',
                             'All cutoffs computed on training only.',
                             'Observed alert counts are not precision against crisis ground truth.',
                             'Apriori rules discovered on training only; descriptive co-occurrence, not causation.',
                             'No future data are used as features.']
    }
    (outdir/f'{run_name}_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return {'summary':summary,'examples':examples,'rules':discovery,'holdout_rules':holdout,'results':results,'metadata':metadata,'fitted':fitted}


def generate_synthetic_levels(n:int=390, seed:int=2026) -> pd.DataFrame:
    """Synthetic fixtures ONLY to validate software; NOT observed market history."""
    rng=np.random.default_rng(seed)
    dates=pd.bdate_range('2024-01-02',periods=n)
    log_returns=rng.normal(0,0.008,n)
    vix_delta=rng.normal(0,0.7,n)
    yield_bp=rng.normal(0,3.8,n)
    for ix in (45,121,239,312,355):
        if ix<n:
            log_returns[ix]-=0.06
            vix_delta[ix]+=6.5
            yield_bp[ix]+=14 if ix%2 else -18
    df=pd.DataFrame({'SP500':4500*np.exp(np.cumsum(log_returns)),
                     'VIXCLS':np.maximum(8,18+np.cumsum(vix_delta)*0.3),
                     'DGS10':4.1+np.cumsum(yield_bp)/100},index=dates)
    df.index.name='DATE'
    return df
