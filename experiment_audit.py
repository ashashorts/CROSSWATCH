"""CROSSWATCH Phase 3 Step 3: leakage-aware, reproducible evaluation diagnostics.

These routines intentionally work on *already scored* observations. They do not
fetch data, fit models, or turn a synthetic software check into a market result.
The phase-3 permission gate remains separate and mandatory for real FRED data.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from crosswatch import engineer_features
from crosswatch_phase3 import evaluation_table, budget_matched_table

from zscore_baseline import ZSCORE_METHODS


METHOD_COLUMNS = {
    "Equity tail baseline": "baseline_alert",
    "Global Isolation Forest": "IFOREST_ALERT",
    "VIX-context Isolation Forest": "REGIME_ALERT",
}


def assert_strict_time_partitions(train: pd.DataFrame, validation: pd.DataFrame,
                                  test: pd.DataFrame) -> dict:
    """Verify monotonic disjoint split membership and time direction."""
    frames={"train":train,"validation":validation,"test":test}
    for name,frame in frames.items():
        if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{name}: empty, duplicated, or unordered observations")
    if not (train.index.max()<validation.index.min()<test.index.min()):
        raise ValueError("Splits overlap or are not strictly chronological")
    union=pd.concat([f.assign(_split=name) for name,f in frames.items()])
    if union.index.has_duplicates: raise ValueError("A date occurs in more than one split")
    return {name:{"rows":len(f),"start":str(f.index.min().date()),
                  "end":str(f.index.max().date())} for name,f in frames.items()}


def strict_gap_check(levels: pd.DataFrame, max_calendar_gap: int=5) -> dict:
    """Audit stretches between common dates (never silently forward-fill)."""
    if max_calendar_gap<1:raise ValueError('max_calendar_gap must be positive')
    if levels.index.has_duplicates or not levels.index.is_monotonic_increasing:
        raise ValueError('Market dates must be strictly ordered and unique')
    gap=levels.index.to_series().diff().dt.days
    bad=gap>max_calendar_gap
    return {"max_gap_days":float(gap.max()) if len(levels)>1 else None,
            "gaps_exceeding_limit":int(bad.sum()),
            "gap_dates":[str(x.date()) for x in gap[bad].index],
            "limit_calendar_days":max_calendar_gap}


def strict_features(levels: pd.DataFrame,max_calendar_gap: int=5) -> pd.DataFrame:
    """Drop daily changes crossing >max_calendar_gap without recomputing stale returns.

    This is a selectable *planned safety upgrade*, not a silent rewrite of Phase 1.
    """
    diagnostic=strict_gap_check(levels,max_calendar_gap)
    feat=engineer_features(levels)
    gaps=levels.index.to_series().diff().dt.days
    crosses=gaps>max_calendar_gap
    feat=feat.loc[~crosses.reindex(feat.index,fill_value=False)].copy()
    feat.attrs['excluded_gap_change_dates']=diagnostic['gap_dates']
    return feat


def _valid_subset(scored:pd.DataFrame, proxy_col:str='forward_proxy') -> pd.DataFrame:
    if proxy_col not in scored:raise KeyError(proxy_col)
    f=scored.loc[scored[proxy_col].notna()].copy()
    if f.empty:raise ValueError('No complete forward-proxy observations available')
    return f


def event_episode_count(proxy:pd.Series) -> int:
    """Consecutive positive proxy days form one event episode, not independent events."""
    p=proxy.astype(bool)
    return int((p & ~p.shift(1,fill_value=False)).sum())


def evaluate_operating_points(scored:pd.DataFrame,proxy_col='forward_proxy') -> pd.DataFrame:
    """Clearly label observed-alert-rate versus future-proxy hit fraction."""
    f=_valid_subset(scored,proxy_col)
    labels=f[proxy_col].astype(bool)
    n=len(f)
    out=[]
    methods = {**METHOD_COLUMNS, **{
        label: f'{key}_alert' for key, label in ZSCORE_METHODS.items() if key in f
    }}
    for name,col in methods.items():
        if f[col].isna().any():
            raise ValueError(f'{name}: missing alerts on evaluable dates.')
        sel=f[col].astype(bool)
        a=int(sel.sum()); hits=int((sel & labels).sum())
        out.append({"method":name,"evaluable_days":n,"proxy_positive_days":int(labels.sum()),
                    "proxy_positive_episodes":event_episode_count(labels),
                    "alerts":a,"alert_rate":a/n,
                    "proxy_hits":hits,"proxy_hit_fraction":hits/a if a else np.nan,
                    "proxy_base_rate":float(labels.mean()),
                    "proxy_hit_lift_vs_base":(hits/a)/float(labels.mean()) if a and labels.any() else np.nan,
                    "proxy_hit_recall":hits/int(labels.sum()) if labels.any() else np.nan,
                    "minimum_10_positive_episodes":event_episode_count(labels)>=10})
    return pd.DataFrame(out)


def regime_diagnostics(scored:dict[str,pd.DataFrame]) -> pd.DataFrame:
    """Detect regimes absent from holdouts; report distribution shift, not success."""
    rows=[]
    for name,frame in scored.items():
        if 'vix_regime' not in frame:raise KeyError('vix_regime')
        high=int((frame['vix_regime']=='Elevated VIX').sum()); n=len(frame)
        rows.append({"split":name,"days":n,"ordinary_vix_days":n-high,
                     "elevated_vix_days":high,"elevated_fraction":high/n if n else np.nan,
                     "one_regime_absent":high==0 or high==n,
                     "regime_dominance_ge_90pct":max(high,n-high)/n>=.90 if n else True})
    return pd.DataFrame(rows)


def _precision(arr:np.ndarray, selection:np.ndarray) -> float:
    return float((arr & selection).sum()/selection.sum()) if selection.sum() else np.nan


def paired_moving_block_bootstrap(scored:pd.DataFrame, a:str='IFOREST_ALERT',b:str='REGIME_ALERT',
                                  n_boot:int=500,block_length:int=10,seed:int=2026) -> dict:
    """Approximate paired descriptive intervals under serial dependence.

    Resample time-contiguous blocks of *fixed test predictions and proxy labels*.
    Thresholds/models stay frozen; there is no holdout re-training or tuned test rule.
    Does not provide formal forecast significance or guarantee calibration.
    """
    f=_valid_subset(scored)
    n=len(f)
    if n<block_length*3 or block_length<2 or n_boot<50:
        raise ValueError('Need >=3 blocks, block_length >=2, n_boot>=50')
    y=f['forward_proxy'].astype(bool).to_numpy()
    sa=f[a].astype(bool).to_numpy();sb=f[b].astype(bool).to_numpy()
    base=_precision(y,sa)-_precision(y,sb)
    rng=np.random.default_rng(seed);sim=[]
    for _ in range(n_boot):
        starts=rng.integers(0,n,size=int(np.ceil(n/block_length)))
        ix=np.concatenate([(np.arange(st,st+block_length)%n) for st in starts])[:n]
        pa=_precision(y[ix],sa[ix]);pb=_precision(y[ix],sb[ix])
        if np.isfinite(pa) and np.isfinite(pb):sim.append(pa-pb)
    ci=np.quantile(sim,[.025,.975]).tolist() if len(sim)>=30 else [None,None]
    return {"pair":f"{a} - {b}","n_evaluable":n,"block_length":block_length,
            "replicates_requested":n_boot,"replicates_valid":len(sim),
            "observed_proxy_hit_fraction_difference":float(base) if np.isfinite(base) else None,
            "descriptive_block_bootstrap_95pct_interval":ci,
            "interpretation":"Exploratory, serial-dependence-sensitive interval, not a valid causal/forecast proof; event days overlap."}


def circular_shift_placebo(scored:pd.DataFrame,method:str='IFOREST_ALERT',
                           n_repeats:int=200,minimum_shift:int=30,seed:int=2026) -> dict:
    """Exploratory placebo: rotate future proxy labels away from fixed observed alerts."""
    f=_valid_subset(scored);n=len(f)
    if n<2*minimum_shift+10 or n_repeats<10:
        raise ValueError('Not enough holdout dates for meaningful shifted-label placebo')
    y=f['forward_proxy'].astype(bool).to_numpy()
    a=f[method].astype(bool).to_numpy()
    if a.sum()==0: raise ValueError('No alerts to evaluate')
    observed=_precision(y,a)
    rng=np.random.default_rng(seed)
    valid_shifts=np.arange(minimum_shift,n-minimum_shift+1)
    vals=np.array([_precision(np.roll(y,int(s)),a) for s in rng.choice(valid_shifts,n_repeats,replace=True)])
    return {'method':method,'n_evaluable':n,'alerts':int(a.sum()),
            'observed_proxy_hit_fraction':float(observed),
            'shifted_proxy_hit_fraction_median':float(np.median(vals)),
            'shifted_proxy_hit_fraction_p05':float(np.quantile(vals,.05)),
            'shifted_proxy_hit_fraction_p95':float(np.quantile(vals,.95)),
            'n_shifted_null_checks':n_repeats,
            'scope':'Descriptive negative control only; nonstationarity and overlapping windows violate simple iid null interpretations.'}


def evaluate_rule_reliability(scored:pd.DataFrame, rules:pd.DataFrame,
                              min_antecedent_count:int=10,min_joint_count:int=5) -> pd.DataFrame:
    """Quantify heldout support for training-discovered rules, no re-mining test."""
    total=len(scored)
    if total==0:raise ValueError('Empty scored split')
    rows=[]
    p_alert=float(scored['IFOREST_ALERT'].mean())
    for _,rule in rules.iterrows():
        names=str(rule['antecedent']).split('+')
        if not set(names).issubset(scored.columns):raise KeyError(names)
        ant=scored[names].astype(bool).all(axis=1)
        both=ant & scored['IFOREST_ALERT'].astype(bool)
        a=int(ant.sum());j=int(both.sum())
        confidence=j/a if a else np.nan
        rows.append({'antecedent':str(rule['antecedent']),
                     'train_support':float(rule['support']),
                     'train_confidence':float(rule['confidence']),
                     'train_lift':float(rule['lift']),
                     'holdout_n':total,'antecedent_count':a,'joint_count':j,
                     'holdout_support':j/total,'holdout_confidence':confidence,
                     'holdout_lift':confidence/p_alert if p_alert else np.nan,
                     'minimum_support_met':a>=min_antecedent_count and j>=min_joint_count,
                     'mechanical_dependence_warning':True})
    return pd.DataFrame(rows)


def audit_all(results:dict,output_dir:str|Path,source_label:str) -> dict:
    """Save diagnostics with an explicit data source label."""
    if source_label not in ('SYNTHETIC_QA_ONLY','SYNTHETIC_INTEGRATION_ONLY','AUTHORIZED_REAL_DATA','FRED_HISTORICAL','RIGHTS_REVIEWED_MARKET_DATA','LOCAL_USER_PROVIDED_QA'):
        raise ValueError('Explicit source type is mandatory')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    scored=results['scored'];partition=assert_strict_time_partitions(scored['train'],scored['validation'],scored['test'])
    regimes=regime_diagnostics(scored);regimes.to_csv(out/f'{source_label}_regime_shift.csv',index=False)
    frames=[];budgets=[]
    for part in ('validation','test'):
        frame=scored[part]
        x=evaluate_operating_points(frame).assign(split=part);frames.append(x)
        budget=budget_matched_table(frame).assign(split=part)
        budgets.append(budget)
    operating=pd.concat(frames,ignore_index=True)
    retrospective=pd.concat(budgets,ignore_index=True)
    operating.to_csv(out/f'{source_label}_operating_proxy_diagnostics.csv',index=False)
    retrospective.to_csv(out/f'{source_label}_RETROSPECTIVE_budget_diagnostics.csv',index=False)
    rule_train=results['rules'];rules=[]
    for part in ('validation','test'):
        e=evaluate_rule_reliability(scored[part],rule_train)
        if not e.empty:rules.append(e.assign(split=part))
    rule_table=pd.concat(rules,ignore_index=True) if rules else pd.DataFrame()
    rule_table.to_csv(out/f'{source_label}_training_rule_holdout_checks.csv',index=False)
    uncertainty={};placebo={}
    for part in ('validation','test'):
        f=scored[part]
        if len(f)>=45:
            uncertainty[part]=paired_moving_block_bootstrap(f)
        if len(f)>=80 and f['IFOREST_ALERT'].sum()>0:
            placebo[part]=circular_shift_placebo(f)
    meta={'source_label':source_label,'partitions':partition,'regime_shift':regimes.to_dict(orient='records'),
          'uncertainty':uncertainty,'negative_controls':placebo,
          'interpretation_limits':['No ground-truth crisis labels',
                'Next-five-day drop proxy is not an independent ground-truth anomaly label',
                'Overlapping evaluation outcomes are serially dependent',
                'Observed historical co-occurrence cannot establish market predictability',
                'Test remains untouched for tuning; any next revisions need a new external holdout',
                'Association rules are inherently dependent on model inputs and should not be explained as causes.'],
          'running_on_real_data':source_label in ('AUTHORIZED_REAL_DATA','FRED_HISTORICAL','RIGHTS_REVIEWED_MARKET_DATA')}
    (out/f'{source_label}_audit_manifest.json').write_text(json.dumps(meta,indent=2,default=str))
    return {'operating':operating,'regime':regimes,'rules':rule_table,
            'budget':retrospective,'uncertainty':uncertainty,'placebo':placebo,'metadata':meta}
