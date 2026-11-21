"""CROSSWATCH Phase 3 Part 1: frozen context thresholds and careful evaluation.

Uses the Phase 2 baseline and detector. Original extensions are explicitly
regime-calibrated alerts and interpretable confirmation, NOT new ML models.
No real-market claim is made by running the synthetic QA fixture.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd
import sklearn
from crosswatch import (
    SERIES, FEATURES, read_fred_csv, join_market_levels, engineer_features,
    chronological_split, fit_crosswatch, apply_crosswatch, mine_alert_rules,
    evaluate_rules_on_holdout, generate_synthetic_levels,
)

from zscore_baseline import ROLLING_WINDOW, ZSCORE_METHODS, ZSCORE_THRESHOLD, zscore_baselines


@dataclass(frozen=True)
class RegimeCalibration:
    vix_quantile: float
    severity_quantile: float
    vix_training_cutoff: float
    ordinary_severity_cutoff: float
    elevated_severity_cutoff: float
    ordinary_train_n: int
    elevated_train_n: int


def fit_regime_calibration(train_scored: pd.DataFrame, *, vix_quantile: float=.75,
                           severity_quantile: float=.95, min_regime_n: int=20) -> RegimeCalibration:
    """Training-only conditional cutoffs; no retraining of Isolation Forest."""
    if not 0 < vix_quantile < 1 or not 0 < severity_quantile < 1:
        raise ValueError('Quantiles must lie between 0 and 1.')
    if train_scored['VIXCLS'].isna().any() or train_scored['anomaly_severity'].isna().any():
        raise ValueError('Training regime fit requires complete values.')
    vix_threshold = float(train_scored['VIXCLS'].quantile(vix_quantile))
    elevated = train_scored['VIXCLS'] >= vix_threshold
    n_high, n_low = int(elevated.sum()), int((~elevated).sum())
    if min(n_high,n_low) < min_regime_n:
        raise ValueError(f'Insufficient training observations per regime: ordinary {n_low}, elevated {n_high}; require {min_regime_n}.')
    ordinary_cutoff = float(train_scored.loc[~elevated,'anomaly_severity'].quantile(severity_quantile))
    elevated_cutoff = float(train_scored.loc[elevated,'anomaly_severity'].quantile(severity_quantile))
    return RegimeCalibration(vix_quantile, severity_quantile,vix_threshold,ordinary_cutoff,elevated_cutoff,n_low,n_high)


def apply_regime_calibration(scored: pd.DataFrame, calibration: RegimeCalibration) -> pd.DataFrame:
    """Apply frozen training-derived calibration to any set of dates."""
    out=scored.copy()
    high=out['VIXCLS'] >= calibration.vix_training_cutoff
    out['vix_regime']=np.where(high,'Elevated VIX','Ordinary VIX')
    out['regime_severity_cutoff']=np.where(high,calibration.elevated_severity_cutoff,calibration.ordinary_severity_cutoff)
    out['REGIME_ALERT']=out['anomaly_severity'] >= out['regime_severity_cutoff']
    out['regime_excess_score']=out['anomaly_severity'] - out['regime_severity_cutoff']
    return out


def future_five_day_decline_proxy(frame: pd.DataFrame, horizon: int=5,
                                  decline_pct: float=2.0) -> pd.Series:
    """Forward test *label*, strictly within the supplied closing-price calendar. Never a model feature.

    True if any closing level in following `horizon` sessions in the supplied closing-price calendar is
    at least `decline_pct` below today's closing level. Last h labels are NA.
    """
    if horizon < 1 or decline_pct <= 0: raise ValueError('Invalid proxy configuration.')
    p=frame['SP500'].astype(float)
    future=pd.concat([p.shift(-i) for i in range(1,horizon+1)],axis=1)
    complete=future.notna().all(axis=1)
    proxy=(future.min(axis=1) <= p*(1-decline_pct/100)).astype('boolean')
    proxy.loc[~complete]=pd.NA
    proxy.name=f'forward_{horizon}d_decline_{decline_pct:g}pct'
    return proxy


def evaluation_table(scored: pd.DataFrame, *, proxy_name: str='forward_proxy') -> pd.DataFrame:
    """Compare fixed natural-operating-point alerts to an auxiliary future-loss proxy.
    The proxy is not a verified anomaly label; there is no claim of causal forecasting.
    """
    valid=scored[proxy_name].notna()
    frame=scored.loc[valid].copy()
    if frame.empty: raise ValueError('No evaluable forward labels.')
    target=frame[proxy_name].astype(bool)
    methods = ['baseline_alert', 'IFOREST_ALERT', 'REGIME_ALERT']
    methods.extend(f'{method}_alert' for method in ZSCORE_METHODS if method in frame)
    rows=[]
    for method in methods:
        if frame[method].isna().any():
            raise ValueError(f'{method}: missing alerts on evaluable dates.')
        alert=frame[method].astype(bool)
        tp=int((alert&target).sum())
        fp=int((alert&~target).sum())
        support=int(target.sum())
        count=int(alert.sum())
        rows.append({
            'method':method,'evaluable_days':int(len(frame)),'proxy_positive_days':support,
            'alerts':count,'alert_rate':count/len(frame),'proxy_hits':tp,
            'proxy_nonhits':fp,'precision_vs_proxy':tp/count if count else float('nan'),
            'recall_vs_proxy':tp/support if support else float('nan'),
            'proxy_base_rate':float(target.mean())
        })
    return pd.DataFrame(rows)


def alert_rankings(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return comparable high-to-low severity rankings on the given dates."""
    scores = {
        'baseline': -frame['sp_return_pct'],
        'global_iforest': frame['anomaly_severity'],
        'regime_iforest': frame['regime_excess_score'],
    }
    scores.update({method: frame[method] for method in ZSCORE_METHODS if method in frame})
    return {
        method: score.sort_values(ascending=False, kind='stable')
        for method, score in scores.items()
    }


def budget_matched_table(scored: pd.DataFrame, *, fraction: float=.05,proxy_name: str='forward_proxy') -> pd.DataFrame:
    """Retrospective top-N comparison. This is *not* a deployable threshold.
    Uses only each method's observed scores, not evaluation labels, for ranking.
    """
    valid=scored[proxy_name].notna()
    frame=scored.loc[valid].copy()
    if not len(frame): raise ValueError('Missing evaluation days.')
    k=max(1,int(np.ceil(len(frame)*fraction)))
    target=frame[proxy_name].astype(bool)
    rows=[]
    for method, rank in alert_rankings(frame).items():
        if rank.isna().any():
            raise ValueError(f'{method}: missing scores on evaluable dates.')
        selected = rank.iloc[:k].index
        hits=int(target.loc[selected].sum())
        rows.append({'method':method,'budget':k,'evaluable_days':len(frame),
                     'proxy_hits':hits,'precision_at_budget':hits/k,
                     'proxy_base_rate':float(target.mean())})
    return pd.DataFrame(rows)


def compare_alerts(scored: pd.DataFrame) -> dict:
    b=scored['baseline_alert'].astype(bool)
    g=scored['IFOREST_ALERT'].astype(bool)
    r=scored['REGIME_ALERT'].astype(bool)
    return {
        'n_days':len(scored),
        'baseline_alerts':int(b.sum()),
        'iforest_alerts':int(g.sum()),
        'regime_alerts':int(r.sum()),
        'global_regime_overlap':int((g&r).sum()),
        'global_regime_disagreement':int((g!=r).sum()),
        'global_regime_jaccard':float((g&r).sum()/(g|r).sum()) if (g|r).any() else None,
        'mean_confirmation_global':float(scored.loc[g,'cross_asset_confirmation'].mean()) if g.any() else None,
        'mean_confirmation_regime':float(scored.loc[r,'cross_asset_confirmation'].mean()) if r.any() else None,
        'ordinary_days':int((scored['vix_regime']=='Ordinary VIX').sum()),
        'elevated_days':int((scored['vix_regime']=='Elevated VIX').sum()),
    }


def run_phase3_part1(levels: pd.DataFrame, out_dir: str|Path,*, source_type: str,
                     extra_provenance: dict|None=None,
                     equity_closes: pd.Series|None=None,
                     model_features: tuple[str, ...]=FEATURES,
                     comparison_features: tuple[str, ...]=(),
                     include_zscore: bool=False) -> dict:
    """Create reproducible real or labeled-synthetic analysis outputs."""
    out_dir=Path(out_dir);out_dir.mkdir(exist_ok=True,parents=True)
    feats=engineer_features(levels)
    train,validation,test=chronological_split(feats)
    # Preserve the original split boundaries while excluding feature warm-up rows.
    required = list(dict.fromkeys((*model_features, *comparison_features)))
    train, validation, test = (
        part.dropna(subset=required) for part in (train, validation, test)
    )
    fit = fit_crosswatch(train, random_state=42, features=model_features)
    scored={n:apply_crosswatch(f,fit) for n,f in [('train',train),('validation',validation),('test',test)]}
    calibration=fit_regime_calibration(scored['train'])
    scored={n:apply_regime_calibration(f,calibration) for n,f in scored.items()}
    if include_zscore:
        zscores = zscore_baselines(train, feats)
        scored = {name: frame.join(zscores) for name, frame in scored.items()}
    # Historical inputs can contain equity sessions excluded by the three-series join.
    if equity_closes is not None:
        equity_closes = equity_closes.dropna().sort_index()
        if not equity_closes.index.is_unique or not np.array_equal(equity_closes.reindex(levels.index), levels['SP500']):
            raise ValueError('Equity closes must be unique and match the aligned SP500 values.')
    for name,frame in scored.items():
        proxy_prices = frame[['SP500']]
        if equity_closes is not None:
            proxy_prices = equity_closes.loc[frame.index[0]:frame.index[-1]].to_frame('SP500')
        frame['forward_proxy'] = future_five_day_decline_proxy(proxy_prices).reindex(frame.index)
        # Avoid disclosing restricted index level values in public outputs.
        drop=['SP500','VIXCLS','DGS10']
        frame.drop(columns=drop).to_csv(out_dir/f'{source_type}_{name}_derived_results.csv',index_label='DATE')
    operating=pd.concat([evaluation_table(scored[n]).assign(split=n) for n in ('validation','test')],ignore_index=True)
    budget=pd.concat([budget_matched_table(scored[n]).assign(split=n) for n in ('validation','test')],ignore_index=True)
    overlaps=pd.DataFrame([{'split':n,**compare_alerts(scored[n])} for n in ('train','validation','test')])
    discovered=mine_alert_rules(scored['train'],min_support=.01,min_confidence=.10,min_lift=1.)
    rule_validation=evaluate_rules_on_holdout(scored['validation'],discovered)
    rule_test=evaluate_rules_on_holdout(scored['test'],discovered)
    if not rule_validation.empty: rule_validation=rule_validation.rename(columns={'test_support':'validation_support','test_confidence':'validation_confidence','test_lift':'validation_lift','antecedent_test_count':'antecedent_validation_count'})
    operating.to_csv(out_dir/f'{source_type}_operating_metrics.csv',index=False)
    budget.to_csv(out_dir/f'{source_type}_budget_matched_metrics.csv',index=False)
    overlaps.to_csv(out_dir/f'{source_type}_alert_overlap.csv',index=False)
    discovered.to_csv(out_dir/f'{source_type}_training_rules.csv',index=False)
    rule_validation.to_csv(out_dir/f'{source_type}_validation_rule_check.csv',index=False)
    rule_test.to_csv(out_dir/f'{source_type}_test_rule_check.csv',index=False)
    top=scored['test'].sort_values('anomaly_severity',ascending=False).head(12)[
        ['sp_return_pct','vix_delta_points','yield_delta_bp','vix_regime','anomaly_severity',
         'IFOREST_ALERT','REGIME_ALERT','cross_asset_confirmation','explanation']]
    top.to_csv(out_dir/f'{source_type}_top_test_days.csv',index_label='DATE')
    # Sensitivity is descriptive on validation only. The test operating point is frozen .75/.95.
    sweep=[]
    for vq in (.70,.75,.80):
        for sq in (.90,.95,.975):
            params=fit_regime_calibration(scored['train'],vix_quantile=vq,severity_quantile=sq)
            val=apply_regime_calibration(scored['validation'],params)
            val['forward_proxy']=scored['validation']['forward_proxy']
            e=evaluation_table(val).query("method == 'REGIME_ALERT'").iloc[0]
            sweep.append({'vix_quantile':vq,'alert_quantile':sq,
                          'validation_alerts':int(e['alerts']),
                          'validation_proxy_hits':int(e['proxy_hits']),
                          'validation_precision_proxy':float(e['precision_vs_proxy'])})
    sensitivity=pd.DataFrame(sweep)
    sensitivity.to_csv(out_dir/f'{source_type}_VALIDATION_ONLY_sensitivity.csv',index=False)
    metadata={
        'source_type':source_type,'data_start':str(train.index.min().date()),'data_end':str(test.index.max().date()),
        'n_rows':sum(len(frame) for frame in scored.values()),'splits':{n:{'n_rows':len(f),'first':str(f.index.min().date()),'last':str(f.index.max().date())} for n,f in scored.items()},
        'features':list(fit.features),'model':'IsolationForest(200 trees, random_state=42) on training-standardized inputs',
        'global_severity_cutoff_95th':fit.anomaly_score_threshold,
        'training_equity_baseline_cutoff_pct':fit.train_cutoffs['equity_return_5th_pct'],
        'regime_calibration':asdict(calibration),
        'forward_proxy':(
            'Any of the next five S&P 500 session closes within the partition is at most 98% of the current close. '
            'Sessions excluded from model scoring are included in the outcome window. Evaluation only.'
            if equity_closes is not None else
            'Any of the next five observed closes within the partition is at most 98% of the current close. Evaluation only.'
        ),
        'analysis_limitations':['No objective ground truth for market-stress events.','Proxy assesses next five-day decline, not true anomaly labels.','Overlapping five-day windows make rows statistically dependent.','Top-budget comparison uses ranking on holdout, not deployable threshold.','Association rules use detector output and input-derived flags, not causal feature attribution.','Rules discovered on training only; test used solely for evaluation.'],
        'python_version':platform.python_version(),'pandas_version':pd.__version__,'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,
        'extra_provenance':extra_provenance or {}
    }
    if include_zscore:
        metadata['zscore_baselines'] = {
            'features': list(FEATURES),
            'score': 'Maximum absolute z-score across the daily changes',
            'threshold_strictly_greater_than': ZSCORE_THRESHOLD,
            'fixed_reference': 'Training-partition mean and sample standard deviation',
            'rolling_reference_observations': ROLLING_WINDOW,
            'rolling_reference_excludes_current_observation': True,
        }
    (out_dir/f'{source_type}_metadata.json').write_text(json.dumps(metadata,indent=2,default=float),encoding='utf-8')
    return {'metadata':metadata,'scored':scored,'operating':operating,'budget':budget,'overlap':overlaps,
            'rules':discovered,'validation_rule_check':rule_validation,'test_rule_check':rule_test,
            'sensitivity':sensitivity,'top_days':top,'calibration':calibration}


def raw_file_hashes(raw_dir: str|Path) -> dict:
    raw_dir=Path(raw_dir)
    return {k:hashlib.sha256((raw_dir/f'{k}.csv').read_bytes()).hexdigest() for k in SERIES}
