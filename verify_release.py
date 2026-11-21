"""Independent read-only integration preflight for one fingerprinted CROSSWATCH run.

Requires all three ORIGINAL local CSVs so we can independently recompute the
five-session evaluation proxy. Does not fit models, fetch data, or send data away.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from historical_cases import load_cases
from release_integrity import read_manifest, VARIANTS


def verify(results_dir: Path, raw_dir: Path) -> dict:
    m=read_manifest(results_dir,raw_dir=raw_dir)
    details={}
    for variant in VARIANTS:
        rows={}
        for split in ('validation','test'):
            f,meta,rules,closes=load_cases(results_dir,raw_dir,variant,split)
            rows[split] = {
                'rows':len(f),
                'proxy_evaluable_rows':int(f['forward_proxy'].notna().sum()),
                'proxy_positive_rows':int(f['forward_proxy'].fillna(False).sum()),
                'forest_global_alerts':int(f['IFOREST_ALERT'].sum()),
                'forest_regime_alerts':int(f['REGIME_ALERT'].sum()),
                'association_rule_rows':int((rules['split']==split).sum()),
                'earliest_scored_date':str(f.index.min().date()),
                'latest_scored_date':str(f.index.max().date()),
            }
        details[variant]=rows
    for split in ('validation','test'):
        if details['three_features'][split]['proxy_evaluable_rows'] != details['five_features'][split]['proxy_evaluable_rows'] or details['three_features'][split]['proxy_positive_rows'] != details['five_features'][split]['proxy_positive_rows']:
            raise ValueError(f'{split}: the two models must share the same outcome clock')
    for name in ('operating','budget'):
        table=pd.read_csv(results_dir/f'feature_comparison_{name}.csv')
        if set(table['feature_set'])!={'three_features','five_features'} or set(table['split'])!={'validation','test'}:
            raise ValueError(f'Incomplete feature_comparison_{name} table')
    for name in ('episode_coverage','pairwise_alert_overlap'):
        table=pd.read_csv(results_dir/f'{name}.csv')
        if set(table['split'])!={'validation','test'}:
            raise ValueError(f'Incomplete {name} table')
    return {
        'kind':'CROSSWATCH_INDEPENDENT_READ_ONLY_INTEGRATION_CHECK',
        'tested_utc':datetime.now(timezone.utc).isoformat(),
        'pass':True,'run_id':m['run_id'],'run_kind':m['run_kind'],
        'source_label':m['source_label'],'artifacts_verified':len(m['artifacts']),
        'all_three_private_csv_hashes_matched':True,
        'scoring_and_five_session_proxy_consistent':True,
        'both_model_variants_verified':True,
        'comparisons_and_episode_outputs_present':True,
        'models':details,
        'legal_scope_not_independently_certified':True,
        'raw_source_values_exported':False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,required=True)
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--report',type=Path)
    a=p.parse_args(argv)
    try:
        out=verify(a.results_dir,a.raw_dir)
    except Exception as exc:
        print('CROSSWATCH RELEASE CHECK FAILED:',type(exc).__name__,str(exc))
        return 2
    if a.report:
        if a.report.resolve().is_relative_to(a.results_dir.resolve()):
            print('Refusing to write a report inside the immutable released run')
            return 2
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('CROSSWATCH RELEASE CHECK PASSED:',out['run_id'],
          'variants',len(out['models']),'fingerprinted files',out['artifacts_verified'])
    return 0
if __name__=='__main__':
    raise SystemExit(main())
