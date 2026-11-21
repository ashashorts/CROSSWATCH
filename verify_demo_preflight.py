"""Read-only CROSSWATCH historical/synthetic demo data and plotting preflight.

Verifies all 2 feature variants x 2 evaluation partitions with the same
manifest-verified loader used by Streamlit. Does NOT launch a browser or prove
that Streamlit itself works. No raw provider observations are exported.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from release_integrity import read_manifest
from historical_cases import VARIANTS, load_cases, select_examples, case_figure, matching_rules


def verify(results_dir: Path, raw_dir: Path) -> dict:
    results_dir, raw_dir=Path(results_dir),Path(raw_dir)
    manifest=read_manifest(results_dir,raw_dir=raw_dir)
    expected_tables=('feature_comparison_operating.csv','feature_comparison_budget.csv',
                     'episode_coverage.csv','pairwise_alert_overlap.csv')
    tables={}
    for filename in expected_tables:
        d=pd.read_csv(results_dir/filename)
        if d.empty or set(d['split'])!={'validation','test'}:
            raise ValueError(f'Missing/incomplete comparison table: {filename}')
        tables[filename]={'rows':len(d),'columns':len(d.columns)}
    variants={}
    for variant in VARIANTS:
        records={}
        for partition in ('validation','test'):
            frame,meta,rules,closes=load_cases(results_dir,raw_dir,variant,partition)
            assert not frame.empty and len(frame)==meta['splits'][partition]['n_rows']
            assert all(c in frame.columns for c in ('IFOREST_ALERT','REGIME_ALERT','forward_proxy'))
            examples=select_examples(frame)
            if not examples:
                # Even quiet partitions must be previewable via any scored date.
                dates=[frame.index[len(frame)//2]]
            else:
                dates=list(dict.fromkeys(examples.values()))[:1]
            charts=0
            rule_matches=0
            for day in dates:
                fig=case_figure(frame,meta,closes,day)
                try:
                    if len(fig.axes)!=5:raise ValueError('Expected five-panels historical case chart')
                    fig.canvas.draw()  # catches renderer/plotting errors, no output files
                    charts+=1
                finally:
                    plt.close(fig)
                matches=matching_rules(frame.loc[day],rules,partition)
                rule_matches+=len(matches)
            records[partition]={
                'scoring_rows':len(frame),
                'proxy_evaluable_rows':int(frame['forward_proxy'].notna().sum()),
                'forest_global_alerts_all_scored_days':int(frame['IFOREST_ALERT'].sum()),
                'forest_context_alerts_all_scored_days':int(frame['REGIME_ALERT'].sum()),
                'example_categories':list(examples),
                'charts_rendered_in_memory':charts,
                'example_rule_matches':rule_matches,
                'rules_evaluation_rows':int((rules['split']==partition).sum()),
            }
        variants[variant]=records
    return {
        'check_name':'CROSSWATCH_NON_BROWSER_DEMO_PREFLIGHT',
        'pass':True,'checked_utc':datetime.now(timezone.utc).isoformat(),
        'run_id':manifest['run_id'],'run_kind':manifest['run_kind'],
        'artifact_files_verified':len(manifest['artifacts']),
        'source_hashes_verified':True,
        'comparison_tables':tables,
        'models':variants,
        'live_browser_tested':False,
        'streamlit_installed_or_executed':False,
        'raw_proprietary_data_in_report':False,
        'limitations':['Does not execute Streamlit or verify browser interactions.',
                       'Does not certify the legal scope of provider authorizations.',
                       'All case charts are generated in-memory only.']
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.results_dir.resolve()):
        parser.error('Write preflight outside immutable results directory')
    try:
        result=verify(args.results_dir,args.raw_dir)
    except Exception as exc:
        print('DEMO PREFLIGHT FAILED:',type(exc).__name__,str(exc))
        return 2
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('NON-BROWSER DEMO PREFLIGHT PASSED',result['run_id'],
          'four model/split combinations; browser remains pending')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
