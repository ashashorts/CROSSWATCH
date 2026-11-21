"""CLI for CROSSWATCH Phase 3 Step 3 diagnostic evaluation.

The synthetic check is NOT financial evidence. A real run must first satisfy
rights, provenance, and source quality using approved_real_run.run().
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from crosswatch import generate_synthetic_levels
from crosswatch_phase3 import run_phase3_part1
from approved_real_run import run as approved_market_run
from experiment_audit import audit_all
from stress_fixtures import alternating_volatility_fixture


def _plot_metrics(analysis: dict,out: Path,tag: str):
    df=analysis['operating']
    fig,ax=plt.subplots(figsize=(9,4.7))
    wide=df.pivot(index='split',columns='method',values='proxy_hit_fraction')
    wide.plot.bar(ax=ax,rot=0,width=.73)
    ax.set_ylabel('Fraction of alerts matching forward-loss proxy')
    ax.set_xlabel('Split')
    ax.set_ylim(0,1)
    ax.set_title(f'{tag} — software test, not historical model performance' if tag.startswith('SYNTHETIC') else f'{tag} — out-of-sample proxy association')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=8)
    ax.grid(axis='y',alpha=.18)
    fig.tight_layout();fig.savefig(out/f'{tag}_proxy_hit_fraction.png',dpi=175,bbox_inches='tight');plt.close(fig)
    regime=analysis['regime']
    fig,ax=plt.subplots(figsize=(7,3.7))
    ax.bar(regime['split'],regime['ordinary_vix_days'],label='Ordinary VIX')
    ax.bar(regime['split'],regime['elevated_vix_days'],bottom=regime['ordinary_vix_days'],label='Elevated VIX')
    ax.set_title(f'{tag} — regime distribution diagnostics')
    ax.set_ylabel('Observed days');ax.legend();ax.grid(axis='y',alpha=.18)
    fig.tight_layout();fig.savefig(out/f'{tag}_regime_diagnostics.png',dpi=175);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['synthetic-check','approved-real'],required=True)
    parser.add_argument('--synthetic-scenario',choices=['random-walk','alternating-regimes'],default='random-walk')
    parser.add_argument('--out-dir',type=Path,default=Path('evidence/stage3_run'))
    parser.add_argument('--raw-dir',type=Path,default=Path('private_raw_data'))
    parser.add_argument('--rights-file',type=Path,default=Path('docs/license_review.json'))
    parser.add_argument('--origin-file',type=Path,default=Path('docs/origin_manifest.json'))
    parser.add_argument('--start',default='2017-01-03')
    parser.add_argument('--end',default='2026-11-18')
    args=parser.parse_args(argv)
    try:
        if args.mode=='synthetic-check':
            args.out_dir.mkdir(parents=True,exist_ok=True)
            label='SYNTHETIC_QA_ONLY'
            fixture=alternating_volatility_fixture() if args.synthetic_scenario=='alternating-regimes' else generate_synthetic_levels(n=1400,seed=2026)
            results=run_phase3_part1(fixture,
                        args.out_dir,source_type=label,
                        extra_provenance={'purpose':'Software QA only. Generated random values, not FRED observations.', 'synthetic_scenario':args.synthetic_scenario})
        else:
            from run_fred_historical import run_authorized
            m = run_authorized(args.raw_dir,args.out_dir,args.rights_file,args.origin_file,args.start,args.end)
            print('COMPLETE HISTORICAL PIPELINE FINISHED:',m['run_id'])
            print('ARTIFACTS:',args.out_dir.resolve())
            return 0
        qa=audit_all(results,args.out_dir,label)
        _plot_metrics(qa,args.out_dir,label)
        print('EXPERIMENT STAGE FINISHED:',label)
        print('OPERATING POINTS (forward loss proxy only):')
        print(qa['operating'][['split','method','evaluable_days','alerts','proxy_hits','proxy_hit_fraction','proxy_positive_episodes']].to_string(index=False))
        print('REGIME COVERAGE:')
        print(qa['regime'].to_string(index=False))
        print('RULE STABILITY ROWS:',len(qa['rules']))
        print('ARTIFACTS:',args.out_dir.resolve())
        return 0
    except Exception as e:
        print('EXPERIMENT STAGE BLOCKED:',type(e).__name__,str(e))
        return 2

if __name__=='__main__':raise SystemExit(main())
