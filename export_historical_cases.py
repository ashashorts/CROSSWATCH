"""Render representative case figures from one fingerprint-verified run."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from historical_cases import LABELS, SIGNALS, case_figure, forward_path, load_cases, select_examples


def export_cases(results_dir: Path, equity_closes: pd.Series, output: Path,
                 *, variant: str='five_features', split: str='test') -> pd.DataFrame:
    frame, metadata, _, closes = load_cases(results_dir, None, variant, split,
                                            equity_closes=equity_closes)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, (category, date) in enumerate(select_examples(frame).items(), 1):
        row = frame.loc[date]
        path = forward_path(closes, date)
        figure = case_figure(frame, metadata, closes, date)
        name = f'case_{number}_{date:%Y-%m-%d}.png'
        figure.savefig(output / name, dpi=150, facecolor='white')
        plt.close(figure)
        rows.append({
            'case':category,'date':date.date(),
            **{key: row[key] for key in SIGNALS},
            **{key: row[key] for key in LABELS if key in row.index},
            'global_cutoff': metadata['global_severity_cutoff_95th'],
            **{key:row[key] for key in ('anomaly_severity','regime_severity_cutoff',
                    'vix_regime','cross_asset_confirmation','explanation','forward_proxy')},
            'worst_next_five_equity_close_pct':path.iloc[1:].min() if len(path)==6 else None,
            'figure':name,
        })
    result = pd.DataFrame(rows)
    result.to_csv(output / 'cases.csv', index=False)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,required=True)
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=('three_features','five_features'),default='five_features')
    p.add_argument('--split',choices=('validation','test'),default='test')
    a=p.parse_args(argv)
    from release_integrity import read_manifest,write_manifest
    m=read_manifest(a.results_dir,raw_dir=a.raw_dir)
    from crosswatch import read_fred_csv
    closes=read_fred_csv('SP500',a.raw_dir,allow_download=False)
    # Exports outside the run root do not mutate the fingerprinted release.
    if a.output.resolve().is_relative_to(a.results_dir.resolve()):
        raise ValueError('Run artifacts are immutable: use an output folder outside the run directory')
    out=export_cases(a.results_dir,closes,a.output,variant=a.variant,split=a.split)
    print('Exported',len(out),'cases from',m['run_id'],'to',a.output)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
