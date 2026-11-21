"""Historical case selection and visualization for the local demonstration."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from association_evaluation import CONTEXT_FLAGS, VOLATILITY_FLAG
from crosswatch import read_fred_csv
from crosswatch_phase3 import future_five_day_decline_proxy
from data_quality_gate import sha256

VARIANTS = ('three_features', 'five_features')
LABELS = {
    'baseline_alert': 'Equity-tail rule',
    'fixed_zscore_alert': 'Training-reference z-score',
    'rolling_zscore_alert': 'Rolling z-score',
    'IFOREST_ALERT': 'Global Isolation Forest',
    'REGIME_ALERT': 'VIX-context Isolation Forest',
}
SIGNALS = {
    'sp_return_pct': 'S&P 500 return (%)',
    'vix_delta_points': 'VIX change (points)',
    'yield_delta_bp': '10-year yield change (bp)',
}


def load_cases(results_dir: Path, raw_dir: Path | None, variant: str, split: str,
               *, equity_closes: pd.Series | None = None):
    """Load only a manifest-verified run; check ALL input CSVs when raw_dir is given.

    A private in-memory equity series is accepted solely for the runner's case
    exporter. In that mode the runner must independently verify the three files.
    """
    from release_integrity import read_manifest, ReleaseIntegrityError
    if variant not in VARIANTS or split not in ('train', 'validation', 'test'):
        raise ValueError('Unsupported model variant or evaluation split')
    manifest = read_manifest(results_dir, raw_dir=raw_dir)
    folder = Path(results_dir) / manifest['variants'][variant]['directory']
    label = manifest['source_label']
    frame = pd.read_csv(
        folder / f'{label}_{split}_derived_results.csv',
        index_col='DATE', parse_dates=['DATE'],
    )
    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ReleaseIntegrityError('Scored dates must be nonempty, unique, and ordered')
    metadata = json.loads((folder / f'{label}_metadata.json').read_text())
    if metadata.get('source_type') != label:
        raise ReleaseIntegrityError('Model and manifest source labels disagree')
    if metadata.get('extra_provenance', {}).get('sha256') != manifest['input_sha256']:
        raise ReleaseIntegrityError('Model source hashes and run manifest disagree')
    if len(frame) != metadata['splits'][split]['n_rows']:
        raise ReleaseIntegrityError('Model metadata partition size mismatch')
    expected_cols = {'anomaly_severity', 'regime_severity_cutoff','vix_regime',
                     'cross_asset_confirmation', 'forward_proxy', 'IFOREST_ALERT',
                     'REGIME_ALERT', 'baseline_alert', 'sp_return_pct',
                     'vix_delta_points', 'yield_delta_bp', 'trailing_5d_return_vol_pct'}
    if {'SP500','VIXCLS','DGS10'}.intersection(frame.columns):
        raise ReleaseIntegrityError('Raw proprietary source levels were embedded in scored outputs')
    if not expected_cols.issubset(frame.columns):
        raise ReleaseIntegrityError(f'Missing scoring columns: {sorted(expected_cols - set(frame.columns))}')
    rules = pd.read_csv(folder / 'association_rules.csv')
    if not {'antecedent','consequent','split'}.issubset(rules.columns):
        raise ReleaseIntegrityError('Association-rule table missing expected columns')
    rule_audit = json.loads((folder / 'association_verification.json').read_text())
    if equity_closes is not None:
        if raw_dir is not None:
            raise ValueError('Supply either raw_dir or pre-verified equity_closes, not both')
        closes = equity_closes.dropna().sort_index()
    else:
        if raw_dir is None:
            raise ValueError('The dashboard requires all three original private-input CSVs')
        closes = read_fred_csv('SP500', raw_dir, allow_download=False).dropna()
    closes = closes.loc[frame.index[0]:frame.index[-1]]
    if frame.index.difference(closes.index).size:
        raise ReleaseIntegrityError('Model scoring dates missing from the source S&P 500 sessions')
    observed = future_five_day_decline_proxy(closes.to_frame('SP500')).reindex(frame.index)
    supplied = frame['forward_proxy'].astype('boolean')
    pd.testing.assert_series_equal(observed, supplied, check_names=False)
    frame['forward_proxy'] = supplied
    frame[VOLATILITY_FLAG] = frame['trailing_5d_return_vol_pct'].ge(
        rule_audit['training_volatility_95th_percentile']
    )
    return frame, metadata, rules, closes


def select_examples(frame: pd.DataFrame) -> dict[str, pd.Timestamp]:
    global_alert = frame['IFOREST_ALERT']
    context_alert = frame['REGIME_ALERT']
    groups = {
        'Both forests alert': global_alert & context_alert,
        'Global forest only': global_alert & ~context_alert,
        'VIX-context forest only': context_alert & ~global_alert,
        'No forest alert before proxy loss': ~global_alert & ~context_alert & frame['forward_proxy'].fillna(False),
    }
    return {
        label: frame.loc[mask, 'anomaly_severity'].idxmax()
        for label, mask in groups.items() if mask.any()
    }


def forward_path(closes: pd.Series, date: pd.Timestamp) -> pd.Series:
    position = closes.index.get_loc(date)
    window = closes.iloc[position:position + 6]
    result = 100 * (window / window.iloc[0] - 1)
    result.index = pd.RangeIndex(len(result), name='Equity sessions after selected close')
    return result


def matching_rules(row: pd.Series, rules: pd.DataFrame, split: str) -> pd.DataFrame:
    active = {flag for flag in CONTEXT_FLAGS if row[flag]}
    current = rules.loc[rules['split'].eq(split)].copy()
    current = current.loc[current['antecedent'].map(
        lambda items: set(items.split('+')).issubset(active)
    )].copy()
    current['alert_on_selected_date'] = current['consequent'].map(
        {method: row[method] for method in ('IFOREST_ALERT', 'REGIME_ALERT')}
    )
    return current


def case_figure(frame, metadata, closes, date):
    """Show scored history and a retrospective forward path on their respective clocks."""
    center = frame.index.get_loc(date)
    history = frame.iloc[max(0, center - 10):center + 11]
    selected = frame.loc[date]
    global_cutoff = metadata['global_severity_cutoff_95th']
    path = forward_path(closes, date)
    fig, axes = plt.subplots(5, 1, figsize=(10.2, 10.0), layout='constrained')
    fig.suptitle(f'{date:%B %d, %Y}   |   {selected.vix_regime}   |   Cross-asset score {selected.cross_asset_confirmation}/3',
                 x=.05, ha='left', fontsize=16, fontweight='bold')
    for ax, (column, label) in zip(axes[:3], SIGNALS.items()):
        ax.bar(history.index, history[column], color=[
            '#b85032' if day == date else '#2f6175' for day in history.index
        ], width=.85)
        ax.set_ylabel(label)
        ax.axhline(0, color='#8d9599', linewidth=.8)
        ax.grid(axis='y', alpha=.22)
        ax.tick_params(axis='x', labelbottom=False)
    score_ax = axes[3]
    score_ax.plot(history.index, history['anomaly_severity'], color='#153947', label='Forest severity')
    score_ax.axhline(global_cutoff, color='#62717b', ls='--', label='Global cutoff')
    score_ax.step(history.index, history['regime_severity_cutoff'], where='mid',
                  color='#b85032', ls=':', label='VIX-context cutoff')
    score_ax.scatter([date], [selected.anomaly_severity], color='#b85032', zorder=4)
    score_ax.set_ylabel('Severity')
    score_ax.legend(loc='best', ncol=3, fontsize=8)
    score_ax.grid(axis='y', alpha=.22)
    forward_ax = axes[4]
    forward_ax.plot(path.index, path, color='#2f6175', marker='o')
    forward_ax.axhline(-2, color='#b85032', ls='--', label='−2% outcome threshold')
    forward_ax.set(xlabel=path.index.name, ylabel='S&P 500 change (%)', xticks=range(6))
    forward_ax.legend(loc='best', fontsize=8)
    forward_ax.grid(alpha=.22)
    for ax in axes[:4]:
        ax.axvline(date, color='#b85032', alpha=.45, linewidth=1)
        ax.set_xlim(history.index.min() - pd.Timedelta(days=1), history.index.max() + pd.Timedelta(days=1))
    return fig
