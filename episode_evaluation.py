"""Compare alert overlap and coverage of contiguous equity-loss proxy episodes."""

from itertools import combinations
from math import ceil

import pandas as pd

from crosswatch_phase3 import alert_rankings, future_five_day_decline_proxy
from zscore_baseline import ZSCORE_METHODS


def first_loss_lead(closes: pd.Series) -> pd.Series:
    """Sessions to the first qualifying close for fully observed windows."""
    evaluation = future_five_day_decline_proxy(closes.to_frame('SP500'))
    leads = pd.Series(pd.NA, index=closes.index, dtype='Int64')
    for day in range(1, 6):
        hit = evaluation.fillna(False) & closes.shift(-day).le(0.98 * closes)
        leads.loc[leads.isna() & hit] = day
    return leads


def evaluate_episodes(outputs: dict, equity_closes: pd.Series) -> dict[str, pd.DataFrame]:
    overlap, coverage = [], []
    closes = equity_closes.dropna().sort_index()
    for split in ('validation', 'test'):
        scored = {variant: result['scored'][split] for variant, result in outputs.items()}
        reference = next(iter(scored.values()))
        labels = reference['forward_proxy']
        if labels.isna().iloc[:labels.notna().sum()].any():
            raise ValueError('Missing outcomes inside the evaluated date range.')
        dates = labels.dropna().index
        positive = labels.loc[dates].astype(bool)
        starts = positive & ~positive.shift(fill_value=False)
        episodes = starts.cumsum().where(positive)
        lead = first_loss_lead(closes.loc[reference.index[0]:reference.index[-1]]).reindex(dates)
        if not lead.notna().equals(positive):
            raise ValueError('Episode timing must match the historical outcome labels.')

        fixed = {'baseline': reference.loc[dates, 'baseline_alert']}
        fixed.update({
            method: reference.loc[dates, f'{method}_alert']
            for method in ZSCORE_METHODS if method in reference
        })
        rankings = alert_rankings(reference.loc[dates])
        for variant, frame in scored.items():
            pd.testing.assert_series_equal(frame['forward_proxy'], labels)
            variant_ranks = alert_rankings(frame.loc[dates])
            for method, column in (
                ('global_iforest', 'IFOREST_ALERT'), ('regime_iforest', 'REGIME_ALERT')
            ):
                key = f'{variant}_{method}'
                fixed[key] = frame.loc[dates, column]
                rankings[key] = variant_ranks[method]
        rankings = {key: rankings[key] for key in fixed}
        budget = ceil(0.05 * len(dates))
        matched = {
            key: pd.Series(dates.isin(rank.iloc[:budget].index), index=dates)
            for key, rank in rankings.items()
        }

        for mode, selections in (('fixed_threshold', fixed), ('top_5pct', matched)):
            for method, mask in selections.items():
                mask = mask.astype(bool)
                first_alerts = episodes.loc[mask & positive].drop_duplicates().index
                leads = lead.loc[first_alerts]
                coverage.append({
                    'split': split, 'mode': mode, 'method': method,
                    'evaluable_days': len(dates), 'alerts': int(mask.sum()),
                    'positive_days_alerted': int((mask & positive).sum()),
                    'proxy_episodes': int(starts.sum()),
                    'episodes_alerted': len(first_alerts),
                    'episode_coverage': len(first_alerts) / starts.sum() if starts.any() else float('nan'),
                    'median_equity_sessions_to_loss': leads.median() if len(leads) else float('nan'),
                    **{f'lead_{day}_sessions': int(leads.eq(day).sum()) for day in range(1, 6)},
                })
            for (left, a), (right, b) in combinations(selections.items(), 2):
                intersection, union = int((a & b).sum()), int((a | b).sum())
                overlap.append({
                    'split': split, 'mode': mode, 'method_a': left, 'method_b': right,
                    'both_alert': intersection, 'either_alert': union,
                    'only_a': int((a & ~b).sum()), 'only_b': int((b & ~a).sum()),
                    'jaccard': intersection / union if union else float('nan'),
                })
    return {'episode_coverage': pd.DataFrame(coverage), 'pairwise_alert_overlap': pd.DataFrame(overlap)}
