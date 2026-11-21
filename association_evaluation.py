"""Audit training-mined alert rules on chronological holdouts."""

from itertools import combinations
import math

import pandas as pd

from crosswatch import ITEMS, apriori_frequent_itemsets, mine_alert_rules

VOLATILITY_FLAG = 'ELEVATED_RECENT_VOLATILITY'
CONTEXT_FLAGS = (*ITEMS[:3], VOLATILITY_FLAG)
MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT = 0.01, 0.10, 1.0
ALERTS = ('IFOREST_ALERT', 'REGIME_ALERT')


def rule_counts(frame: pd.DataFrame, antecedent: tuple, consequent: str) -> dict:
    condition = frame.loc[:, list(antecedent)].all(axis=1)
    alert = frame[consequent]
    n, a, b, both = len(frame), int(condition.sum()), int(alert.sum()), int((condition & alert).sum())
    confidence = both / a if a else math.nan
    return {
        'days': n, 'antecedent_count': a, 'alert_count': b, 'joint_count': both,
        'support': both / n, 'confidence': confidence,
        'lift': confidence / (b / n) if b else math.nan,
    }


def evaluate_associations(scored: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Freeze detection context bins on eligible training rows and audit both consequents."""
    training_volatility = scored['train']['trailing_5d_return_vol_pct'].dropna()
    if training_volatility.empty:
        raise ValueError('Need valid trailing volatility on training dates.')
    cutoff = float(training_volatility.quantile(0.95))
    transactions = {}
    for split, data in scored.items():
        usable = data.loc[data['trailing_5d_return_vol_pct'].notna()]
        if usable.empty:
            raise ValueError(f'{split}: no transactions with complete volatility history.')
        transactions[split] = usable.loc[:, [*ITEMS, 'REGIME_ALERT']].assign(
            **{VOLATILITY_FLAG: usable['trailing_5d_return_vol_pct'].ge(cutoff)}
        )
    train = transactions['train']
    rows, summaries = [], {}
    try:
        from mlxtend.frequent_patterns import apriori
    except ImportError:
        apriori = None

    for consequent in ALERTS:
        items = (*CONTEXT_FLAGS, consequent)
        baskets = train.loc[:, list(items)]
        mined = apriori_frequent_itemsets(baskets, MIN_SUPPORT, items=items)
        actual = {frozenset(row['items']): row['count'] for row in mined}
        oracle = {}
        for size in range(1, len(items) + 1):
            for keys in combinations(items, size):
                count = int(baskets.loc[:, list(keys)].all(axis=1).sum())
                if count / len(baskets) >= MIN_SUPPORT:
                    oracle[frozenset(keys)] = count
        if oracle != actual:
            raise ValueError(f'{consequent}: Apriori disagrees with exhaustive itemset counts.')
        if apriori is not None:
            library = apriori(baskets, min_support=MIN_SUPPORT, use_colnames=True)
            library_counts = {keys: round(support * len(baskets)) for support, keys in library.itertuples(index=False)}
            if library_counts != oracle:
                raise ValueError(f'{consequent}: mlxtend disagrees with exhaustive counts.')

        rules = mine_alert_rules(
            baskets, MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT, items=items, consequent=consequent
        )
        expected = set()
        for flags in oracle:
            if consequent in flags and len(flags) > 1:
                antecedent = tuple(sorted(flags - {consequent}))
                metrics = rule_counts(train, antecedent, consequent)
                if metrics['confidence'] >= MIN_CONFIDENCE and metrics['lift'] >= MIN_LIFT:
                    expected.add('+'.join(antecedent))
        if set(rules['antecedent']) != expected:
            raise ValueError(f'{consequent}: rule list does not match exhaustive counts.')

        for rule in rules.itertuples(index=False):
            antecedent = tuple(rule.antecedent.split('+'))
            measured = rule_counts(train, antecedent, consequent)
            for metric in ('support', 'confidence', 'lift', 'joint_count'):
                if not math.isclose(measured[metric], getattr(rule, metric), abs_tol=1e-12):
                    raise ValueError(f'{consequent}: incorrect {metric} for {rule.antecedent}')
            subset_confidences = [
                rule_counts(train, subset, consequent)['confidence']
                for size in range(1, len(antecedent))
                for subset in combinations(antecedent, size)
            ]
            gain = measured['confidence'] - max(subset_confidences) if subset_confidences else math.nan
            for split, frame in transactions.items():
                metrics = rule_counts(frame, antecedent, consequent)
                rows.append({
                    'consequent': consequent, 'antecedent': rule.antecedent, 'split': split,
                    **metrics, 'training_confidence_gain_over_best_subset': gain,
                    'adds_training_confidence': not subset_confidences or gain > 1e-12,
                    'meets_count_screen': metrics['antecedent_count'] >= 10 and metrics['joint_count'] >= 5,
                })
        summaries[consequent] = {
            'frequent_itemsets_checked': len(actual), 'training_rules_checked': len(rules),
        }
    return pd.DataFrame(rows), {
        'training_volatility_95th_percentile': cutoff, 'flags': list(CONTEXT_FLAGS),
        'transactions_by_split': {split: len(frame) for split, frame in transactions.items()},
        'thresholds': {'support': MIN_SUPPORT, 'confidence': MIN_CONFIDENCE, 'lift': MIN_LIFT},
        'exhaustive_count_verification': True, 'mlxtend_verification': apriori is not None,
        'consequents': summaries,
    }
