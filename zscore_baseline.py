"""Statistical baselines for the three daily market changes."""

import pandas as pd

from crosswatch import FEATURES

ZSCORE_THRESHOLD = 2.0
ROLLING_WINDOW = 60
ZSCORE_METHODS = {
    'fixed_zscore': 'Training-reference z-score',
    'rolling_zscore': 'Rolling z-score',
}


def zscore_baselines(train: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Return maximum absolute z-scores from fixed and past-only references."""
    reference = train.loc[:, list(FEATURES)]
    changes = features.loc[:, list(FEATURES)]
    fixed_std = reference.std(ddof=1)
    fixed_z = (changes - reference.mean()) / fixed_std.where(fixed_std > 0)

    # Include earlier partitions and only observations prior to the date scored.
    history = changes.shift(1).rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
    rolling_std = history.std(ddof=1)
    rolling_z = (changes - history.mean()) / rolling_std.where(rolling_std > 0)

    scores = pd.DataFrame({
        'fixed_zscore': fixed_z.abs().max(axis=1),
        'rolling_zscore': rolling_z.abs().max(axis=1),
    }, index=changes.index)
    for method in ZSCORE_METHODS:
        score = scores[method]
        scores[f'{method}_alert'] = score.gt(ZSCORE_THRESHOLD).astype('boolean').where(score.notna())
    return scores
