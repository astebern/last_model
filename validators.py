"""
Time-Series Cross-Validation & Evaluation
==========================================
- Expanding-window time-series splits with purge gap
- Approximation of AW-MAE (Augmented Weighted MAE)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time-Series Cross-Validation
# ---------------------------------------------------------------------------

def create_time_series_splits(df, date_col="date", n_splits=5, purge_days=30):
    """
    Create expanding-window time-series splits with purge gap.

    Each fold trains on all data before a cutoff, skips a purge window,
    then validates on data after the purge until the next cutoff.

    Parameters
    ----------
    df : pd.DataFrame with a parseable date column
    date_col : str
    n_splits : int
    purge_days : int, gap between train and validation to prevent leakage

    Returns
    -------
    list of (train_indices, val_indices, fold_info_dict)
    """
    dates = pd.to_datetime(df[date_col])

    # Define cutoffs based on the data range
    min_year = dates.dt.year.min()
    max_year = dates.dt.year.max()

    # For train spanning 1872-2011, use these cutoffs:
    cutoffs = [
        ("2001-01-01", "2003-12-31"),
        ("2003-06-01", "2006-05-31"),
        ("2006-01-01", "2008-12-31"),
        ("2008-06-01", "2010-05-31"),
        ("2010-01-01", "2011-08-31"),
    ]

    # Use only the requested number of splits (from the end for recency)
    cutoffs = cutoffs[-n_splits:]

    splits = []
    for i, (val_start, val_end) in enumerate(cutoffs):
        val_start_dt = pd.Timestamp(val_start)
        val_end_dt = pd.Timestamp(val_end)
        purge_dt = val_start_dt - pd.Timedelta(days=purge_days)

        train_mask = dates < purge_dt
        val_mask = (dates >= val_start_dt) & (dates <= val_end_dt)

        train_idx = df.index[train_mask].tolist()
        val_idx = df.index[val_mask].tolist()

        if len(train_idx) > 0 and len(val_idx) > 0:
            info = {
                "fold": i,
                "train_end": str(purge_dt.date()),
                "val_start": val_start,
                "val_end": val_end,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
            splits.append((train_idx, val_idx, info))

    return splits


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

# Tournament importance weights for AW-MAE
TOURNAMENT_WEIGHTS = {
    5: 2.0,   # World Cup
    4: 1.5,   # Continental championship
    3: 1.2,   # Nations League, Gold Cup, etc.
    2: 1.0,   # Qualifiers
    1: 0.6,   # Friendlies
}


def compute_awmae(y_true_team, y_true_opp, y_pred_team, y_pred_opp,
                   tournament_importance=None):
    """
    Approximate the Augmented Weighted MAE (AW-MAE).

    Components:
    1. Score MAE: |actual - predicted| for each goal column
    2. Result accuracy bonus/penalty: correct W/D/L reduces error
    3. Goal difference accuracy: |actual_gd - pred_gd|
    4. Tournament weighting: important matches weighted more

    Parameters
    ----------
    y_true_team, y_true_opp : array-like, actual goals
    y_pred_team, y_pred_opp : array-like, predicted goals (integers)
    tournament_importance : array-like or None, importance scores (1-5)

    Returns
    -------
    float : AW-MAE score (lower is better)
    """
    y_true_team = np.asarray(y_true_team, dtype=float)
    y_true_opp = np.asarray(y_true_opp, dtype=float)
    y_pred_team = np.asarray(y_pred_team, dtype=float)
    y_pred_opp = np.asarray(y_pred_opp, dtype=float)

    n = len(y_true_team)

    # 1. Base score MAE
    mae_team = np.abs(y_true_team - y_pred_team)
    mae_opp = np.abs(y_true_opp - y_pred_opp)
    score_error = (mae_team + mae_opp) / 2.0

    # 2. Match result (W/D/L)
    actual_result = np.sign(y_true_team - y_true_opp)
    pred_result = np.sign(y_pred_team - y_pred_opp)
    result_correct = (actual_result == pred_result).astype(float)
    result_penalty = 1.0 - 0.3 * result_correct  # 30% reduction if result correct

    # 3. Goal difference accuracy
    actual_gd = y_true_team - y_true_opp
    pred_gd = y_pred_team - y_pred_opp
    gd_error = np.abs(actual_gd - pred_gd) * 0.2  # 20% weight on GD

    # 4. Tournament weights
    if tournament_importance is not None:
        tournament_importance = np.asarray(tournament_importance)
        weights = np.array([TOURNAMENT_WEIGHTS.get(int(t), 1.0)
                           for t in tournament_importance])
    else:
        weights = np.ones(n)

    # Combine: augmented error per sample
    augmented_error = (score_error * result_penalty + gd_error) * weights

    return augmented_error.mean()


def compute_simple_mae(y_true_team, y_true_opp, y_pred_team, y_pred_opp):
    """Simple MAE baseline for comparison."""
    mae_team = np.mean(np.abs(np.asarray(y_true_team) - np.asarray(y_pred_team)))
    mae_opp = np.mean(np.abs(np.asarray(y_true_opp) - np.asarray(y_pred_opp)))
    return (mae_team + mae_opp) / 2.0


def compute_result_accuracy(y_true_team, y_true_opp, y_pred_team, y_pred_opp):
    """Percentage of matches where W/D/L is predicted correctly."""
    actual = np.sign(np.asarray(y_true_team) - np.asarray(y_true_opp))
    pred = np.sign(np.asarray(y_pred_team) - np.asarray(y_pred_opp))
    return np.mean(actual == pred)
