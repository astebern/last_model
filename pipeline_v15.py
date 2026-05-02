"""
Solution V2: Robust International Football Score Prediction
============================================================
KEY FIXES vs V1:
  1. NO fillna(-999)  → LightGBM handles NaN natively
  2. NO iterative test prediction → eliminates error accumulation
  3. Vectorized pandas ops → O(n) instead of O(n²)
  4. Bayesian-smoothed team encodings → robust for rare teams
  5. Single file → easy to debug and submit

Strategy:
  - Elo: sequential loop on match-level data (fast: ~39K matches),
         carry forward final Elo to test
  - Team/Opponent encoding: expanding mean via groupby.transform,
         final aggregate for test
  - H2H: expanding cumulative via groupby(['team','opponent']),
         final aggregate for test
  - Shared features: temporal, geographic, socio-economic, tournament
  - Model: LightGBM + XGBoost Poisson ensemble
  - Rounding: simple clip + round, consistency enforcement via match_id

Usage:  python pipeline_v2.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "submission_v15.csv")

RUN_TUNING = False
TUNING_TRIALS = 30

RANDOM_STATE = 42
INITIAL_ELO = 1500
HOME_ADV = 100
PRIOR_WEIGHT = 10        # Bayesian smoothing strength

LGB_PARAMS = dict(
    objective="poisson", metric="mae", boosting_type="gbdt",
    n_estimators=3000, learning_rate=0.02,
    num_leaves=31, max_depth=6, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=2.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)

XGB_PARAMS = dict(
    objective="count:poisson", eval_metric="mae",
    n_estimators=3000, learning_rate=0.02,
    max_depth=6, min_child_weight=50,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=2.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)

W_LGB, W_XGB = 0.8, 0.2   # ensemble weights


# ============================================================================
# ELO SIMULATOR  (sequential but efficient — match-level, not row-level)
# ============================================================================

TOURNAMENT_K = {
    "fifa world cup": 60, "confederations cup": 55,
    "european championship": 50, "copa amér": 50, "copa amer": 50,
    "african cup of nations": 50, "africa cup of nations": 50,
    "afc asian cup": 50, "concacaf gold cup": 45, "ofc nations cup": 40,
    "world cup qualification": 40, "euro qualification": 40,
    "nations league": 35, "friendly": 20,
}


def _get_k(tournament):
    t = tournament.lower()
    for pattern, k in TOURNAMENT_K.items():
        if pattern in t:
            return k
    return 30


def _gd_mult(gd):
    gd = abs(gd)
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    return (11.0 + gd) / 8.0


def compute_elo(matches_df):
    """
    Compute Elo for every match.

    Parameters
    ----------
    matches_df : DataFrame with one row per match, columns:
        match_id, gender, team_a, team_b, goals_a, goals_b,
        is_home_a, neutral, tournament

    Returns
    -------
    elo_before : dict  (match_id, team) -> elo_before_this_match
    final_elos : dict  (gender, team) -> final elo after all matches
    """
    ratings = {}     # (gender, team) -> elo
    elo_before = {}  # (match_id, team) -> elo before match

    for row in matches_df.itertuples(index=False):
        g = row.gender
        ta, tb = row.team_a, row.team_b
        ga, gb = int(row.goals_a), int(row.goals_b)

        ea = ratings.get((g, ta), INITIAL_ELO)
        eb = ratings.get((g, tb), INITIAL_ELO)
        elo_before[(row.match_id, ta)] = ea
        elo_before[(row.match_id, tb)] = eb

        # home advantage adjustment
        if row.neutral:
            ea_adj, eb_adj = ea, eb
        elif row.is_home_a:
            ea_adj, eb_adj = ea + HOME_ADV, eb
        else:
            ea_adj, eb_adj = ea, eb + HOME_ADV

        exp_a = 1.0 / (1.0 + 10.0 ** ((eb_adj - ea_adj) / 400.0))
        actual_a = 1.0 if ga > gb else (0.5 if ga == gb else 0.0)

        k = _get_k(row.tournament)
        gm = _gd_mult(ga - gb)
        delta = k * gm * (actual_a - exp_a)

        ratings[(g, ta)] = ea + delta
        ratings[(g, tb)] = eb - delta

    return elo_before, ratings


def extract_match_level(df):
    """
    Convert paired-row format to one row per match for Elo.
    Uses groupby on match_id; takes first/last rows per group.
    """
    grouped = df.sort_values(["date", "match_id"]).groupby("match_id", sort=False)

    matches = pd.DataFrame({
        "match_id": grouped["match_id"].first(),
        "date": grouped["date"].first(),
        "gender": grouped["gender"].first(),
        "team_a": grouped["team"].first(),
        "team_b": grouped["team"].last(),
        "goals_a": grouped["team_goals"].first(),
        "goals_b": grouped["team_goals"].last(),
        "is_home_a": grouped["is_home"].first(),
        "neutral": grouped["neutral"].first(),
        "tournament": grouped["tournament"].first(),
    }).reset_index(drop=True)

    return matches.sort_values("date").reset_index(drop=True)


# ============================================================================
# TEAM ENCODING  (vectorized via groupby — Bayesian smoothed)
# ============================================================================

def compute_team_encoding_train(df, global_avg_gf, global_avg_ga, global_wr):
    """
    Compute per-row, per-team expanding statistics (shifted by 1).
    Bayesian-smoothed to handle teams with few matches.
    """
    df = df.copy()
    df["is_win"] = (df["team_goals"] > df["opp_goals"]).astype(int)
    df["is_draw"] = (df["team_goals"] == df["opp_goals"]).astype(int)
    df["gd"] = df["team_goals"] - df["opp_goals"]

    grp = ["gender", "team"]

    # Cumulative sums (shifted = only past matches)
    for col, src in [("_cgf", "team_goals"), ("_cga", "opp_goals"),
                     ("_cgd", "gd"), ("_cw", "is_win")]:
        df[col] = df.groupby(grp)[src].transform(lambda x: x.cumsum().shift(1))

    df["_cn"] = df.groupby(grp).cumcount()  # 0 for first match of each team

    # Bayesian-smoothed averages
    n = df["_cn"]
    df["team_enc_goals"] = (df["_cgf"].fillna(0) + PRIOR_WEIGHT * global_avg_gf) / (n + PRIOR_WEIGHT)
    df["team_enc_conceded"] = (df["_cga"].fillna(0) + PRIOR_WEIGHT * global_avg_ga) / (n + PRIOR_WEIGHT)
    df["team_enc_gd"] = df["team_enc_goals"] - df["team_enc_conceded"]
    df["team_enc_wr"] = (df["_cw"].fillna(0) + PRIOR_WEIGHT * global_wr) / (n + PRIOR_WEIGHT)
    df["team_enc_n"] = n

    # Rolling recent form (last 10, shifted) — captures current form
    for col, src in [("team_recent_goals", "team_goals"),
                     ("team_recent_conceded", "opp_goals"),
                     ("team_recent_wr", "is_win")]:
        df[col] = df.groupby(grp)[src].transform(
            lambda x: x.rolling(10, min_periods=1).mean().shift(1)
        )
    df["team_recent_gd"] = df["team_recent_goals"] - df["team_recent_conceded"]

    # Days since last match
    df["team_prev_date"] = df.groupby(grp)["date"].shift(1)
    df["team_days_since"] = (df["date"] - df["team_prev_date"]).dt.days

    # Cleanup temp columns
    df.drop(columns=["_cgf", "_cga", "_cgd", "_cw", "_cn",
                      "team_prev_date", "is_win", "is_draw", "gd"],
            inplace=True, errors="ignore")

    return df


def compute_opp_encoding_train(df):
    """
    Get opponent features by merging with the partner row via match_id.
    For match M001: row A (team=Brazil) gets row B (team=Argentina)'s team_ stats.
    """
    team_cols = ["team_enc_goals", "team_enc_conceded", "team_enc_gd",
                 "team_enc_wr", "team_enc_n",
                 "team_recent_goals", "team_recent_conceded",
                 "team_recent_wr", "team_recent_gd", "team_days_since"]

    lookup = df[["match_id", "team"] + team_cols].copy()
    rename = {"team": "opponent"}
    for c in team_cols:
        rename[c] = c.replace("team_", "opp_")
    lookup = lookup.rename(columns=rename)

    df = df.merge(lookup, on=["match_id", "opponent"], how="left")
    return df


def compute_team_encoding_test(train, test, global_avg_gf, global_avg_ga, global_wr):
    """
    For test: merge static team-level stats (final from train) per (gender, team).
    """
    train_cp = train.copy()
    train_cp["is_win"] = (train_cp["team_goals"] > train_cp["opp_goals"]).astype(int)

    # --- All-time stats ---
    stats = train_cp.groupby(["gender", "team"]).agg(
        _gf=("team_goals", "sum"), _ga=("opp_goals", "sum"),
        _w=("is_win", "sum"), _n=("team_goals", "count"),
    ).reset_index()

    stats["team_enc_goals"] = (stats["_gf"] + PRIOR_WEIGHT * global_avg_gf) / (stats["_n"] + PRIOR_WEIGHT)
    stats["team_enc_conceded"] = (stats["_ga"] + PRIOR_WEIGHT * global_avg_ga) / (stats["_n"] + PRIOR_WEIGHT)
    stats["team_enc_gd"] = stats["team_enc_goals"] - stats["team_enc_conceded"]
    stats["team_enc_wr"] = (stats["_w"] + PRIOR_WEIGHT * global_wr) / (stats["_n"] + PRIOR_WEIGHT)
    stats["team_enc_n"] = stats["_n"]

    # --- Recent form (last 10 matches per team) ---
    recent = train_cp.sort_values("date").groupby(["gender", "team"]).tail(10)
    recent_stats = recent.groupby(["gender", "team"]).agg(
        team_recent_goals=("team_goals", "mean"),
        team_recent_conceded=("opp_goals", "mean"),
        team_recent_wr=("is_win", "mean"),
    ).reset_index()
    recent_stats["team_recent_gd"] = recent_stats["team_recent_goals"] - recent_stats["team_recent_conceded"]

    keep = ["gender", "team", "team_enc_goals", "team_enc_conceded", "team_enc_gd",
            "team_enc_wr", "team_enc_n"]
    team_final = stats[keep].merge(recent_stats, on=["gender", "team"], how="left")

    # --- Days since last match ---
    last_date = train_cp.groupby(["gender", "team"])["date"].max().reset_index()
    last_date.columns = ["gender", "team", "_last_date"]
    team_final = team_final.merge(last_date, on=["gender", "team"], how="left")

    # Merge as TEAM features
    test = test.merge(team_final, on=["gender", "team"], how="left")
    test["team_days_since"] = (test["date"] - test["_last_date"]).dt.days.clip(upper=365)
    test.drop(columns=["_last_date"], inplace=True, errors="ignore")

    # Fill unseen teams with global mean
    test["team_enc_goals"] = test["team_enc_goals"].fillna(global_avg_gf)
    test["team_enc_conceded"] = test["team_enc_conceded"].fillna(global_avg_ga)
    test["team_enc_gd"] = test["team_enc_gd"].fillna(0)
    test["team_enc_wr"] = test["team_enc_wr"].fillna(global_wr)
    test["team_enc_n"] = test["team_enc_n"].fillna(0)
    test["team_recent_goals"] = test["team_recent_goals"].fillna(global_avg_gf)
    test["team_recent_conceded"] = test["team_recent_conceded"].fillna(global_avg_ga)
    test["team_recent_wr"] = test["team_recent_wr"].fillna(global_wr)
    test["team_recent_gd"] = test["team_recent_gd"].fillna(0)

    # Merge as OPPONENT features
    opp_final = team_final.drop(columns=["_last_date"], errors="ignore").copy()
    opp_rename = {"team": "opponent"}
    for c in opp_final.columns:
        if c.startswith("team_"):
            opp_rename[c] = c.replace("team_", "opp_")
    opp_final = opp_final.rename(columns=opp_rename)

    test = test.merge(opp_final, on=["gender", "opponent"], how="left")

    # Fill unseen opponents
    test["opp_enc_goals"] = test["opp_enc_goals"].fillna(global_avg_gf)
    test["opp_enc_conceded"] = test["opp_enc_conceded"].fillna(global_avg_ga)
    test["opp_enc_gd"] = test["opp_enc_gd"].fillna(0)
    test["opp_enc_wr"] = test["opp_enc_wr"].fillna(global_wr)
    test["opp_enc_n"] = test["opp_enc_n"].fillna(0)
    test["opp_recent_goals"] = test["opp_recent_goals"].fillna(global_avg_gf)
    test["opp_recent_conceded"] = test["opp_recent_conceded"].fillna(global_avg_ga)
    test["opp_recent_wr"] = test["opp_recent_wr"].fillna(global_wr)
    test["opp_recent_gd"] = test["opp_recent_gd"].fillna(0)

    # Opponent days_since: we don't have it cleanly for test, leave as NaN
    test["opp_days_since"] = np.nan

    return test


# ============================================================================
# H2H FEATURES
# ============================================================================

def compute_h2h_train(df):
    """Expanding H2H stats per (gender, team, opponent) pair, shifted."""
    df = df.copy()
    df["_is_win"] = (df["team_goals"] > df["opp_goals"]).astype(int)
    grp = ["gender", "team", "opponent"]

    df["h2h_cum_scored"] = df.groupby(grp)["team_goals"].transform(
        lambda x: x.cumsum().shift(1))
    df["h2h_cum_conceded"] = df.groupby(grp)["opp_goals"].transform(
        lambda x: x.cumsum().shift(1))
    df["h2h_cum_wins"] = df.groupby(grp)["_is_win"].transform(
        lambda x: x.cumsum().shift(1))
    df["h2h_n"] = df.groupby(grp).cumcount()  # number of prior meetings

    # Averages (NaN for first meeting)
    h2h_n_safe = df["h2h_n"].replace(0, np.nan)
    df["h2h_avg_scored"] = df["h2h_cum_scored"] / h2h_n_safe
    df["h2h_avg_conceded"] = df["h2h_cum_conceded"] / h2h_n_safe
    df["h2h_wr"] = df["h2h_cum_wins"] / h2h_n_safe

    df.drop(columns=["h2h_cum_scored", "h2h_cum_conceded", "h2h_cum_wins", "_is_win"],
            inplace=True, errors="ignore")
    return df


def compute_h2h_test(train, test):
    """Final H2H stats from all of train, merged to test."""
    tc = train.copy()
    tc["_is_win"] = (tc["team_goals"] > tc["opp_goals"]).astype(int)

    h2h = tc.groupby(["gender", "team", "opponent"]).agg(
        h2h_avg_scored=("team_goals", "mean"),
        h2h_avg_conceded=("opp_goals", "mean"),
        h2h_wr=("_is_win", "mean"),
        h2h_n=("team_goals", "count"),
    ).reset_index()

    test = test.merge(h2h, on=["gender", "team", "opponent"], how="left")
    # NaN for unseen matchups → LightGBM handles this
    return test


# ============================================================================
# SHARED FEATURES (available in both train and test)
# ============================================================================

TOURNAMENT_IMPORTANCE = {
    "FIFA World Cup": 5, "UEFA European Championship": 4,
    "Copa América": 4, "Copa America": 4,
    "African Cup of Nations": 4, "Africa Cup of Nations": 4,
    "AFC Asian Cup": 4, "CONCACAF Gold Cup": 3, "Confederations Cup": 3,
    "OFC Nations Cup": 3, "Nations League": 3,
    "qualification": 2, "Qualifying": 2, "Friendly": 1,
}


def _tourn_imp(name):
    if pd.isna(name): return 2
    for k, v in TOURNAMENT_IMPORTANCE.items():
        if k.lower() in str(name).lower():
            return v
    return 2


def build_shared_features(df):
    """All features computable from the 20 shared columns."""
    df = df.copy()
    dt = pd.to_datetime(df["date"])

    # --- Temporal ---
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["era"] = pd.cut(df["year"],
                       bins=[1870, 1930, 1960, 1990, 2010, 2030],
                       labels=[0, 1, 2, 3, 4]).astype(float)

    # --- Tournament ---
    df["tourn_imp"] = df["tournament"].apply(_tourn_imp)
    df["is_friendly"] = (df["tourn_imp"] == 1).astype(int)
    df["is_major"] = (df["tourn_imp"] >= 4).astype(int)

    # --- Match context ---
    df["is_away"] = ((df["is_home"] == 0) & (df["neutral"] == 0)).astype(int)
    df["same_confed"] = (df["confederation_team"] == df["confederation_opp"]).astype(int)
    df["gender_enc"] = (df["gender"] == "M").astype(int)

    # --- Confederation encoding ---
    cmap = {"UEFA": 0, "CONMEBOL": 1, "CAF": 2, "AFC": 3, "CONCACAF": 4, "OFC": 5}
    df["confed_team_enc"] = df["confederation_team"].map(cmap).fillna(6)
    df["confed_opp_enc"] = df["confederation_opp"].map(cmap).fillna(6)

    # --- Geographic ---
    df["travel_diff"] = df["distance_travel_team"] - df["distance_travel_opp"]
    df["travel_ratio"] = df["distance_travel_team"] / (df["distance_travel_opp"].clip(lower=1))
    df["high_altitude"] = (df["altitude_venue"] > 1500).astype(float)
    df["temp_extreme"] = (df["temperature_venue"] - 20.0).abs()

    # --- Socio-economic ---
    df["gdp_ratio"] = df["gdp_per_capita_team"] / df["gdp_per_capita_opp"].clip(lower=1)
    df["gdp_diff"] = df["gdp_per_capita_team"] - df["gdp_per_capita_opp"]
    df["log_gdp_team"] = np.log1p(df["gdp_per_capita_team"].clip(lower=0))
    df["log_gdp_opp"] = np.log1p(df["gdp_per_capita_opp"].clip(lower=0))
    df["pop_ratio"] = df["population_team"] / df["population_opp"].clip(lower=1)
    df["log_pop_team"] = np.log1p(df["population_team"].clip(lower=0))
    df["log_pop_opp"] = np.log1p(df["population_opp"].clip(lower=0))

    return df


# ============================================================================
# INTERACTION & DIFF FEATURES
# ============================================================================

def build_derived_features(df):
    """Interactions and differences between team/opponent encodings."""
    df = df.copy()

    # Strength diffs
    df["enc_goals_diff"] = df["team_enc_goals"] - df["opp_enc_goals"]
    df["enc_conceded_diff"] = df["team_enc_conceded"] - df["opp_enc_conceded"]
    df["enc_wr_diff"] = df["team_enc_wr"] - df["opp_enc_wr"]
    df["enc_gd_diff"] = df["team_enc_gd"] - df["opp_enc_gd"]
    df["recent_goals_diff"] = df["team_recent_goals"] - df["opp_recent_goals"]
    df["recent_wr_diff"] = df["team_recent_wr"] - df["opp_recent_wr"]

    # Elo interactions
    if "elo_diff" in df.columns:
        df["elo_diff_x_home"] = df["elo_diff"] * df["is_home"]
        df["elo_diff_x_neutral"] = df["elo_diff"] * df["neutral"]
        df["elo_diff_x_importance"] = df["elo_diff"] * df["tourn_imp"]
        df["elo_expected"] = 1.0 / (1.0 + 10.0 ** (-df["elo_diff"] / 400.0))

    # Cross-category interactions
    df["goals_x_home"] = df["team_enc_goals"] * df["is_home"]
    df["wr_x_importance"] = df["team_enc_wr"] * df["tourn_imp"]

    # --- NEW INTERACTION FEATURES FROM V10 ---
    if "distance_travel_team" in df.columns and "altitude_venue" in df.columns:
        df["travel_x_altitude"] = df["distance_travel_team"] * df["altitude_venue"]
    if "team_recent_wr" in df.columns and "elo_team" in df.columns:
        df["form_x_elo"] = df["team_recent_wr"] * df["elo_team"]
    if "elo_expected" in df.columns:
        df["elo_upset_potential"] = (0.5 - df["elo_expected"]).abs()


    return df


# ============================================================================
# FEATURE COLUMN SELECTION
# ============================================================================

FEATURE_COLS = [
    # Elo
    "elo_team", "elo_opp", "elo_diff", "elo_sum", "elo_expected",
    # Team encoding
    "team_enc_goals", "team_enc_conceded", "team_enc_gd", "team_enc_wr", "team_enc_n",
    "team_recent_goals", "team_recent_conceded", "team_recent_wr", "team_recent_gd",
    "team_days_since",
    # Opponent encoding
    "opp_enc_goals", "opp_enc_conceded", "opp_enc_gd", "opp_enc_wr", "opp_enc_n",
    "opp_recent_goals", "opp_recent_conceded", "opp_recent_wr", "opp_recent_gd",
    "opp_days_since",
    # H2H
    "h2h_avg_scored", "h2h_avg_conceded", "h2h_wr", "h2h_n",
    # Diffs
    "enc_goals_diff", "enc_conceded_diff", "enc_wr_diff", "enc_gd_diff",
    "recent_goals_diff", "recent_wr_diff",
    # Elo interactions
    "elo_diff_x_home", "elo_diff_x_neutral", "elo_diff_x_importance",
    # Cross interactions
    "goals_x_home", "wr_x_importance",
    # Match context
    "is_home", "neutral", "is_away", "same_confed", "gender_enc",
    "tourn_imp", "is_friendly", "is_major",
    "confed_team_enc", "confed_opp_enc",
    # Geographic
    "distance_travel_team", "distance_travel_opp", "travel_diff", "travel_ratio",
    "altitude_venue", "high_altitude", "temperature_venue", "temp_extreme",
    # Socio-economic
    "gdp_per_capita_team", "gdp_per_capita_opp", "gdp_ratio", "gdp_diff",
    "log_gdp_team", "log_gdp_opp",
    "population_team", "population_opp", "pop_ratio", "log_pop_team", "log_pop_opp",
    # Temporal
    "year", "month", "day_of_week", "is_summer", "era",
    # New V10
    "travel_x_altitude", "form_x_elo", "elo_upset_potential",
]


# ============================================================================
# CROSS-VALIDATION (time-series expanding window)
# ============================================================================

def time_series_cv(df, feature_cols, n_folds=5, purge_days=30):
    """Expanding-window CV with purge gap. Returns fold metrics."""
    cutoffs = [
        ("2001-01-01", "2003-12-31"),
        ("2003-06-01", "2006-05-31"),
        ("2006-01-01", "2008-12-31"),
        ("2008-06-01", "2010-05-31"),
        ("2010-01-01", "2011-08-31"),
    ][-n_folds:]

    results = []
    for fold, (vs, ve) in enumerate(cutoffs):
        vs_dt = pd.Timestamp(vs)
        ve_dt = pd.Timestamp(ve)
        purge_dt = vs_dt - pd.Timedelta(days=purge_days)

        tr = df[df["date"] < purge_dt]
        va = df[(df["date"] >= vs_dt) & (df["date"] <= ve_dt)]

        if len(tr) == 0 or len(va) == 0:
            continue

        # Use only columns that exist
        fcols = [c for c in feature_cols if c in tr.columns and c in va.columns]

        X_tr, X_va = tr[fcols], va[fcols]
        yt_tr, yo_tr = tr["team_goals"], tr["opp_goals"]
        yt_va, yo_va = va["team_goals"], va["opp_goals"]

        # LightGBM
        m_t = lgb.LGBMRegressor(**LGB_PARAMS)
        m_o = lgb.LGBMRegressor(**LGB_PARAMS)
        
        if "sample_weight" in tr.columns:
            w_tr = tr["sample_weight"]
            m_t.fit(X_tr, yt_tr, sample_weight=w_tr, eval_set=[(X_va, yt_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
            m_o.fit(X_tr, yo_tr, sample_weight=w_tr, eval_set=[(X_va, yo_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
        else:
            m_t.fit(X_tr, yt_tr, eval_set=[(X_va, yt_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
            m_o.fit(X_tr, yo_tr, eval_set=[(X_va, yo_va)], callbacks=[lgb.early_stopping(100, verbose=False)])

        # XGBoost
        mx_t = xgb.XGBRegressor(**XGB_PARAMS)
        mx_o = xgb.XGBRegressor(**XGB_PARAMS)
        if "sample_weight" in tr.columns:
            mx_t.fit(X_tr, yt_tr, sample_weight=w_tr, eval_set=[(X_va, yt_va)], verbose=False)
            mx_o.fit(X_tr, yo_tr, sample_weight=w_tr, eval_set=[(X_va, yo_va)], verbose=False)
        else:
            mx_t.fit(X_tr, yt_tr, eval_set=[(X_va, yt_va)], verbose=False)
            mx_o.fit(X_tr, yo_tr, eval_set=[(X_va, yo_va)], verbose=False)

        pt = np.clip(W_LGB * m_t.predict(X_va) + W_XGB * mx_t.predict(X_va), 0, 15)
        po = np.clip(W_LGB * m_o.predict(X_va) + W_XGB * mx_o.predict(X_va), 0, 15)

        pt_int = np.round(pt).astype(int)
        po_int = np.round(po).astype(int)

        mae_t = np.mean(np.abs(yt_va.values - pt_int))
        mae_o = np.mean(np.abs(yo_va.values - po_int))
        mae = (mae_t + mae_o) / 2

        # Result accuracy
        actual_res = np.sign(yt_va.values - yo_va.values)
        pred_res = np.sign(pt_int - po_int)
        res_acc = np.mean(actual_res == pred_res)

        print(f"  Fold {fold} | train {len(tr):,}  val {len(va):,} | "
              f"MAE {mae:.4f} | ResultAcc {res_acc:.3f}")
        results.append({"fold": fold, "mae": mae, "res_acc": res_acc})

    if results:
        avg_mae = np.mean([r["mae"] for r in results])
        avg_acc = np.mean([r["res_acc"] for r in results])
        print(f"\n  CV MEAN → MAE: {avg_mae:.4f} | ResultAcc: {avg_acc:.3f}\n")
    return results



# ============================================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================================
def tune_hyperparameters(train, feature_cols):
    print("=" * 60)
    print(f"OPTUNA HYPERPARAMETER TUNING ({TUNING_TRIALS} Trials)")
    print("=" * 60)
    
    split_idx = int(len(train) * 0.9)
    train_split = train.iloc[:split_idx]
    val_split = train.iloc[split_idx:]
    
    X_tr = train_split[feature_cols]
    y_tr = train_split["team_goals"]
    w_tr = train_split["sample_weight"] if "sample_weight" in train.columns else None
    
    X_va = val_split[feature_cols]
    y_va = val_split["team_goals"]

    def objective(trial):
        params = {
            "objective": "poisson",
            "metric": "mae",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
            "n_estimators": 500,
        }
        
        model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE)
        if w_tr is not None:
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
            
        preds = model.predict(X_va)
        return np.mean(np.abs(y_va - preds))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=TUNING_TRIALS)
    
    print(f"  Best MAE: {study.best_value:.4f}")
    return study.best_params

# ============================================================================
# SMART ROUNDING (From V10)
# ============================================================================
def smart_round(team_raw, opp_raw):
    diff = team_raw - opp_raw
    tg = int(round(max(0, team_raw)))
    og = int(round(max(0, opp_raw)))
    raw_result = np.sign(diff)
    rounded_result = np.sign(tg - og)

    if raw_result != rounded_result and abs(diff) > 0.2:
        if raw_result > 0 and tg <= og:
            if team_raw - tg > og - opp_raw: tg = og + 1
            else: og = max(0, tg - 1)
        elif raw_result < 0 and tg >= og:
            if opp_raw - og > tg - team_raw: og = tg + 1
            else: tg = max(0, og - 1)
    return tg, og

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    t0 = time.time()

    # ---------------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    
    # Rule 1: Handling Missing Values
    train.replace([-9999, -9999.0], np.nan, inplace=True)
    test.replace([-9999, -9999.0], np.nan, inplace=True)

    # Rule 2: Time Trimming
    old_count = len(train)
    train = train[train["date"] >= '1950-01-01'].reset_index(drop=True)
    print(f"  [Cleaning] Trimming: Membuang {old_count - len(train):,} baris data sebelum 1 Jan 1950")

    # Rule 3: Outlier Clipping
    train["team_goals"] = train["team_goals"].clip(upper=4)
    train["opp_goals"] = train["opp_goals"].clip(upper=4)
    print(f"  [Cleaning] Clipping: Membatasi target goals maksimal di angka 4")

    # --- ADVANCED PREPROCESSING: Smart Imputation Geografis dari V10 ---
    print("  [Cleaning] Imputasi nilai geografis yang kosong...")
    for df_data in [train, test]:
        for col in ["altitude_venue", "temperature_venue"]:
            if col in df_data.columns:
                df_data[col] = df_data.groupby("venue_country")[col].transform(lambda x: x.fillna(x.mean()))
                df_data[col] = df_data[col].fillna(df_data[col].median())

    # Rule 4: Sort Data
    train = train.sort_values(["date", "match_id"]).reset_index(drop=True)
    test = test.sort_values(["date", "match_id"]).reset_index(drop=True)

    print(f"\n  Final Train: {train.shape[0]:,} rows | Test: {test.shape[0]:,} rows")
    print(f"  Train date range: {train['date'].min().date()} → {train['date'].max().date()}")
    print(f"  Goals distribution (after clip): mean={train['team_goals'].mean():.2f}")

    global_avg_gf = train["team_goals"].mean()
    global_avg_ga = train["opp_goals"].mean()
    global_wr = (train["team_goals"] > train["opp_goals"]).mean()
    print(f"  Global avg GF={global_avg_gf:.3f}  GA={global_avg_ga:.3f}  WR={global_wr:.3f}")

    # ---------------------------------------------------------------
    # 2. ELO COMPUTATION
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: ELO COMPUTATION")
    print("=" * 60)
    matches = extract_match_level(train)
    print(f"  {len(matches):,} unique matches")

    elo_before, final_elos = compute_elo(matches)
    print(f"  Elo computed for {len(final_elos)} (gender, team) pairs")

    # Merge Elo (before match) into train
    train["elo_team"] = train.apply(
        lambda r: elo_before.get((r["match_id"], r["team"]), INITIAL_ELO), axis=1)
    train["elo_opp"] = train.apply(
        lambda r: elo_before.get((r["match_id"], r["opponent"]), INITIAL_ELO), axis=1)

    # Merge Elo (final) into test
    test["elo_team"] = test.apply(
        lambda r: final_elos.get((r["gender"], r["team"]), INITIAL_ELO), axis=1)
    test["elo_opp"] = test.apply(
        lambda r: final_elos.get((r["gender"], r["opponent"]), INITIAL_ELO), axis=1)

    for df in [train, test]:
        df["elo_diff"] = df["elo_team"] - df["elo_opp"]
        df["elo_sum"] = df["elo_team"] + df["elo_opp"]

    print("  ✓ Elo merged to train and test")

    # ---------------------------------------------------------------
    # 3. TEAM & OPPONENT ENCODING
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: TEAM & OPPONENT ENCODING")
    print("=" * 60)

    # Train: expanding stats (vectorized)
    train = compute_team_encoding_train(train, global_avg_gf, global_avg_ga, global_wr)
    train = compute_opp_encoding_train(train)
    print(f"  ✓ Train team+opp encoding done. Shape: {train.shape}")

    # Test: final stats from train (static merge)
    test = compute_team_encoding_test(train, test, global_avg_gf, global_avg_ga, global_wr)
    print(f"  ✓ Test team+opp encoding done. Shape: {test.shape}")

    # ---------------------------------------------------------------
    # 4. H2H FEATURES
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: H2H FEATURES")
    print("=" * 60)
    train = compute_h2h_train(train)
    test = compute_h2h_test(train, test)
    print(f"  ✓ H2H done. Train: {train.shape}  Test: {test.shape}")

    # ---------------------------------------------------------------
    # 5. SHARED + DERIVED FEATURES
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: SHARED & DERIVED FEATURES")
    print("=" * 60)
    train = build_shared_features(train)
    test = build_shared_features(test)
    train = build_derived_features(train)
    test = build_derived_features(test)
    print(f"  ✓ Features done. Train: {train.shape}  Test: {test.shape}")

    # ---------------------------------------------------------------
    # 6. FEATURE SELECTION
    # ---------------------------------------------------------------
    fcols = [c for c in FEATURE_COLS if c in train.columns and c in test.columns]
    print(f"\n  Using {len(fcols)} features")

    # --- ADVANCED PREPROCESSING: Exponential Time Decay (V10) ---
    print("  [Feature] Menghitung Exponential Time Decay (sample_weight)")
    train["sample_weight"] = train["tourn_imp"].map({
        5: 2.0, 4: 1.5, 3: 1.2, 2: 1.0, 1: 0.6
    }).fillna(1.0)
    ref_date = pd.to_datetime(train["date"]).max()
    days_diff = (ref_date - pd.to_datetime(train["date"])).dt.days
    alpha = 0.0005 
    train["sample_weight"] *= np.exp(-alpha * days_diff)

    # ---------------------------------------------------------------
    # 7. CROSS-VALIDATION
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: CROSS-VALIDATION")
    print("=" * 60)
    cv_results = time_series_cv(train, fcols, n_folds=5, purge_days=30)

    # ---------------------------------------------------------------
    # 8. TRAIN FINAL MODELS
    # ---------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: TRAINING FINAL MODELS")
    print("=" * 60)

    if RUN_TUNING:
        best_params = tune_hyperparameters(train, fcols)
        LGB_PARAMS.update(best_params)
        LGB_PARAMS["n_estimators"] = 3000

    X_full = train[fcols]
    yt_full = train["team_goals"]
    yo_full = train["opp_goals"]
    w_full = train["sample_weight"] if "sample_weight" in train.columns else None

    # LightGBM
    print("  Training LightGBM (team_goals)...")
    lgb_team = lgb.LGBMRegressor(**LGB_PARAMS)
    if w_full is not None:
        lgb_team.fit(X_full, yt_full, sample_weight=w_full)
    else:
        lgb_team.fit(X_full, yt_full)

    print("  Training LightGBM (opp_goals)...")
    lgb_opp = lgb.LGBMRegressor(**LGB_PARAMS)
    if w_full is not None:
        lgb_opp.fit(X_full, yo_full, sample_weight=w_full)
    else:
        lgb_opp.fit(X_full, yo_full)

    # XGBoost
    print("  Training XGBoost (team_goals)...")
    xgb_team = xgb.XGBRegressor(**XGB_PARAMS)
    if w_full is not None:
        xgb_team.fit(X_full, yt_full, sample_weight=w_full)
    else:
        xgb_team.fit(X_full, yt_full)

    print("  Training XGBoost (opp_goals)...")
    xgb_opp = xgb.XGBRegressor(**XGB_PARAMS)
    if w_full is not None:
        xgb_opp.fit(X_full, yo_full, sample_weight=w_full)
    else:
        xgb_opp.fit(X_full, yo_full)

    # Feature importance
    imp = pd.Series(lgb_team.feature_importances_, index=fcols).sort_values(ascending=False)
    print("\n  Top 15 features (LightGBM team_goals):")
    for feat, val in imp.head(15).items():
        print(f"    {feat:35s}  {val:6.0f}")

    # ---------------------------------------------------------------
    # 9. PREDICT TEST
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 9: PREDICTING TEST")
    print("=" * 60)

    X_test = test[fcols]

    # Ensemble raw predictions — NO fillna! LightGBM handles NaN natively
    pred_team_raw = W_LGB * lgb_team.predict(X_test) + W_XGB * xgb_team.predict(X_test)
    pred_opp_raw = W_LGB * lgb_opp.predict(X_test) + W_XGB * xgb_opp.predict(X_test)

    # Clip to [0, 15]
    pred_team_raw = np.clip(pred_team_raw, 0, 15)
    pred_opp_raw = np.clip(pred_opp_raw, 0, 15)

    # ---------------------------------------------------------------
    # 10. CONSISTENCY ENFORCEMENT + ROUNDING
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 10: CONSISTENCY & ROUNDING")
    print("=" * 60)

    test["pred_team_raw"] = pred_team_raw
    test["pred_opp_raw"] = pred_opp_raw

    # For each match_id, enforce consistency:
    # Row A's team_goals = Row B's opp_goals (and vice versa)
    # Average the two predictions
    consistent_team = np.zeros(len(test))
    consistent_opp = np.zeros(len(test))

    match_groups = test.groupby("match_id")
    consistency_ok = 0
    total_matches = 0

    for mid, grp in match_groups:
        if len(grp) != 2:
            # Fallback: use raw predictions
            for i in grp.index:
                consistent_team[i] = test.at[i, "pred_team_raw"]
                consistent_opp[i] = test.at[i, "pred_opp_raw"]
            continue

        total_matches += 1
        ia, ib = grp.index[0], grp.index[1]

        # Row A: team=A, opp=B → pred_team = goals_A, pred_opp = goals_B
        # Row B: team=B, opp=A → pred_team = goals_B, pred_opp = goals_A
        # Average: goals_A = (row_A_pred_team + row_B_pred_opp) / 2
        goals_a = (test.at[ia, "pred_team_raw"] + test.at[ib, "pred_opp_raw"]) / 2.0
        goals_b = (test.at[ib, "pred_team_raw"] + test.at[ia, "pred_opp_raw"]) / 2.0

        consistent_team[ia] = goals_a
        consistent_opp[ia] = goals_b
        consistent_team[ib] = goals_b
        consistent_opp[ib] = goals_a
        consistency_ok += 1

    # Round (Using Smart Rounding from V10)
    final_team = np.zeros(len(test), dtype=int)
    final_opp = np.zeros(len(test), dtype=int)
    for i in range(len(test)):
        t, o = smart_round(consistent_team[i], consistent_opp[i])
        final_team[i] = min(t, 15)
        final_opp[i] = min(o, 15)

    print(f"  Matched {consistency_ok}/{total_matches} matches for consistency")
    print(f"  Pred team_goals: mean={final_team.mean():.2f} max={final_team.max()}")
    print(f"  Pred opp_goals:  mean={final_opp.mean():.2f} max={final_opp.max()}")

    # Goal distribution
    print("\n  Goal distribution (team_goals):")
    vals, counts = np.unique(final_team, return_counts=True)
    for v, c in zip(vals[:8], counts[:8]):
        print(f"    {v}: {c:,} ({c/len(final_team)*100:.1f}%)")

    # ---------------------------------------------------------------
    # 11. SUBMISSION
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 11: GENERATING SUBMISSION")
    print("=" * 60)

    submission = pd.DataFrame({
        "Id": test["Id"],
        "team_goals": final_team,
        "opp_goals": final_opp,
    })

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"  ✓ Saved {len(submission):,} rows to {SUBMISSION_PATH}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
