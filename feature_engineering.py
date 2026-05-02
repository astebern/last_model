"""
Feature Engineering for International Football Score Prediction
===============================================================
Contains:
- TeamStatsTracker: Rolling performance statistics per team
- H2HTracker: Head-to-head statistics between team pairs
- Shared feature builders: temporal, geographic, socio-economic, tournament
"""

from collections import defaultdict
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rolling Team Statistics Tracker
# ---------------------------------------------------------------------------

class TeamStatsTracker:
    """
    Maintains a chronological history of match results per team.
    Computes rolling features (last-5, last-10, all-time) on demand.

    Usage:
        tracker = TeamStatsTracker()
        # Process matches chronologically:
        features = tracker.get_features("Brazil")   # BEFORE the match
        tracker.add_result("Brazil", date, gf, ga)  # AFTER  the match
    """

    def __init__(self):
        # team -> list of dicts, ordered chronologically
        self.history = defaultdict(list)

    def add_result(self, team, date, goals_for, goals_against):
        """Record a match result for a team."""
        self.history[team].append({
            "date": date,
            "gf": goals_for,
            "ga": goals_against,
            "gd": goals_for - goals_against,
            "pts": 3 if goals_for > goals_against else (1 if goals_for == goals_against else 0),
            "win": int(goals_for > goals_against),
            "draw": int(goals_for == goals_against),
            "loss": int(goals_for < goals_against),
        })

    def get_features(self, team):
        """
        Compute rolling features from accumulated history.
        Call BEFORE add_result() for the current match to avoid leakage.

        Returns dict of ~25 features.
        """
        hist = self.history.get(team, [])
        feat = {}

        for n in [5, 10]:
            recent = hist[-n:] if hist else []
            if recent:
                gf = [m["gf"] for m in recent]
                ga = [m["ga"] for m in recent]
                gd = [m["gd"] for m in recent]
                pts = [m["pts"] for m in recent]
                wins = [m["win"] for m in recent]
                feat[f"avg_goals_last{n}"] = np.mean(gf)
                feat[f"avg_conceded_last{n}"] = np.mean(ga)
                feat[f"avg_gd_last{n}"] = np.mean(gd)
                feat[f"total_pts_last{n}"] = sum(pts)
                feat[f"win_rate_last{n}"] = np.mean(wins)
                feat[f"max_goals_last{n}"] = max(gf)
                feat[f"clean_sheets_last{n}"] = sum(1 for g in ga if g == 0)
                feat[f"failed_to_score_last{n}"] = sum(1 for g in gf if g == 0)
            else:
                for suffix in [
                    "avg_goals", "avg_conceded", "avg_gd", "total_pts",
                    "win_rate", "max_goals", "clean_sheets", "failed_to_score",
                ]:
                    feat[f"{suffix}_last{n}"] = np.nan

        # Days since last match
        if hist:
            feat["days_since_last"] = None  # Will be filled when date is known
            feat["hist_avg_goals"] = np.mean([m["gf"] for m in hist])
            feat["hist_avg_conceded"] = np.mean([m["ga"] for m in hist])
            feat["hist_win_rate"] = np.mean([m["win"] for m in hist])
            feat["total_matches"] = len(hist)
        else:
            feat["days_since_last"] = np.nan
            feat["hist_avg_goals"] = np.nan
            feat["hist_avg_conceded"] = np.nan
            feat["hist_win_rate"] = np.nan
            feat["total_matches"] = 0

        return feat

    def get_days_since_last(self, team, current_date):
        """Compute days since this team's last match."""
        hist = self.history.get(team, [])
        if hist:
            return (current_date - hist[-1]["date"]).days
        return np.nan


# ---------------------------------------------------------------------------
# Head-to-Head Tracker
# ---------------------------------------------------------------------------

class H2HTracker:
    """
    Tracks head-to-head match history between pairs of teams.
    Provides H2H features from a specific team's perspective.
    """

    def __init__(self):
        # (team_a, team_b) sorted -> list of match records
        self.history = defaultdict(list)

    @staticmethod
    def _pair_key(team_a, team_b):
        return tuple(sorted([team_a, team_b]))

    def add_result(self, team_a, team_b, goals_a, goals_b, date):
        """Record an H2H match result."""
        key = self._pair_key(team_a, team_b)
        self.history[key].append({
            "date": date,
            "team_a": team_a, "team_b": team_b,
            "goals_a": goals_a, "goals_b": goals_b,
        })

    def get_features(self, team, opponent):
        """
        Compute H2H features from `team`'s perspective against `opponent`.
        Call BEFORE add_result() for the current match.
        """
        key = self._pair_key(team, opponent)
        hist = self.history.get(key, [])
        feat = {}

        if hist:
            # Extract goals from team's perspective
            team_goals, opp_goals = [], []
            wins, draws = 0, 0
            for m in hist:
                if m["team_a"] == team:
                    tg, og = m["goals_a"], m["goals_b"]
                else:
                    tg, og = m["goals_b"], m["goals_a"]
                team_goals.append(tg)
                opp_goals.append(og)
                if tg > og:
                    wins += 1
                elif tg == og:
                    draws += 1

            n = len(hist)
            feat["h2h_matches"] = n
            feat["h2h_win_rate"] = wins / n
            feat["h2h_draw_rate"] = draws / n
            feat["h2h_avg_scored"] = np.mean(team_goals)
            feat["h2h_avg_conceded"] = np.mean(opp_goals)

            # Recent H2H (last 5 meetings)
            recent_tg = team_goals[-5:]
            recent_og = opp_goals[-5:]
            feat["h2h_recent_scored"] = np.mean(recent_tg)
            feat["h2h_recent_conceded"] = np.mean(recent_og)
        else:
            feat["h2h_matches"] = 0
            feat["h2h_win_rate"] = np.nan
            feat["h2h_draw_rate"] = np.nan
            feat["h2h_avg_scored"] = np.nan
            feat["h2h_avg_conceded"] = np.nan
            feat["h2h_recent_scored"] = np.nan
            feat["h2h_recent_conceded"] = np.nan

        return feat


# ---------------------------------------------------------------------------
# Shared Feature Builders (applicable to both train and test)
# ---------------------------------------------------------------------------

def build_temporal_features(df):
    """Extract temporal features from the 'date' column."""
    df = df.copy()
    dt = pd.to_datetime(df["date"])

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_year"] = dt.dt.dayofyear
    df["decade"] = (df["year"] // 10) * 10
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

    # Era encoding: football quality changed drastically over time
    df["era"] = pd.cut(
        df["year"],
        bins=[1870, 1930, 1960, 1990, 2010, 2030],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    return df


def build_geo_features(df):
    """Build geographic features from distance / altitude / temperature."""
    df = df.copy()

    # Travel asymmetry
    df["travel_diff"] = df["distance_travel_team"] - df["distance_travel_opp"]
    df["travel_ratio"] = df["distance_travel_team"] / (df["distance_travel_opp"] + 1.0)
    df["is_long_haul_team"] = (df["distance_travel_team"] > 5000).astype(int)
    df["is_long_haul_opp"] = (df["distance_travel_opp"] > 5000).astype(int)

    # Altitude effect (high-altitude venues like La Paz are notorious)
    df["high_altitude"] = (df["altitude_venue"] > 1500).astype(float)

    # Temperature deviation from optimal (~20°C)
    df["temp_extreme"] = (df["temperature_venue"] - 20.0).abs()

    return df


def build_socioeconomic_features(df):
    """Build socio-economic features from GDP and population."""
    df = df.copy()

    # GDP features
    df["gdp_ratio"] = df["gdp_per_capita_team"] / (df["gdp_per_capita_opp"] + 1.0)
    df["gdp_diff"] = df["gdp_per_capita_team"] - df["gdp_per_capita_opp"]
    df["log_gdp_team"] = np.log1p(df["gdp_per_capita_team"].clip(lower=0))
    df["log_gdp_opp"] = np.log1p(df["gdp_per_capita_opp"].clip(lower=0))

    # Population features
    df["pop_ratio"] = df["population_team"] / (df["population_opp"] + 1.0)
    df["log_pop_team"] = np.log1p(df["population_team"].clip(lower=0))
    df["log_pop_opp"] = np.log1p(df["population_opp"].clip(lower=0))

    # National wealth proxy (total GDP)
    df["total_wealth_team"] = df["gdp_per_capita_team"] * df["population_team"]
    df["total_wealth_opp"] = df["gdp_per_capita_opp"] * df["population_opp"]
    df["log_wealth_team"] = np.log1p(df["total_wealth_team"].clip(lower=0))
    df["log_wealth_opp"] = np.log1p(df["total_wealth_opp"].clip(lower=0))

    return df


# Tournament importance mapping
TOURNAMENT_IMPORTANCE_MAP = {
    "FIFA World Cup": 5,
    "UEFA European Championship": 4,
    "Copa América": 4,
    "Copa America": 4,
    "African Cup of Nations": 4,
    "Africa Cup of Nations": 4,
    "AFC Asian Cup": 4,
    "CONCACAF Gold Cup": 3,
    "Confederations Cup": 3,
    "OFC Nations Cup": 3,
    "Nations League": 3,
    "UEFA Nations League": 3,
    "CONCACAF Nations League": 3,
    "qualification": 2,
    "Qualifying": 2,
    "Friendly": 1,
}


def _get_tournament_importance(name):
    """Map tournament name to ordinal importance score."""
    if pd.isna(name):
        return 2
    name_str = str(name)
    for pattern, score in TOURNAMENT_IMPORTANCE_MAP.items():
        if pattern.lower() in name_str.lower():
            return score
    return 2  # default: mid importance


def build_tournament_features(df):
    """Build tournament-related features."""
    df = df.copy()

    df["tournament_importance"] = df["tournament"].apply(_get_tournament_importance)
    df["is_friendly"] = (df["tournament_importance"] == 1).astype(int)
    df["is_major_tournament"] = (df["tournament_importance"] >= 4).astype(int)

    return df


def build_match_context_features(df):
    """Build match context features from is_home, neutral, confederations."""
    df = df.copy()

    # Home/Away/Neutral encoding
    df["is_away"] = ((df["is_home"] == 0) & (df["neutral"] == 0)).astype(int)

    # Same confederation (inter- vs intra-confederation match)
    df["same_confederation"] = (
        df["confederation_team"] == df["confederation_opp"]
    ).astype(int)

    return df


def encode_confederation(df):
    """One-hot encode confederation columns."""
    df = df.copy()
    confed_map = {"UEFA": 0, "CONMEBOL": 1, "CAF": 2, "AFC": 3,
                  "CONCACAF": 4, "OFC": 5}

    df["confed_team_enc"] = df["confederation_team"].map(confed_map).fillna(6)
    df["confed_opp_enc"] = df["confederation_opp"].map(confed_map).fillna(6)

    return df


def encode_gender(df):
    """Binary encode gender."""
    df = df.copy()
    df["gender_enc"] = (df["gender"] == "M").astype(int)
    return df


# ---------------------------------------------------------------------------
# Master Feature Builder
# ---------------------------------------------------------------------------

def build_all_shared_features(df):
    """
    Apply all shared feature engineering steps.
    These features can be computed for both train and test without leakage.
    """
    df = build_temporal_features(df)
    df = build_geo_features(df)
    df = build_socioeconomic_features(df)
    df = build_tournament_features(df)
    df = build_match_context_features(df)
    df = encode_confederation(df)
    df = encode_gender(df)
    return df


# ---------------------------------------------------------------------------
# Historical feature computation (process chronologically)
# ---------------------------------------------------------------------------

def compute_historical_features(df, elo_system, team_tracker, h2h_tracker,
                                 update_state=True, goals_col_team="team_goals",
                                 goals_col_opp="opp_goals"):
    """
    Process a DataFrame chronologically, computing Elo, rolling stats, and H2H
    features for each row. Optionally updates tracker state with results.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by date. Each match_id has exactly 2 rows.
    elo_system : EloSystem
    team_tracker : TeamStatsTracker
    h2h_tracker : H2HTracker
    update_state : bool
        If True, update trackers with actual/predicted goals after computing features.
    goals_col_team, goals_col_opp : str
        Column names for goals (use defaults for train; for test these may be predictions).

    Returns
    -------
    pd.DataFrame with new feature columns added.
    """
    df = df.copy()
    df["date_ts"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date_ts", "match_id"]).reset_index(drop=True)

    # --- Pre-allocate feature arrays ---
    n = len(df)
    feature_names = []

    # Elo features
    elo_cols = ["elo_team_custom", "elo_opp_custom", "elo_diff", "elo_sum",
                "elo_expected"]
    for c in elo_cols:
        df[c] = np.nan

    # Rolling features (team + opp perspective) — will be filled from tracker
    rolling_suffixes = [
        "avg_goals_last5", "avg_conceded_last5", "avg_gd_last5",
        "total_pts_last5", "win_rate_last5", "max_goals_last5",
        "clean_sheets_last5", "failed_to_score_last5",
        "avg_goals_last10", "avg_conceded_last10", "avg_gd_last10",
        "total_pts_last10", "win_rate_last10", "max_goals_last10",
        "clean_sheets_last10", "failed_to_score_last10",
        "days_since_last", "hist_avg_goals", "hist_avg_conceded",
        "hist_win_rate", "total_matches",
    ]
    for prefix in ["team_", "opp_"]:
        for s in rolling_suffixes:
            col = prefix + s
            df[col] = np.nan

    # Diff features for rolling stats
    diff_suffixes = ["avg_goals_last5", "avg_conceded_last5", "win_rate_last5",
                     "avg_goals_last10", "win_rate_last10",
                     "hist_avg_goals", "hist_win_rate"]
    for s in diff_suffixes:
        df[f"diff_{s}"] = np.nan

    # H2H features
    h2h_cols = ["h2h_matches", "h2h_win_rate", "h2h_draw_rate",
                "h2h_avg_scored", "h2h_avg_conceded",
                "h2h_recent_scored", "h2h_recent_conceded"]
    for c in h2h_cols:
        df[c] = np.nan

    # --- Process by unique match ---
    processed_matches = set()

    for idx, row in df.iterrows():
        mid = row["match_id"]
        if mid in processed_matches:
            continue
        processed_matches.add(mid)

        # Find partner row
        match_mask = df["match_id"] == mid
        match_rows = df.loc[match_mask]
        if len(match_rows) != 2:
            continue

        row_a = match_rows.iloc[0]
        row_b = match_rows.iloc[1]
        idx_a = match_rows.index[0]
        idx_b = match_rows.index[1]

        team_a = row_a["team"]
        team_b = row_b["team"]
        gender = row_a["gender"]
        date = row_a["date_ts"]

        # ---- Elo features (before match) ----
        elo_a = elo_system.get_elo(gender, team_a)
        elo_b = elo_system.get_elo(gender, team_b)

        df.at[idx_a, "elo_team_custom"] = elo_a
        df.at[idx_a, "elo_opp_custom"] = elo_b
        df.at[idx_a, "elo_diff"] = elo_a - elo_b
        df.at[idx_a, "elo_sum"] = elo_a + elo_b
        df.at[idx_a, "elo_expected"] = elo_system.expected_score(
            elo_a + (100 if row_a["is_home"] and not row_a["neutral"] else 0),
            elo_b + (100 if row_b["is_home"] and not row_b["neutral"] else 0),
        )

        df.at[idx_b, "elo_team_custom"] = elo_b
        df.at[idx_b, "elo_opp_custom"] = elo_a
        df.at[idx_b, "elo_diff"] = elo_b - elo_a
        df.at[idx_b, "elo_sum"] = elo_a + elo_b
        df.at[idx_b, "elo_expected"] = elo_system.expected_score(
            elo_b + (100 if row_b["is_home"] and not row_b["neutral"] else 0),
            elo_a + (100 if row_a["is_home"] and not row_a["neutral"] else 0),
        )

        # ---- Rolling team stats (before match) ----
        feat_a = team_tracker.get_features(team_a)
        feat_b = team_tracker.get_features(team_b)

        # Days since last match (needs date)
        feat_a["days_since_last"] = team_tracker.get_days_since_last(team_a, date)
        feat_b["days_since_last"] = team_tracker.get_days_since_last(team_b, date)

        for s in rolling_suffixes:
            df.at[idx_a, f"team_{s}"] = feat_a.get(s, np.nan)
            df.at[idx_a, f"opp_{s}"] = feat_b.get(s, np.nan)
            df.at[idx_b, f"team_{s}"] = feat_b.get(s, np.nan)
            df.at[idx_b, f"opp_{s}"] = feat_a.get(s, np.nan)

        # Diff features
        for s in diff_suffixes:
            val_a = feat_a.get(s, np.nan)
            val_b = feat_b.get(s, np.nan)
            if pd.notna(val_a) and pd.notna(val_b):
                df.at[idx_a, f"diff_{s}"] = val_a - val_b
                df.at[idx_b, f"diff_{s}"] = val_b - val_a
            else:
                df.at[idx_a, f"diff_{s}"] = np.nan
                df.at[idx_b, f"diff_{s}"] = np.nan

        # ---- H2H features (before match) ----
        h2h_feat_a = h2h_tracker.get_features(team_a, team_b)
        h2h_feat_b = h2h_tracker.get_features(team_b, team_a)

        for c in h2h_cols:
            df.at[idx_a, c] = h2h_feat_a.get(c, np.nan)
            df.at[idx_b, c] = h2h_feat_b.get(c, np.nan)

        # ---- Update state AFTER computing features ----
        if update_state:
            goals_a = row_a[goals_col_team]
            goals_b = row_b[goals_col_team]

            if pd.notna(goals_a) and pd.notna(goals_b):
                goals_a = int(goals_a)
                goals_b = int(goals_b)

                team_tracker.add_result(team_a, date, goals_a, goals_b)
                team_tracker.add_result(team_b, date, goals_b, goals_a)
                h2h_tracker.add_result(team_a, team_b, goals_a, goals_b, date)
                elo_system.update_match(
                    gender, team_a, team_b, goals_a, goals_b,
                    is_home_a=row_a["is_home"], neutral=row_a["neutral"],
                    tournament=row_a["tournament"],
                )

    # Drop helper column
    df.drop(columns=["date_ts"], inplace=True, errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Interaction features (computed after all base features are ready)
# ---------------------------------------------------------------------------

def build_interaction_features(df):
    """Build interaction features between key variables."""
    df = df.copy()

    # Elo × Match context
    df["elo_diff_x_home"] = df["elo_diff"] * df["is_home"]
    df["elo_diff_x_neutral"] = df["elo_diff"] * df["neutral"]
    df["elo_diff_x_importance"] = df["elo_diff"] * df["tournament_importance"]

    # Form × Team strength
    if "team_win_rate_last5" in df.columns and "elo_team_custom" in df.columns:
        df["form_x_elo"] = df["team_win_rate_last5"] * df["elo_team_custom"]

    # Travel × Altitude
    df["travel_x_altitude"] = df["distance_travel_team"] * df["altitude_venue"]

    # Expected result difference (from Elo)
    df["elo_upset_potential"] = (0.5 - df["elo_expected"]).abs()

    return df
