"""
Custom Elo Rating System for International Football
====================================================
Implements a modified World Football Elo Rating system with:
- Tournament-dependent K-factors (World Cup > Friendly)
- Goal-difference-weighted updates
- Home advantage adjustment
- Gender-separated rating pools
"""

from collections import defaultdict
import numpy as np


class EloSystem:
    """
    Modified Elo rating system based on World Football Elo Ratings methodology.
    Maintains separate rating pools per gender (M/W).
    """

    # K-factor mapping: higher K = more volatile updates for important matches
    TOURNAMENT_K = {
        "fifa world cup": 60,
        "confederations cup": 55,
        "uefa european championship": 50,
        "copa america": 50,
        "copa américa": 50,
        "african cup of nations": 50,
        "africa cup of nations": 50,
        "afc asian cup": 50,
        "concacaf gold cup": 45,
        "ofc nations cup": 40,
        "fifa world cup qualification": 40,
        "uefa euro qualification": 40,
        "uefa nations league": 35,
        "concacaf nations league": 35,
        "afc asian cup qualification": 35,
        "african nations championship": 35,
        "friendly": 20,
    }
    DEFAULT_K = 30

    # Tournament importance weights (for AW-MAE metric awareness)
    TOURNAMENT_IMPORTANCE = {
        "fifa world cup": 1.0,
        "confederations cup": 0.85,
        "uefa european championship": 0.9,
        "copa america": 0.85,
        "copa américa": 0.85,
        "african cup of nations": 0.8,
        "africa cup of nations": 0.8,
        "afc asian cup": 0.8,
        "concacaf gold cup": 0.75,
        "ofc nations cup": 0.65,
        "fifa world cup qualification": 0.7,
        "uefa euro qualification": 0.65,
        "uefa nations league": 0.55,
        "friendly": 0.3,
    }
    DEFAULT_IMPORTANCE = 0.5

    def __init__(self, initial_elo=1500, home_advantage=100):
        self.initial_elo = initial_elo
        self.home_advantage = home_advantage
        # Ratings keyed by (gender, team)
        self.ratings = {}

    def _key(self, gender, team):
        return (gender, team)

    def get_elo(self, gender, team):
        """Get current Elo rating for a team. Returns initial_elo if unseen."""
        return self.ratings.get(self._key(gender, team), self.initial_elo)

    def get_all_ratings(self):
        """Return a copy of all current ratings."""
        return dict(self.ratings)

    @staticmethod
    def expected_score(elo_a, elo_b):
        """Standard Elo expected score: P(A wins) given ratings."""
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    @staticmethod
    def _goal_diff_multiplier(goal_diff):
        """
        World Football Elo goal-difference multiplier.
        Amplifies updates for decisive victories.
        """
        gd = abs(goal_diff)
        if gd <= 1:
            return 1.0
        elif gd == 2:
            return 1.5
        else:
            return (11.0 + gd) / 8.0

    def _get_k(self, tournament):
        """Lookup K-factor by tournament name (fuzzy match)."""
        tournament_lower = tournament.lower().strip()
        for pattern, k in self.TOURNAMENT_K.items():
            if pattern in tournament_lower:
                return k
        return self.DEFAULT_K

    def get_tournament_importance(self, tournament):
        """Get tournament importance weight for AW-MAE."""
        tournament_lower = tournament.lower().strip()
        for pattern, imp in self.TOURNAMENT_IMPORTANCE.items():
            if pattern in tournament_lower:
                return imp
        return self.DEFAULT_IMPORTANCE

    def update_match(self, gender, team_a, team_b, goals_a, goals_b,
                     is_home_a, neutral, tournament):
        """
        Update Elo ratings after a single match.

        Parameters
        ----------
        gender : str ('M' or 'W')
        team_a, team_b : str (team names)
        goals_a, goals_b : int (goals scored)
        is_home_a : bool/int (1 if team_a is home)
        neutral : bool/int (1 if neutral venue)
        tournament : str

        Returns
        -------
        (new_elo_a, new_elo_b) : tuple of floats
        """
        elo_a = self.get_elo(gender, team_a)
        elo_b = self.get_elo(gender, team_b)

        # Adjust for home advantage
        if neutral:
            elo_a_adj, elo_b_adj = elo_a, elo_b
        elif is_home_a:
            elo_a_adj = elo_a + self.home_advantage
            elo_b_adj = elo_b
        else:
            elo_a_adj = elo_a
            elo_b_adj = elo_b + self.home_advantage

        # Expected scores
        exp_a = self.expected_score(elo_a_adj, elo_b_adj)

        # Actual scores (1 = win, 0.5 = draw, 0 = loss)
        if goals_a > goals_b:
            actual_a = 1.0
        elif goals_a == goals_b:
            actual_a = 0.5
        else:
            actual_a = 0.0

        # K-factor & goal difference multiplier
        k = self._get_k(tournament)
        gd_mult = self._goal_diff_multiplier(goals_a - goals_b)

        # Update (zero-sum)
        delta = k * gd_mult * (actual_a - exp_a)
        new_elo_a = elo_a + delta
        new_elo_b = elo_b - delta

        self.ratings[self._key(gender, team_a)] = new_elo_a
        self.ratings[self._key(gender, team_b)] = new_elo_b

        return new_elo_a, new_elo_b
