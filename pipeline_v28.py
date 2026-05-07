"""
Pipeline V28: distribution-based scoreline selector.

This pipeline does not read test_result.csv or external result tables. It uses:
  - V27/V20 regression base for expected goals.
  - Goal-bin classifiers for P(goals = 0, 1, ..., 5, 6+).
  - Outcome classifier for win/draw/loss consistency.
  - Smoothed gender+tournament scoreline priors from train.csv.
  - A match-level selector over candidate scorelines, so paired rows remain
    consistent.

Usage:
    python solution/pipeline_v28.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

import pipeline_v20 as v20
import pipeline_v27 as v27


warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "submission_v28.csv")

RANDOM_STATE = 42
MAX_BIN = 6
MAX_SCORE_CANDIDATE = 8

GOAL_CLS_PARAMS = dict(
    objective="multiclass",
    num_class=MAX_BIN + 1,
    metric="multi_logloss",
    boosting_type="gbdt",
    n_estimators=1400,
    learning_rate=0.025,
    num_leaves=31,
    max_depth=6,
    min_child_samples=45,
    subsample=0.85,
    colsample_bytree=0.78,
    reg_alpha=0.8,
    reg_lambda=2.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

OUTCOME_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    metric="multi_logloss",
    boosting_type="gbdt",
    n_estimators=1000,
    learning_rate=0.025,
    num_leaves=31,
    max_depth=6,
    min_child_samples=60,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

# Selector weights. Raw expected goals anchor the search; goal-bin probabilities
# and priors correct the under-predicted zero/high-score distribution.
RAW_W = 1.20
GOAL_W = 0.45
OUTCOME_W = 0.25
PRIOR_W = 0.06
TOTAL_W = 0.05


def build_classifier_features(train, test):
    extra = [
        "elo_abs_diff", "elo_ratio", "draw_proxy", "goal_pressure", "defense_gap",
        "rest_diff", "is_women", "women_x_elo_abs_diff", "mismatch_blowout",
    ]
    cols = list(v20.FEATURE_COLS) + extra
    return [c for c in cols if c in train.columns and c in test.columns]


def cap_goal(values):
    return np.minimum(values.astype(int), MAX_BIN)


def fit_classifier_models(train, fcols):
    X = train[fcols]
    w = train["sample_weight"].copy()

    # Mildly up-weight rare tails for calibration. This affects probabilities,
    # not direct score output, because final selection still uses raw-goal anchor.
    high = (train["target_team_goals_raw"] >= 5) | (train["target_opp_goals_raw"] >= 5)
    nil = (train["target_team_goals_raw"] == 0) | (train["target_opp_goals_raw"] == 0)
    w.loc[high] *= 1.30
    w.loc[nil] *= 1.08

    y_team_bin = cap_goal(train["target_team_goals_raw"])
    y_opp_bin = cap_goal(train["target_opp_goals_raw"])
    y_outcome = np.sign(train["target_team_goals_raw"] - train["target_opp_goals_raw"]).map({-1: 0, 0: 1, 1: 2})

    print("  Training goal-bin classifiers...")
    team_goal = lgb.LGBMClassifier(**GOAL_CLS_PARAMS)
    opp_goal = lgb.LGBMClassifier(**GOAL_CLS_PARAMS)
    team_goal.fit(X, y_team_bin, sample_weight=w)
    opp_goal.fit(X, y_opp_bin, sample_weight=w)

    print("  Training outcome classifier...")
    outcome = lgb.LGBMClassifier(**OUTCOME_PARAMS)
    outcome.fit(X, y_outcome, sample_weight=train["sample_weight"])

    return {"team_goal": team_goal, "opp_goal": opp_goal, "outcome": outcome}


def predict_classifier_probs(models, test, fcols):
    X = test[fcols]
    return {
        "team_goal": models["team_goal"].predict_proba(X),
        "opp_goal": models["opp_goal"].predict_proba(X),
        "outcome": models["outcome"].predict_proba(X),
    }


def bin_prob(probs, goals):
    if goals >= MAX_BIN:
        return probs[MAX_BIN]
    return probs[goals]


def build_scoreline_priors(train):
    priors = {}
    totals = {}

    work = train.copy()
    work["_tg"] = work["target_team_goals_raw"].clip(upper=MAX_SCORE_CANDIDATE).astype(int)
    work["_og"] = work["target_opp_goals_raw"].clip(upper=MAX_SCORE_CANDIDATE).astype(int)
    global_counts = work.groupby(["_tg", "_og"]).size()
    global_total = float(global_counts.sum())

    for (tg, og), count in global_counts.items():
        priors[("ALL", "ALL", int(tg), int(og))] = count / global_total

    for key_cols in [["gender"], ["gender", "tournament"]]:
        grouped = work.groupby(key_cols + ["_tg", "_og"]).size().reset_index(name="count")
        group_totals = work.groupby(key_cols).size().to_dict()
        for _, row in grouped.iterrows():
            if key_cols == ["gender"]:
                key = (row["gender"], "ALL", int(row["_tg"]), int(row["_og"]))
                total = group_totals[row["gender"]]
            else:
                key = (row["gender"], row["tournament"], int(row["_tg"]), int(row["_og"]))
                total = group_totals[(row["gender"], row["tournament"])]
            global_prior = priors.get(("ALL", "ALL", int(row["_tg"]), int(row["_og"])), 1e-6)
            priors[key] = (row["count"] + 30.0 * global_prior) / (total + 30.0)

    total_counts = work.groupby(["gender", "tournament"])["_tg"].agg(["count", "mean"]).reset_index()
    global_mean = work["_tg"].mean()
    for row in total_counts.itertuples(index=False):
        totals[(row.gender, row.tournament)] = (row.count * row.mean + 80.0 * global_mean) / (row.count + 80.0)
    for gender, group in work.groupby("gender"):
        totals[(gender, "ALL")] = group["_tg"].mean()
    totals[("ALL", "ALL")] = global_mean
    return priors, totals


def prior_prob(priors, row, a, b):
    a = min(int(a), MAX_SCORE_CANDIDATE)
    b = min(int(b), MAX_SCORE_CANDIDATE)
    return (
        priors.get((row["gender"], row["tournament"], a, b))
        or priors.get((row["gender"], "ALL", a, b))
        or priors.get(("ALL", "ALL", a, b))
        or 1e-6
    )


def expected_total(totals, row):
    return (
        totals.get((row["gender"], row["tournament"]))
        or totals.get((row["gender"], "ALL"))
        or totals.get(("ALL", "ALL"))
        or 1.5
    ) * 2.0


def candidate_grid(raw_a, raw_b, row):
    candidates = set()
    for a in range(MAX_SCORE_CANDIDATE + 1):
        for b in range(MAX_SCORE_CANDIDATE + 1):
            if a + b <= 10:
                candidates.add((a, b))

    # Always include V27's scoreline and its calibrated rule variant.
    base_a, base_b = v27.smart_round_v27(raw_a, raw_b)
    candidates.add((int(np.clip(base_a, 0, MAX_SCORE_CANDIDATE)), int(np.clip(base_b, 0, MAX_SCORE_CANDIDATE))))
    candidates.add(v27.apply_scoreline_rules(base_a, base_b, row))
    return candidates


def outcome_loss(a, b, probs):
    idx = 2 if a > b else (1 if a == b else 0)
    return -np.log(max(probs[idx], 1e-6))


def candidate_loss(a, b, ctx):
    pa = max(bin_prob(ctx["goal_a"], a), 1e-6)
    pb = max(bin_prob(ctx["goal_b"], b), 1e-6)
    raw_loss = abs(a - ctx["raw_a"]) + abs(b - ctx["raw_b"])
    goal_loss = -np.log(pa) - np.log(pb)
    prior_loss = -np.log(max(ctx["prior"], 1e-6))
    total_loss = abs((a + b) - ctx["expected_total"]) / 4.0
    loss = (
        RAW_W * raw_loss
        + GOAL_W * goal_loss
        + OUTCOME_W * outcome_loss(a, b, ctx["outcome"])
        + PRIOR_W * prior_loss
        + TOTAL_W * total_loss
    )
    cand_outcome = np.sign(a - b)
    if cand_outcome != ctx["base_outcome"]:
        loss += 10.0
    if abs(a - ctx["raw_a"]) > 2.25 or abs(b - ctx["raw_b"]) > 2.25:
        loss += 2.0
    return loss


def select_scorelines(test, raw_team, raw_opp, probs, priors, totals):
    test = test.copy()
    test["_raw_team"] = raw_team
    test["_raw_opp"] = raw_opp
    final_team = np.zeros(len(test), dtype=int)
    final_opp = np.zeros(len(test), dtype=int)

    for _, group in test.groupby("match_id", sort=False):
        if len(group) != 2:
            for idx in group.index:
                a, b = v27.smart_round_v27(test.at[idx, "_raw_team"], test.at[idx, "_raw_opp"])
                a, b = v27.apply_scoreline_rules(a, b, test.loc[idx])
                final_team[idx] = int(np.clip(a, 0, MAX_SCORE_CANDIDATE))
                final_opp[idx] = int(np.clip(b, 0, MAX_SCORE_CANDIDATE))
            continue

        ia, ib = group.index[0], group.index[1]
        row_a = test.loc[ia]
        raw_a = (test.at[ia, "_raw_team"] + test.at[ib, "_raw_opp"]) / 2.0
        raw_b = (test.at[ib, "_raw_team"] + test.at[ia, "_raw_opp"]) / 2.0
        base_a, base_b = v27.smart_round_v27(raw_a, raw_b)
        base_a, base_b = v27.apply_scoreline_rules(base_a, base_b, row_a)
        base_outcome = np.sign(base_a - base_b)

        goal_a = (probs["team_goal"][ia] + probs["opp_goal"][ib]) / 2.0
        goal_b = (probs["team_goal"][ib] + probs["opp_goal"][ia]) / 2.0

        p_a_win = (probs["outcome"][ia, 2] + probs["outcome"][ib, 0]) / 2.0
        p_draw = (probs["outcome"][ia, 1] + probs["outcome"][ib, 1]) / 2.0
        p_b_win = (probs["outcome"][ia, 0] + probs["outcome"][ib, 2]) / 2.0
        outcome = np.array([p_b_win, p_draw, p_a_win])
        outcome = outcome / outcome.sum()

        best = None
        best_loss = np.inf
        for a, b in candidate_grid(raw_a, raw_b, row_a):
            if np.sign(a - b) != base_outcome:
                continue
            ctx = {
                "raw_a": raw_a,
                "raw_b": raw_b,
                "goal_a": goal_a,
                "goal_b": goal_b,
                "outcome": outcome,
                "prior": prior_prob(priors, row_a, a, b),
                "expected_total": expected_total(totals, row_a),
                "base_outcome": base_outcome,
            }
            loss = candidate_loss(a, b, ctx)
            if loss < best_loss:
                best_loss = loss
                best = (a, b)

        a, b = best
        a, b = v27.apply_scoreline_rules(a, b, row_a)
        a = int(np.clip(a, 0, MAX_SCORE_CANDIDATE))
        b = int(np.clip(b, 0, MAX_SCORE_CANDIDATE))
        final_team[ia] = a
        final_opp[ia] = b
        final_team[ib] = b
        final_opp[ib] = a

    return final_team, final_opp


def main():
    t0 = time.time()
    print("=" * 60)
    print("PIPELINE V28: GOAL DISTRIBUTION SCORELINE SELECTOR")
    print("=" * 60)

    print("STEP 1: Loading and cleaning data")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train, test = v27.clean_and_sort(train, test)
    print(f"  Train rows: {len(train):,} | Test rows: {len(test):,}")

    print("STEP 2: Building V20/V27 features")
    train, test = v27.add_v20_features(train, test)
    reg_fcols = v27.build_feature_list(train, test)
    cls_fcols = build_classifier_features(train, test)
    v27.add_sample_weight(train)
    print(f"  Regression features: {len(reg_fcols)} | Classifier features: {len(cls_fcols)}")

    print("STEP 3: Training regression base")
    reg_models = v27.fit_models(train, reg_fcols)

    print("STEP 4: Training distribution models")
    cls_models = fit_classifier_models(train, cls_fcols)

    print("STEP 5: Predicting and selecting scorelines")
    raw_team, raw_opp = v27.predict_raw(reg_models, test, reg_fcols)
    probs = predict_classifier_probs(cls_models, test, cls_fcols)
    priors, totals = build_scoreline_priors(train)
    final_team, final_opp = select_scorelines(test, raw_team, raw_opp, probs, priors, totals)

    submission = pd.DataFrame({
        "Id": test["Id"],
        "team_goals": final_team,
        "opp_goals": final_opp,
    })
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"  Predicted team_goals mean={final_team.mean():.3f}, max={final_team.max()}")
    print(f"  Predicted zero rate={(final_team == 0).mean():.3f}")
    print(f"  Predicted high rate={(final_team >= 5).mean():.3f}")
    print(f"  Predicted draws: {(final_team == final_opp).sum():,} rows")
    print(f"  Saved: {SUBMISSION_PATH}")
    print(f"Done in {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
