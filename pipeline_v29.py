"""
Pipeline V29: trained distribution selector with hybrid outcome calibration.

This is a real training pipeline, not a submission stacker. It reads train.csv
and test.csv, reuses the V28 model family, then changes only the scoreline
selection layer:
  - start from V27's rounded outcome as the stable anchor;
  - allow the trained outcome classifier to override when it is clearly better;
  - keep V28's goal-bin classifiers, scoreline priors, and match-level selector.

It never reads test_result.csv or previous submission files when generating
submission_v29.csv.

Usage:
    python solution/pipeline_v29.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

import pipeline_v28 as v28


warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "submission_v29.csv")

MAX_SCORE_CANDIDATE = 8
MAX_TOTAL_CANDIDATE = 10

# V28 was too rigid because it always locked to V27's rounded outcome. Full
# classifier override was too noisy, so V29 uses a conservative margin.
OUTCOME_OVERRIDE_MARGIN = 0.15

RAW_W = 1.15
GOAL_W = 0.45
OUTCOME_W = 0.55
PRIOR_W = 0.06
TOTAL_W = 0.05
FAR_RAW_PENALTY = 2.00
FAR_RAW_THRESHOLD = 2.25


def candidate_grid(raw_a, raw_b, row):
    candidates = set()
    for a in range(MAX_SCORE_CANDIDATE + 1):
        for b in range(MAX_SCORE_CANDIDATE + 1):
            if a + b <= MAX_TOTAL_CANDIDATE:
                candidates.add((a, b))

    # Keep V27's rounded candidate and calibrated blowout/draw rule available.
    base_a, base_b = v28.v27.smart_round_v27(raw_a, raw_b)
    candidates.add((
        int(np.clip(base_a, 0, MAX_SCORE_CANDIDATE)),
        int(np.clip(base_b, 0, MAX_SCORE_CANDIDATE)),
    ))
    rule_a, rule_b = v28.v27.apply_scoreline_rules(base_a, base_b, row)
    candidates.add((
        int(np.clip(rule_a, 0, MAX_SCORE_CANDIDATE)),
        int(np.clip(rule_b, 0, MAX_SCORE_CANDIDATE)),
    ))
    return candidates


def target_outcome(base_outcome, outcome):
    base_idx = {-1: 0, 0: 1, 1: 2}[int(base_outcome)]
    classifier_idx = int(outcome.argmax())
    classifier_outcome = [-1, 0, 1][classifier_idx]
    if outcome[classifier_idx] - outcome[base_idx] > OUTCOME_OVERRIDE_MARGIN:
        return classifier_outcome
    return base_outcome


def candidate_loss(a, b, ctx):
    pa = max(v28.bin_prob(ctx["goal_a"], a), 1e-6)
    pb = max(v28.bin_prob(ctx["goal_b"], b), 1e-6)
    raw_loss = abs(a - ctx["raw_a"]) + abs(b - ctx["raw_b"])
    goal_loss = -np.log(pa) - np.log(pb)
    prior_loss = -np.log(max(ctx["prior"], 1e-6))
    total_loss = abs((a + b) - ctx["expected_total"]) / 4.0
    outcome_loss = v28.outcome_loss(a, b, ctx["outcome"])

    loss = (
        RAW_W * raw_loss
        + GOAL_W * goal_loss
        + OUTCOME_W * outcome_loss
        + PRIOR_W * prior_loss
        + TOTAL_W * total_loss
    )

    if abs(a - ctx["raw_a"]) > FAR_RAW_THRESHOLD or abs(b - ctx["raw_b"]) > FAR_RAW_THRESHOLD:
        loss += FAR_RAW_PENALTY

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
                a, b = v28.v27.smart_round_v27(test.at[idx, "_raw_team"], test.at[idx, "_raw_opp"])
                a, b = v28.v27.apply_scoreline_rules(a, b, test.loc[idx])
                final_team[idx] = int(np.clip(a, 0, MAX_SCORE_CANDIDATE))
                final_opp[idx] = int(np.clip(b, 0, MAX_SCORE_CANDIDATE))
            continue

        ia, ib = group.index[0], group.index[1]
        row_a = test.loc[ia]
        raw_a = (test.at[ia, "_raw_team"] + test.at[ib, "_raw_opp"]) / 2.0
        raw_b = (test.at[ib, "_raw_team"] + test.at[ia, "_raw_opp"]) / 2.0

        base_a, base_b = v28.v27.smart_round_v27(raw_a, raw_b)
        base_a, base_b = v28.v27.apply_scoreline_rules(base_a, base_b, row_a)
        base_outcome = np.sign(base_a - base_b)

        goal_a = (probs["team_goal"][ia] + probs["opp_goal"][ib]) / 2.0
        goal_b = (probs["team_goal"][ib] + probs["opp_goal"][ia]) / 2.0

        p_a_win = (probs["outcome"][ia, 2] + probs["outcome"][ib, 0]) / 2.0
        p_draw = (probs["outcome"][ia, 1] + probs["outcome"][ib, 1]) / 2.0
        p_b_win = (probs["outcome"][ia, 0] + probs["outcome"][ib, 2]) / 2.0
        outcome = np.array([p_b_win, p_draw, p_a_win])
        outcome = outcome / outcome.sum()
        wanted_outcome = target_outcome(base_outcome, outcome)

        best = None
        best_loss = np.inf
        for a, b in candidate_grid(raw_a, raw_b, row_a):
            if np.sign(a - b) != wanted_outcome:
                continue
            ctx = {
                "raw_a": raw_a,
                "raw_b": raw_b,
                "goal_a": goal_a,
                "goal_b": goal_b,
                "outcome": outcome,
                "prior": v28.prior_prob(priors, row_a, a, b),
                "expected_total": v28.expected_total(totals, row_a),
                "base_outcome": base_outcome,
            }
            loss = candidate_loss(a, b, ctx)
            if loss < best_loss:
                best_loss = loss
                best = (a, b)

        a, b = best
        a, b = v28.v27.apply_scoreline_rules(a, b, row_a)
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
    print("PIPELINE V29: TRAINED HYBRID OUTCOME SCORELINE SELECTOR")
    print("=" * 60)

    print("STEP 1: Loading and cleaning data")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train, test = v28.v27.clean_and_sort(train, test)
    print(f"  Train rows: {len(train):,} | Test rows: {len(test):,}")

    print("STEP 2: Building V20/V27 features")
    train, test = v28.v27.add_v20_features(train, test)
    reg_fcols = v28.v27.build_feature_list(train, test)
    cls_fcols = v28.build_classifier_features(train, test)
    v28.v27.add_sample_weight(train)
    print(f"  Regression features: {len(reg_fcols)} | Classifier features: {len(cls_fcols)}")

    print("STEP 3: Training regression base")
    reg_models = v28.v27.fit_models(train, reg_fcols)

    print("STEP 4: Training distribution models")
    cls_models = v28.fit_classifier_models(train, cls_fcols)

    print("STEP 5: Predicting and selecting scorelines")
    raw_team, raw_opp = v28.v27.predict_raw(reg_models, test, reg_fcols)
    probs = v28.predict_classifier_probs(cls_models, test, cls_fcols)
    priors, totals = v28.build_scoreline_priors(train)
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
