"""
Pipeline V30: adaptive probabilistic scoreline selector.

This version keeps the proven V27/V28 training stack and improves the final
scoreline layer. V29 used a hard classifier outcome gate; V30 uses a soft
outcome penalty, Poisson-shaped raw-goal likelihood, and rule-aware candidate
scoring so post-processing cannot silently flip the selected outcome.

It never reads test_result.csv or previous submission files when generating
submission_v30.csv.

Usage:
    python pipeline_v30.py
"""

import math
import os
import time
import warnings

import numpy as np
import pandas as pd

import pipeline_v28 as v28


warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(HERE, "..")
DATA_DIR = HERE if os.path.exists(os.path.join(HERE, "train.csv")) else PARENT
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "submission_v30.csv")

MAX_SCORE_CANDIDATE = 8
MAX_TOTAL_CANDIDATE = 10

# Main selector weights. RAW_NLL_W adds a discrete Poisson shape around the raw
# regressor means; RAW_ABS_W keeps the selector from drifting too far.
RAW_ABS_W = 1.05
RAW_NLL_W = 0.16
GOAL_W = 0.40
OUTCOME_W = 0.42
ANCHOR_W = 0.60
PRIOR_W = 0.06
TOTAL_W = 0.12

FAR_RAW_THRESHOLD = 2.50
FAR_RAW_PENALTY = 1.50

# Override behavior is adaptive. The base value is conservative, then the
# selector relaxes near draws and tightens when raw goals are decisive.
BASE_OVERRIDE_MARGIN = 0.15
DRAW_RAW_WINDOW = 0.42
DECISIVE_RAW_DIFF = 0.85


OUTCOME_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_OUTCOME = np.array([-1, 0, 1])


def _outcome_idx(a, b):
    return 2 if a > b else (1 if a == b else 0)


def _poisson_nll(k, mu):
    mu = max(float(mu), 0.05)
    k = int(max(k, 0))
    return mu - k * math.log(mu) + math.lgamma(k + 1)


def _normalize_probs(probs):
    probs = np.asarray(probs, dtype=float)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(probs) / len(probs)
    return probs / total


def candidate_grid(raw_a, raw_b, row):
    candidates = set()
    for a in range(MAX_SCORE_CANDIDATE + 1):
        for b in range(MAX_SCORE_CANDIDATE + 1):
            if a + b <= MAX_TOTAL_CANDIDATE:
                candidates.add((a, b))

    # Preserve the stable V27 anchor and its historical calibration variant.
    base_a, base_b = v28.v27.smart_round_v27(raw_a, raw_b)
    base_a = int(np.clip(base_a, 0, MAX_SCORE_CANDIDATE))
    base_b = int(np.clip(base_b, 0, MAX_SCORE_CANDIDATE))
    candidates.add((base_a, base_b))

    rule_a, rule_b = v28.v27.apply_scoreline_rules(base_a, base_b, row)
    candidates.add((
        int(np.clip(rule_a, 0, MAX_SCORE_CANDIDATE)),
        int(np.clip(rule_b, 0, MAX_SCORE_CANDIDATE)),
    ))

    # Add a dense local neighborhood around raw means. It is redundant for most
    # rows, but protects against future max-total changes and unusual raw means.
    center_a = int(round(max(0.0, raw_a)))
    center_b = int(round(max(0.0, raw_b)))
    for a in range(center_a - 3, center_a + 4):
        for b in range(center_b - 3, center_b + 4):
            if 0 <= a <= MAX_SCORE_CANDIDATE and 0 <= b <= MAX_SCORE_CANDIDATE:
                if a + b <= MAX_TOTAL_CANDIDATE:
                    candidates.add((a, b))

    return candidates


def adaptive_outcome_context(base_outcome, raw_a, raw_b, outcome):
    outcome = _normalize_probs(outcome)
    base_idx = OUTCOME_TO_INDEX[int(base_outcome)]
    clf_idx = int(outcome.argmax())
    clf_outcome = int(INDEX_TO_OUTCOME[clf_idx])
    raw_diff = float(raw_a - raw_b)
    raw_abs = abs(raw_diff)
    raw_outcome = int(np.sign(raw_diff))

    margin = BASE_OVERRIDE_MARGIN
    if raw_abs <= DRAW_RAW_WINDOW:
        margin -= 0.04
    if clf_outcome == 0 and raw_abs <= 0.65:
        margin -= 0.03
    if raw_outcome != 0 and clf_outcome not in (0, raw_outcome) and raw_abs >= DECISIVE_RAW_DIFF:
        margin += 0.10
    if raw_outcome != 0 and clf_outcome == raw_outcome:
        margin -= 0.02
    margin = float(np.clip(margin, 0.05, 0.30))

    override_gap = float(outcome[clf_idx] - outcome[base_idx])
    target = clf_outcome if override_gap > margin else int(base_outcome)

    return {
        "outcome": outcome,
        "base_idx": base_idx,
        "base_outcome": int(base_outcome),
        "clf_idx": clf_idx,
        "clf_outcome": clf_outcome,
        "raw_outcome": raw_outcome,
        "raw_abs": raw_abs,
        "margin": margin,
        "override_gap": override_gap,
        "target_outcome": int(target),
    }


def apply_scoreline_rules_v30(a, b, row, ctx):
    """Apply V27 rules only when their outcome change has model support."""
    outcome = ctx["outcome"]
    raw_abs = ctx["raw_abs"]

    if row["neutral"] == 1 and ((a == 1 and b == 0) or (a == 0 and b == 1)):
        draw_prob = outcome[1]
        win_idx = 2 if a > b else 0
        win_prob = outcome[win_idx]
        draw_supported = (
            raw_abs <= 0.34
            or draw_prob >= win_prob - 0.03
            or ctx["target_outcome"] == 0
        )
        if draw_supported:
            return 1, 1
        return a, b

    # V27 found many 4-0 blowouts were under-scaled. Keep the bump only when
    # strength/raw signals agree; otherwise allow 4-0 to stay 4-0.
    mismatch = row.get("mismatch_blowout", 0)
    if a == 4 and b == 0:
        if (
            raw_abs >= 2.20
            or mismatch >= 1
            or ctx["goal_a"][5:].sum() >= 0.12
            or (ctx["raw_a"] >= 3.35 and ctx["raw_b"] <= 0.70)
        ):
            return 5, 0
    if a == 0 and b == 4:
        if (
            raw_abs >= 2.20
            or mismatch >= 1
            or ctx["goal_b"][5:].sum() >= 0.12
            or (ctx["raw_b"] >= 3.35 and ctx["raw_a"] <= 0.70)
        ):
            return 0, 5
    return a, b


def candidate_loss(a, b, ctx):
    pa = max(v28.bin_prob(ctx["goal_a"], a), 1e-6)
    pb = max(v28.bin_prob(ctx["goal_b"], b), 1e-6)
    prior = max(ctx["prior"], 1e-6)

    raw_abs_loss = abs(a - ctx["raw_a"]) + abs(b - ctx["raw_b"])
    raw_nll_loss = _poisson_nll(a, ctx["raw_a"]) + _poisson_nll(b, ctx["raw_b"])
    goal_loss = -math.log(pa) - math.log(pb)
    outcome_loss = v28.outcome_loss(a, b, ctx["outcome"])
    prior_loss = -math.log(prior)
    total_loss = abs((a + b) - ctx["expected_total"]) / 4.0

    cand_outcome = int(np.sign(a - b))
    cand_idx = _outcome_idx(a, b)
    base_prob = ctx["outcome"][ctx["base_idx"]]
    cand_prob = ctx["outcome"][cand_idx]
    clf_prob = ctx["outcome"][ctx["clf_idx"]]

    # Soft anchor: leaving V27's rounded outcome is allowed, but it needs
    # probability support from the trained outcome classifier.
    anchor_loss = 0.0
    if cand_outcome != ctx["base_outcome"]:
        support = max(0.0, cand_prob - base_prob)
        raw_guard = min(0.55, ctx["raw_abs"] * 0.25)
        anchor_loss += max(0.0, 0.36 + raw_guard - 2.6 * support)

    if cand_outcome != ctx["target_outcome"]:
        confidence_gap = max(0.0, clf_prob - cand_prob)
        anchor_loss += 0.18 + 0.65 * confidence_gap

    if ctx["raw_outcome"] != 0 and cand_outcome not in (0, ctx["raw_outcome"]):
        anchor_loss += max(0.0, ctx["raw_abs"] - 0.55) * 0.32

    loss = (
        RAW_ABS_W * raw_abs_loss
        + RAW_NLL_W * raw_nll_loss
        + GOAL_W * goal_loss
        + OUTCOME_W * outcome_loss
        + ANCHOR_W * anchor_loss
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
    override_rows = 0

    for _, group in test.groupby("match_id", sort=False):
        if len(group) != 2:
            for idx in group.index:
                a, b = v28.v27.smart_round_v27(test.at[idx, "_raw_team"], test.at[idx, "_raw_opp"])
                a = int(np.clip(a, 0, MAX_SCORE_CANDIDATE))
                b = int(np.clip(b, 0, MAX_SCORE_CANDIDATE))
                final_team[idx] = a
                final_opp[idx] = b
            continue

        ia, ib = group.index[0], group.index[1]
        row_a = test.loc[ia]

        raw_a = (test.at[ia, "_raw_team"] + test.at[ib, "_raw_opp"]) / 2.0
        raw_b = (test.at[ib, "_raw_team"] + test.at[ia, "_raw_opp"]) / 2.0

        base_a, base_b = v28.v27.smart_round_v27(raw_a, raw_b)
        base_a, base_b = v28.v27.apply_scoreline_rules(base_a, base_b, row_a)
        base_outcome = int(np.sign(base_a - base_b))

        goal_a = _normalize_probs((probs["team_goal"][ia] + probs["opp_goal"][ib]) / 2.0)
        goal_b = _normalize_probs((probs["team_goal"][ib] + probs["opp_goal"][ia]) / 2.0)

        p_a_win = (probs["outcome"][ia, 2] + probs["outcome"][ib, 0]) / 2.0
        p_draw = (probs["outcome"][ia, 1] + probs["outcome"][ib, 1]) / 2.0
        p_b_win = (probs["outcome"][ia, 0] + probs["outcome"][ib, 2]) / 2.0
        outcome = _normalize_probs(np.array([p_b_win, p_draw, p_a_win]))

        octx = adaptive_outcome_context(base_outcome, raw_a, raw_b, outcome)
        if octx["target_outcome"] != base_outcome:
            override_rows += 2

        base_ctx = {
            "raw_a": raw_a,
            "raw_b": raw_b,
            "goal_a": goal_a,
            "goal_b": goal_b,
            "outcome": outcome,
            "expected_total": v28.expected_total(totals, row_a),
            **octx,
        }

        best = None
        best_loss = np.inf
        seen_adjusted = set()
        for cand_a, cand_b in candidate_grid(raw_a, raw_b, row_a):
            ctx = {
                **base_ctx,
                "prior": v28.prior_prob(priors, row_a, cand_a, cand_b),
            }
            adj_a, adj_b = apply_scoreline_rules_v30(cand_a, cand_b, row_a, ctx)
            adj_a = int(np.clip(adj_a, 0, MAX_SCORE_CANDIDATE))
            adj_b = int(np.clip(adj_b, 0, MAX_SCORE_CANDIDATE))
            if (adj_a, adj_b) in seen_adjusted:
                continue
            seen_adjusted.add((adj_a, adj_b))

            ctx["prior"] = v28.prior_prob(priors, row_a, adj_a, adj_b)
            loss = candidate_loss(adj_a, adj_b, ctx)
            if loss < best_loss:
                best_loss = loss
                best = (adj_a, adj_b)

        a, b = best
        final_team[ia] = a
        final_opp[ia] = b
        final_team[ib] = b
        final_opp[ib] = a

    diagnostics = {"override_rows": override_rows}
    return final_team, final_opp, diagnostics


def validate_submission(test, final_team, final_opp):
    bad_pairs = 0
    for _, group in test.assign(_team=final_team, _opp=final_opp).groupby("match_id", sort=False):
        if len(group) != 2:
            continue
        a = group.iloc[0]
        b = group.iloc[1]
        if int(a["_team"]) != int(b["_opp"]) or int(a["_opp"]) != int(b["_team"]):
            bad_pairs += 1
    return bad_pairs


def main():
    t0 = time.time()
    print("=" * 60)
    print("PIPELINE V30: ADAPTIVE PROBABILISTIC SCORELINE SELECTOR")
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
    final_team, final_opp, diagnostics = select_scorelines(test, raw_team, raw_opp, probs, priors, totals)

    submission = pd.DataFrame({
        "Id": test["Id"],
        "team_goals": final_team,
        "opp_goals": final_opp,
    })
    submission.to_csv(SUBMISSION_PATH, index=False)

    bad_pairs = validate_submission(test, final_team, final_opp)
    print(f"  Predicted team_goals mean={final_team.mean():.3f}, max={final_team.max()}")
    print(f"  Predicted zero rate={(final_team == 0).mean():.3f}")
    print(f"  Predicted high rate={(final_team >= 5).mean():.3f}")
    print(f"  Predicted draws: {(final_team == final_opp).sum():,} rows")
    print(f"  Classifier override rows: {diagnostics['override_rows']:,}")
    print(f"  Inconsistent match pairs: {bad_pairs:,}")
    print(f"  Saved: {SUBMISSION_PATH}")
    print(f"Done in {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
