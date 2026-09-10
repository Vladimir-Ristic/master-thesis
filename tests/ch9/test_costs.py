"""The cost function is the answer to half of RQ1, so its arithmetic is checked
against hand-computed values rather than against itself."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import config as C, costs


Y = np.array([1, 1, 0, 0])
AMT = np.array([100.0, 2.0, 500.0, 30.0])


def test_example_dependent_cost_is_hand_computable():
    # Review the first and third transactions: TP (ca) + FP (ca) + FN (2.0).
    yhat = np.array([1, 0, 1, 0])
    expected = C.CA_REVIEW + C.CA_REVIEW + 2.0
    assert costs.example_dependent_cost(Y, yhat, AMT) == pytest.approx(expected)


def test_reviewing_nothing_costs_the_full_fraud_amount():
    yhat = np.zeros(4, dtype=np.int8)
    assert costs.example_dependent_cost(Y, yhat, AMT) == pytest.approx(102.0)


def test_reviewing_everything_costs_the_review_fee_only():
    yhat = np.ones(4, dtype=np.int8)
    assert costs.example_dependent_cost(Y, yhat, AMT) == pytest.approx(4 * C.CA_REVIEW)


def test_fixed_ratio_cost_is_hand_computable():
    yhat = np.array([1, 0, 1, 0])
    expected = C.FIXED_COST_FN * 1 + C.FIXED_COST_FP * 1
    assert costs.fixed_ratio_cost(Y, yhat) == pytest.approx(expected)


def test_savings_are_zero_for_the_do_nothing_policy():
    base = costs.baseline_costs(Y, AMT)
    cost = costs.example_dependent_cost(Y, np.zeros(4, dtype=np.int8), AMT)
    assert costs.savings(cost, base["ed_none"]) == pytest.approx(0.0)


def test_transactions_below_the_review_cost_are_never_reviewed():
    """Under this matrix a $2 fraud is not worth a $4 review, at any probability."""
    p = np.array([0.99, 0.99, 0.99, 0.99])
    amount = np.array([100.0, 2.0, 3.99, 500.0])
    yhat = costs.elkan_predictions(p, amount)
    assert yhat.tolist() == [1, 0, 0, 1]


def test_per_transaction_rule_is_row_wise_optimal():
    """The analytic rule must match a brute-force per-row cost comparison."""
    rng = np.random.default_rng(0)
    n = 2_000
    p = rng.random(n)
    amount = np.exp(rng.normal(4, 1.2, n))
    y = (rng.random(n) < p).astype(np.int8)

    analytic = costs.elkan_predictions(p, amount)
    # Expected cost of reviewing row i is ca; of not reviewing it, p_i * amt_i.
    brute = (p * amount > C.CA_REVIEW).astype(np.int8)
    assert np.array_equal(analytic, brute)
    assert costs.example_dependent_cost(y, analytic, amount) <= costs.example_dependent_cost(
        y, np.zeros(n, dtype=np.int8), amount
    )


def test_fixed_ratio_threshold_matches_elkan_formula():
    assert costs.fixed_ratio_threshold(100.0, 1.0) == pytest.approx(1 / 101)


def test_global_threshold_search_finds_a_minimum():
    rng = np.random.default_rng(1)
    n = 5_000
    y = (rng.random(n) < 0.035).astype(np.int8)
    p = np.clip(0.02 + 0.5 * y + rng.normal(0, 0.12, n), 1e-6, 1 - 1e-6)
    amount = np.exp(rng.normal(4, 1.0, n))

    t, cost = costs.best_global_threshold(y, p, amount, cost_model="ed")
    curve = costs.cost_curve(y, p, amount)
    assert cost <= curve["cost_ed"].min() + 1e-9
    for other in (0.5, 0.1, 0.01):
        alt = costs.example_dependent_cost(y, (p >= other).astype(np.int8), amount)
        assert cost <= alt + 1e-9


def test_the_two_cost_models_can_disagree_about_the_threshold():
    """If they always agreed, reporting both would be padding."""
    rng = np.random.default_rng(2)
    n = 8_000
    y = (rng.random(n) < 0.035).astype(np.int8)
    p = np.clip(0.02 + 0.4 * y + rng.normal(0, 0.1, n), 1e-6, 1 - 1e-6)
    amount = np.exp(rng.normal(4, 1.4, n))
    t_ed, _ = costs.best_global_threshold(y, p, amount, cost_model="ed")
    t_fx, _ = costs.best_global_threshold(y, p, amount, cost_model="fx")
    assert t_ed != t_fx


def test_all_decision_rules_beats_the_default_threshold():
    rng = np.random.default_rng(3)
    n = 6_000
    y = (rng.random(n) < 0.035).astype(np.int8)
    p = np.clip(0.02 + 0.45 * y + rng.normal(0, 0.1, n), 1e-6, 1 - 1e-6)
    amount = np.exp(rng.normal(4, 1.1, n))

    rules = {d.rule: d for d in costs.all_decision_rules(y, p, amount)}
    assert len(rules) == 5
    default = rules["default 0.50"]
    chosen = rules["example-dependent, global"]
    assert chosen.cost_ed <= default.cost_ed


def test_review_cost_sensitivity_is_monotone_in_the_review_cost():
    """A dearer review can only justify reviewing fewer transactions."""
    rng = np.random.default_rng(4)
    n = 6_000
    y = (rng.random(n) < 0.035).astype(np.int8)
    p = np.clip(0.02 + 0.45 * y + rng.normal(0, 0.1, n), 1e-6, 1 - 1e-6)
    amount = np.exp(rng.normal(4, 1.1, n))

    rows = costs.ca_sensitivity(y, p, amount, ca_values=(1.0, 4.0, 16.0, 64.0))
    rates = [r["elkan_review_rate"] for r in rows]
    assert rates == sorted(rates, reverse=True)
