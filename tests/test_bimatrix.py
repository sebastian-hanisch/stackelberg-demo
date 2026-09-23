"""Mini-Spiel: Handrechnung (Kostenmatrizen, reine und gemischtes Gleichgewicht, Erstzug von Lkw 1)."""

import numpy as np
import pytest

import sk_bimatrix as B


def test_cost_matrices_by_hand_at_delta_one():
    K1, K2 = B.cost_matrices(1.0)
    assert K1.tolist() == [[9.0, 7.0], [8.0, 10.0]]         # Zeile = Tor Lkw 1, Spalte = Tor Lkw 2
    assert K2.tolist() == [[9.0, 8.0], [7.0, 10.0]]


def test_pure_and_mixed_equilibria_by_hand():
    K1, K2 = B.cost_matrices(1.0)
    assert B.pure_equilibria(K1, K2) == [(0, 1), (1, 0)]
    p, q = B.mixed_equilibrium(K1, K2)
    assert p == pytest.approx(0.75) and q == pytest.approx(0.75)
    assert B.expected_costs(K1, K2, np.outer([p, 1 - p], [q, 1 - q])) == pytest.approx((8.5, 8.5))


def test_dominant_gate_for_delta_two_or_more():
    for delta in (2.5, 3.0, 4.0):
        K1, K2 = B.cost_matrices(delta)
        assert B.pure_equilibria(K1, K2) == [(0, 0)] and B.mixed_equilibrium(K1, K2) is None


def test_stackelberg_by_hand_at_delta_one():
    """Legt sich Lkw 1 auf A fest, antwortet Lkw 2 mit B (8 statt 9): Kosten (7, 8). Auf B: Lkw 2 antwortet mit A (7 statt 10): (8, 7). Lkw 1 wählt A."""
    K1, K2 = B.cost_matrices(1.0)
    assert B.follower_responses(K2, 0) == [1] and B.follower_responses(K2, 1) == [0]
    rows, best = B.stackelberg(K1, K2)
    assert rows == [(0, 1, 7.0, 8.0), (1, 0, 8.0, 7.0)] and best == 0


def test_the_leader_gets_the_better_pure_equilibrium_and_the_follower_the_worse():
    for delta in (0.5, 1.0, 1.5):
        K1, K2 = B.cost_matrices(delta)
        rows, best = B.stackelberg(K1, K2)
        assert (rows[best][0], rows[best][1]) == (0, 1) and rows[best][2] < rows[best][3]


def test_dominant_case_the_first_move_changes_nothing():
    K1, K2 = B.cost_matrices(3.0)
    rows, best = B.stackelberg(K1, K2)
    assert (rows[best][0], rows[best][1]) == (0, 0) and rows[best][2] == K1[0, 0] and rows[best][3] == K2[0, 0]


def test_first_move_beats_the_mixed_equilibrium_for_both_trucks_at_delta_one():
    K1, K2 = B.cost_matrices(1.0)
    p, q = B.mixed_equilibrium(K1, K2)
    mixed = B.expected_costs(K1, K2, np.outer([p, 1 - p], [q, 1 - q]))
    rows, best = B.stackelberg(K1, K2)
    assert rows[best][2] < mixed[0] and rows[best][3] < mixed[1]


def test_ties_follow_the_selection_rule():
    """delta = 2: Lkw 2 ist gegenüber Lkw 1 auf A indifferent (9 auf A und 9 auf B). Optimistisch antwortet er mit B (Lkw 1: 7), pessimistisch mit A (Lkw 1: 9)."""
    K1, K2 = B.cost_matrices(2.0)
    assert B.follower_responses(K2, 0) == [0, 1]
    assert B.stackelberg(K1, K2, "optimistic")[0][0] == (0, 1, 7.0, 9.0)
    assert B.stackelberg(K1, K2, "pessimistic")[0][0] == (0, 0, 9.0, 9.0)
    with pytest.raises(ValueError):
        B.stackelberg(K1, K2, "bogus")
