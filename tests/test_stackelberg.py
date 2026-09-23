"""Stackelberg-Kern: Zustände gegen die Definition, Folger-Gleichgewichte und Festlegungen gegen eine unabhängige Schleife, Handrechnung im 2x2-Fall, Eigenschaften (Nash-Schranke, Auswahlregeln)."""

import itertools

import numpy as np
import pytest

import sk_bimatrix as B
import sk_gates as G
import sk_stackelberg as S


def cost_by_definition(inst, assign, i):
    load = sum(inst.w[j] for j in range(inst.n) if assign[j] == assign[i])
    return inst.a[assign[i]] + inst.b[assign[i]] * load


def deviation_by_definition(inst, assign, i, h):
    moved = list(assign)
    moved[i] = h
    return cost_by_definition(inst, moved, i)


def brute_force(inst, leaders, objective, selection):
    """Stackelberg direkt aus der Definition: Festlegungen der Anführer durchprobieren, Folger-Gleichgewichte durch vollständiges Durchsuchen der Folger-Zuordnungen."""
    n, m = inst.n, inst.m
    followers = [i for i in range(n) if i not in leaders]
    best = None
    for s_l in itertools.product(range(m), repeat=len(leaders)):
        outcomes = []
        for s_f in itertools.product(range(m), repeat=len(followers)):
            s = [0] * n
            for pos, i in enumerate(leaders):
                s[i] = s_l[pos]
            for pos, i in enumerate(followers):
                s[i] = s_f[pos]
            if all(cost_by_definition(inst, s, i) <= min(deviation_by_definition(inst, s, i, h) for h in range(m)) + 1e-9 for i in followers):
                if objective == "own" and leaders:
                    outcomes.append(sum(cost_by_definition(inst, s, i) for i in leaders))
                else:
                    outcomes.append(sum(cost_by_definition(inst, s, i) for i in range(n)))
        if not outcomes:
            continue
        val = min(outcomes) if selection == "optimistic" else max(outcomes)
        best = val if best is None else min(best, val)
    return best


def test_states_agree_with_the_definition():
    inst = G.generate(4, 3, "mixed", 3)
    st = S.States(inst)
    for k in range(st.N):
        s = [(k // inst.m ** j) % inst.m for j in range(inst.n)]
        assert list(st.A[k]) == s
        for i in range(inst.n):
            assert st.own[k, i] == pytest.approx(cost_by_definition(inst, s, i))
            for h in range(inst.m):
                assert st.dev[k, i, h] == pytest.approx(deviation_by_definition(inst, s, i, h))


def test_too_many_assignments_raise_and_unknown_options_raise():
    with pytest.raises(ValueError):
        S.States(G.generate(11, 3, "mixed", 0))
    inst = G.generate(4, 3, "mixed", 1)
    st = S.States(inst)
    with pytest.raises(ValueError):
        S.solve(st, [0], "bogus", "optimistic")
    with pytest.raises(ValueError):
        S.solve(st, [0], "own", "bogus")
    with pytest.raises(ValueError):
        S.leaders_by_rule(inst, 1, "bogus")


def test_leaders_by_rule():
    inst = G.Instance(np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([1.0, 3.0, 2.0, 3.0, 1.0]), 0)
    assert S.leaders_by_rule(inst, 2, "largest") == [1, 3]
    assert S.leaders_by_rule(inst, 2, "smallest") == [0, 4]
    assert S.leaders_by_rule(inst, 3, "index") == [0, 1, 2]
    assert S.leaders_by_rule(inst, 0, "largest") == []


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("objective", ["own", "social"])
@pytest.mark.parametrize("selection", ["optimistic", "pessimistic"])
def test_solve_agrees_with_the_brute_force_definition(seed, objective, selection):
    inst = G.generate(5, 2, "mixed", seed)
    st = S.States(inst)
    for k in (1, 2):
        leaders = S.leaders_by_rule(inst, k, "largest")
        expected = brute_force(inst, leaders, objective, selection)
        res = S.solve(st, leaders, objective, selection)
        assert res["goal"] == pytest.approx(expected, abs=1e-9)


def test_follower_ne_mask_agrees_with_the_scalar_definition():
    inst = G.generate(5, 3, "mixed", 4)
    st = S.States(inst)
    followers = [1, 3, 4]
    mask = st.is_ne_for(followers)
    for k in range(st.N):
        s = [(k // inst.m ** j) % inst.m for j in range(inst.n)]
        by_def = all(cost_by_definition(inst, s, i) <= min(deviation_by_definition(inst, s, i, h) for h in range(inst.m)) + 1e-9 for i in followers)
        assert bool(mask[k]) == by_def


@pytest.mark.parametrize("seed", range(8))
def test_optimistic_stackelberg_is_never_worse_than_the_best_nash_equilibrium(seed):
    inst = G.generate(6, 3, "mixed", seed)
    st = S.States(inst)
    nash = st.is_ne_for(range(inst.n))
    for k in (1, 2, 3):
        leaders = S.leaders_by_rule(inst, k, "largest")
        for objective in ("own", "social"):
            goal = st.own[:, leaders].sum(axis=1) if objective == "own" else st.social
            res = S.solve(st, leaders, objective, "optimistic")
            assert res["goal"] <= goal[nash].min() + 1e-9
            pes = S.solve(st, leaders, objective, "pessimistic")
            assert pes["goal"] >= res["goal"] - 1e-9


def test_without_leaders_the_value_is_the_nash_range():
    inst = G.generate(6, 3, "mixed", 2)
    st = S.States(inst)
    nash = st.is_ne_for(range(inst.n))
    assert S.solve(st, [], "social", "optimistic")["goal"] == pytest.approx(st.social[nash].min())
    assert S.solve(st, [], "social", "pessimistic")["goal"] == pytest.approx(st.social[nash].max())
    assert S.solve(st, [], "own", "optimistic")["n_keys"] == 1


def test_all_trucks_leading_gives_the_optimum():
    inst = G.generate(5, 3, "mixed", 6)
    st = S.States(inst)
    for objective in ("own", "social"):
        res = S.solve(st, list(range(inst.n)), objective, "pessimistic")
        assert res["goal"] == pytest.approx(st.social.min()) and res["social"] == pytest.approx(st.social.min())


def test_solve_reports_a_consistent_state():
    inst = G.generate(5, 3, "mixed", 7)
    st = S.States(inst)
    leaders = [0, 2]
    res = S.solve(st, leaders, "own", "optimistic")
    state = res["state"]
    assert res["leader_cost"] == pytest.approx(st.own[state, leaders].sum()) and res["social"] == pytest.approx(st.social[state])
    assert st.is_ne_for([1, 3, 4])[state]
    assert res["n_keys"] == 9 and len(res["commitments"]) <= 9


def two_by_two(delta):
    return G.Instance(np.array([5.0, 5.0 + delta]), np.array([2.0, 2.0]), np.array([1.0, 1.0]), 0)


@pytest.mark.parametrize("delta", [0.0, 1.0, 2.0, 3.0])
def test_two_by_two_stackelberg_agrees_with_the_bimatrix_module(delta):
    K1, K2 = B.cost_matrices(delta)
    rows, best = B.stackelberg(K1, K2)
    inst = two_by_two(delta)
    st = S.States(inst)
    res = S.solve(st, [0], "own", "optimistic")
    assert res["goal"] == pytest.approx(rows[best][2], abs=1e-9)
    for g1, g2, c1, c2 in rows:
        assert K1[g1, g2] == c1 and K2[g1, g2] == c2
