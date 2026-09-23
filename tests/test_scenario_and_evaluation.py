"""Auswertung: Analyse einer Instanz, Steuerungs-Kurve und die beiden Experimente - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import sk_constants as C
import sk_evaluation as E


def test_analyse_default_is_consistent():
    a = E.analyse(E.Settings(n=6))
    lo, hi = a.nash_goal_range()
    assert a.leaders
    assert a.opt_res["goal"] <= lo + 1e-9 and a.pes_res["goal"] >= a.opt_res["goal"] - 1e-9 and lo <= hi + 1e-12
    slo, shi = a.nash_social_range()
    assert a.opt_social <= slo + 1e-9 and slo <= shi + 1e-12


def test_goal_is_the_leaders_own_cost_or_the_social_cost():
    a = E.analyse(E.Settings(n=6, k=2, objective="own"))
    assert np.allclose(a.goal(), a.st.own[:, a.leaders].sum(axis=1))
    b = E.analyse(E.Settings(n=6, k=2, objective="social"))
    assert np.allclose(b.goal(), b.st.social)
    c = E.analyse(E.Settings(n=6, k=0, objective="own"))
    assert np.allclose(c.goal(), c.st.social)


def test_commitment_table_is_sorted_and_complete():
    a = E.analyse(E.Settings(n=6, k=2, objective="own"))
    rows, total = a.commitment_table(top=100)
    assert total == 9 and len(rows) == 9 and all(r1[1] <= r2[1] + 1e-12 for r1, r2 in zip(rows, rows[1:]))
    assert rows[0][1] == pytest.approx(a.opt_res["goal"])
    assert all(len(r[0]) == 2 and len(r[3]) == 6 for r in rows)
    for gates, _, _, assign in rows:
        assert all(assign[i] == gates[j] for j, i in enumerate(a.leaders))                # die Anführer halten ihre Festlegung ein
    pes_rows, _ = a.commitment_table("pessimistic", top=100)
    assert pes_rows[0][1] == pytest.approx(a.pes_res["goal"])


def test_control_curve_shape_and_endpoints():
    rows = E.control_curve(E.Settings(n=6, rule="largest"))
    assert [r["k"] for r in rows] == list(range(7)) and rows[0]["share"] == 0.0 and rows[-1]["share"] == pytest.approx(1.0)
    assert rows[-1]["optimistic"] == pytest.approx(1.0) and rows[-1]["pessimistic"] == pytest.approx(1.0)
    assert all(r["pessimistic"] >= r["optimistic"] - 1e-9 and r["optimistic"] >= 1 - 1e-9 for r in rows)
    assert all(y2["share"] >= y1["share"] for y1, y2 in zip(rows, rows[1:]))


def test_first_mover_experiment_shape_and_invariants():
    rows = E.first_mover_experiment(ks=(1, 2), rules=("largest",), n=6, seeds=range(830000, 830010))
    assert [(r["rule"], r["k"]) for r in rows] == [("largest", 1), ("largest", 2)]
    for r in rows:
        assert r["mean_gain_vs_best"] >= -1e-12 and r["mean_gain_vs_worst"] >= -1e-12 and 0 <= r["share_gain_vs_best"] <= 1 and 0 <= r["share_pes_worse_than_best"] <= 1
        assert r["social_nash_best"] <= r["social_nash_worst"] + 1e-12 and r["social_nash_best"] >= 1 - 1e-12


def test_control_experiment_shape_and_invariants():
    out = E.control_experiment(rules=("largest", "smallest"), n=5, seeds=range(840000, 840004))
    for rule in ("largest", "smallest"):
        v = out[rule]
        assert len(v["k"]) == 6 and v["share"][0] == 0.0 and v["optimistic"][-1] == pytest.approx(1.0) and np.all(v["pessimistic"] >= v["optimistic"] - 1e-9)
    assert out["largest"]["share"][2] >= out["smallest"]["share"][2]


def test_experiments_are_reproducible():
    a = E.control_experiment(rules=("largest",), n=5, seeds=range(840000, 840003))
    b = E.control_experiment(rules=("largest",), n=5, seeds=range(840000, 840003))
    assert np.array_equal(a["largest"]["optimistic"], b["largest"]["optimistic"])
