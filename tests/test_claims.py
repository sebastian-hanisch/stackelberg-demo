"""Jede im README/PRESET_HELP/App genannte Zahl wird hier nachgerechnet - keine Behauptung ohne Test.

Alle Werte sind exakte Aufzählungen fester Instanzen (kein Zufall im Verfahren): Einzelinstanzen mit Toleranz für Rundung, Mittel über Instanzen mit Bändern
(feedback_ci_platform_robust_tests / feedback_ci_unpinned_numeric_asserts)."""

import numpy as np
import pytest

import sk_bimatrix as B
import sk_constants as C
import sk_evaluation as E


def _preset(name):
    p = C.PRESETS[name]
    return E.analyse(E.Settings(p["n"], p["m"], p["size_mode"], p["seed"], p["rule"], p["k"], p["objective"], p["selection"]))


# --- PRESET_HELP: exakte Werte ---------------------------------------------------------------------------------------------------------------


def test_standardfall_numbers():
    a = _preset("Standardfall (ein Anführer)")
    assert a.leaders == [1] and a.inst.w[1] == 3.0
    assert a.nash_goal_range() == pytest.approx((12.35, 13.30), abs=0.006)
    assert a.opt_res["goal"] == pytest.approx(12.35, abs=0.006) and a.pes_res["goal"] == pytest.approx(12.35, abs=0.006)
    assert (a.nash_goal_range()[1] - a.opt_res["goal"]) / a.nash_goal_range()[1] == pytest.approx(0.072, abs=0.002)


def test_pessimistische_folger_numbers():
    a = _preset("Pessimistische Folger")
    assert a.nash_goal_range() == pytest.approx((12.57, 14.89), abs=0.006)
    assert a.opt_res["goal"] == pytest.approx(12.57, abs=0.006) and a.pes_res["goal"] == pytest.approx(14.35, abs=0.006)


def test_disposition_3_numbers():
    a = _preset("Disposition steuert 3 Lkw")
    assert a.leaders == [1, 2, 3] and a.inst.w[a.leaders].sum() / a.inst.w.sum() == pytest.approx(0.5)
    assert a.opt_res["goal"] == pytest.approx(a.opt_social) == pytest.approx(95.46, abs=0.006) and a.pes_res["goal"] == pytest.approx(95.46, abs=0.006)
    assert a.nash_goal_range() == pytest.approx((96.90, 100.32), abs=0.006)
    e = E.analyse(E.Settings(k=2, objective="social"))
    assert e.opt_res["goal"] > e.opt_social + 0.1                             # mit 2 Lkw noch nicht das Optimum


def test_kleine_lkw_numbers():
    a = _preset("Kleine Lkw führen (3)")
    assert a.leaders == [0, 5, 7]
    assert a.nash_goal_range() == pytest.approx((33.87, 37.48), abs=0.006) and a.opt_res["goal"] == pytest.approx(33.65, abs=0.006)
    assert a.opt_res["social"] == pytest.approx(98.91, abs=0.006) and a.nash_social_range()[0] == pytest.approx(96.90, abs=0.006) and a.opt_res["social"] > a.nash_social_range()[0]


def test_einheitliche_lkw_numbers():
    a = _preset("Einheitliche Lkw")
    assert a.nash_goal_range() == pytest.approx((7.57, 10.12), abs=0.006) and a.opt_res["goal"] == pytest.approx(7.57, abs=0.006)
    assert a.nash_social_range() == pytest.approx((78.27, 78.27), abs=0.006) and a.opt_social == pytest.approx(75.07, abs=0.006)


def test_disposition_5_numbers():
    a = _preset("Disposition steuert 5 Lkw")
    assert a.inst.w[a.leaders].sum() / a.inst.w.sum() == pytest.approx(0.833, abs=0.001)
    assert a.opt_res["goal"] == pytest.approx(95.46, abs=0.006) and a.pes_res["goal"] == pytest.approx(95.46, abs=0.006)


# --- Experiment 1: Erstzug-Vorteil (100 Instanzen, 8 Lkw) ---------------------------------------------------------------------------------------


def test_first_mover_experiment_numbers():
    rows = {(r["rule"], r["k"]): r for r in E.first_mover_experiment()}
    l1, l4, s1, s4 = rows[("largest", 1)], rows[("largest", 4)], rows[("smallest", 1)], rows[("smallest", 4)]
    assert l1["n_inst"] == 100
    for r in (l1, s1, rows[("index", 1)], rows[("largest", 2)]):
        assert r["share_gain_vs_best"] == 0.0 and r["mean_gain_vs_best"] == 0.0                # ein einzelner Anführer verbessert sich nie über das beste Nash hinaus (und zwei größte auch nicht)
    assert l1["mean_gain_vs_worst"] == pytest.approx(0.144, abs=0.006) and l1["share_pes_worse_than_best"] == pytest.approx(0.59, abs=0.05) and l1["mean_pes_loss_vs_best"] == pytest.approx(0.060, abs=0.006)
    assert s4["share_gain_vs_best"] == pytest.approx(0.34, abs=0.05) and s4["mean_gain_vs_best"] == pytest.approx(0.0118, abs=0.003) and l4["share_gain_vs_best"] == pytest.approx(0.06, abs=0.04)
    assert s1["share_pes_worse_than_best"] == pytest.approx(0.84, abs=0.05)
    assert l1["social_stack"] == pytest.approx(1.0454, abs=0.003) and l1["social_nash_best"] == pytest.approx(1.0208, abs=0.002) and l1["social_nash_worst"] == pytest.approx(1.0568, abs=0.002)
    assert l1["social_nash_best"] < l1["social_stack"] < l1["social_nash_worst"] + 0.01


def test_no_first_mover_gain_over_the_best_nash_is_a_measurement_not_a_theorem():
    """Die Schranke 'optimistisch nie schlechter als bestes Nash' ist ein Satz (getestet in test_stackelberg); dass sie hier mit Gleichheit erreicht wird, ist die Messung."""
    rows = E.first_mover_experiment(ks=(1,), rules=("largest",), seeds=range(850000, 850040))
    assert rows[0]["share_gain_vs_best"] == 0.0


# --- Experiment 2: Steuerungsanteil (30 Instanzen, 8 Lkw) ---------------------------------------------------------------------------------------


def test_control_experiment_numbers():
    out = E.control_experiment()
    lg, sm, ix = out["largest"], out["smallest"], out["index"]
    assert lg["optimistic"][0] == pytest.approx(1.0202, abs=0.002) and lg["pessimistic"][0] == pytest.approx(1.0626, abs=0.003)
    assert lg["share"][3] == pytest.approx(0.547, abs=0.01) and lg["optimistic"][3] == pytest.approx(1.0036, abs=0.002) and lg["pessimistic"][3] == pytest.approx(1.0088, abs=0.003)
    k_opt = next(k for k, v in zip(lg["k"], lg["optimistic"]) if v <= 1.0005)
    assert k_opt == 5 and lg["share"][5] == pytest.approx(0.769, abs=0.01)
    assert sm["share"][3] == pytest.approx(0.231, abs=0.01) and sm["optimistic"][3] == pytest.approx(1.0141, abs=0.003) and sm["pessimistic"][3] == pytest.approx(1.0474, abs=0.004)
    assert ix["optimistic"][3] == pytest.approx(1.0093, abs=0.003)
    assert lg["optimistic"][3] < ix["optimistic"][3] < sm["optimistic"][3] and lg["pessimistic"][3] < ix["pessimistic"][3] < sm["pessimistic"][3]
    assert sm["share"][6] == pytest.approx(0.603, abs=0.01) and sm["optimistic"][6] == pytest.approx(1.0093, abs=0.003) and sm["pessimistic"][6] == pytest.approx(1.0161, abs=0.004)
    assert sm["share"][6] > lg["share"][3] and sm["optimistic"][6] > lg["optimistic"][3] + 0.003
    for r in (lg, sm, ix):
        assert r["optimistic"][-1] == pytest.approx(1.0) and np.all(r["pessimistic"] >= r["optimistic"] - 1e-9)
    assert lg["optimistic"][3] < sm["optimistic"][3] - 0.005 and lg["pessimistic"][3] < sm["pessimistic"][3] - 0.02


# --- Mini-Spiel ----------------------------------------------------------------------------------------------------------------------------------


def test_mini_game_numbers_at_the_default_delta():
    K1, K2 = B.cost_matrices(C.DEFAULT_MINI_DELTA)
    rows, best = B.stackelberg(K1, K2)
    assert rows[best] == (0, 1, 7.0, 8.0) and best == 0
    p, q = B.mixed_equilibrium(K1, K2)
    mixed = B.expected_costs(K1, K2, np.outer([p, 1 - p], [q, 1 - q]))
    assert mixed == pytest.approx((8.5, 8.5)) and rows[best][2] + rows[best][3] == pytest.approx(15.0) and sum(mixed) == pytest.approx(17.0)
    sigma = np.zeros((2, 2))
    sigma[0, 1] = sigma[1, 0] = 0.5
    assert B.expected_costs(K1, K2, sigma) == pytest.approx((7.5, 7.5))
