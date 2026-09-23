"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Anführerzahl-Klammer, Würfel-Knopf, Permalink-Grenzen, Extremwerte, Mini-Spiel, zwei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import sk_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_has_no_exception_and_says_no_advantage_over_the_good_equilibrium():
    at = _run()
    _ok(at)
    assert at.metric and any("Kein Erstzug-Vorteil" in i.value for i in at.info)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["n_slider"] == p["n"] and at.session_state["k_slider"] == p["k"] and at.session_state["objective_select"] == p["objective"]
    assert at.metric


def test_pessimistic_warning_and_planner_success():
    at = _run(seed_input=10, selection_select="pessimistic")
    _ok(at)
    assert any("schlechteste für die Anführer" in w.value for w in at.warning)
    at = _run(objective_select="social", k_slider=3)
    _ok(at)
    assert any("erreicht das Optimum" in s.value for s in at.success)


def test_no_leader_info_and_all_leaders():
    at = _run(k_slider=0)
    _ok(at)
    assert any("Ohne Anführer" in i.value for i in at.info)
    at = _run(k_slider=8, n_slider=8, objective_select="social")
    _ok(at)


def test_leader_count_is_clamped_after_shrinking_the_instance():
    at = _run(n_slider=8, k_slider=6)
    _ok(at)
    at.session_state["n_slider"] = 4
    at.run()
    _ok(at)
    assert at.session_state["k_slider"] <= 4


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Vehikel generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["k"] = "9999"
    at.query_params["delta"] = "0.7"
    at.query_params["rule"] = "smallest"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["k_slider"] <= C.N_MAX and at.session_state["rule_select"] == "smallest"
    assert abs(at.session_state["delta_slider"] - 0.5) < 1e-9 or abs(at.session_state["delta_slider"] - 1.0) < 1e-9


@pytest.mark.parametrize("kw", [dict(n_slider=C.N_MIN, k_slider=1), dict(m_slider=C.M_MIN), dict(size_select="uniform"), dict(rule_select="smallest", k_slider=3), dict(rule_select="index", k_slider=2),
                                dict(selection_select="pessimistic", k_slider=2), dict(objective_select="social", k_slider=2), dict(delta_slider=C.MINI_DELTA_MIN), dict(delta_slider=2.0), dict(delta_slider=C.MINI_DELTA_MAX)])
def test_extreme_settings_run(kw):
    _ok(_run(**{"n_slider": 6, **kw}))


def test_mini_game_shows_the_first_move_and_the_dominant_case():
    at = _run(delta_slider=1.0)
    assert any("Summe der Wartezeiten 15,0 min statt 17,0 im gemischten Gleichgewicht" in c.value for c in at.caption)
    assert not any("{" in c.value and "de(" in c.value for c in at.caption)               # keine unaufgelösten f-String-Ausdrücke
    at = _run(delta_slider=3.0)
    _ok(at)
    assert any("Ab δ ≥ 2" in c.value for c in at.caption)


def test_first_mover_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "FIRST_SEEDS", tuple(range(830000, 830004)))
    monkeypatch.setattr(C, "FIRST_N", 6)
    at = _run()
    next(b for b in at.button if b.key == "first_start").click().run()
    _ok(at)
    assert at.session_state["first_on"] and at.get("plotly_chart")


def test_control_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "CONTROL_SEEDS", tuple(range(840000, 840003)))
    monkeypatch.setattr(C, "CONTROL_N", 6)
    at = _run()
    next(b for b in at.button if b.key == "control_start").click().run()
    _ok(at)
    assert at.session_state["control_on"]


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
