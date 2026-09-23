"""Stackelberg - wer sich zuerst festlegt, lenkt das Ergebnis - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Linie "Spieltheorie & Mechanism Design" der "Konzepte"-Reihe (Nachfolger von korreliert-demo): ein Anführer legt sich zuerst auf ein Tor fest, die übrigen Lkw reagieren.
Was bringt der Erstzug - und wie viel Last muss eine Disposition steuern, um das Optimum zu erreichen?

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import sk_bimatrix as B
import sk_constants as C
from sk_evaluation import Settings, analyse, control_curve, control_experiment, first_mover_experiment
from sk_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from sk_visualization import build_bimatrix, build_commitments, build_control_curve, build_control_experiment, build_first_mover, build_goal_bars, build_pessimistic

st.set_page_config(page_title="Stackelberg – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _first():
    return first_mover_experiment()


@st.cache_data(show_spinner=False)
def _control():
    return control_experiment()


st.title("🥇 Stackelberg – wer sich zuerst festlegt, lenkt das Ergebnis")
st.markdown(
    """
Bisher wählten alle Lkw gleichzeitig. Ein **Anführer** legt sich stattdessen **zuerst** auf ein Tor fest - verbindlich -, und die übrigen Lkw, die **Folger**, reagieren darauf mit einem Nash-Gleichgewicht ihres Restspiels
(von Stackelberg 1934). Wer festlegt, weiß, wie die Folger antworten, und kann sich das beste Tor aussuchen. Die Demo rechnet das exakt für das Torwahl-Spiel: für einen eigennützigen Anführer
(ein Spediteur mit k Lkw) und für eine Disposition, die k Lkw nach dem Gemeinwohl steuert und alle anderen sich selbst überlässt (Stackelberg-Routing). Gefragt wird: was bringt der Erstzug, wem nützt er -
und wie viel Last muss man steuern?
"""
)
st.caption(
    "Sechstes Stück der Linie \"Spieltheorie & Mechanism Design\" der \"Konzepte\"-Reihe, Nachfolger von **korreliert-demo**: dort lenkte ein Vermittler durch Empfehlungen, hier durch Festlegung. "
    "Die Bindung an den Erstzug und volle Information der Folger über ihn sind Modellannahmen."
)

with st.expander("So funktioniert der Erstzug", expanded=True):
    st.markdown(
        """
1. **Anführer und Folger.** Eine Regel bestimmt, welche *k* Lkw führen (die größten, die kleinsten oder die ersten nach Nummer). Sie legen ihre Tore fest; danach spielen die übrigen Lkw ein Nash-Gleichgewicht:
   kein Folger kann sich durch einen Alleingang verbessern.
2. **Mehrere Folger-Gleichgewichte.** Gibt es mehrere, entscheidet die Auswahlregel: **optimistisch** (die Folger wählen das für die Anführer beste Gleichgewicht, "starkes" Stackelberg) oder **pessimistisch** (das schlechteste).
3. **Ziel des Anführers.** *Eigennützig:* kleinste Summe der Wartezeiten der eigenen *k* Lkw. *Gemeinwohl:* kleinste Summe aller Wartezeiten.
4. **Die beste Festlegung.** Der Anführer probiert alle Festlegungen seiner Lkw durch (m<sup>k</sup> Stück), sieht jeweils die Folger-Antwort und wählt die mit dem besten Ziel. Alles wird aus der Definition aufgezählt.
        """,
        unsafe_allow_html=True,
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_trucks = st.slider("Lkw", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Anzahl der Lkw. Die Aufzählung hat eine Zeile je Zuordnung (m^n).")
    m_gates = st.slider("Tore", *bounds("m_slider"), key="m_slider", step=C.M_STEP)
    size_mode = st.selectbox("Lkw-Größen", C.SIZE_MODES, key="size_select", format_func=lambda k: C.SIZE_MODE_LABELS[k],
                             help="Gemischt: Transporter, Lkw und Sattelzüge belegen ein Tor unterschiedlich stark. Einheitlich: alle Größe 1.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1, help="Legt Tore und Lkw-Größen fest.")
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    st.markdown("**Anführer**")
    rule = st.selectbox("Wer führt?", C.RULES, key="rule_select", format_func=lambda k: C.RULE_LABELS[k])
    if "k_slider" in st.session_state:
        st.session_state["k_slider"] = min(int(st.session_state["k_slider"]), int(n_trucks))          # die Obergrenze hängt von der Lkw-Zahl ab
    k_leaders = st.slider("Zahl der Anführer k", 0, int(n_trucks), key="k_slider", help="0 = kein Anführer (reines Nash-Spiel).")
    objective = st.selectbox("Ziel des Anführers", C.OBJECTIVES, key="objective_select", format_func=lambda k: C.OBJECTIVE_LABELS[k])
    selection = st.selectbox("Auswahl unter mehreren Folger-Gleichgewichten", C.SELECTIONS, key="selection_select", format_func=lambda k: C.SELECTION_LABELS[k])
    st.markdown("**Mini-Spiel**")
    delta = st.slider("δ – Tor B langsamer um [min]", *bounds("delta_slider"), key="delta_slider", step=C.MINI_DELTA_STEP, format="%.1f")

sync_query_params({"n_slider": int(n_trucks), "m_slider": int(m_gates), "size_select": size_mode, "rule_select": rule, "k_slider": int(k_leaders), "objective_select": objective,
                   "selection_select": selection, "delta_slider": float(delta), "seed_input": int(seed)})

settings = Settings(int(n_trucks), int(m_gates), size_mode, int(seed), rule, int(k_leaders), objective, selection)
with st.spinner("Rechne alle Festlegungen durch..."):
    a = analyse(settings)
inst = a.inst
goal_name = "Wartezeiten der Anführer" if (objective == "own" and a.leaders) else "Wartezeiten aller Lkw"
nash_lo, nash_hi = a.nash_goal_range()
soc_lo, soc_hi = a.nash_social_range()
opt_res, pes_res = a.opt_res, a.pes_res
share = float(inst.w[a.leaders].sum() / inst.w.sum()) if a.leaders else 0.0

# --- Der Anführer -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Der Anführer")
if a.leaders:
    st.caption(f"Anführer: {', '.join(f'Lkw {i + 1} (Größe {int(inst.w[i])})' for i in a.leaders)} - zusammen {pct(share)} der Last. Folger: die übrigen {inst.n - len(a.leaders)} Lkw.")
else:
    st.caption("Kein Anführer: alle Lkw spielen gleichzeitig ein Nash-Gleichgewicht.")
st.markdown(f"**Ziel des Anführers: Summe der {goal_name} (min)**")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bestes Nash-Gleichgewicht", de(nash_lo, 2), help="Kleinster Wert des Ziels über alle reinen Nash-Gleichgewichte aller Lkw (gleichzeitiges Spiel).")
m2.metric("Schlechtestes Nash-Gleichgewicht", de(nash_hi, 2))
m3.metric("Stackelberg, optimistisch", de(opt_res["goal"], 2), delta=f"{de(opt_res['goal'] - nash_lo, 2)} gegenüber bestem Nash", delta_color="off", help="Die Folger wählen unter mehreren Gleichgewichten das beste für die Anführer.")
m4.metric("Stackelberg, pessimistisch", de(pes_res["goal"], 2), delta=f"{de(pes_res['goal'] - nash_lo, 2)} gegenüber bestem Nash", delta_color="off", help="Die Folger wählen unter mehreren Gleichgewichten das schlechteste für die Anführer.")
st.plotly_chart(build_goal_bars([("Bestes Nash", nash_lo), ("Schlechtestes Nash", nash_hi), ("Stackelberg optimistisch", opt_res["goal"]), ("Stackelberg pessimistisch", pes_res["goal"])], f"Summe der {goal_name} (min)"),
                width="stretch", key="goal_chart")

if not a.leaders:
    st.info("Ohne Anführer ist das Spiel das gleichzeitige Nash-Spiel der Vorgänger-Demos; die Stackelberg-Werte stimmen mit den Nash-Werten überein.")
elif objective == "own":
    gain_worst = (nash_hi - opt_res["goal"]) / nash_hi
    if opt_res["goal"] < nash_lo - 1e-9:
        st.success(f"✅ Der Erstzug lohnt sich: die Anführer warten {de(opt_res['goal'], 2)} statt {de(nash_lo, 2)} min - weniger als in jedem Nash-Gleichgewicht.")
    else:
        st.info(f"Kein Erstzug-Vorteil über das gute Gleichgewicht hinaus: die Anführer erreichen {de(opt_res['goal'], 2)} min - so viel wie im besten Nash-Gleichgewicht ({de(nash_lo, 2)} min). Gegenüber dem schlechtesten "
                f"({de(nash_hi, 2)} min) sparen sie {pct(gain_worst, 1)}: der Erstzug schützt vor dem schlechten Gleichgewicht.")
    if pes_res["goal"] > nash_lo + 1e-9:
        st.warning(f"⚠️ Wählen die Folger unter mehreren Gleichgewichten das schlechteste für die Anführer, kostet die Festlegung {de(pes_res['goal'], 2)} min - {de(pes_res['goal'] - nash_lo, 2)} min mehr als das beste Nash-Gleichgewicht.")
    if opt_res["social"] > soc_lo + 1e-9:
        st.caption(f"Für alle zusammen liegt die Summe der Wartezeiten bei {de(opt_res['social'], 2)} min, das Nash-Spiel bei {de(soc_lo, 2)} bis {de(soc_hi, 2)} und das Optimum bei {de(a.opt_social, 2)} min.")
else:
    if opt_res["goal"] <= a.opt_social + 1e-9:
        st.success(f"✅ Die Disposition steuert {len(a.leaders)} Lkw ({pct(share)} der Last) und erreicht das Optimum: {de(opt_res['goal'], 2)} min gegenüber {de(nash_lo, 2)} bis {de(nash_hi, 2)} im Nash-Spiel.")
    else:
        st.info(f"Die Disposition steuert {len(a.leaders)} Lkw ({pct(share)} der Last): {de(opt_res['goal'], 2)} min (optimistisch), {de(pes_res['goal'], 2)} min (pessimistisch); Nash-Spiel {de(nash_lo, 2)} bis {de(nash_hi, 2)}, Optimum {de(a.opt_social, 2)} min.")

st.markdown("**Was die Anführer festlegen können** (beste Festlegungen zuerst, nach der gewählten Auswahlregel)")
table, n_commit = a.commitment_table(top=8)
st.dataframe({**{f"Lkw {i + 1} (Anführer)": [f"Tor {gates[j] + 1}" for gates, _, _, _ in table] for j, i in enumerate(a.leaders)}, f"Ziel: {goal_name} (min)": [de(v, 2) for _, v, _, _ in table],
              "Summe aller Wartezeiten (min)": [de(s, 2) for _, _, s, _ in table],
              "Antwort der Folger (Tor je Lkw)": [" ".join(str(g + 1) for g in assign) for _, _, _, assign in table]}, hide_index=True)
st.caption(f"Insgesamt {n_commit} mögliche Festlegungen; gezeigt sind die besten {len(table)}. Die letzte Spalte zeigt die Tore aller Lkw nach der Folger-Antwort (auch die der Anführer).")
if a.leaders and len(table) > 1:
    st.plotly_chart(build_commitments(table, a.leaders, "optimistisch" if selection == "optimistic" else "pessimistisch"), width="stretch", key="commit_chart")

st.markdown("**Wie viel Last muss man steuern?** Eine Disposition mit Gemeinwohl-Ziel steuert nacheinander mehr Lkw (Anführer nach der gewählten Regel):")
rows_c = control_curve(settings)
st.plotly_chart(build_control_curve(rows_c), width="stretch", key="control_curve_chart")
first_opt = next((r for r in rows_c if r["optimistic"] <= 1 + 1e-9), None)
if first_opt is not None:
    st.caption(f"In dieser Instanz genügt bei optimistischer Auswahl eine Steuerung von {first_opt['k']} Lkw ({pct(first_opt['share'])} der Last) für das Optimum; ohne Steuerung liegt die Summe {pct(rows_c[0]['optimistic'] - 1, 1)} bis "
               f"{pct(rows_c[0]['pessimistic'] - 1, 1)} darüber.")

st.markdown("---")

# --- Experiment 1: Erstzug-Vorteil -----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Was bringt der Erstzug einem eigennützigen Anführer?")
st.caption(f"{C.FIRST_N} Lkw, {C.FIRST_M} Tore, gemischte Größen, {len(C.FIRST_SEEDS)} feste Instanzen; eigennützige Anführer (k = {C.FIRST_KS[0]} bis {C.FIRST_KS[-1]} Lkw, drei Regeln). Verglichen wird das Ziel der "
           "Anführer (Summe ihrer eigenen Wartezeiten) mit dem besten und dem schlechtesten Nash-Gleichgewicht aus ihrer Sicht.")
if st.button("Erstzug-Vorteil messen (dauert etwa 10 Sekunden)", key="first_start"):
    st.session_state["first_on"] = True
if st.session_state.get("first_on"):
    with st.spinner("Rechne 100 Instanzen × 12 Konfigurationen..."):
        rows_f = _first()
    st.plotly_chart(build_first_mover(rows_f), width="stretch", key="first_chart")
    st.plotly_chart(build_pessimistic(rows_f), width="stretch", key="pessimistic_chart")
    by = {(r["rule"], r["k"]): r for r in rows_f}
    l1, s4, l4 = by[("largest", 1)], by[("smallest", 4)], by[("largest", 4)]
    st.warning(
        f"**Befund:** Ein einzelner Anführer verbessert sich gegenüber dem **besten** Nash-Gleichgewicht in {('keiner der ' + str(l1['n_inst']) + ' Instanzen') if l1['share_gain_vs_best'] == 0 else pct(l1['share_gain_vs_best']) + ' der Instanzen'}. Gegenüber dem **schlechtesten** spart der größte Lkw im Mittel {pct(l1['mean_gain_vs_worst'], 1)} "
        f"seiner Wartezeit: der Erstzug ist eine Versicherung gegen das schlechte Gleichgewicht, kein Bonus über das gute hinaus. Erst mehrere Anführer können mehr: vier kleinste Lkw gewinnen in {pct(s4['share_gain_vs_best'])} der Instanzen "
        f"(im Mittel {pct(s4['mean_gain_vs_best'], 1)}), vier größte in {pct(l4['share_gain_vs_best'])}. Wählen die Folger das für die Anführer schlechteste Gleichgewicht, ist die Festlegung beim größten Einzel-Anführer in "
        f"{pct(l1['share_pes_worse_than_best'])} der Instanzen schlechter als das beste Nash-Gleichgewicht (im Mittel {pct(l1['mean_pes_loss_vs_best'], 1)} mehr). Für die Gesamtheit liegt die Summe der Wartezeiten mit dem größten Anführer bei "
        f"{de(l1['social_stack'], 3)} mal Optimum - zwischen bestem ({de(l1['social_nash_best'], 3)}) und schlechtestem ({de(l1['social_nash_worst'], 3)}) Nash-Gleichgewicht."
    )

st.markdown("---")

# --- Experiment 2: Steuerungsanteil ------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viel Steuerung braucht die Disposition?")
st.caption(f"{C.CONTROL_N} Lkw, {C.CONTROL_M} Tore, gemischte Größen, {len(C.CONTROL_SEEDS)} feste Instanzen; die Disposition steuert k Lkw nach dem Gemeinwohl-Ziel, die übrigen spielen Nash. "
           "Mittel der Summe aller Wartezeiten geteilt durch das Optimum über dem gesteuerten Lastanteil.")
if st.button("Steuerungsanteil messen (dauert etwa 30 Sekunden)", key="control_start"):
    st.session_state["control_on"] = True
if st.session_state.get("control_on"):
    with st.spinner("Rechne 30 Instanzen × 3 Regeln × 9 Anführerzahlen..."):
        out_c = _control()
    st.plotly_chart(build_control_experiment(out_c), width="stretch", key="control_chart")
    lg = out_c["largest"]
    k_opt = next(k for k, v in zip(lg["k"], lg["optimistic"]) if v <= 1.0005)
    sm = out_c["smallest"]
    st.warning(
        f"**Befund:** Ohne Steuerung liegt die Summe der Wartezeiten im Mittel {pct(lg['optimistic'][0] - 1, 1)} (beste Folger-Wahl) bis {pct(lg['pessimistic'][0] - 1, 1)} (schlechteste) über dem Optimum. "
        f"Steuert die Disposition die größten Lkw, sinkt das schnell: mit {pct(lg['share'][3])} der Last (3 Lkw) sind es {pct(lg['optimistic'][3] - 1, 1)} bis {pct(lg['pessimistic'][3] - 1, 1)}, und ab {k_opt} Lkw "
        f"({pct(lg['share'][k_opt])} der Last) ist das Optimum im Mittel erreicht. Steuert sie stattdessen die kleinsten Lkw, hilft dieselbe Zahl kaum: mit 3 Lkw ({pct(sm['share'][3])} der Last) bleibt es bei "
        f"{pct(sm['optimistic'][3] - 1, 1)} bis {pct(sm['pessimistic'][3] - 1, 1)}. Es zählt also nicht nur die Zahl der Lkw, sondern welche: selbst mit {pct(sm['share'][6])} der Last (6 kleinste Lkw) bleibt es bei {pct(sm['optimistic'][6] - 1, 1)} bis {pct(sm['pessimistic'][6] - 1, 1)}, "
        f"während die größten Lkw mit {pct(lg['share'][3])} der Last {pct(lg['optimistic'][3] - 1, 1)} bis {pct(lg['pessimistic'][3] - 1, 1)} erreichen. Bei wenig Steuerung entscheidet außerdem die Auswahlregel der Folger "
        "über den Unterschied zwischen dem optimistischen und dem pessimistischen Ergebnis."
    )

st.markdown("---")

# --- Mini-Spiel --------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🥇 Mini-Spiel: zwei Lkw, zwei Tore - der erste Zug")
st.caption(f"Beide Lkw Größe 1, beide Tore Grundzeit {C.MINI_A:g} min und Zuschlag {C.MINI_B:g} min; Tor B ist um δ Minuten langsamer (Regler links). Lkw 1 legt sich zuerst fest, Lkw 2 reagiert.")
K1, K2 = B.cost_matrices(float(delta))
names = ("Tor A", "Tor B")
table_md = "| Lkw 1 ↓ / Lkw 2 → | Tor A | Tor B |\n|---|---|---|\n" + "\n".join(f"| **{names[i]}** | " + " | ".join(f"{K1[i, j]:g} / {K2[i, j]:g}" for j in range(2)) + " |" for i in range(2))
st.markdown(table_md)
rows_b, best_b = B.stackelberg(K1, K2)
st.markdown("**Festlegungen von Lkw 1 und Antwort von Lkw 2**")
st.dataframe({"Lkw 1 legt sich fest auf": [names[r[0]] for r in rows_b], "Lkw 2 antwortet mit": [names[r[1]] for r in rows_b], "Wartezeit Lkw 1 (min)": [de(r[2], 2) for r in rows_b],
              "Wartezeit Lkw 2 (min)": [de(r[3], 2) for r in rows_b], "Beste Festlegung": ["✓" if i == best_b else "" for i in range(2)]}, hide_index=True)
g1, g2, c1, c2 = rows_b[best_b]
chart = {f"Erstzug von Lkw 1 ({names[g1]})": (c1, c2)}
mixed = B.mixed_equilibrium(K1, K2)
if mixed is not None:
    p, q = mixed
    chart["Gemischtes Nash"] = B.expected_costs(K1, K2, np.outer([p, 1 - p], [q, 1 - q]))
pure = B.pure_equilibria(K1, K2)
if len(pure) >= 2:
    sigma = np.zeros((2, 2))
    for pg in pure[:2]:
        sigma[pg] = 0.5
    chart["Abwechselnde Ampel"] = B.expected_costs(K1, K2, sigma)
st.plotly_chart(build_bimatrix(chart), width="stretch", key="bimatrix_chart")
if len(pure) >= 2 and mixed is not None:
    st.caption(f"Zwei Lkw, die sich aus dem Weg gehen wollen: der Anführer nimmt sich das bessere der beiden reinen Gleichgewichte und zwingt den Folger ins schlechtere - der Folger wartet länger als in der abwechselnden "
               f"Ampel, aber kürzer als im gemischten Gleichgewicht. Der Erstzug löst die Koordination (Summe der Wartezeiten {de(c1 + c2, 1)} min statt {de(sum(chart['Gemischtes Nash']), 1)} im gemischten Gleichgewicht) und verteilt den Gewinn ungleich; "
               "die Ampel aus der Vorgänger-Demo verteilt ihn gleich.")
else:
    st.caption("Ab δ ≥ 2 ist Tor A für beide die bessere Wahl, egal was der andere tut: der Erstzug ändert nichts.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Festlegung ist verbindlich und die Folger sehen sie** | Kann der Anführer nachträglich wechseln oder sehen die Folger ihn nicht, ist es wieder ein gleichzeitiges Spiel. | - |
| **Die Folger spielen ein reines Nash-Gleichgewicht** | Lernende Folger enden nicht immer in einem Gleichgewicht (siehe No-Regret-Lernen); gemischte Folger-Gleichgewichte sind hier nicht abgebildet. | [No-Regret-Lernen](https://sebastianhanisch-noregret-demo.streamlit.app/) |
| **Die Disposition kennt das ganze Spiel und steuert ihre Lkw verlässlich** | Sie löst eine Aufzählung über alle Festlegungen (m^k mal Folger-Spiel); in echten Flotten kommen Ausfälle, Verspätungen und fremde Lkw dazu. | [Multi-Agenten-Koordination](https://sebastianhanisch-marl-demo.streamlit.app/) |
| **Steuerung ist kostenlos** | Ein gesteuerter Lkw verzichtet auf seinen eigenen Vorteil: die Anführer warten im Gemeinwohl-Fall womöglich länger als im Nash-Spiel. Wer das ausgleichen soll, ist eine Preisfrage. | [Maut](https://sebastianhanisch-maut-demo.streamlit.app/) |
| **Eine einzige Runde** | Über viele Runden können Anführer Ruf aufbauen oder Folger Strafen androhen: ein anderes Spiel. | - |
"""
)
st.caption(
    "Verwandt: [korreliert-demo](https://sebastianhanisch-korreliert-demo.streamlit.app/) (Vorgänger: Vermittler mit Empfehlungen), [nash-demo](https://sebastianhanisch-nash-demo.streamlit.app/) "
    "(gleichzeitiges Torwahl-Spiel), [maut-demo](https://sebastianhanisch-maut-demo.streamlit.app/) (Preise statt Führung)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Spiel.** Lkw $i$ mit Größe $w_i$, Tore $g$ mit $a_g, b_g$: Wartezeit $c_i(s) = a_{s_i} + b_{s_i} \sum_{j: s_j = s_i} w_j$; Summe der Wartezeiten $SC(s) = \sum_i c_i(s)$.

**Stackelberg-Spiel.** Anführer $L \subseteq \{1,\dots,n\}$, Folger $F$ = übrige Lkw. Der Anführer wählt $s_L \in \{1,\dots,m\}^{L}$. Zu jeder Festlegung sind die Folger-Gleichgewichte
$NE_F(s_L) = \{ s_F : c_i(s_L, s_F) \le c_i(g, s_{-i}) \text{ für alle } i \in F,\ g \}$ die Antworten. Mit Zielfunktion $\phi$ (eigene Wartezeiten $\sum_{i \in L} c_i$ oder $SC$):

$$\text{optimistisch: } \min_{s_L} \min_{s_F \in NE_F(s_L)} \phi(s_L, s_F), \qquad \text{pessimistisch: } \min_{s_L} \max_{s_F \in NE_F(s_L)} \phi(s_L, s_F).$$

Ohne Anführer ($L = \emptyset$) ist das der Bereich der reinen Nash-Gleichgewichte. Der Wert des optimistischen Stackelberg-Spiels ist nie schlechter als das beste Nash-Gleichgewicht aus Sicht des Ziels,
denn der Anführer kann sich auf seinen Teil dieses Gleichgewichts festlegen - die Folger können dann (unter mehreren Antworten) dazu passend antworten.

**Stackelberg-Routing.** Steuert eine Disposition einen Lastanteil $\alpha$ und ist $\phi = SC$, wächst mit $\alpha$ die Menge der Zuordnungen, die sie erreichen kann; ab welchem Lastanteil das Optimum erreichbar ist, misst das
Experiment (Roughgarden 2001 untersucht die Strategie "größte Latenz zuerst", Korilis/Lazar/Orda 1997 die Erreichbarkeit des Optimums durch Steuerung eines Teils des Verkehrs).

Implementiert in `sk_stackelberg.py` (Zuordnungen, Folger-Gleichgewichte, Festlegungen), `sk_bimatrix.py` (Mini-Spiel), `sk_evaluation.py` (Analyse, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
