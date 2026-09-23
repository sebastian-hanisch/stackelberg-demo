"""Konstanten der Stackelberg-Demo: Vehikel "Torwahl" aus nash-demo, Anführer-Regeln, Mini-Spiel, Experimente (Presets nach den Messungen)."""

EPS = 1e-9                         # ein Wechsel/eine Verbesserung zählt nur, wenn sie echt ist
SEED_MAX = 999999

# --- Vehikel "Torwahl" (wortgleich zu nash-demo) ----------------------------------------------------------------------------------------------

A_MIN, A_MAX = 2.0, 10.0           # Grundwartezeit a_g je Tor in Minuten
B_MIN, B_MAX = 0.5, 3.0            # Zuschlag b_g je Ladungseinheit in Minuten
SIZES = (1, 2, 3)                  # Lkw-Größen: Transporter, Lkw, Sattelzug
SIZE_PROBS = (0.5, 0.3, 0.2)
SIZE_MODES = ("mixed", "uniform")
SIZE_MODE_LABELS = {"mixed": "Gemischt (1/2/3)", "uniform": "Einheitlich (alle 1)"}
ENUM_MAX_ASSIGNMENTS = 100_000     # Aufzählung aller Zuordnungen: m^n darf diese Grenze nicht übersteigen

N_MIN, N_MAX, DEFAULT_N, N_STEP = 4, 10, 8, 1
M_MIN, M_MAX, DEFAULT_M, M_STEP = 2, 3, 3, 1
DEFAULT_SEED = 35                  # Vehikel-Seed (wie nash-demo)

# --- Anführer ---------------------------------------------------------------------------------------------------------------------------------

RULES = ("largest", "smallest", "index")
RULE_LABELS = {"largest": "Die größten Lkw führen", "smallest": "Die kleinsten Lkw führen", "index": "Die ersten Lkw (nach Nummer) führen"}
OBJECTIVES = ("own", "social")
OBJECTIVE_LABELS = {"own": "Eigennützig: kleinste Summe der eigenen Wartezeiten (Spediteur mit k Lkw)", "social": "Gemeinwohl: kleinste Summe aller Wartezeiten (Disposition steuert k Lkw)"}
SELECTIONS = ("optimistic", "pessimistic")
SELECTION_LABELS = {"optimistic": "Optimistisch: die Folger wählen das für den Anführer beste Gleichgewicht", "pessimistic": "Pessimistisch: die Folger wählen das für den Anführer schlechteste"}
DEFAULT_K = 1

# --- Mini-Spiel: zwei Lkw, zwei Tore (wie nash-demo) ------------------------------------------------------------------------------------------

MINI_A, MINI_B = 5.0, 2.0          # beide Tore: Grundzeit 5, Zuschlag 2 je Einheit; Tor B ist um DELTA langsamer
MINI_DELTA_MIN, MINI_DELTA_MAX, DEFAULT_MINI_DELTA, MINI_DELTA_STEP = 0.0, 4.0, 1.0, 0.5

# --- Experimente (feste Seeds) ---------------------------------------------------------------------------------------------------------------

FIRST_N, FIRST_M = 8, 3
FIRST_SEEDS = tuple(range(830000, 830100))
FIRST_KS = (1, 2, 3, 4)
CONTROL_N, CONTROL_M = 8, 3
CONTROL_SEEDS = tuple(range(840000, 840030))

# --- Presets (Werte nach den Messungen) ----------------------------------------------------------------------------------------------------


def _preset(n=DEFAULT_N, m=DEFAULT_M, size_mode="mixed", seed=DEFAULT_SEED, rule="largest", k=DEFAULT_K, objective="own", selection="optimistic", delta=DEFAULT_MINI_DELTA):
    return {"n": n, "m": m, "size_mode": size_mode, "seed": seed, "rule": rule, "k": k, "objective": objective, "selection": selection, "delta": delta}


PRESETS = {
    "Standardfall (ein Anführer)": _preset(),
    "Pessimistische Folger": _preset(seed=10, selection="pessimistic"),
    "Disposition steuert 3 Lkw": _preset(objective="social", k=3),
    "Kleine Lkw führen (3)": _preset(rule="smallest", k=3),
    "Einheitliche Lkw": _preset(size_mode="uniform"),
    "Disposition steuert 5 Lkw": _preset(objective="social", k=5),
}
PRESET_HELP = {
    "Standardfall (ein Anführer)": "Der größte Lkw (Nr. 2) legt sich zuerst auf ein Tor fest, die übrigen 7 reagieren: seine Wartezeit liegt bei 12,35 min - genau so viel wie im besten Nash-Gleichgewicht (schlechtestes: 13,3). Der Erstzug schützt vor dem schlechten Gleichgewicht, bringt aber nichts über das gute hinaus.",
    "Pessimistische Folger": "Anderes Vehikel (Seed 10): Wählen die Folger von mehreren Gleichgewichten das schlechteste für den Anführer, kostet ihn die Festlegung 14,35 min - mehr als im besten Nash-Gleichgewicht (12,57 min), fast so viel wie im schlechtesten (14,89).",
    "Disposition steuert 3 Lkw": "Eine Disposition steuert die 3 größten Lkw nach dem Ziel der kleinsten Summe aller Wartezeiten (die Hälfte der Last), die anderen 5 spielen Nash: die Summe fällt auf das Optimum, 95,46 min - besser als jedes Nash-Gleichgewicht (96,9 bis 100,3).",
    "Kleine Lkw führen (3)": "Die 3 kleinsten Lkw führen eigennützig: ihre gemeinsame Wartezeit sinkt von 33,87 (bestes Nash) auf 33,65 min, aber die Summe aller Wartezeiten steigt auf 98,91 - über das beste Nash-Gleichgewicht (96,9).",
    "Einheitliche Lkw": "Alle Lkw Größe 1: alle Nash-Gleichgewichte haben dieselbe Summe der Wartezeiten (78,27 min); der eine Anführer wartet 7,57 min - im schlechtesten Gleichgewicht wären es 10,12.",
    "Disposition steuert 5 Lkw": "Steuert die Disposition 5 der 8 Lkw (83 % der Last), erreicht sie ebenfalls das Optimum (95,46 min) - mit weniger Lkw ist es bei dieser Instanz schon mit 3 der Fall.",
}
