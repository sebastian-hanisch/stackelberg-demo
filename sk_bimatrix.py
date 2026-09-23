"""Mini-Spiel: zwei Lkw (Größe 1), zwei Tore mit gleicher Grundzeit; Tor B ist um delta langsamer. Kostenmatrizen (wie nash-demo), reine und gemischtes Gleichgewicht, Stackelberg mit Lkw 1 als Anführer."""

import numpy as np

import sk_constants as C


def cost_matrices(delta, a=C.MINI_A, b=C.MINI_B):
    """(K1, K2): Kosten von Lkw 1/2, Zeile = Tor von Lkw 1, Spalte = Tor von Lkw 2 (0 = Tor A, 1 = Tor B)."""
    a_g = (a, a + delta)
    K1, K2 = np.zeros((2, 2)), np.zeros((2, 2))
    for g1 in range(2):
        for g2 in range(2):
            K1[g1, g2] = a_g[g1] + b * (1 + (g1 == g2))
            K2[g1, g2] = a_g[g2] + b * (1 + (g1 == g2))
    return K1, K2


def pure_equilibria(K1, K2):
    return [(g1, g2) for g1 in range(2) for g2 in range(2)
            if K1[g1, g2] <= K1[1 - g1, g2] + C.EPS and K2[g1, g2] <= K2[g1, 1 - g2] + C.EPS]


def mixed_equilibrium(K1, K2):
    """Vollständig gemischtes Gleichgewicht (p = Wahrscheinlichkeit, dass Lkw 1 Tor A wählt, q = dito Lkw 2) oder None (Indifferenzformeln)."""
    d1 = K1[0, 0] - K1[0, 1] - K1[1, 0] + K1[1, 1]
    d2 = K2[0, 0] - K2[1, 0] - K2[0, 1] + K2[1, 1]
    if abs(d1) < C.EPS or abs(d2) < C.EPS:
        return None
    q = (K1[1, 1] - K1[0, 1]) / d1
    p = (K2[1, 1] - K2[1, 0]) / d2
    return (float(p), float(q)) if 0.0 < p < 1.0 and 0.0 < q < 1.0 else None


def expected_costs(K1, K2, sigma):
    sigma = np.asarray(sigma).reshape(2, 2)
    return float((sigma * K1).sum()), float((sigma * K2).sum())


def follower_responses(K2, g1):
    """Alle besten Antworten von Lkw 2, wenn Lkw 1 sich auf g1 festgelegt hat (mehrere bei Gleichstand)."""
    best = K2[g1, :].min()
    return [g2 for g2 in range(2) if K2[g1, g2] <= best + C.EPS]


def stackelberg(K1, K2, selection="optimistic"):
    """Lkw 1 führt: für jede Festlegung g1 antwortet Lkw 2 bestmöglich (bei Gleichstand nach der Auswahlregel: optimistisch = das für Lkw 1 beste, pessimistisch = das schlechteste);
    Lkw 1 wählt die Festlegung mit den kleinsten eigenen Kosten. Rückgabe: Liste (g1, g2, Kosten Lkw 1, Kosten Lkw 2) je Festlegung und der Index der besten Festlegung."""
    if selection not in ("optimistic", "pessimistic"):
        raise ValueError(selection)
    rows = []
    for g1 in range(2):
        cands = follower_responses(K2, g1)
        g2 = min(cands, key=lambda g: K1[g1, g]) if selection == "optimistic" else max(cands, key=lambda g: K1[g1, g])
        rows.append((g1, g2, float(K1[g1, g2]), float(K2[g1, g2])))
    best = int(np.argmin([r[2] for r in rows]))
    return rows, best
