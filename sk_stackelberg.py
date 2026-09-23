"""Stackelberg im Torwahl-Spiel (von Stackelberg 1934): eine Gruppe von Lkw, die Anführer, legt ihre Tore zuerst fest; die übrigen Lkw, die Folger, spielen danach ein Nash-Gleichgewicht des Restspiels
(reine Gleichgewichte, Vollaufzählung). Gibt es mehrere Folger-Gleichgewichte, entscheidet die Auswahlregel: optimistisch (das für den Anführer beste, "starkes" Stackelberg) oder pessimistisch (das
schlechteste, "schwaches" Stackelberg). Anführer-Ziel: 'own' = kleinste Summe der eigenen Wartezeiten (ein Spediteur mit k Lkw), 'social' = kleinste Summe aller Wartezeiten (eine Disposition,
die k Lkw steuert und alle anderen sich selbst überlässt - Stackelberg-Routing, Korilis/Lazar/Orda 1997, Roughgarden 2001)."""

import numpy as np

import sk_constants as C
import sk_gates as G


class States:
    """Alle m^n Zuordnungen mit Kosten je Lkw und Abweichungskosten (Wartezeit an Tor h, wenn nur der Lkw wechselt)."""

    def __init__(self, inst):
        if inst.m ** inst.n > C.ENUM_MAX_ASSIGNMENTS:
            raise ValueError("zu viele Zuordnungen")
        n, m = inst.n, inst.m
        N = m ** n
        idx = np.arange(N, dtype=np.int64)
        A = np.empty((N, n), dtype=np.int64)
        for i in range(n):
            A[:, i] = idx % m
            idx //= m
        L = np.zeros((N, m))
        for i in range(n):
            L[np.arange(N), A[:, i]] += inst.w[i]
        wait = inst.a[None, :] + inst.b[None, :] * L
        self.inst, self.A, self.N = inst, A, N
        self.own = np.take_along_axis(wait, A, axis=1)
        dev = np.empty((N, n, m))
        for i in range(n):
            Li = L.copy()
            Li[np.arange(N), A[:, i]] -= inst.w[i]
            dev[:, i, :] = inst.a[None, :] + inst.b[None, :] * (Li + inst.w[i])
        self.dev = dev
        self.social = self.own.sum(axis=1)

    def is_ne_for(self, trucks):
        """Maske der Zuordnungen, in denen keiner der Lkw `trucks` sich durch einen Alleingang strikt verbessert."""
        ok = np.ones(self.N, dtype=bool)
        for i in trucks:
            ok &= self.own[:, i] <= self.dev[:, i, :].min(axis=1) + C.EPS
        return ok


def leaders_by_rule(inst, k, rule):
    """Die k Anführer: 'largest' = die größten Lkw (bei Gleichstand kleinere Nummer zuerst), 'smallest' = die kleinsten, 'index' = die ersten k nach Nummer."""
    if rule == "largest":
        order = sorted(range(inst.n), key=lambda i: (-inst.w[i], i))
    elif rule == "smallest":
        order = sorted(range(inst.n), key=lambda i: (inst.w[i], i))
    elif rule == "index":
        order = list(range(inst.n))
    else:
        raise ValueError(rule)
    return sorted(order[:k])


def solve(st, leaders, objective="own", selection="optimistic"):
    """Stackelberg-Ergebnis: Für jede Festlegung der Anführer die Folger-Gleichgewichte; der Anführer wählt die Festlegung mit dem besten Ergebnis bei angenommener Auswahlregel.
    Rückgabe: dict mit 'state' (Index der resultierenden Zuordnung), 'leader_cost', 'social', 'commitments' (je Festlegung: Ziel bei Auswahlregel und Index der gewählten Zuordnung) oder None,
    falls es kein Folger-Gleichgewicht gibt (kommt im Torwahl-Spiel nicht vor: Potenzialspiel)."""
    leaders = list(leaders)
    followers = [i for i in range(st.inst.n) if i not in leaders]
    mask = st.is_ne_for(followers)
    goal = st.own[:, leaders].sum(axis=1) if (objective == "own" and leaders) else st.social
    if objective not in ("own", "social") or selection not in ("optimistic", "pessimistic"):
        raise ValueError((objective, selection))
    key = np.zeros(st.N, dtype=np.int64)
    for pos, l in enumerate(leaders):
        key += st.A[:, l] * (st.inst.m ** pos)
    n_keys = st.inst.m ** len(leaders)
    commitments = {}
    idx_all = np.flatnonzero(mask)
    for k in np.unique(key[idx_all]):
        cand = idx_all[key[idx_all] == k]
        vals = goal[cand]
        pick = cand[np.argmin(vals)] if selection == "optimistic" else cand[np.argmax(vals)]
        commitments[int(k)] = (float(goal[pick]), int(pick))
    if not commitments:
        return None
    best_key = min(commitments, key=lambda k: commitments[k][0])
    state = commitments[best_key][1]
    return {"state": state, "leader_cost": float(st.own[state, leaders].sum()) if leaders else 0.0, "social": float(st.social[state]), "goal": commitments[best_key][0],
            "commitments": commitments, "n_keys": n_keys}
