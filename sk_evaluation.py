"""Auswertung: eine Stackelberg-Analyse je Instanz und Einstellungen, Steuerungs-Kurve über die Zahl der Anführer, zwei Experimente (Erstzug-Vorteil, Steuerungsanteil)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import sk_constants as C
import sk_gates as G
import sk_stackelberg as S


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    size_mode: str = "mixed"
    seed: int = C.DEFAULT_SEED
    rule: str = "largest"
    k: int = C.DEFAULT_K
    objective: str = "own"
    selection: str = "optimistic"


@lru_cache(maxsize=256)
def instance(n, m, size_mode, seed):
    return G.generate(n, m, size_mode, seed)


@lru_cache(maxsize=8)
def states(n, m, size_mode, seed):
    return S.States(instance(n, m, size_mode, seed))


@dataclass
class Analysis:
    settings: Settings
    inst: object
    st: object
    leaders: list
    nash: np.ndarray          # Maske der Nash-Gleichgewichte (alle Lkw)
    opt_res: dict             # Stackelberg-Ergebnis, optimistische Auswahl
    pes_res: dict             # pessimistische Auswahl

    @property
    def opt_social(self):
        return float(self.st.social.min())

    def goal(self):
        """Ziel des Anführers je Zuordnung (Summe der eigenen Wartezeiten bzw. aller Wartezeiten)."""
        if self.settings.objective == "own" and self.leaders:
            return self.st.own[:, self.leaders].sum(axis=1)
        return self.st.social

    def nash_goal_range(self):
        g = self.goal()[self.nash]
        return float(g.min()), float(g.max())

    def nash_social_range(self):
        s = self.st.social[self.nash]
        return float(s.min()), float(s.max())

    def result(self, selection=None):
        return self.opt_res if (selection or self.settings.selection) == "optimistic" else self.pes_res

    def commitment_table(self, selection=None, top=8):
        """Festlegungen der Anführer, sortiert nach ihrem Ziel bei der Auswahlregel: (Tore der Anführer, Ziel, Summe aller Wartezeiten, gewählte Zuordnung)."""
        res = self.result(selection)
        m = self.inst.m
        rows = []
        for key, (val, state) in res["commitments"].items():
            gates = tuple((key // m ** pos) % m for pos in range(len(self.leaders)))
            rows.append((gates, val, float(self.st.social[state]), tuple(int(x) for x in self.st.A[state])))
        rows.sort(key=lambda r: r[1])
        return rows[:top], len(rows)


@lru_cache(maxsize=64)
def analyse(settings):
    inst = instance(settings.n, settings.m, settings.size_mode, settings.seed)
    st = states(settings.n, settings.m, settings.size_mode, settings.seed)
    leaders = S.leaders_by_rule(inst, settings.k, settings.rule)
    nash = st.is_ne_for(range(inst.n))
    return Analysis(settings, inst, st, leaders, nash, S.solve(st, leaders, settings.objective, "optimistic"), S.solve(st, leaders, settings.objective, "pessimistic"))


def control_curve(settings, st=None):
    """Über alle Anführerzahlen k = 0..n (Gemeinwohl-Ziel): Summe aller Wartezeiten / Optimum bei optimistischer und pessimistischer Auswahl, dazu der Lastanteil der Anführer."""
    inst = instance(settings.n, settings.m, settings.size_mode, settings.seed)
    st = states(settings.n, settings.m, settings.size_mode, settings.seed) if st is None else st
    o = float(st.social.min())
    rows = []
    for k in range(inst.n + 1):
        lead = S.leaders_by_rule(inst, k, settings.rule)
        rows.append({"k": k, "share": float(inst.w[lead].sum() / inst.w.sum()) if lead else 0.0, "optimistic": S.solve(st, lead, "social", "optimistic")["social"] / o,
                     "pessimistic": S.solve(st, lead, "social", "pessimistic")["social"] / o})
    return rows


# --- Experiment 1: Erstzug-Vorteil ------------------------------------------------------------------------------------------------------------


def first_mover_experiment(ks=None, rules=None, n=None, m=None, seeds=None, size_mode="mixed"):
    """Eigennützige Anführer (k Lkw, Regel): Ziel = Summe ihrer eigenen Wartezeiten. Gegenüber dem besten und dem schlechtesten Nash-Gleichgewicht (aus Sicht der Anführer) und bei pessimistischer Auswahl."""
    ks = C.FIRST_KS if ks is None else ks
    rules = C.RULES if rules is None else rules
    n = C.FIRST_N if n is None else n
    m = C.FIRST_M if m is None else m
    seeds = C.FIRST_SEEDS if seeds is None else seeds
    acc = {(rule, k): {"g_best": [], "g_worst": [], "opt_v": [], "pes_v": [], "soc_opt": [], "soc_lo": [], "soc_hi": []} for rule in rules for k in ks}
    opt_soc = []
    for s in seeds:
        inst = instance(n, m, size_mode, s)
        st = S.States(inst)
        nash = st.is_ne_for(range(n))
        opt_soc.append(st.social.min())
        for rule in rules:
            for k in ks:
                lead = S.leaders_by_rule(inst, k, rule)
                g = st.own[:, lead].sum(axis=1)
                r_o, r_p = S.solve(st, lead, "own", "optimistic"), S.solve(st, lead, "own", "pessimistic")
                a = acc[(rule, k)]
                a["g_best"].append(g[nash].min())
                a["g_worst"].append(g[nash].max())
                a["opt_v"].append(r_o["leader_cost"])
                a["pes_v"].append(r_p["leader_cost"])
                a["soc_opt"].append(r_o["social"])
                a["soc_lo"].append(st.social[nash].min())
                a["soc_hi"].append(st.social[nash].max())
    opt_soc = np.array(opt_soc)
    rows = []
    for (rule, k), a in acc.items():
        g_best, g_worst, opt_v, pes_v, soc_opt, soc_lo, soc_hi = (np.array(a[x]) for x in ("g_best", "g_worst", "opt_v", "pes_v", "soc_opt", "soc_lo", "soc_hi"))
        rows.append({"rule": rule, "k": k, "n_inst": len(seeds),
                     "share_gain_vs_best": float(np.mean(opt_v < g_best - 1e-9)), "mean_gain_vs_best": float(np.mean((g_best - opt_v) / g_best)),
                     "mean_gain_vs_worst": float(np.mean((g_worst - opt_v) / g_worst)), "share_pes_worse_than_best": float(np.mean(pes_v > g_best + 1e-9)),
                     "mean_pes_loss_vs_best": float(np.mean((pes_v - g_best) / g_best)),
                     "social_stack": float(np.mean(soc_opt / opt_soc)), "social_nash_best": float(np.mean(soc_lo / opt_soc)), "social_nash_worst": float(np.mean(soc_hi / opt_soc))})
    return rows


# --- Experiment 2: Steuerungsanteil ------------------------------------------------------------------------------------------------------------


def control_experiment(rules=None, n=None, m=None, seeds=None, size_mode="mixed"):
    """Eine Disposition steuert k Lkw (Gemeinwohl-Ziel), die übrigen spielen Nash: Mittel von Summe der Wartezeiten / Optimum und Lastanteil über feste Instanzen, je Regel und Auswahlregel."""
    rules = C.RULES if rules is None else rules
    n = C.CONTROL_N if n is None else n
    m = C.CONTROL_M if m is None else m
    seeds = C.CONTROL_SEEDS if seeds is None else seeds
    per = {rule: [] for rule in rules}
    for s in seeds:
        st = S.States(instance(n, m, size_mode, s))
        for rule in rules:
            per[rule].append(control_curve(Settings(n, m, size_mode, s, rule), st))
    out = {}
    for rule in rules:
        out[rule] = {"k": list(range(n + 1)), "share": np.mean([[r["share"] for r in c] for c in per[rule]], axis=0),
                     "optimistic": np.mean([[r["optimistic"] for r in c] for c in per[rule]], axis=0), "pessimistic": np.mean([[r["pessimistic"] for r in c] for c in per[rule]], axis=0)}
    return out
