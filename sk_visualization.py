"""Plotly-Abbildungen der Stackelberg-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sk_constants as C

LINE_COLOR = "#4c78a8"
REF_COLOR = "#7f7f7f"
GOOD = "#54a24b"
BAD = "#e45756"
WARN = "#f58518"
RULE_COLORS = {"largest": LINE_COLOR, "smallest": WARN, "index": GOOD}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_goal_bars(labels_values, y_title, colors=None):
    """Balken für vier bis fünf Größen (bestes/schlechtestes Nash, Stackelberg optimistisch/pessimistisch, ...); `labels_values` = [(Name, Wert), ...]."""
    names = [x[0] for x in labels_values]
    vals = [x[1] for x in labels_values]
    colors = colors or [GOOD, BAD, LINE_COLOR, WARN, REF_COLOR][:len(vals)]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors, text=[f"{v:.2f}" for v in vals], textposition="outside"))
    fig.update_yaxes(title_text=y_title, range=[0, max(vals) * 1.12])
    return _base(fig, 300)


def build_commitments(rows, leaders, selection_label):
    """Wie gut ist jede Festlegung der Anführer für sie (nach der Auswahlregel)? Balken je Festlegung, beste zuerst."""
    labels = [", ".join(f"L{leaders[j] + 1}→T{g + 1}" for j, g in enumerate(gates)) or "keine" for gates, _, _, _ in rows]
    vals = [r[1] for r in rows]
    colors = [GOOD if i == 0 else "#c9c9c9" for i in range(len(rows))]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors, text=[f"{v:.2f}" for v in vals], textposition="outside"))
    fig.update_yaxes(title_text=f"Ziel der Anführer ({selection_label})", range=[min(vals) * 0.9, max(vals) * 1.05])
    fig.update_xaxes(title_text="Festlegung (Lkw → Tor)")
    return _base(fig, 300)


def build_control_curve(rows):
    """Summe aller Wartezeiten / Optimum, wenn eine Disposition den angegebenen Lastanteil steuert (Anführer nach der gewählten Regel), bei optimistischer und pessimistischer Auswahl."""
    xs = [100 * r["share"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["optimistic"] for r in rows], mode="lines+markers", name="Folger wählen das beste Gleichgewicht", line=dict(color=GOOD, width=2.5),
                             text=[f"{r['k']} Lkw" for r in rows], hovertemplate="%{text}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[r["pessimistic"] for r in rows], mode="lines+markers", name="Folger wählen das schlechteste", line=dict(color=BAD, width=2.5),
                             text=[f"{r['k']} Lkw" for r in rows], hovertemplate="%{text}: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Gesteuerter Lastanteil (%)")
    fig.update_yaxes(title_text="Summe der Wartezeiten / Optimum")
    return _base(fig, 320)


def build_first_mover(rows):
    """Links: mittlere Ersparnis des eigennützigen Anführers gegenüber dem schlechtesten Nash-Gleichgewicht; rechts: Anteil der Instanzen, in denen er sich gegenüber dem besten Nash-Gleichgewicht verbessert."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Ersparnis gegenüber dem schlechtesten Nash (%)", "Besser als das beste Nash (Anteil der Instanzen, %)"), horizontal_spacing=0.12)
    for rule in C.RULES:
        sel = [r for r in rows if r["rule"] == rule]
        if not sel:
            continue
        fig.add_trace(go.Bar(x=[r["k"] for r in sel], y=[100 * r["mean_gain_vs_worst"] for r in sel], name=C.RULE_LABELS[rule], marker_color=RULE_COLORS[rule]), row=1, col=1)
        fig.add_trace(go.Bar(x=[r["k"] for r in sel], y=[100 * r["share_gain_vs_best"] for r in sel], name=C.RULE_LABELS[rule], marker_color=RULE_COLORS[rule], showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="Anführer k", dtick=1)
    fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_pessimistic(rows):
    """Wie oft und wie stark schadet die Festlegung dem eigennützigen Anführer, wenn die Folger das für ihn schlechteste Gleichgewicht wählen? Anteil der Instanzen, in denen es schlechter ist als das beste Nash."""
    fig = go.Figure()
    for rule in C.RULES:
        sel = [r for r in rows if r["rule"] == rule]
        fig.add_trace(go.Bar(x=[r["k"] for r in sel], y=[100 * r["share_pes_worse_than_best"] for r in sel], name=C.RULE_LABELS[rule], marker_color=RULE_COLORS[rule]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Anführer k", dtick=1)
    fig.update_yaxes(title_text="Instanzen, in denen die Festlegung schadet (%)")
    return _base(fig, 300).update_layout(legend=dict(orientation="h", y=-0.3))


def build_control_experiment(out):
    """Mittlere Summe der Wartezeiten / Optimum über dem gesteuerten Lastanteil, je Anführer-Regel; durchgezogen: Folger wählen das beste Gleichgewicht, gestrichelt: das schlechteste."""
    fig = go.Figure()
    for rule in C.RULES:
        if rule not in out:
            continue
        x = 100 * np.asarray(out[rule]["share"])
        fig.add_trace(go.Scatter(x=x, y=out[rule]["optimistic"], mode="lines+markers", name=C.RULE_LABELS[rule], line=dict(color=RULE_COLORS[rule], width=2.5)))
        fig.add_trace(go.Scatter(x=x, y=out[rule]["pessimistic"], mode="lines", name=C.RULE_LABELS[rule] + " (pessimistisch)", line=dict(color=RULE_COLORS[rule], width=2, dash="dash"), showlegend=False))
    fig.update_xaxes(title_text="Gesteuerter Lastanteil (%)")
    fig.update_yaxes(title_text="Summe der Wartezeiten / Optimum")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_bimatrix(costs):
    """Erwartete Wartezeiten der beiden Lkw im Mini-Spiel je Ergebnis; `costs` = {Name: (Lkw 1, Lkw 2)}."""
    names = list(costs)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[costs[k][0] for k in names], name="Lkw 1 (Anführer)", marker_color=LINE_COLOR, text=[f"{costs[k][0]:.2f}" for k in names], textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=[costs[k][1] for k in names], name="Lkw 2", marker_color=WARN, text=[f"{costs[k][1]:.2f}" for k in names], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="(Erwartete) Wartezeit (min)", range=[0, max(max(v) for v in costs.values()) * 1.15])
    return _base(fig, 300)
