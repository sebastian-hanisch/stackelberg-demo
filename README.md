# 🥇 Stackelberg – wer sich zuerst festlegt, lenkt das Ergebnis

Sechstes Stück der **Spieltheorie-&-Mechanism-Design-Linie** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Nachfolger von
[korreliert-demo](https://sebastianhanisch-korreliert-demo.streamlit.app/): dort lenkte ein Vermittler durch Empfehlungen, hier durch **Festlegung**. Die Bindung an den Erstzug und volle Information der
Folger über ihn sind Modellannahmen.

## Warum dieses Problem

Bisher wählten alle Lkw gleichzeitig. Ein **Anführer** legt sich stattdessen zuerst auf ein Tor fest – verbindlich –, und die übrigen Lkw, die **Folger**, reagieren darauf mit einem Nash-Gleichgewicht ihres Restspiels
(von Stackelberg 1934). Wer festlegt, weiß, wie die Folger antworten, und kann sich das beste Tor aussuchen. Gefragt wird: was bringt der Erstzug, wem nützt er – und wie viel Last muss eine Disposition steuern,
um das Optimum zu erreichen (Stackelberg-Routing; Korilis/Lazar/Orda 1997, Roughgarden 2001)?

## Modell

**Torwahl-Vehikel** (`sk_gates.py`): wortgleich aus [nash-demo](https://github.com/sebastian-hanisch/nash-demo) (Tore mit Wartezeit $a_g + b_g \cdot$ Last, Lkw mit Größe 1, 2 oder 3; Gütemaß = Summe der Wartezeiten aller Lkw).

**Stackelberg-Spiel** (`sk_stackelberg.py`): eine Regel wählt die $k$ Anführer (die größten, die kleinsten oder die ersten nach Nummer). Sie legen ihre Tore fest ($m^k$ Festlegungen); zu jeder Festlegung werden alle
reinen Nash-Gleichgewichte der Folger aus der Definition aufgezählt. Bei mehreren Folger-Gleichgewichten entscheidet die Auswahlregel: **optimistisch** (das für die Anführer beste, "starkes" Stackelberg) oder
**pessimistisch** (das schlechteste, "schwaches"). Ziel der Anführer: *eigennützig* (kleinste Summe der Wartezeiten der eigenen $k$ Lkw, ein Spediteur mit $k$ Lkw) oder *Gemeinwohl* (kleinste Summe aller Wartezeiten,
eine Disposition, die $k$ Lkw steuert und die anderen sich selbst überlässt).

**Mini-Spiel** (`sk_bimatrix.py`): zwei Lkw, zwei Tore (wie nash-demo); Lkw 1 legt sich zuerst fest, Lkw 2 antwortet.

## Methodik

- Alles ist Vollaufzählung aller $m^n$ Zuordnungen (n ≤ 10, m ≤ 3), kein Sampling: Festlegungen, Folger-Gleichgewichte und Nash-Bereiche werden direkt aus der Definition gerechnet und gegen eine unabhängige
  Schleife nachgeprüft.
- **Satz** (getestet): der optimistische Stackelberg-Wert ist nie schlechter als das beste Nash-Gleichgewicht aus Sicht des Ziels – der Anführer kann sich auf seinen Teil dieses Gleichgewichts festlegen. Ob er
  *strikt besser* sein kann, ist eine Messung (siehe Befunde).
- **Literatur** (per Recherche geprüft, nicht nachgebaut): von Stackelberg 1934 ("Marktform und Gleichgewicht"), Korilis/Lazar/Orda 1997 (Erreichen des Optimums durch Steuerung eines Teils des Verkehrs),
  Roughgarden 2001 (Stackelberg-Strategien, u. a. "größte Latenz zuerst").

## Befunde (gemessen, keine Behauptungen)

8 Lkw, 3 Tore, gemischte Größen, 100 Instanzen (Experiment 1) bzw. 30 Instanzen (Experiment 2).

| Frage | Befund | Test |
|---|---|---|
| Bringt der Erstzug einem einzelnen eigennützigen Anführer mehr als das beste Nash? | Nein: in keiner der 100 Instanzen verbessert sich der größte, kleinste oder erste Lkw gegenüber seinem Wert im besten Nash-Gleichgewicht (auch nicht zwei größte Lkw). Gegenüber dem **schlechtesten** Nash-Gleichgewicht spart der größte Lkw im Mittel 14,4 % seiner Wartezeit: der Erstzug ist eine Versicherung gegen das schlechte Gleichgewicht, kein Bonus über das gute hinaus. | `test_first_mover_experiment_numbers`, `test_no_first_mover_gain_over_the_best_nash_is_a_measurement_not_a_theorem` |
| Können mehrere Anführer mehr? | Ja, vier kleinste Lkw verbessern sich in 34 % der Instanzen über das beste Nash hinaus (im Mittel 1,2 %), vier größte in 6 %. | dito |
| Und wenn die Folger pessimistisch wählen? | Wählen sie unter mehreren Gleichgewichten das für den Anführer schlechteste, ist die Festlegung beim größten Einzel-Anführer in 59 % der Instanzen schlechter als das beste Nash-Gleichgewicht (im Mittel 6,0 % mehr; beim kleinsten Lkw in 84 %). Festlegen ohne Einfluss auf die Auswahl kann schaden. | dito |
| Wem nützt der Erstzug? | Für alle zusammen liegt die Summe der Wartezeiten mit dem größten Anführer bei 1,045 mal Optimum – zwischen bestem (1,021) und schlechtestem (1,057) Nash-Gleichgewicht: der Anführer nimmt sich das für ihn beste Gleichgewicht, das für die Gesamtheit nicht das beste ist. | dito |
| Wie viel Steuerung braucht eine Disposition? | Ohne Steuerung liegt die Summe der Wartezeiten 2,0 % (beste Folger-Wahl) bis 6,3 % (schlechteste) über dem Optimum. Steuert sie die größten Lkw, sind es mit 55 % der Last (3 Lkw) 0,4 % bis 0,9 %; ab 5 Lkw (77 % der Last) ist das Optimum im Mittel erreicht. | `test_control_experiment_numbers` |
| Kommt es nur auf die Zahl der Lkw an? | Nein: mit den 3 kleinsten Lkw (23 % der Last) bleibt es bei 1,4 % bis 4,7 %; selbst mit den 6 kleinsten (60 % der Last) bei 0,9 % bis 1,6 % – die größten Lkw erreichen mit 55 % der Last 0,4 % bis 0,9 %. Die Regel "nach Nummer" liegt dazwischen. | dito |
| Standardinstanz (Seed 35) | Der größte Lkw (Nr. 2, Größe 3, 17 % der Last) wartet als Anführer 12,35 min – genau so viel wie im besten Nash-Gleichgewicht (schlechtestes: 13,30, also 7,2 % mehr). Eine Disposition, die die 3 größten Lkw steuert (50 % der Last), erreicht das Optimum (95,46 min) – besser als jedes Nash-Gleichgewicht (96,90 bis 100,32); mit 2 Lkw noch nicht. Die 3 kleinsten Lkw als eigennützige Anführer: 33,65 statt 33,87 min (bestes Nash), aber die Summe aller Wartezeiten steigt auf 98,91. | `test_standardfall_numbers`, `test_disposition_3_numbers`, `test_kleine_lkw_numbers` |
| Mini-Spiel | δ = 1: legt sich Lkw 1 auf Tor A fest, antwortet Lkw 2 mit B: Kosten (7, 8). Auf B wäre es (8, 7). Der Erstzug löst die Koordination (Summe 15 min statt 17 im gemischten Gleichgewicht, das 8,5 min für beide kostet) und verteilt den Gewinn ungleich; die abwechselnde Ampel aus korreliert-demo gäbe beiden 7,5 min. | `test_mini_game_numbers_at_the_default_delta`, `test_stackelberg_by_hand_at_delta_one` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Festlegung ist verbindlich und die Folger sehen sie** | Kann der Anführer nachträglich wechseln oder sehen die Folger ihn nicht, ist es wieder ein gleichzeitiges Spiel. | – |
| **Die Folger spielen ein reines Nash-Gleichgewicht** | Lernende Folger enden nicht immer in einem Gleichgewicht; gemischte Folger-Gleichgewichte sind hier nicht abgebildet. | [No-Regret-Lernen](https://sebastianhanisch-noregret-demo.streamlit.app/) |
| **Die Disposition kennt das ganze Spiel und steuert verlässlich** | Echte Flotten haben Ausfälle, Verspätungen und fremde Lkw. | [Multi-Agenten-Koordination](https://sebastianhanisch-marl-demo.streamlit.app/) |
| **Steuerung ist kostenlos** | Ein gesteuerter Lkw verzichtet auf seinen Vorteil; wer das ausgleicht, ist eine Preisfrage. | [Maut](https://sebastianhanisch-maut-demo.streamlit.app/) |
| **Eine einzige Runde** | Über viele Runden können Anführer Ruf aufbauen oder Folger Strafen androhen. | – |

Die Messreihen gelten für diese eine Spielfamilie, diese Instanzgröße (8 Lkw, 3 Tore) und diese Auswahl der Anführer-Regeln. Nur reine Gleichgewichte; die Aufzählung ist auf $m^n \le 100\,000$ Zuordnungen begrenzt.

Verwandt: [korreliert-demo](https://sebastianhanisch-korreliert-demo.streamlit.app/) (Vorgänger), [nash-demo](https://sebastianhanisch-nash-demo.streamlit.app/) (gleichzeitiges Torwahl-Spiel),
[maut-demo](https://sebastianhanisch-maut-demo.streamlit.app/) (Preise statt Führung).

## Tests

Pytest-Suite (`pytest tests/ -v`): Zustände, Kosten und Abweichungskosten gegen die Definition; Folger-Gleichgewichte und Stackelberg-Werte gegen eine unabhängige Schleife über alle Festlegungen (5 Lkw, 2 Tore,
beide Ziele, beide Auswahlregeln); Satz "optimistisch nie schlechter als bestes Nash" und "pessimistisch nie besser als optimistisch"; ohne Anführer = Nash-Bereich, alle Lkw führen = Optimum; Mini-Spiel per Handrechnung
(inklusive Gleichstand bei δ = 2 und Auswahlregel) und das 2×2-Spiel im Vergleich mit dem Aufzählungs-Kern; AppTest-Rauchtests (jedes Preset, Anführerzahl-Klammer, Permalink-Grenzen, Mini-Spiel, zwei Experimente auf Abruf)
und `test_claims.py` (jede Zahl aus diesem README; exakte Aufzählungen mit Toleranz für Rundung, Mittel über Instanzen mit Bändern).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `sk_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Experiment-Seeds, Presets |
| `sk_presets.py` | Permalink/Presets-Mechanik |
| `sk_gates.py` | Torwahl-Vehikel, Vollaufzählung der reinen Gleichgewichte (aus nash-demo) |
| `sk_stackelberg.py` | Zustände, Folger-Gleichgewichte, Festlegungen der Anführer, Anführer-Regeln |
| `sk_bimatrix.py` | Mini-Spiel mit Lkw 1 als Anführer |
| `sk_evaluation.py` | Analyse einer Instanz, Steuerungs-Kurve, zwei Experimente |
| `sk_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Mehr als 10 Lkw und mehr als 3 Tore (die Aufzählung wächst mit $m^n$).
- Gemischte Strategien von Anführern oder Folgern und wiederholte Spiele.
- Optimale Anführer-Auswahl (welche $k$ Lkw eine Disposition steuern sollte): hier nur drei einfache Regeln.
- Ein PDF-Export – wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
