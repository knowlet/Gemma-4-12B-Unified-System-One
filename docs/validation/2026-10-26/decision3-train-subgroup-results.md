# Same-row TRAIN diagnosis exposes operation tradeoffs

Independent FP64 replay compared W0, the accepted 153-row operation readout and
the accepted 225-row readout on exactly the same 225 TRAIN rows. Every row kept
its original ordered operation criteria, soft targets and legal mask. No model,
browser or optimizer was run. The [aggregate metrics](decision3-train-subgroup-results.json)
include soft CE, valid-target probability margins and empty groups as null.

| TRAIN population | W0 correct | 153-row readout | 225-row readout |
| --- | ---: | ---: | ---: |
| All 225 | 84/225 | 177/225 | 202/225 |
| Retained 119 | 52/119 | 116/119 | 109/119 |
| Earlier admitted 34 | 10/34 | 31/34 | 28/34 |
| Navigation 72 | 22/72 | 30/72 | 65/72 |
| Navigation CLICK teachers | 0/33 | 10/33 | 32/33 |
| Navigation TYPE teachers | 5/9 | 1/9 | 7/9 |
| Navigation SELECT teachers | 3/6 | 3/6 | 2/6 |
| Navigation DONE teachers | 14/24 | 16/24 | 24/24 |

The larger fit gains on new rows while losing ten correct decisions on the
retained 153-row prefix. Four of its six navigation SELECT rows predict CLICK.
The earlier TYPE group also falls from 4/4 to 2/4, with both errors predicting
CLICK. Overall soft CE improves from 0.706 to 0.474 despite these regressions.
This supports a concrete operation tradeoff; it does not prove class frequency
is its only cause or explain an unseen browser trajectory by itself.

Literal state classification covers only the separately admitted 34 and 72
rows. Neither group contains a TYPE teacher with nonempty real action history.
Of the three navigation SELECT teachers with history, all three readouts miss
all three; only one of those rows also has an observed checked radio. Earlier
TYPE rows with a checked radio and empty history decline from 3/3 to 1/3.
Here "checked" means any offered radio is checked, not that its value satisfies
the goal. The retained 119 rows were not classified by these state conditions.

The [new independent collection](decision3-scroll-state-native-collection-results.md)
and its [verified frozen features](decision3-scroll-state-gpu-feature-results.md)
provide the next training inputs. A separate classification of those actual
48 wires verifies six TYPE and six SELECT rows with a checked radio and
nonempty history, directly adding joint-state coverage to the diagnosed
admitted populations. Compare an unchanged-objective reference
with explicit teacher-stratum weighting while tracking retained-prefix and
rare-operation regressions. This remains a prospective experiment: no new fit
or deployed readout is established here.

The owned CPU replay completed successfully with five cores and a separate
terminal closure. A separate bootstrap supplied only the exact empty Root log
identity authenticated by the previously accepted fit receipt, then invoked
the unchanged replay leaf's finalizer. The original numerical source, cached
results and owner evidence were preserved. These TRAIN metrics are neither
browser success rates nor either JevBench score; no winning-model claim follows.
