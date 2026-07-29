# Research architecture

## Observation levels

The system separates:

1. atomic evidence;
2. episodes;
3. lived, interpretive, and prospective day views;
4. daily state vectors;
5. learned states and macro-states;
6. transition dynamics;
7. forecasts and simulated trajectories.

This prevents message volume from being treated as importance and prevents
later reflection from being silently rewritten as an event on the original day.

## State estimation

Daily vectors combine reduced semantic coordinates with source-normalized
structural features. Clustering is unsupervised. State IDs are arbitrary and
must be interpreted from held-out profiles or independent validation.

## Temporal validation

Forecasts use expanding-window, walk-forward evaluation. Future-linked impact,
later-reference counts, forward recurrence, and hindsight hinge labels are
excluded from the predictor state.

## Blind validation

External labels are created without exposing the classifier to cluster labels,
profiles, or state descriptions. Association is evaluated only afterward using
contingency tables, enrichment, mutual information, and permutation tests.

## Causal boundary

Observed conditional transitions estimate:

```text
P(future state | observed action)
```

They do not by themselves estimate:

```text
P(future state | do(action))
```

Interventional estimates require repeated deliberate interventions, natural
experiments, matched untreated periods, or an explicit identification design.

