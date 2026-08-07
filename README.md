# Life Topology Lab

**Interactive demo:** deployment link available from the repository homepage.

The demo exposes the evaluation architecture—retrieval benchmarking, blind
state validation, sampled topology, negative-result tracking, and walk-forward
forecasting—using synthetic display records. It contains no private corpus
content.

A privacy-preserving research toolkit for modeling a longitudinal personal
record as a dynamical system.

The repository preserves the **methods**, not a person's life:

- multi-view daily state construction;
- semantic and structural state vectors;
- six-state clustering and transition analysis;
- persistent topology and reconstructed dynamics;
- Conley–Morse approximations;
- symbolic transition grammar;
- hinge and topological-echo analysis;
- walk-forward state forecasting;
- Monte Carlo trajectory simulation;
- blind cognitive-operation validation.

No messages, names, contacts, locations, embeddings, daily labels, findings,
or subject-level outputs are stored here.

## Privacy boundary

```text
private corpus
    ↓ local derivation
daily features + vectors
    ↓
Life Topology Lab
    ↓
private local outputs
```

The private corpus and generated results remain outside Git. Paths are supplied
at runtime. See [PRIVACY.md](PRIVACY.md).

## Model supply-chain boundary

Embedding models are treated as executable dependencies rather than arbitrary
user input. The runtime accepts only the audited model allowlist in
`life_topology.config`, pins each download to an exact repository commit, and
disables remote model code. Models that publish safetensors are required to use
that format. The security workflow audits both Python and npm dependencies on
every dependency change and once each week.

## Blind cognitive-operation experiment

The operation experiment independently assigns every day one primary operation:

1. Observe
2. Experience
3. Evaluate
4. Model
5. Execute

The classifier receives daily text but no learned state IDs or state profiles.
Only after classification are the labels joined to the six learned states.
The analysis reports:

- the 6×5 confusion matrix;
- operation enrichment relative to its global base rate;
- chi-square independence;
- mutual information;
- a 10,000-permutation null;
- replication across two embedding models;
- agreement-only results.

## Status

This repository is the sanitized extraction of an active longitudinal research
system. The source modules are preserved now; the next extraction pass will
replace the remaining project-local loaders with a stable public input API and
synthetic fixtures.
