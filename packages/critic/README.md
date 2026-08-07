# ZeroModel Critic

ZeroModel Critic is a tiny learned readout over identified numeric evidence.

It is not an LLM, a universal verifier, a text model, semantic truth, or a replacement for expensive validation. Domain systems produce declared numeric features; Critic fits and replays one narrow judgement over that prepared feature surface.

```text
domain producer
↓
feature surface
↓
CriticFeatureBatch
↓
CriticReadout
↓
score / rank / triage
↓
VPM / receipt / replay
```

The v1 runtime is deliberately small: directionality, standardisation, a linear logit, sigmoid score, and optional Platt calibration. The executable portable payload is canonical JSON and defaults to a 50 KiB limit. Many specialised critics may read the same feature surface while answering different declared targets.

Safe claim: ZeroModel Critic can compile a declared numeric feature schema and a fitted lightweight linear classifier into an identified critic artifact that scores compatible evidence deterministically, exposes feature-level contributions, exports a portable arithmetic-only inference payload, and reproduces scoring through artifact replay.

