# VIPER, criticality, and verifiable policy artifacts

The closest research precedent for ZeroModel's compiled-versus-invoked distinction is VIPER: *Verifiable Reinforcement Learning via Policy Extraction* by Osbert Bastani, Yewen Pu, and Armando Solar-Lezama (NeurIPS 2018).

VIPER uses a neural policy and its Q-function as an upstream oracle, then extracts a smaller decision-tree policy that can be analyzed with techniques that are impractical on the original network.

```text
neural policy oracle
        ↓
policy extraction
        ↓
structured decision-tree policy
        ↓
runtime execution and verification
```

The model remains upstream. The extracted structure becomes the runtime object.

That is a direct precedent for a central ZeroModel proposition:

> A powerful model may produce policy without remaining inside every later decision.

## Structured policy enables new operations

VIPER's primary objective is verification. Its extracted decision trees expose a structure that existing verification tools can analyze more efficiently than the source neural policy.

The lesson for ZeroModel is broader than speed:

> Changing the representation of a policy can make downstream operations practical that were difficult on the source model.

For VIPER, the downstream operation is formal verification of the extracted tree.

For ZeroModel, current downstream operations include:

- deterministic state addressing;
- source-to-view mapping;
- whole-field rendering;
- artifact identity;
- serialization;
- policy comparison;
- per-decision artifact-and-cell traces;
- exhaustive row-level policy properties.

The shared principle is compilation into a representation designed for a different consumer.

## The alternatives carry information

VIPER's Q-DAGGER algorithm does not treat every sampled state equally. It prioritizes states using a Q-derived criticality measure:

```text
criticality(s) = max Q(s, a) - min Q(s, a)
```

A large value means the difference between a good and a poor action is consequential.

ZeroModel 1.0.11 preserves that signal as a non-action evidence metric. It also preserves a second, distinct value:

```text
decision_margin(s)
    = best Q(s, a) - second_best Q(s, a)
```

These answer different questions:

```text
criticality:
How costly could a poor action be?

decision margin:
How decisively does the winner beat its nearest alternative?
```

A state can therefore be:

- important and unambiguous;
- important and fragile;
- low-consequence and ambiguous;
- low-consequence and clear.

A Q-bearing VPM can retain:

```text
candidate Q-values
criticality
decision margin
selected action
source coordinates
view coordinates
artifact identity
```

`VPMPolicyLookup` selects only over declared action columns. Criticality and decision margin remain evidence and can be returned with the runtime trace.

The terminology boundary matters. Best-minus-worst is VIPER-style criticality only when the source values have Q-value or equivalent consequence semantics. On arbitrary scores, it is only score spread.

## Verification and identity are complementary

VIPER asks:

> Does this structured policy satisfy the checked property?

ZeroModel asks:

> Which exact policy artifact and value produced this action?

ZeroModel 1.0.11 connects those questions with a verification artifact:

```text
policy artifact
        ↓ checked by
finite property checker
        ↓ produces
verification artifact
```

The verification artifact records:

- checked policy artifact ID;
- checker version;
- property IDs and versions;
- property-spec digest;
- rows checked;
- pass or fail;
- exact counterexample states;
- selected candidates and evidence;
- source and view coordinates.

Its provenance contains a parent relation named `verifies` pointing to the exact policy artifact checked.

The resulting claim is precise:

> This identified finite policy artifact passed these named row-level properties under this identified checker and property specification.

The hash does not prove authorship, authorization, universal safety, or that the property set is sufficient.

## Counterexample, repair, and re-verification

VIPER's Pong experiment provides an important operational pattern:

```text
structured policy
        ↓
verification
        ↓
counterexample
        ↓
localized repair
        ↓
re-verification
```

ZeroModel makes that loop an artifact lineage:

```text
original policy artifact
        ↓
unsafe policy artifact
        ↓
failed verification artifact
        ↓
localized counterexample
        ↓
repaired policy artifact
        ↓
passing verification artifact
```

Every policy and verification result has an identity. Every failure can retain its state, selected action, candidate values, evidence, and exact source cell.

The committed `examples/criticality_verification.py` fixture demonstrates the loop by seeding one invalid `FIRE` winner, locating the exact row, rebuilding a repaired artifact, and producing a passing re-verification artifact.

Automatic repair is not part of the current release. The example records a reviewable repair and promotion path.

## Criticality-aware representation

VIPER also suggests a research direction for larger state spaces.

When every state cannot receive equal representation capacity, criticality may help allocate:

- finer discretization;
- more testing;
- greater inspection priority;
- stronger fallback requirements.

```text
high criticality:
finer representation and stronger review

low criticality:
coarser representation may be acceptable
```

VIPER motivates this direction but does not prove that its Q-DAGGER bound transfers to VPM quantization or hierarchical addressing. That requires a separate fixed-budget comparison between uniform and criticality-weighted representations.

## Relationship

VIPER and ZeroModel are not competitors solving the same problem.

```text
VIPER:
extract a structured policy
so behavioural properties become tractable to verify

ZeroModel:
turn scored policy and evidence surfaces
into identified, spatially declared, traceable artifacts
```

A VIPER tree could become a ZeroModel policy producer. ZeroModel can preserve its candidate evidence and attach verification results to the exact deployed artifact.

The research extension is:

> Compile policy not only so it can run without the source model, but so its critical states, checked properties, counterexamples, repairs, and runtime decisions become durable parts of the artifact record.

## Reference

Osbert Bastani, Yewen Pu, and Armando Solar-Lezama. “Verifiable Reinforcement Learning via Policy Extraction.” *Advances in Neural Information Processing Systems 31*, 2018.
