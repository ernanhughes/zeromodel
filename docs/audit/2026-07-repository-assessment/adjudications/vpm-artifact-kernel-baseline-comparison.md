# Baseline Comparison

A fair baseline is an immutable dataclass or dictionary containing a NumPy float64 matrix, ordered row IDs, ordered metric IDs, JSON metadata, SHA-256 over canonical JSON plus big-endian matrix bytes, deterministic argmax, and JSON/zip serialization. That baseline can already preserve scores, identifiers, metadata, content identity, finite lookup, deterministic tie-breaking, and basic traces.

ZeroModel adds an explicit layout recipe, normalized view matrix, row/column permutations, cell-level source/view mapping, provenance and parent conventions, validation at construction, a bundle API, rendering helpers, action/evidence separation, compiled plan identity, and Lua source generation. These additions are material when the same source must be inspected through multiple views, persisted, audited, or consumed outside Python.

Costs: more concepts, more public surface, more identity fields that change hashes even for behaviourally equivalent policies, and terminology that can sound stronger than the mechanism. For a single in-process lookup table, the baseline is simpler and probably preferable. For reviewed finite policy artifacts with trace and portability requirements, ZeroModel is a useful integration contract rather than a fundamentally new data structure.
