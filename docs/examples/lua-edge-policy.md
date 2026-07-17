# Lua edge policy consumer

ZeroModel can compile an immutable VPM policy and fixed reader configuration into a dependency-free Lua module.

This is a portability fixture, not yet a microcontroller benchmark. It proves that the same identified policy can be consumed outside Python without NumPy or a model at decision time.

## Export the arcade policy

From the repository root:

```bash
python examples/lua_edge_policy.py \
  --output build/lua/arcade_policy.lua
```

The generated module contains:

- the exact VPM `artifact_id`;
- a separate deterministic `plan_id` for the reader configuration;
- stable row identifiers;
- action and evidence metric identifiers;
- precompiled candidate values;
- precompiled winning-action indices;
- source and view coordinates;
- `choose(row_id)` for the minimal hot path;
- `read(row_id)` for a complete candidate/evidence trace.

The plan ID is derived from:

- artifact identity;
- action metrics;
- evidence metrics;
- raw or normalized value source;
- tie-break rule;
- compiled values and coordinates.

The plan is a consumer-side derivative. It does not replace or mutate the VPM artifact identity.

## Run with Lua

```bash
lua5.4 examples/lua/run_arcade_policy.lua \
  build/lua/arcade_policy.lua
```

The Lua fixture simulates the same four-target arcade wave used by the Python example. It must:

- clear all four targets;
- finish inside the declared step bound;
- return the expected `FIRE` decision for an aligned ready state;
- retain the same VPM artifact identity in its trace.

The generated module requires only a standard Lua interpreter. It does not import Python, NumPy, a JSON library, a model runtime or the ZeroModel package.

## Why Lua

Lua is useful here because it is:

- small;
- embeddable;
- widely available in constrained and game runtimes;
- simple enough that the consumer logic remains inspectable.

The current fixture demonstrates language and runtime portability. It does **not** yet establish:

- a specific microcontroller memory footprint;
- sub-millisecond latency;
- energy consumption;
- real-time guarantees;
- equivalence across every Lua implementation;
- signed deployment provenance.

Those require named hardware and measured benchmarks.

## Python hot paths

The Python reader now exposes two paths:

```python
reader.choose(row_id)  # action only
reader.read(row_id)    # action plus full forensic trace
```

Construction compiles the immutable artifact into row/metric indices, value matrices, winning actions and coordinates. Runtime lookup no longer constructs one `VPMCell` per candidate or recomputes the argmax.

A benchmark that reports dictionary and VPM paths separately is available at:

```bash
python examples/policy_lookup_benchmark.py \
  --lookups 200000 \
  --repeat 5
```

Do not compare a dictionary returning one string with a VPM reader returning a complete decision trace as though they perform identical work. The benchmark reports action-only and trace-rich paths separately.
