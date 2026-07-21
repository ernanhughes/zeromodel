# Video Action-Set Final Decision Rule Options

The implemented deterministic evaluator supports only rules that are already
fixed in an approved protocol. The initial supported rule shape is:

```json
{
  "kind": "fixed_metric_threshold",
  "aggregate": "mean",
  "metric_id": "METRIC-ID",
  "operator": "gte",
  "threshold": "0.0"
}
```

Supported aggregates are `mean`, `minimum`, and `maximum`. Supported operators
are `gte` and `lte`. The mapping must contain exactly these five keys. Missing
or unknown kinds, extra keys, unsupported aggregates/operators, and binary
floating-point thresholds are rejected.

Thresholds and evidence metrics are integers or decimal strings. Aggregation
uses Python `Decimal`; `gte` and `lte` include equality. Evidence is sorted by
family ID, episode ID, frame ordinal, frame ID, and provider ID before
aggregation. Therefore row input order cannot change result bytes or digests.

The evaluator refuses tuning or selection fields in the protocol and refuses
non-final evidence rows. If required final evidence is incomplete, the outcome is
`indeterminate`.

Choosing metric IDs, thresholds, providers, candidate sets, and operating points
is a scientific review decision and is not performed by Codex.
