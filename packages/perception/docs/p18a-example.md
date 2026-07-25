# P18A Example

Given a two-field grayscale observation:

```text
before: [unchanged actor field] [unchanged target field]
after:  [changed actor field]   [unchanged target field]
```

P18A produces:

- one field record with non-zero absolute and signed change;
- one field record with zero change;
- annotation identities bound to their declared fields;
- a grayscale transition VPM whose changed field is bright and unchanged field is dark;
- a transition identity binding both observations, the schema, threshold, measurements, annotations, and rendered PNG digest.

The result says where measured pixels changed. It does not say why they changed.
