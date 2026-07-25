# P18A Measurement Contract

For each field with values `before[i]` and `after[i]`:

- `before_mean = mean(before) / 255`
- `after_mean = mean(after) / 255`
- `mean_absolute_change = mean(abs(after - before)) / 255`
- `mean_signed_change = mean(after - before) / 255`
- `changed_value_count = count(abs(after - before) >= change_threshold)`
- `changed_fraction = changed_value_count / total_value_count`

All arithmetic uses signed intermediate deltas so downward changes cannot wrap around unsigned integer boundaries.
