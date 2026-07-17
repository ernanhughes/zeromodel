# Visual local baseline showdown

**Status:** selected next experiment after System B v2 adjudication  
**Selection date:** July 17, 2026

## Why this is next

System B v2 showed:

- raw top-1 ranking remains materially informative;
- distinguishable rejection requires an operating point so strict that benign coverage collapses to zero;
- translation remains a concentrated failure family.

That evidence does not justify a full factorized Visual State Compiler yet.

It first justifies testing simpler local mechanisms that directly target alignment and locality.

## Ordered baseline set

The next bounded experiments should be:

1. registration plus local normalized-pixel comparison;
2. translation-equivariant template correlation;
3. deterministic connected-components and geometry extraction.

Only if those fail should the repository promote:

4. a tiny native-resolution CNN; or
5. a factorized multi-head evidence model.

## Current status

This document records the selected next experiment class.

It does **not** claim that the showdown has been implemented or executed yet.
