# Fixed-camera status-panel benchmark

**Status:** scaffold specification only  
**As of July 17, 2026:** no empirical fixed-camera captures are committed

## Purpose

The arcade fixture is synthetic and render-perfect.

The next bounded real environment should be a fixed camera observing a small status panel with known valid states and independently recorded ground truth.

## Current repository state

This benchmark is not yet implemented as a completed empirical dataset or executed evaluation.

What exists now is only the declared requirement to preserve:

- capture identity;
- session identity;
- camera/setup identity;
- ground-truth state;
- partition and family identity;
- expected evaluation role.

## Required future families

- baseline lighting
- dim lighting
- bright lighting
- glare
- camera translation
- camera rotation
- focus variation
- compression
- partial occlusion
- background change
- invalid object configuration
- missing evidence
- out-of-domain frame

## Evidence boundary

Until real captures are collected and evaluated, this benchmark remains:

`implemented only as a research direction, not measured evidence`
