# Documentation

This directory contains supporting documentation for the Consolidation Detection Framework.

## Contents

- [`paper.md`](paper.md): formal research note describing the framework, design rationale, and methodology
- [`methodology.md`](methodology.md): detection logic, validation design, and output interpretation

## Operating Assumptions

- Input data includes `snapshotTime`, `open`, `high`, `low`, and `close`.
- The input series can be ordered by timestamp and treated as a time-indexed OHLC sequence.
- Parameter selection uses rolling-origin validation rather than shuffled sampling.

## Execution Path

1. Load and validate OHLC data.
2. Engineer price, volatility, momentum, statistical, and Sutte-derived features.
3. Optimize detector parameters with time-series-aware validation.
4. Detect consolidation windows and estimate breakout probabilities.
5. Calibrate probabilities and export dashboard artifacts.

For citation metadata, see [`../CITATION.cff`](../CITATION.cff).
