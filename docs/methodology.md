# Methodology

For the longer research-style document, see [`paper.md`](paper.md).

## Objective

Identify windows where price action exhibits sustained compression and structural stability consistent with consolidation rather than active expansion.

## Signal Construction

The detector combines multiple transparent components:

- contraction and range-width measures over rolling windows
- price-position and mean-reversion features
- Hurst-style persistence diagnostics used as a structural penalty
- attention-style context matching against comparable historical windows
- periodic context features derived from intraday and day-of-week cycles
- scale-anchor features based on local price structure
- Alpha-Sutte diagnostics for directional alignment and conviction

Each component is normalized before aggregation so parameter optimization can weight them on a comparable scale.

## Validation Design

Parameter search uses Optuna with rolling-origin cross-validation. The study objective combines predictive performance with a stability penalty so the selected parameter set is not driven by a single favorable validation fold.

## Outputs

The pipeline produces:

- consolidation events with component-level scores
- breakout probabilities and directions
- summary metrics for regime behavior and probability dispersion
- an HTML dashboard for inspection of events, component behavior, and optimization results

## Practical Limitations

- The implementation is a single-file reference system rather than a packaged library.
- Reported metrics are diagnostic and should not be treated as execution-ready performance estimates.
- The default workflow assumes local CSV access and optional access to a companion loader module.
