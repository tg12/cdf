# CDF: Consolidation Detection Framework

**Author:** James Sawyer  
**Project type:** Research software and reference implementation  
**Primary audience:** Quantitative researchers, systematic traders, market structure analysts, and technical reviewers who want an interpretable consolidation detector rather than an opaque prediction stack

## Abstract

CDF studies market consolidation as a measurable structural regime in OHLC time series. The framework does not treat consolidation as a loose chart annotation. It treats it as a composite state defined by compression, bounded price exploration, non-persistent local dynamics, contextual similarity to prior episodes, and directional candle diagnostics. The implementation then estimates whether that state resolves into a directional breakout, calibrates the resulting probabilities, and exports an inspection-oriented dashboard.

This repository was created by **James Sawyer** to answer a practical research question: can consolidation be detected and ranked with transparent statistics before any breakout model is applied? The design goal is not to hide signal generation inside a black-box architecture. The design goal is to separate structural regime detection from directional expansion scoring so that a reviewer can inspect each stage, challenge each assumption, and reproduce the full pipeline.

The resulting system is intentionally technical. It combines volatility contraction, price-position stability, Hurst-style persistence diagnostics [1], RSI and momentum-style trend context derived from Wilder-era technical analysis [6], attention-like historical context matching, periodic time encoding, a local autoregressive anchor inspired by the long-short temporal decomposition used in LSTNet [5], Bayesian hyperparameter search with TPE through Optuna [2][3], rolling-origin cross-validation for time-order preservation [4], and a monotonic post-hoc probability calibration stage. The emphasis is on interpretable research machinery, not minimal code golf.

For a longer technical treatment, see [`docs/paper.md`](docs/paper.md). For citation metadata, see [`CITATION.cff`](CITATION.cff).

## Why This Exists

Many breakout systems fail because they conflate two different problems:

1. Is the market actually in a compressed regime?
2. If it is, is the subsequent move strong enough to treat as expansion rather than noise?

CDF separates those questions. James Sawyer built this repository for users who care about that separation:

- researchers testing whether consolidation is a statistically coherent state rather than a discretionary label
- systematic traders who want to audit component scores instead of trusting a single indicator
- technical leads and reviewers who need deterministic, inspectable behavior before considering productionization

The project is aimed at users who prefer falsifiable structure over narrative trading heuristics. It is not primarily a beginner tutorial. It is a working research instrument.

## Research Positioning

CDF sits between classical technical analysis and modern time-series modeling:

- From classical technical analysis it inherits bounded-range reasoning, momentum normalization, and candle-structure interpretation [6].
- From time-series analysis it adopts rolling-origin validation so that model selection does not leak future information into the past [4].
- From modern machine learning systems it borrows efficient hyperparameter optimization through TPE and Optuna [2][3].
- From neural temporal modeling it borrows the idea that long-term periodic structure, short-term local structure, and autoregressive anchoring should be modeled as distinct but interacting sources of evidence [5].

The result is not a neural network. It is a transparent scoring framework that borrows useful ideas without surrendering interpretability.

## Intended User

CDF was written for the following reader:

- someone comfortable reading feature engineering code and validation logic
- someone who wants to know why a consolidation score is high, not just that it is high
- someone evaluating market structure in replay, research, or hybrid discretionary-systematic workflows

If that is not the target user, the repository will feel over-specified. That is intentional.

## System Overview

The implementation in [`cdf.py`](cdf.py) follows a single end-to-end pipeline:

1. Load OHLC data and construct a time-indexed frame.
2. Engineer price, volatility, momentum, statistical, temporal, and Sutte-derived features.
3. Optimize consolidation and breakout parameters with rolling-origin cross-validation.
4. Score each candidate consolidation window.
5. Generate breakout probabilities and directions from validated consolidation zones.
6. Calibrate probabilities from out-of-fold empirical hit rates.
7. Export metrics and an interactive dashboard.

This repository is currently a **single-file reference implementation** rather than a packaged library. That choice keeps the research logic in one place for audit and review.

## Data Contract

The pipeline expects OHLC input with at least the following columns:

- `snapshotTime`
- `open`
- `high`
- `low`
- `close`

The datetime column is parsed using the configured format in `cdf.py`, then promoted to the index. The framework does not currently require exchange metadata, volume, order book depth, or trade prints. It is therefore a price-structure model rather than a full microstructure model.

## Methodology

### 1. Feature Engineering

CDF constructs several families of interpretable features from raw OHLC bars.

**Price geometry**

- `hl_range` and `hl_range_pct` measure absolute and normalized bar range.
- `oc_range` and `oc_range_pct` capture candle body magnitude.
- `body_to_range_ratio` and `wick_balance` characterize candle compression and symmetry.

**Volatility contraction**

- Rolling highs and lows define local price envelopes.
- `range_width_pct_p` measures envelope width over multiple horizons.
- `vol_ratio_p` compares recent mean range to a longer baseline, providing a direct compression signal.

**Momentum and positional structure**

- Rate of change and RSI are computed over short and medium windows [6].
- `trend_strength_p` measures directional persistence relative to typical bar range.
- Rolling price position tracks where closing prices sit inside the evolving local envelope.

**Statistical regime features**

- Z-scores normalize price against rolling mean and standard deviation.
- A rolling Hurst-style estimate is computed from the log-log relation between lag and differenced standard deviation, which serves as a persistence diagnostic rather than as a strict textbook long-memory estimator [1].

**Temporal context encoding**

- hour, minute-of-day, and day-of-week are encoded as sine/cosine pairs
- this allows the detector to compare windows by phase rather than raw timestamp

**Sutte-derived directional diagnostics**

- the framework derives low-ratio and high-ratio moving averages from close relative to low and high
- the resulting `sutte_signal`, `sutte_conviction`, and `sutte_direction` form an interpretable directional subscore
- an exponentially weighted variant is also retained for faster local diagnostics

### 2. Consolidation Scoring

Each candidate window receives a composite consolidation score. In simplified form:

```text
score =
    w_contraction * contraction_component
  + w_range       * range_component
  + w_position    * position_component
  + w_hurst       * hurst_component
  + w_attention   * attention_context_component
  + w_periodic    * periodic_context_component
  + w_scale       * scale_anchor_component
  + w_sutte       * sutte_component
```

Where:

- `contraction_component` rewards low recent volatility relative to local history
- `range_component` rewards narrow realized price width after volatility normalization
- `position_component` rewards stable price placement within the evolving local range
- `hurst_component` rewards non-persistent local behavior and penalizes persistent trend structure
- `attention_context_component` rewards similarity to prior historically compression-like contexts
- `periodic_context_component` rewards similarity in intraday and weekly phase space
- `scale_anchor_component` rewards closeness to a small autoregressive local anchor
- `sutte_component` rewards coherent Sutte conviction and direction stability

This score is not hand-tuned once and left alone. The weights and structural thresholds are optimized from data.

### 3. Context Matching Logic

One of the more distinctive aspects of CDF is that it does not only look at the current window in isolation. It also asks whether the current context resembles prior compressed contexts.

The implementation:

- builds robustly scaled context vectors using medians and robust dispersion estimates
- applies weighted L1 distance in feature space
- ranks historical analogs by `exp(-distance / temperature)`
- aggregates the compression signatures of the top-ranked historical matches

This is not transformer attention in the strict architectural sense. It is an interpretable analog of attention-style similarity weighting. The periodic context path uses the same logic on cyclical time encodings, while the scale-anchor path uses a small autoregressive fit. The division between long-range temporal context and local autoregressive anchoring is conceptually aligned with the long-short decomposition ideas described in LSTNet [5], but implemented here with explicit statistics rather than deep latent states.

### 4. Directional Alignment and Divergence

CDF also measures whether the Sutte-derived directional bias agrees with recent price direction inside the consolidation window. The outcome is represented as:

- `ALIGNED`
- `DIVERGENT`
- `NEUTRAL`

This alignment signal does not define consolidation on its own. It acts as a confidence modifier for subsequent breakout scoring.

### 5. Breakout Probability Model

Once a consolidation zone is accepted, CDF evaluates post-zone prices relative to the zone high and zone low.

For each future bar inside a bounded hold horizon, it computes:

- directional distance beyond the zone boundary
- current volatility relative to in-zone volatility
- a context bias term derived from the strength of the accepted consolidation

That expansion strength is then mapped through a logistic function:

```text
probability = sigmoid(
    logistic_steepness * (
        breakout_strength - min_expansion - logistic_threshold
    )
)
```

The raw probability is then adjusted by:

- current Sutte directional support
- consolidation-window Sutte/price alignment

This makes the breakout score conditional on both structural compression and directional coherence.

### 6. Optimization and Validation

Hyperparameter search uses **Optuna** [3] with the **TPESampler**, which implements the Tree-structured Parzen Estimator approach described by Bergstra et al. [2]. The search space includes:

- consolidation lookback and lookforward
- minimum bar count
- range threshold
- position sensitivity
- consolidation threshold
- breakout expansion threshold
- breakout hold period
- logistic steepness and threshold
- all normalized score-component weights

Validation uses rolling-origin splits so that each validation fold only sees history available up to that point in time [4]. This is a deliberate anti-lookahead choice. The fold score combines:

```text
0.3 * F1
+ 0.3 * Sharpe
+ 0.4 * mean(precision, recall)
```

The study objective then subtracts a stability penalty based on:

- downside deviation across folds
- lower-quartile underperformance
- worst-fold gap
- degradation trend across fold order

This means the optimizer is asked to find parameterizations that are not only strong on average, but also resistant to fold-to-fold fragility.

### 7. Probability Calibration

Raw breakout probabilities are not assumed to be well calibrated. CDF therefore fits a monotonic, piecewise-linear mapping from out-of-fold predicted probabilities to empirical directional hit rates.

The calibration procedure:

- collects non-zero out-of-fold signals
- bins probabilities by quantiles
- computes empirical hit rates within each bin
- blends raw confidence with empirical hit rate
- enforces monotonic anchors
- interpolates calibrated probabilities from the resulting anchor curve

This stage exists because ranking quality and probability calibration are different problems.

### 8. Diagnostics and Reporting

The framework reports:

- total consolidations
- breakout signal count
- component means across accepted consolidations
- signal probability dispersion and bootstrap confidence intervals
- simulated directional return statistics
- recent-versus-baseline regime volatility ratios
- feature instability diagnostics
- optimization parameter influence estimates

When Plotly is available, the pipeline also generates an HTML dashboard in `results/consolidation_dashboard_latest.html` that includes the primary chart, component diagnostics, optimization summaries, and recent signal context.

## Technical Contribution

What makes CDF interesting is not any single indicator. It is the way the implementation composes several interpretable ideas into a consolidated structural test:

- It treats consolidation as a regime-scoring problem, not as a simple range breakout trigger.
- It uses robust history matching rather than only current-window thresholding.
- It separates structural detection from directional expansion scoring.
- It optimizes for both predictive quality and fold stability.
- It calibrates probabilities instead of treating raw sigmoid output as trustworthy.
- It exports enough intermediate state to make review possible.

That is the central technical statement of the repository.

## Reproducibility

CDF is designed to be reproducible at the research level:

- Python and NumPy random generators are seeded
- configuration is centralized near the top of `cdf.py`
- logging is explicit and structured
- rolling-origin validation preserves time order
- the parameter set can be hashed deterministically for run tracking

The project should therefore be read as a reproducible research pipeline, not as a collection of ad hoc scripts.

## Repository Layout

- `cdf.py`: full reference implementation
- `docs/`: supplemental documentation
- `results/`: generated dashboards and artifacts
- `requirements.txt`: runtime dependencies

## Running The Pipeline

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Provide OHLC input by placing `backtest_prices.csv` next to `cdf.py`, or by updating `DATA_FILE_PATH` at the top of the module.

Run:

```bash
python3 cdf.py
```

Notes:

- Plotly is required for dashboard generation.
- If `REJECT_STALE_BACKTEST_DATA` is set to `False`, CDF can load the local CSV directly.
- If stale-data rejection is enabled, the optional companion module `backtest_loader.py` must be available.

## Limitations

CDF is intentionally serious, but it is still a research repository. Its current limitations should be stated plainly:

- it is a single-file research implementation, not yet a packaged library
- simulated returns do not model transaction costs, slippage, or execution latency
- the feature set is OHLC-based and does not ingest order flow or market depth
- the Hurst calculation is a practical rolling proxy, not a full long-memory research treatment
- the Sutte-derived features are implementation choices inside this repository, not the sole foundation of the method
- dashboard output is diagnostic, not a substitute for formal experiment tracking

## Attribution

This repository was created by **James Sawyer**.

If you reference the project in research notes, internal reports, or derivative work, attribute it as:

```text
Sawyer, James. CDF: Consolidation Detection Framework. Research software, 2026.
```

GitHub-compatible citation metadata is available in [`CITATION.cff`](CITATION.cff).

## References

1. Hurst, H. E. *Long-Term Storage Capacity of Reservoirs*. American Society of Civil Engineers, 1950. https://doi.org/10.1061/TACEAT.0006518
2. Bergstra, J. S., Bardenet, R., Bengio, Y., and Kegl, B. *Algorithms for Hyper-Parameter Optimization*. Advances in Neural Information Processing Systems 24, 2011. https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization
3. Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. *Optuna: A Next-generation Hyperparameter Optimization Framework*. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019. https://doi.org/10.1145/3292500.3330701
4. Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed., Section 5.10, Time Series Cross-Validation. OTexts. https://otexts.com/fpp3/tscv.html
5. Lai, G., Chang, W. C., Yang, Y., and Liu, H. *Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks*. arXiv:1703.07015, 2018. https://doi.org/10.48550/arXiv.1703.07015
6. Wilder, J. W. *New Concepts in Technical Trading Systems*. Trend Research, 1978. https://books.google.com/books/about/New_Concepts_in_Technical_Trading_System.html?id=WesJAQAAMAAJ
