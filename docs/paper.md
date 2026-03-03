# CDF Research Note

**Title:** Consolidation Detection Framework  
**Author:** James Sawyer  
**Document type:** Technical research note  
**Repository:** https://github.com/tg12/cdf

## Abstract

This note describes the design of the Consolidation Detection Framework (CDF), a research-oriented system for identifying consolidation regimes in OHLC time series and estimating directional breakout probability after structural compression. The framework was built by James Sawyer for quantitative researchers and systematic traders who want an interpretable market-structure model rather than an opaque end-to-end predictor. CDF treats consolidation as a composite statistical state defined by volatility contraction, bounded price exploration, non-persistent local dynamics, historical-context similarity, temporal phase structure, and directional candle diagnostics. It then applies time-series-aware hyperparameter selection, monotonic probability calibration, and dashboard reporting to produce an auditable research workflow.

## 1. Motivation

Consolidation is routinely referenced in discretionary trading, but it is often defined informally. In practice that creates three problems:

1. Different analysts use different implicit definitions of compression.
2. Breakout logic is often applied before structural validation is complete.
3. Post-hoc interpretation hides which component actually supported the signal.

CDF was designed to address those problems directly. The project asks whether consolidation can be represented as an explicit scoring function over transparent statistical features, then separated from the later task of directional expansion assessment.

The intended reader is:

- a quantitative researcher evaluating whether consolidation is a coherent regime class
- a systematic trader who wants to inspect component scores before acting on a breakout
- a technical reviewer who needs deterministic validation logic and clear failure modes

The intended reader is not a user looking for a minimalist trading script or a single-indicator rule set.

## 2. Design Principles

CDF follows a small set of research principles:

- interpretability before model opacity
- time-order preservation before convenience sampling
- component inspection before aggregate storytelling
- probability calibration before confidence claims
- deterministic configuration before hidden state

These principles shape both the feature set and the evaluation procedure.

## 3. Formal Problem Statement

Let each bar at time `t` be represented by:

```text
X_t = (o_t, h_t, l_t, c_t)
```

where `o_t`, `h_t`, `l_t`, and `c_t` are open, high, low, and close. For a candidate lookback length `L`, define a trailing window:

```text
W_t(L) = {X_(t-L), ..., X_(t-1)}
```

CDF estimates a consolidation score:

```text
S_t in [0, +inf)
```

from a weighted collection of normalized structural components. A window is accepted as a consolidation event when:

```text
S_t > tau
```

for an optimized threshold `tau`.

Conditional on acceptance, CDF evaluates whether subsequent prices move beyond the high-low envelope of `W_t(L)` strongly enough to justify a directional breakout probability estimate.

This turns the overall problem into two linked but distinct tasks:

1. Structural regime detection.
2. Conditional expansion scoring after structural acceptance.

## 4. Feature Families

### 4.1 Price Geometry

CDF first constructs local geometric features from OHLC data:

- high-low range and normalized range
- open-close range and normalized body size
- body-to-range ratio
- wick balance

These features attempt to capture whether candles are compressive, symmetric, and locally bounded rather than expansionary.

### 4.2 Volatility Contraction

Rolling highs and lows define local envelopes over short, medium, and long horizons. Recent realized range is then compared against a broader baseline. A consolidation candidate should exhibit local compression relative to its recent historical context rather than only absolute narrowness. This is important because a narrow range in one regime can be large in another.

### 4.3 Momentum and Positional Stability

The framework computes rate-of-change, momentum, RSI-style signals, and trend-strength ratios derived from price change relative to local bar range [6]. These features are not used as direct trade triggers. They are used to measure whether price is repeatedly exploring the same bounded region or behaving like an incipient directional move.

### 4.4 Persistence Diagnostics

CDF uses a rolling Hurst-style estimate as a practical persistence diagnostic [1]. The implementation fits a log-log relation between lag and differenced standard deviation within a bounded lag set. The resulting value is not treated as a full long-memory research estimator. It is used operationally:

- values at or below 0.5 support non-persistent interpretation
- larger values are increasingly penalized as evidence of directional persistence

This matters because a strongly persistent local process is generally inconsistent with the intuition of consolidation.

### 4.5 Temporal Phase Encoding

Intraday hour, minute-of-day, and day-of-week are encoded as sine and cosine pairs. This lets the model compare windows by cyclical phase rather than by raw timestamp distance. The purpose is to preserve recurring temporal structure without forcing the detector to learn arbitrary calendar offsets.

### 4.6 Sutte-Derived Directional Diagnostics

CDF includes a Sutte-inspired signal derived from close relative to low and high, smoothed with short moving averages. From this it derives:

- directional sign
- directional conviction
- an exponentially weighted adaptive variant

The Sutte path does not define consolidation on its own. It provides directional coherence diagnostics that later modulate breakout confidence.

### 4.7 Local Autoregressive Anchor

CDF fits a small autoregressive local anchor on the trailing close path and measures how well the latest close aligns with that anchor. This is conceptually adjacent to the long-short separation described by LSTNet [5], but here it is implemented with an explicit linear autoregressive fit instead of a deep recurrent-convolutional architecture.

## 5. Consolidation Score Construction

Each candidate window receives a composite score:

```text
S_t =
    w_c * C_t
  + w_r * R_t
  + w_p * P_t
  + w_h * H_t
  + w_a * A_t
  + w_q * Q_t
  + w_s * G_t
  + w_u * U_t
```

where:

- `C_t` is the contraction component
- `R_t` is the range-width component
- `P_t` is the price-position and interaction component
- `H_t` is the non-persistence component
- `A_t` is the attention-style historical context component
- `Q_t` is the periodic-context component
- `G_t` is the scale-anchor component
- `U_t` is the Sutte component

The weights are constrained and normalized during optimization, which forces the score to remain a compositional model rather than a loosely interacting set of arbitrary thresholds.

## 6. Historical Context Matching

The attention-style component is a central part of the framework.

For each candidate window, CDF:

1. Builds a feature vector from selected structural columns.
2. Estimates robust centers and scales from prior history using medians, interquartile range, and median absolute deviation.
3. Robust-scales the reference and candidate vectors.
4. Measures weighted L1 distance in feature space.
5. Converts distance to similarity with `exp(-distance / temperature)`.
6. Aggregates the compression signatures of the top historical matches.

This is not attention in the strict neural-network sense. It is a deliberately interpretable approximation of similarity-weighted contextual recall. A separate periodic path applies the same logic to cyclical time encodings alone.

## 7. Directional Alignment and Breakout Logic

After a consolidation is accepted, CDF checks whether Sutte-derived bias is aligned or divergent relative to recent in-window price direction. The alignment state is represented as:

- `ALIGNED`
- `DIVERGENT`
- `NEUTRAL`

The system then evaluates future prices against the accepted zone's high and low. For each bar within a bounded hold horizon, it computes:

- signed distance beyond the boundary
- current volatility relative to in-zone volatility
- a context bias term derived from accepted structural strength

The expansion statistic is mapped through a logistic transfer function. That probability is then adjusted by:

- current Sutte directional support
- Sutte-price directional alignment measured inside the accepted window

This design enforces an important separation: structural acceptance precedes directional confidence.

## 8. Hyperparameter Search and Validation

The hyperparameter search layer uses Optuna [3] with a Tree-structured Parzen Estimator sampler [2]. The optimized space includes:

- lookback and lookforward windows
- minimum required bar count
- range threshold
- position sensitivity
- consolidation threshold
- breakout expansion threshold
- breakout hold horizon
- logistic steepness and threshold
- normalized score-component weights

Validation uses rolling-origin splits to preserve temporal order [4]. This matters because random shuffling would allow future structure to influence parameter selection indirectly.

Each fold is evaluated with a blended score formed from:

- weighted F1
- weighted precision and recall
- a simplified Sharpe-style statistic computed from directional returns

CDF then subtracts a fold-stability penalty based on downside deviation, lower-quartile underperformance, worst-fold gap, and cross-fold degradation trend. This pushes the optimizer away from parameter sets that only work well on isolated slices.

## 9. Probability Calibration

Raw breakout probabilities are not treated as calibrated beliefs. Instead, CDF fits a monotonic piecewise-linear calibration map from out-of-fold predictions to empirical directional hit rates. The pipeline:

- gathers out-of-fold non-zero signals
- groups them into probability quantiles
- measures empirical hit rates in each bin
- blends raw confidence with empirical outcome rate
- enforces monotonic anchors
- interpolates calibrated probabilities from the anchor curve

This is a practical reliability layer. It exists because ranking performance and probability calibration are not the same problem.

## 10. Diagnostics and Interpretability

CDF reports more than final directional calls. It also exposes:

- component means across accepted consolidations
- bootstrap intervals for mean active probabilities
- signal bias between long and short calls
- simulated return aggregates
- recent-versus-baseline volatility regime shifts
- rolling instability in core engineered features
- optimization parameter influence estimates

This is essential to the purpose of the repository. James Sawyer built CDF to be inspectable. The dashboard is not an afterthought. It is part of the evidence trail.

## 11. Implementation Mapping

The conceptual pipeline maps directly to the current code structure in `cdf.py`:

- `FeatureEngineer`: raw OHLC transformation and feature construction
- `ConsolidationDetector`: window scoring and acceptance
- `BreakoutPredictor`: conditional breakout probability and direction scoring
- `ParameterOptimizer`: TPE-driven search with rolling-origin validation
- `ProbabilityCalibrator`: out-of-fold monotonic confidence adjustment
- `PerformanceAnalyzer`: aggregate metrics, regime diagnostics, and uncertainty summaries
- `ConsolidationDashboard`: HTML reporting and visualization

This direct mapping is intentional. The repository is meant to be read by humans as well as executed by Python.

## 12. Intended Use

CDF is aimed at:

- research on consolidation as a structural market regime
- strategy prototyping where interpretability matters
- hybrid discretionary-systematic workflows that need auditable diagnostics
- internal technical review of market-structure assumptions

CDF is not currently aimed at:

- latency-sensitive execution systems
- order-book microstructure modeling
- cost-aware production portfolio management
- a one-indicator retail trading workflow

## 13. Limitations

Several limitations are deliberate and should be read plainly:

- the implementation is currently a single-file reference system
- execution frictions such as slippage and fees are not modeled
- the feature set is OHLC-centric and excludes depth and order-flow signals
- the Hurst calculation is a practical proxy rather than a full econometric treatment
- the Sutte path is a supporting diagnostic, not an exclusive thesis of the method
- the optimization objective is useful for ranking configurations, not for proving deployable alpha

The framework is therefore best understood as an interpretable research instrument.

## 14. Conclusion

The central claim of CDF is that consolidation can be treated as a structured, inspectable, multi-component regime rather than as a vague visual pattern. James Sawyer created the framework for technically demanding users who want that claim expressed in code, diagnostics, and validation logic rather than in rhetoric. The contribution of the repository is not that it predicts everything. The contribution is that it makes regime detection explicit, testable, and reviewable.

## References

1. Hurst, H. E. *Long-Term Storage Capacity of Reservoirs*. American Society of Civil Engineers, 1950. https://doi.org/10.1061/TACEAT.0006518
2. Bergstra, J. S., Bardenet, R., Bengio, Y., and Kegl, B. *Algorithms for Hyper-Parameter Optimization*. Advances in Neural Information Processing Systems 24, 2011. https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization
3. Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. *Optuna: A Next-generation Hyperparameter Optimization Framework*. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019. https://doi.org/10.1145/3292500.3330701
4. Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed., Section 5.10, Time Series Cross-Validation. OTexts. https://otexts.com/fpp3/tscv.html
5. Lai, G., Chang, W. C., Yang, Y., and Liu, H. *Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks*. arXiv:1703.07015, 2018. https://doi.org/10.48550/arXiv.1703.07015
6. Wilder, J. W. *New Concepts in Technical Trading Systems*. Trend Research, 1978. https://books.google.com/books/about/New_Concepts_in_Technical_Trading_System.html?id=WesJAQAAMAAJ
