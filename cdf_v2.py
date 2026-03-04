import hashlib
import html
import inspect
import json
import logging
import random
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from backtest_loader import load_backtest_prices
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION - ALL VARIABLES AT TOP OF FILE
# =============================================================================

# Data configuration
DATA_FILE_PATH = None
TIMESTAMP_COLUMN = "snapshotTime"
DATETIME_FORMAT = "%Y:%m:%d-%H:%M:%S"
OPEN_COLUMN = "open"
HIGH_COLUMN = "high"
LOW_COLUMN = "low"
CLOSE_COLUMN = "close"
REJECT_STALE_BACKTEST_DATA = False
MAX_BACKTEST_CSV_AGE_HOURS = 2.0

# Optimization configuration
OPTIMIZATION_TRIALS = 100
VALIDATION_SPLITS = 5
RANDOM_SEED = 42
OPTIMIZATION_DIRECTION = "maximize"
MIN_VALIDATION_FOLDS = 2
ROLLING_CV_TRAIN_WINDOW_MULTIPLIER = 3
PROBABILITY_CALIBRATION_ENABLED = True
PROBABILITY_CALIBRATION_BINS = 7
PROBABILITY_CALIBRATION_MIN_SAMPLES = 25
PROBABILITY_CALIBRATION_BLEND = 0.65

# Baseline benchmark configuration
BASELINE_BENCHMARK_ENABLED = True
BASELINE_FEATURE_COLUMNS = (
    "hl_range_pct",
    "vol_ratio_20",
    "trend_strength_20",
    "price_position_20",
    "body_to_range_ratio",
    "wick_balance",
    "hurst_20",
    "rsi_20",
)
BASELINE_LOGISTIC_C = 1.0
BASELINE_LOGISTIC_MAX_ITER = 1000

# Real-time forecasting configuration
REALTIME_DIRECTION_THRESHOLD = 0.55
REALTIME_SIMILAR_LOOKBACK = 25
REALTIME_BOOTSTRAP_SAMPLES = 250
REALTIME_CI_LOW = 5.0
REALTIME_CI_HIGH = 95.0

# Feature calculation periods
SHORT_PERIODS = [5, 10]
MEDIUM_PERIODS = [20, 50]
LONG_PERIODS = [100]

# Consolidation detection bounds
MIN_LOOKBACK = 5
MAX_LOOKBACK = 50
MIN_LOOKFORWARD = 5
MAX_LOOKFORWARD = 30
MIN_CONSOLIDATION_BARS = 3
MAX_CONSOLIDATION_BARS = 15
MIN_CONSOLIDATION_THRESHOLD = 0.10
MAX_CONSOLIDATION_THRESHOLD = 0.50

# Weight bounds for optimization
MIN_WEIGHT = 0.1
MAX_WEIGHT = 0.5

# Breakout prediction bounds
MIN_HOLD_PERIODS = 5
MAX_HOLD_PERIODS = 30
MIN_LOGISTIC_STEEPNESS = 1.0
MAX_LOGISTIC_STEEPNESS = 5.0
MIN_LOGISTIC_THRESHOLD = 0.3
MAX_LOGISTIC_THRESHOLD = 1.5
HOLD_DURATION_MULTIPLIER = 1.5

# Hurst feature configuration
HURST_DEFAULT = 0.5
HURST_MIN_POINTS = 20
HURST_MIN_LAG = 2
HURST_MAX_LAG = 20
HURST_NON_PERSISTENT_THRESHOLD = 0.50
HURST_PERSISTENCE_PENALTY_MAX = 0.70

# Probability uncertainty configuration
PROBABILITY_CI_BOOTSTRAP_SAMPLES = 500
PROBABILITY_CI_LOW = 5.0
PROBABILITY_CI_MEDIAN = 50.0
PROBABILITY_CI_HIGH = 95.0

# Regime diagnostics
REGIME_RECENT_WINDOW = 20
REGIME_BASELINE_WINDOW = 80
REGIME_SHIFT_HIGH_THRESHOLD = 1.5
REGIME_SHIFT_LOW_THRESHOLD = 0.67
REGIME_EW_HALF_LIFE = 20

# Dashboard configuration
DASHBOARD_ENABLED = True
# Set adaptive dashboard windows to None so each panel uses the full dataset span
# unless a caller explicitly overrides it.
DASHBOARD_DAYS_BACK = None
DASHBOARD_HEIGHT = 1225
DASHBOARD_HEATMAP_DAYS = None
DASHBOARD_FORECAST_BARS = None
DASHBOARD_MAX_ZONES = None
DASHBOARD_MAX_COMPONENT_POINTS = None
DASHBOARD_SIGNAL_PROBABILITY_THRESHOLD = 0.55
DASHBOARD_OUTPUT_DIR = "results"
DASHBOARD_OUTPUT_NAME = "consolidation_dashboard_latest.html"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Numerical stability configuration
PRICE_EPSILON = 1e-10

# Robust context similarity configuration
CONTEXT_WEIGHT_DEFAULT = 1.0
CONTEXT_SCALE_CLIP = 5.0
CONTEXT_IQR_TO_STD = 1.349
CONTEXT_MAD_TO_STD = 1.4826

# Alpha-Sutte configuration
SUTTE_MA_PERIOD = 3
SUTTE_WEIGHT = 0.35
SUTTE_THRESHOLD = 0.02
SUTTE_MIN_HISTORY = 10
SUTTE_SCORE_TYPICAL_MIN = 0.30
SUTTE_SCORE_TYPICAL_MAX = 0.80
SUTTE_DIVERGENCE_WINDOW = 20
SUTTE_DIVERGENCE_SIGNAL_THRESHOLD = 0.10
SUTTE_CONFIDENCE_BOOST = 0.30
SUTTE_CONFIDENCE_PENALTY = 0.50

# LSTNet-inspired detector enhancements
ATTENTION_HISTORY_MULTIPLIER = 8
ATTENTION_MIN_HISTORY_BARS = 20
ATTENTION_TOP_K = 12
ATTENTION_TEMPERATURE = 0.35
PERIODIC_TOP_K = 8
PERIODIC_TEMPERATURE = 0.20
AUTOREGRESSIVE_MAX_LAGS = 3
AUTOREGRESSIVE_MIN_SAMPLES = 10
AUTOREGRESSIVE_TOLERANCE = 0.75
BREAKOUT_CONTEXT_BONUS = 0.35
AUTO_CALIBRATE_THRESHOLD_ON_EMPTY = True
AUTO_CALIBRATION_PERCENTILE = 70.0
THRESHOLD_DIAGNOSTIC_SAMPLE_SIZE = 500
THRESHOLD_DIAGNOSTIC_LEVELS = (0.10, 0.20, 0.30, 0.40, 0.50)
ATTENTION_SCORE_TYPICAL_MIN = 0.30
ATTENTION_SCORE_TYPICAL_MAX = 0.80
PERIODIC_SCORE_TYPICAL_MIN = 0.30
PERIODIC_SCORE_TYPICAL_MAX = 0.80
SCALE_SCORE_TYPICAL_MIN = 0.30
SCALE_SCORE_TYPICAL_MAX = 0.80
ATTENTION_CONTEXT_WEIGHTS = {
    "hl_range_pct": 1.10,
    "oc_range_pct": 0.90,
    "vol_ratio_20": 1.30,
    "trend_strength_5": 0.85,
    "trend_strength_20": 1.10,
    "hurst_20": 1.25,
    "rsi_5": 0.70,
    "rsi_20": 0.85,
    "body_to_range_ratio": 1.10,
    "wick_balance": 1.00,
    "sutte_signal": 1.15,
    "sutte_conviction": 1.05,
    "minute_sin": 0.50,
    "minute_cos": 0.50,
    "dow_sin": 0.35,
    "dow_cos": 0.35,
}
PERIODIC_CONTEXT_WEIGHTS = {
    "minute_sin": 1.00,
    "minute_cos": 1.00,
    "dow_sin": 0.80,
    "dow_cos": 0.80,
}

# Context vector definitions
ATTENTION_CONTEXT_COLUMNS = (
    "hl_range_pct",
    "oc_range_pct",
    "vol_ratio_20",
    "trend_strength_5",
    "trend_strength_20",
    "hurst_20",
    "rsi_5",
    "rsi_20",
    "body_to_range_ratio",
    "wick_balance",
    "sutte_signal",
    "sutte_conviction",
    "minute_sin",
    "minute_cos",
    "dow_sin",
    "dow_cos",
)
PERIODIC_CONTEXT_COLUMNS = (
    "minute_sin",
    "minute_cos",
    "dow_sin",
    "dow_cos",
)

# =============================================================================
# LOGGING SETUP
# =============================================================================


def configure_logging(name: str = __name__, level: str = LOG_LEVEL) -> logging.Logger:
    """Configure module-level logging."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    module_logger = logging.getLogger(name)

    if not module_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        module_logger.addHandler(handler)

    module_logger.setLevel(level_map.get(level.upper(), logging.INFO))
    module_logger.propagate = False
    logging.getLogger("optuna").setLevel(logging.WARNING)
    return module_logger


logger = configure_logging()


def setup_random_seed(seed: int = RANDOM_SEED) -> None:
    """Seed deterministic generators used by optimization and diagnostics."""
    random.seed(seed)
    np.random.seed(seed)
    logger.debug("Seeded random generators with seed=%d", seed)


def _build_rolling_origin_splits(
    df_length: int,
    n_splits: int,
    train_window_multiplier: int = ROLLING_CV_TRAIN_WINDOW_MULTIPLIER,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build contiguous rolling-origin splits with a fixed training window."""
    safe_length = max(0, int(df_length))
    safe_splits = max(1, int(n_splits))
    safe_multiplier = max(1, int(train_window_multiplier))
    total_buckets = safe_splits + safe_multiplier

    if safe_length <= total_buckets:
        raise ValueError(
            f"Insufficient rows for rolling CV: rows={safe_length}, required>{total_buckets}."
        )

    validation_window = max(1, safe_length // total_buckets)
    train_window = safe_length - validation_window * safe_splits
    if train_window < validation_window:
        raise ValueError(
            "Rolling CV could not allocate a fixed training window: "
            f"rows={safe_length}, splits={safe_splits}, "
            f"train_window={train_window}, validation_window={validation_window}."
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for split_idx in range(safe_splits):
        train_start = split_idx * validation_window
        train_end = train_start + train_window
        val_start = train_end
        val_end = val_start + validation_window
        if split_idx == safe_splits - 1:
            val_end = safe_length

        if train_end > safe_length or val_start >= safe_length:
            break

        train_idx = np.arange(train_start, train_end, dtype=np.int64)
        val_idx = np.arange(val_start, min(val_end, safe_length), dtype=np.int64)
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        splits.append((train_idx, val_idx))

    if len(splits) < safe_splits:
        raise ValueError(
            f"Rolling CV produced too few splits: requested={safe_splits}, produced={len(splits)}."
        )

    return splits


def _build_future_direction_targets(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Return future breakout direction labels for the given lookahead horizon."""
    future_returns = df[CLOSE_COLUMN].pct_change(horizon).shift(-horizon)
    return pd.Series(np.sign(future_returns), index=df.index, dtype=np.float64)


def _predict_validation_fold_with_history(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    params: "OptimizationParams",
) -> tuple[pd.Series, pd.Series]:
    """Predict a validation fold while preserving the preceding train history."""
    fold_start = int(train_idx[0])
    fold_end = int(val_idx[-1]) + 1
    fold_df = df.iloc[fold_start:fold_end].copy()
    evaluation_index = df.index[val_idx]

    detector = ConsolidationDetector(params)
    predictor = BreakoutPredictor(params)
    fold_cons = detector.detect(fold_df)
    fold_probs, fold_dirs = predictor.predict(fold_df, fold_cons)

    val_probs = fold_probs.reindex(evaluation_index).fillna(0.0)
    val_dirs = fold_dirs.reindex(evaluation_index).fillna(0).astype(int)
    return val_probs, val_dirs


def _map_directional_probabilities_to_up_probability(
    probabilities: pd.Series,
    directions: pd.Series,
) -> pd.Series:
    """Convert directional breakout probabilities into a binary up-probability."""
    up_probability = pd.Series(0.5, index=probabilities.index, dtype=np.float64)
    clipped = probabilities.astype(np.float64).clip(0.0, 1.0)
    up_mask = directions > 0
    down_mask = directions < 0
    up_probability.loc[up_mask] = clipped.loc[up_mask]
    up_probability.loc[down_mask] = 1.0 - clipped.loc[down_mask]
    return up_probability.clip(0.0, 1.0)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ConsolidationFeatures:
    """Container for consolidation detection features."""

    start_idx: int
    end_idx: int
    duration: int
    range_width: float
    range_width_pct: float
    volatility_ratio: float
    mean_reversion_strength: float
    hurst_exponent: float
    consolidation_score: float = 0.0
    attention_context_score_raw: float = 0.5
    periodic_context_score_raw: float = 0.5
    scale_anchor_score_raw: float = 0.5
    attention_context_score: float = 0.5
    periodic_context_score: float = 0.5
    scale_anchor_score: float = 0.5
    sutte_score_raw: float = 0.5
    sutte_score: float = 0.5
    sutte_divergence: float = 0.0
    sutte_divergence_confidence: float = 0.0
    sutte_divergence_signal: str = "NEUTRAL"
    interaction_score: float = 0.0
    timestamp_start: datetime | None = None
    timestamp_end: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "duration": self.duration,
            "range_width": self.range_width,
            "range_width_pct": self.range_width_pct,
            "volatility_ratio": self.volatility_ratio,
            "hurst_exponent": self.hurst_exponent,
            "consolidation_score": self.consolidation_score,
            "attention_context_score_raw": self.attention_context_score_raw,
            "periodic_context_score_raw": self.periodic_context_score_raw,
            "scale_anchor_score_raw": self.scale_anchor_score_raw,
            "attention_context_score": self.attention_context_score,
            "periodic_context_score": self.periodic_context_score,
            "scale_anchor_score": self.scale_anchor_score,
            "sutte_score_raw": self.sutte_score_raw,
            "sutte_score": self.sutte_score,
            "sutte_divergence": self.sutte_divergence,
            "sutte_divergence_confidence": self.sutte_divergence_confidence,
            "sutte_divergence_signal": self.sutte_divergence_signal,
            "interaction_score": self.interaction_score,
        }


@dataclass
class OptimizationParams:
    """Container for optimized parameters."""

    lookback_period: int
    lookforward_period: int
    min_bars: int
    range_threshold: float
    position_sensitivity: float
    consolidation_threshold: float
    min_expansion: float
    weight_contraction: float
    weight_range: float
    weight_position: float
    weight_hurst: float
    max_hold: int
    logistic_steepness: float
    logistic_threshold: float
    weight_periodic: float = 0.0
    weight_attention: float = 0.0
    weight_scale: float = 0.0
    weight_sutte: float = SUTTE_WEIGHT

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}

    def get_hash(self) -> str:
        """Generate deterministic hash of parameters."""
        param_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(param_string.encode()).hexdigest()[:8]


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


class FeatureEngineer:
    """Calculate technical features from OHLC data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer with OHLC dataframe.

        Args:
            df: DataFrame with OHLC columns
        """
        self.df = df.copy()
        self.logger = logging.getLogger(__name__)

    def calculate_all_features(self) -> pd.DataFrame:
        """Calculate complete feature set from OHLC data."""
        self.logger.info("Calculating features from OHLC data")

        self._calculate_price_features()
        self._calculate_volatility_features()
        self._calculate_momentum_features()
        self._calculate_statistical_features()
        self._calculate_sutte_features()

        # Keep rows usable by downstream detector/predictor while tolerating
        # partial NaNs in non-critical engineered features.
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(axis=1, how="all", inplace=True)
        required = [OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, CLOSE_COLUMN, "hl_range_pct"]
        self.df = self.df.dropna(subset=required)

        if self.df.empty:
            raise ValueError(
                "Feature engineering produced an empty dataset. "
                "Increase input history or reduce feature windows."
            )

        self.logger.debug(f"Feature calculation complete. Shape: {self.df.shape}")
        return self.df

    def _calculate_price_features(self) -> None:
        """Calculate basic price-derived features."""
        self.df["typical_price"] = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        self.df["hl_range"] = self.df["high"] - self.df["low"]
        self.df["hl_range_pct"] = (self.df["hl_range"] / self.df["close"]) * 100
        self.df["oc_range"] = abs(self.df["open"] - self.df["close"])
        self.df["oc_range_pct"] = (self.df["oc_range"] / self.df["close"]) * 100
        self.df["volume_proxy"] = self.df["hl_range"] * self.df["hl_range_pct"]
        self.df["body_size"] = self.df["oc_range"]
        self.df["upper_wick"] = self.df["high"] - self.df[["open", "close"]].max(axis=1)
        self.df["lower_wick"] = self.df[["open", "close"]].min(axis=1) - self.df["low"]
        self.df["body_to_range_ratio"] = self.df["body_size"] / (
            self.df["hl_range"] + PRICE_EPSILON
        )
        self.df["wick_balance"] = 1.0 - (
            abs(self.df["upper_wick"] - self.df["lower_wick"])
            / (self.df["hl_range"] + PRICE_EPSILON)
        )
        self.df["wick_balance"] = self.df["wick_balance"].clip(0.0, 1.0)
        self._calculate_temporal_context_features()

    def _calculate_temporal_context_features(self) -> None:
        """Add intraday and weekly context features for regime diagnostics."""
        if not isinstance(self.df.index, pd.DatetimeIndex):
            return

        self.df["hour_utc"] = self.df.index.hour.astype(np.float64)
        self.df["day_of_week"] = self.df.index.dayofweek.astype(np.float64)
        self.df["minute_of_day"] = (self.df.index.hour * 60 + self.df.index.minute).astype(
            np.float64
        )

        hour_radians = 2.0 * np.pi * self.df["hour_utc"] / 24.0
        self.df["hour_sin"] = np.sin(hour_radians)
        self.df["hour_cos"] = np.cos(hour_radians)

        minute_radians = 2.0 * np.pi * self.df["minute_of_day"] / (24.0 * 60.0)
        self.df["minute_sin"] = np.sin(minute_radians)
        self.df["minute_cos"] = np.cos(minute_radians)

        day_radians = 2.0 * np.pi * self.df["day_of_week"] / 7.0
        self.df["dow_sin"] = np.sin(day_radians)
        self.df["dow_cos"] = np.cos(day_radians)

    def _calculate_volatility_features(self) -> None:
        """Calculate volatility metrics across multiple periods."""
        all_periods = SHORT_PERIODS + MEDIUM_PERIODS + LONG_PERIODS
        self.df["log_return"] = np.log(self.df["close"] / self.df["close"].shift(1))

        for period in all_periods:
            self.df[f"high_{period}"] = self.df["high"].rolling(period).max()
            self.df[f"low_{period}"] = self.df["low"].rolling(period).min()
            self.df[f"range_width_{period}"] = self.df[f"high_{period}"] - self.df[f"low_{period}"]
            self.df[f"range_width_pct_{period}"] = (
                self.df[f"range_width_{period}"] / self.df["close"] * 100
            )

            self.df[f"vol_ratio_{period}"] = (
                self.df["hl_range_pct"].rolling(period, min_periods=period).mean()
                / self.df["hl_range_pct"].rolling(period * 3, min_periods=period).mean()
            )

            self.df[f"price_position_{period}"] = (self.df["close"] - self.df[f"low_{period}"]) / (
                self.df[f"range_width_{period}"] + PRICE_EPSILON
            )

            self.df[f"realized_vol_{period}"] = self.df["log_return"].rolling(
                period
            ).std() * np.sqrt(252)

    def _calculate_momentum_features(self) -> None:
        """Calculate momentum and trend features."""
        for period in SHORT_PERIODS + MEDIUM_PERIODS:
            self.df[f"roc_{period}"] = self.df["close"].pct_change(period) * 100
            self.df[f"rsi_{period}"] = self._calculate_rsi(period)
            self.df[f"momentum_{period}"] = self.df["close"] - self.df["close"].shift(period)
            self.df[f"trend_strength_{period}"] = abs(
                self.df[f"roc_{period}"].rolling(period).mean()
            ) / (self.df["hl_range_pct"].rolling(period).mean() + PRICE_EPSILON)

    def _calculate_rsi(self, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = self.df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_statistical_features(self) -> None:
        """Calculate advanced statistical features."""
        for period in MEDIUM_PERIODS:
            rolling_mean = self.df["close"].rolling(period).mean()
            rolling_std = self.df["close"].rolling(period).std()
            self.df[f"zscore_{period}"] = (self.df["close"] - rolling_mean) / rolling_std
            self.df[f"hurst_{period}"] = self._calculate_hurst(period)

    def _calculate_sutte_features(self) -> None:
        """Calculate Alpha-Sutte candle structure features."""
        self.df["sutte_low_ratio"] = self.df["close"] / (self.df["low"] + PRICE_EPSILON)
        self.df["sutte_high_ratio"] = self.df["close"] / (self.df["high"] + PRICE_EPSILON)

        self.df["sutte_low_ma"] = (
            self.df["sutte_low_ratio"]
            .rolling(
                window=SUTTE_MA_PERIOD,
                min_periods=2,
            )
            .mean()
        )
        self.df["sutte_high_ma"] = (
            self.df["sutte_high_ratio"]
            .rolling(
                window=SUTTE_MA_PERIOD,
                min_periods=2,
            )
            .mean()
        )

        self.df["sutte_signal"] = (self.df["sutte_low_ma"] - self.df["sutte_high_ma"]) / (
            self.df["sutte_low_ma"] + self.df["sutte_high_ma"] + PRICE_EPSILON
        )
        self.df["sutte_conviction"] = self.df["sutte_signal"].abs().clip(0.0, 1.0)
        self.df["sutte_direction"] = np.where(
            self.df["sutte_signal"] > SUTTE_THRESHOLD,
            1,
            np.where(self.df["sutte_signal"] < -SUTTE_THRESHOLD, -1, 0),
        )

        # Exponential smoothing keeps a faster-reacting variant available for live diagnostics.
        self.df["sutte_adaptive_low"] = (
            self.df["sutte_low_ratio"]
            .ewm(
                span=SUTTE_MA_PERIOD,
                adjust=False,
            )
            .mean()
        )
        self.df["sutte_adaptive_high"] = (
            self.df["sutte_high_ratio"]
            .ewm(
                span=SUTTE_MA_PERIOD,
                adjust=False,
            )
            .mean()
        )
        self.df["sutte_adaptive_signal"] = (
            self.df["sutte_adaptive_low"] - self.df["sutte_adaptive_high"]
        ) / (self.df["sutte_adaptive_low"] + self.df["sutte_adaptive_high"] + PRICE_EPSILON)

    def _calculate_hurst(self, period: int) -> pd.Series:
        """Calculate rolling Hurst exponent with bounded lag set."""
        prices = self.df["close"].to_numpy(dtype=np.float64, copy=False)
        hurst = np.full(len(prices), HURST_DEFAULT, dtype=np.float64)

        for i in range(period, len(prices)):
            window = prices[i - period : i]
            if len(window) < HURST_MIN_POINTS:
                continue

            max_lag = min(HURST_MAX_LAG, len(window) // 2)
            if max_lag <= HURST_MIN_LAG:
                continue

            lags: list[int] = []
            tau_vals: list[float] = []
            for lag in range(HURST_MIN_LAG, max_lag + 1):
                diffs = window[lag:] - window[:-lag]
                tau = np.float64(np.std(diffs))
                if np.isfinite(tau) and tau > 0.0:
                    lags.append(lag)
                    tau_vals.append(tau)

            if len(tau_vals) < 2:
                continue

            try:
                slope, _ = np.polyfit(np.log(lags), np.log(tau_vals), 1)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue

            if np.isfinite(slope):
                hurst[i] = np.float64(slope)

        return pd.Series(hurst, index=self.df.index, dtype=np.float64)


# =============================================================================
# CONSOLIDATION DETECTOR
# =============================================================================


class ConsolidationDetector:
    """Detect consolidation zones using statistical methods."""

    def __init__(self, params: OptimizationParams):
        """
        Initialize detector with optimized parameters.

        Args:
            params: Optimized parameters for detection
        """
        self.params = params
        self.logger = logging.getLogger(__name__)

    def detect(self, df: pd.DataFrame) -> list[ConsolidationFeatures]:
        """
        Identify consolidation zones in the data.

        Args:
            df: DataFrame with calculated features

        Returns:
            List of consolidation features
        """
        self.logger.info(f"Detecting consolidations with parameters: {self.params.get_hash()}")
        consolidations = []

        for i in range(len(df)):
            cons = self._evaluate_window(df, i)
            if cons is not None:
                consolidations.append(cons)

        self.logger.info(f"Found {len(consolidations)} consolidation zones")
        return consolidations

    def _normalize_component_score(
        self,
        score: float,
        typical_min: float,
        typical_max: float,
    ) -> float:
        """Rescale a bounded component score to a more useful 0-1 range."""
        if typical_max <= typical_min:
            return np.float64(min(max(score, 0.0), 1.0))
        if score <= typical_min:
            return 0.0
        if score >= typical_max:
            return 1.0
        return np.float64((score - typical_min) / (typical_max - typical_min))

    def _get_history_bounds(self, start_idx: int, lookback: int) -> tuple[int, int]:
        """Return the non-overlapping history span used for context matching."""
        history_end = max(0, start_idx)
        history_span = max(ATTENTION_MIN_HISTORY_BARS, lookback * ATTENTION_HISTORY_MULTIPLIER)
        history_start = max(0, history_end - history_span)
        return history_start, history_end

    def _build_context_vector(self, row: pd.Series, columns: tuple[str, ...]) -> np.ndarray:
        """Return a stable numeric feature vector for similarity matching."""
        values: list[float] = []
        for col in columns:
            raw_value = row[col] if col in row.index else 0.0
            value = np.float64(raw_value) if pd.notna(raw_value) else 0.0
            if not np.isfinite(value):
                value = 0.0
            values.append(value)
        return np.asarray(values, dtype=np.float64)

    def _resolve_context_weights(self, columns: tuple[str, ...]) -> np.ndarray:
        """Return per-feature weights for a given context family."""
        if columns == ATTENTION_CONTEXT_COLUMNS:
            weight_map = ATTENTION_CONTEXT_WEIGHTS
        elif columns == PERIODIC_CONTEXT_COLUMNS:
            weight_map = PERIODIC_CONTEXT_WEIGHTS
        else:
            weight_map = {}

        return np.asarray(
            [np.float64(weight_map.get(col, CONTEXT_WEIGHT_DEFAULT)) for col in columns],
            dtype=np.float64,
        )

    def _build_context_statistics(
        self,
        history_frame: pd.DataFrame,
        columns: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate robust per-feature centers and scales from the matching history."""
        centers: list[float] = []
        scales: list[float] = []

        for col in columns:
            if col not in history_frame.columns:
                centers.append(0.0)
                scales.append(1.0)
                continue

            series = pd.to_numeric(history_frame[col], errors="coerce")
            values = (
                series.replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=np.float64, copy=False)
            )
            if values.size == 0:
                centers.append(0.0)
                scales.append(1.0)
                continue

            median_value = np.float64(np.median(values))
            q1_value, q3_value = np.percentile(values, [25.0, 75.0])
            iqr_scale = np.float64((q3_value - q1_value) / CONTEXT_IQR_TO_STD)
            mad_scale = np.float64(CONTEXT_MAD_TO_STD * np.median(np.abs(values - median_value)))
            robust_scale = max(iqr_scale, mad_scale)
            if not np.isfinite(robust_scale) or robust_scale <= PRICE_EPSILON:
                std_scale = np.float64(np.std(values))
                robust_scale = std_scale if std_scale > PRICE_EPSILON else 1.0

            centers.append(median_value)
            scales.append(robust_scale)

        return np.asarray(centers, dtype=np.float64), np.asarray(scales, dtype=np.float64)

    def _scale_context_vector(
        self,
        vector: np.ndarray,
        centers: np.ndarray,
        scales: np.ndarray,
    ) -> np.ndarray:
        """Robust-scale a context vector to reduce dominance from high-variance features."""
        safe_scales = np.where(scales > PRICE_EPSILON, scales, 1.0)
        scaled = (vector - centers) / safe_scales
        return np.clip(scaled, -CONTEXT_SCALE_CLIP, CONTEXT_SCALE_CLIP)

    def _normalized_distance(
        self,
        left: np.ndarray,
        right: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Measure weighted L1 distance between already robust-scaled context vectors."""
        total_weight = np.float64(np.sum(weights))
        if total_weight <= PRICE_EPSILON:
            return 0.0
        return np.float64(np.sum(weights * np.abs(left - right)) / total_weight)

    def _hurst_non_persistence_score(self, hurst_value: float) -> float:
        """Reward non-persistent regimes while penalizing persistence above 0.5."""
        value = np.float64(hurst_value)
        if not np.isfinite(value):
            return 0.5
        if value <= HURST_NON_PERSISTENT_THRESHOLD:
            return 1.0
        if value >= HURST_PERSISTENCE_PENALTY_MAX:
            return 0.0

        penalty_progress = (value - HURST_NON_PERSISTENT_THRESHOLD) / (
            HURST_PERSISTENCE_PENALTY_MAX - HURST_NON_PERSISTENT_THRESHOLD
        )
        return np.float64(1.0 - penalty_progress)

    def _row_compression_signature(self, row: pd.Series) -> float:
        """Approximate how consolidation-like a row looks using local features."""
        components: list[float] = []

        vol_ratio = row.get("vol_ratio_20")
        if pd.notna(vol_ratio):
            components.append(1.0 - min(max(np.float64(vol_ratio), 0.0), 2.0) / 2.0)

        body_to_range = row.get("body_to_range_ratio")
        if pd.notna(body_to_range):
            components.append(1.0 - min(max(np.float64(body_to_range), 0.0), 1.0))

        wick_balance = row.get("wick_balance")
        if pd.notna(wick_balance):
            components.append(min(max(np.float64(wick_balance), 0.0), 1.0))

        trend_strength = row.get("trend_strength_20")
        if pd.notna(trend_strength):
            components.append(1.0 - min(abs(np.float64(trend_strength)) / 2.0, 1.0))

        hurst_value = row.get("hurst_20")
        if pd.notna(hurst_value):
            components.append(self._hurst_non_persistence_score(np.float64(hurst_value)))

        if not components:
            return 0.5
        return np.float64(np.mean(components))

    def _similarity_weighted_history_score(
        self,
        df: pd.DataFrame,
        reference_idx: int,
        start_idx: int,
        lookback: int,
        columns: tuple[str, ...],
        top_k: int,
        temperature: float,
    ) -> float:
        """Aggregate compression signatures from similar historical contexts."""
        history_start, history_end = self._get_history_bounds(start_idx, lookback)
        if history_end - history_start < self.params.min_bars:
            return 0.5

        history_frame = df.iloc[history_start:history_end]
        centers, scales = self._build_context_statistics(history_frame, columns)
        weights = self._resolve_context_weights(columns)
        reference_vector = self._scale_context_vector(
            self._build_context_vector(df.iloc[reference_idx], columns),
            centers,
            scales,
        )
        weighted_signatures: list[tuple[float, float]] = []

        for candidate_idx in range(history_start, history_end):
            candidate_row = df.iloc[candidate_idx]
            candidate_vector = self._scale_context_vector(
                self._build_context_vector(candidate_row, columns),
                centers,
                scales,
            )
            distance = self._normalized_distance(reference_vector, candidate_vector, weights)
            weight = np.float64(np.exp(-distance / max(temperature, PRICE_EPSILON)))
            signature = self._row_compression_signature(candidate_row)
            weighted_signatures.append((weight, signature))

        if not weighted_signatures:
            return 0.5

        weighted_signatures.sort(key=lambda item: item[0], reverse=True)
        top_matches = weighted_signatures[:top_k]
        total_weight = sum(weight for weight, _ in top_matches)
        if total_weight <= PRICE_EPSILON:
            return 0.5

        weighted_sum = sum(weight * signature for weight, signature in top_matches)
        return np.float64(weighted_sum / total_weight)

    def _calculate_attention_context_score(
        self,
        df: pd.DataFrame,
        reference_idx: int,
        start_idx: int,
        lookback: int,
    ) -> float:
        """Approximate context matching using similarity over engineered features."""
        return self._similarity_weighted_history_score(
            df=df,
            reference_idx=reference_idx,
            start_idx=start_idx,
            lookback=lookback,
            columns=ATTENTION_CONTEXT_COLUMNS,
            top_k=ATTENTION_TOP_K,
            temperature=ATTENTION_TEMPERATURE,
        )

    def _calculate_periodic_context_score(
        self,
        df: pd.DataFrame,
        reference_idx: int,
        start_idx: int,
        lookback: int,
    ) -> float:
        """Approximate periodic skip-memory using time-phase similarity."""
        if "minute_sin" not in df.columns or "dow_sin" not in df.columns:
            return 0.5

        return self._similarity_weighted_history_score(
            df=df,
            reference_idx=reference_idx,
            start_idx=start_idx,
            lookback=lookback,
            columns=PERIODIC_CONTEXT_COLUMNS,
            top_k=PERIODIC_TOP_K,
            temperature=PERIODIC_TEMPERATURE,
        )

    def _calculate_price_positions(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
    ) -> list[float]:
        """Track how tightly price stays within the growing local range."""
        positions: list[float] = []
        for row_idx in range(start_idx, end_idx + 1):
            pos_window = df.iloc[start_idx : row_idx + 1]
            pos_range = pos_window["high"].max() - pos_window["low"].min()
            if pos_range <= 0:
                continue
            pos = (df["close"].iloc[row_idx] - pos_window["low"].min()) / pos_range
            positions.append(np.float64(pos))
        return positions

    def _calculate_interaction_score(self, window: pd.DataFrame) -> float:
        """Measure candle compression from joint OHLC interactions."""
        if "body_to_range_ratio" not in window.columns or "wick_balance" not in window.columns:
            return 0.5

        body_compression = 1.0 - min(
            max(np.float64(window["body_to_range_ratio"].mean()), 0.0), 1.0
        )
        wick_balance = min(max(np.float64(window["wick_balance"].mean()), 0.0), 1.0)
        return np.float64((body_compression + wick_balance) / 2.0)

    def _calculate_sutte_score(self, window: pd.DataFrame) -> float:
        """Score Sutte signal strength, stability, and recent acceleration."""
        if (
            len(window) < SUTTE_MIN_HISTORY
            or "sutte_conviction" not in window.columns
            or window["sutte_conviction"].dropna().empty
        ):
            return 0.5

        conviction = window["sutte_conviction"].dropna()
        mean_conviction = np.float64(conviction.mean())
        consistency = 0.5

        if "sutte_direction" in window.columns:
            directions = window["sutte_direction"].dropna()
            if len(directions) > 1:
                direction_std = np.float64(directions.std())
                consistency = 1.0 - min(direction_std, 1.0)
            elif len(directions) == 1:
                consistency = 1.0 if int(directions.iloc[0]) != 0 else 0.5

        if len(conviction) >= 6:
            recent_conviction = np.float64(conviction.iloc[-3:].mean())
            earlier_conviction = np.float64(conviction.iloc[:3].mean())
            conviction_trend = (recent_conviction - earlier_conviction) / (
                earlier_conviction + PRICE_EPSILON
            )
            trend_score = 0.5 + np.clip(conviction_trend, -0.5, 0.5)
        else:
            trend_score = 0.5

        score = 0.4 * mean_conviction + 0.3 * consistency + 0.3 * trend_score
        return np.float64(min(max(score, 0.0), 1.0))

    def detect_sutte_stacking_divergence(
        self,
        df: pd.DataFrame,
        consolidation_score: float,
        window_size: int = SUTTE_DIVERGENCE_WINDOW,
    ) -> dict[str, Any]:
        """Detect agreement or divergence between Sutte bias and recent price direction."""
        if (
            "sutte_signal" not in df.columns
            or "close" not in df.columns
            or len(df) < max(2, SUTTE_MIN_HISTORY)
        ):
            return {
                "divergence": 0.0,
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "sutte_value": 0.0,
                "price_trend": 0.0,
            }

        effective_window = min(max(2, int(window_size)), len(df))
        recent_window = df.iloc[-effective_window:]
        recent_sutte_series = recent_window["sutte_signal"].dropna()
        if recent_sutte_series.empty:
            return {
                "divergence": 0.0,
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "sutte_value": 0.0,
                "price_trend": 0.0,
            }

        recent_sutte = np.float64(recent_sutte_series.mean())
        start_close = np.float64(recent_window["close"].iloc[0])
        end_close = np.float64(recent_window["close"].iloc[-1])
        recent_return = (end_close - start_close) / (start_close + PRICE_EPSILON)
        price_trend = np.float64(np.sign(recent_return))
        sutte_trend = np.float64(np.sign(recent_sutte))
        stacking_score = min(max(np.float64(consolidation_score), 0.0), 1.0)

        if (
            abs(recent_sutte) <= SUTTE_DIVERGENCE_SIGNAL_THRESHOLD
            or price_trend == 0.0
            or sutte_trend == 0.0
        ):
            return {
                "divergence": 0.0,
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "sutte_value": recent_sutte,
                "price_trend": price_trend,
            }

        confidence_multiplier = 1.0 + 0.5 * stacking_score
        if sutte_trend != price_trend:
            divergence = np.float64(-price_trend)
            signal = "FADE"
            confidence = min(abs(recent_sutte) * confidence_multiplier * 2.0, 1.0)
        else:
            divergence = price_trend
            signal = "TRUST"
            confidence = min(abs(recent_sutte) * confidence_multiplier * 1.5, 1.0)

        return {
            "divergence": divergence,
            "signal": signal,
            "confidence": np.float64(confidence),
            "sutte_value": recent_sutte,
            "price_trend": price_trend,
        }

    def _estimate_autoregressive_forecast(self, prices: np.ndarray) -> float:
        """Fit a small linear autoregressive model and forecast one step ahead."""
        if prices.size == 0:
            return 0.0
        if prices.size < AUTOREGRESSIVE_MIN_SAMPLES:
            return np.float64(prices[-1])

        lag_count = min(AUTOREGRESSIVE_MAX_LAGS, prices.size - 1)
        if lag_count <= 0:
            return np.float64(prices[-1])

        rows = prices.size - lag_count
        lagged_columns = [
            prices[lag_count - lag - 1 : lag_count - lag - 1 + rows] for lag in range(lag_count)
        ]
        design_matrix = np.column_stack(lagged_columns + [np.ones(rows, dtype=np.float64)])
        targets = prices[lag_count:]

        try:
            coefficients, _, _, _ = np.linalg.lstsq(design_matrix, targets, rcond=None)
        except np.linalg.LinAlgError:
            return np.float64(prices[-1])

        latest_lags = np.asarray([prices[-lag - 1] for lag in range(lag_count)], dtype=np.float64)
        forecast = np.float64(np.dot(coefficients[:-1], latest_lags) + coefficients[-1])
        if not np.isfinite(forecast):
            return np.float64(prices[-1])
        return forecast

    def _calculate_scale_anchor_score(self, close_window: pd.Series) -> float:
        """Score how well the last close follows a local linear price anchor."""
        prices = close_window.to_numpy(dtype=np.float64, copy=False)
        if prices.size < 2:
            return 0.5

        training_prices = prices[:-1]
        latest_price = np.float64(prices[-1])
        forecast = self._estimate_autoregressive_forecast(training_prices)
        window_range = max(np.float64(np.max(prices) - np.min(prices)), PRICE_EPSILON)
        anchor_error = abs(latest_price - forecast) / window_range
        return np.float64(max(0.0, 1.0 - min(anchor_error / AUTOREGRESSIVE_TOLERANCE, 1.0)))

    def _score_window(self, df: pd.DataFrame, idx: int) -> ConsolidationFeatures | None:
        """Calculate consolidation features and score for a single window."""
        lookback = min(self.params.lookback_period, idx)
        lookforward = min(self.params.lookforward_period, len(df) - idx)

        if lookback < self.params.min_bars:
            self.logger.debug(
                "Reject idx=%d: lookback=%d < min_bars=%d",
                idx,
                lookback,
                self.params.min_bars,
            )
            return None
        if lookforward < self.params.min_bars:
            self.logger.debug(
                "Reject idx=%d: lookforward=%d < min_bars=%d",
                idx,
                lookforward,
                self.params.min_bars,
            )
            return None

        start_idx = idx - lookback
        end_idx = idx - 1
        past_window = df.iloc[start_idx:idx]
        if past_window.empty:
            self.logger.debug("Reject idx=%d: past window is empty", idx)
            return None

        close_mean = np.float64(past_window["close"].mean())
        if not np.isfinite(close_mean) or close_mean <= 0.0:
            self.logger.debug("Reject idx=%d: invalid close mean %.6f", idx, close_mean)
            return None

        past_range = np.float64(past_window["high"].max() - past_window["low"].min())
        past_range_pct = past_range / (close_mean + PRICE_EPSILON) * 100.0

        past_vol = np.float64(past_window["hl_range_pct"].mean())
        historical_vol = df["hl_range_pct"].iloc[max(0, start_idx - lookback * 3) : idx].mean()
        vol_contraction = past_vol / historical_vol if historical_vol > 0 else 1.0

        price_positions = self._calculate_price_positions(df, start_idx, end_idx)
        pos_std = np.float64(np.std(price_positions)) if price_positions else 1.0

        hurst_value = HURST_DEFAULT
        if "hurst_20" in df.columns:
            hurst_value = np.float64(df["hurst_20"].iloc[end_idx])

        interaction_score = self._calculate_interaction_score(past_window)
        position_component = (
            0.5 * (1.0 - min(pos_std * self.params.position_sensitivity, 1.0))
            + 0.5 * interaction_score
        )
        attention_context_score = self._calculate_attention_context_score(
            df=df,
            reference_idx=end_idx,
            start_idx=start_idx,
            lookback=lookback,
        )
        attention_context_score_norm = self._normalize_component_score(
            attention_context_score,
            ATTENTION_SCORE_TYPICAL_MIN,
            ATTENTION_SCORE_TYPICAL_MAX,
        )
        periodic_context_score = self._calculate_periodic_context_score(
            df=df,
            reference_idx=end_idx,
            start_idx=start_idx,
            lookback=lookback,
        )
        periodic_context_score_norm = self._normalize_component_score(
            periodic_context_score,
            PERIODIC_SCORE_TYPICAL_MIN,
            PERIODIC_SCORE_TYPICAL_MAX,
        )
        scale_anchor_score = self._calculate_scale_anchor_score(past_window["close"])
        scale_anchor_score_norm = self._normalize_component_score(
            scale_anchor_score,
            SCALE_SCORE_TYPICAL_MIN,
            SCALE_SCORE_TYPICAL_MAX,
        )
        sutte_score_raw = self._calculate_sutte_score(past_window)
        sutte_score_norm = self._normalize_component_score(
            sutte_score_raw,
            SUTTE_SCORE_TYPICAL_MIN,
            SUTTE_SCORE_TYPICAL_MAX,
        )

        consolidation_score = (
            self.params.weight_contraction * (1.0 - min(vol_contraction, 1.0))
            + self.params.weight_range
            * (
                1.0
                - min(
                    past_range_pct / ((past_vol + PRICE_EPSILON) * self.params.range_threshold),
                    1.0,
                )
            )
            + self.params.weight_position * position_component
            + self.params.weight_hurst * self._hurst_non_persistence_score(hurst_value)
            + self.params.weight_attention * attention_context_score_norm
            + self.params.weight_periodic * periodic_context_score_norm
            + self.params.weight_scale * scale_anchor_score_norm
            + self.params.weight_sutte * sutte_score_norm
        )
        sutte_divergence = self.detect_sutte_stacking_divergence(
            past_window,
            consolidation_score,
        )

        return ConsolidationFeatures(
            start_idx=start_idx,
            end_idx=end_idx,
            duration=lookback,
            range_width=past_range,
            range_width_pct=past_range_pct,
            volatility_ratio=vol_contraction,
            mean_reversion_strength=position_component,
            hurst_exponent=hurst_value,
            consolidation_score=consolidation_score,
            attention_context_score_raw=attention_context_score,
            periodic_context_score_raw=periodic_context_score,
            scale_anchor_score_raw=scale_anchor_score,
            attention_context_score=attention_context_score_norm,
            periodic_context_score=periodic_context_score_norm,
            scale_anchor_score=scale_anchor_score_norm,
            sutte_score_raw=sutte_score_raw,
            sutte_score=sutte_score_norm,
            sutte_divergence=np.float64(sutte_divergence["divergence"]),
            sutte_divergence_confidence=np.float64(sutte_divergence["confidence"]),
            sutte_divergence_signal=str(sutte_divergence["signal"]),
            interaction_score=interaction_score,
            timestamp_start=df.index[start_idx] if hasattr(df.index, "__getitem__") else None,
            timestamp_end=df.index[end_idx] if hasattr(df.index, "__getitem__") else None,
        )

    def score_live_window(
        self,
        df: pd.DataFrame,
        idx: int | None = None,
    ) -> ConsolidationFeatures | None:
        """
        Score the latest available window without requiring future bars.

        This is the inference-only path used for real-time forecasting. It reuses
        the consolidation scoring logic but skips the backtest-only lookforward gate.
        """
        if df.empty:
            return None

        live_idx = len(df) - 1 if idx is None else int(idx)
        if live_idx <= 0 or live_idx >= len(df):
            return None

        lookback = min(self.params.lookback_period, live_idx)
        if lookback < self.params.min_bars:
            return None

        start_idx = live_idx - lookback
        end_idx = live_idx - 1
        past_window = df.iloc[start_idx:live_idx]
        if past_window.empty:
            return None

        close_mean = np.float64(past_window["close"].mean())
        if not np.isfinite(close_mean) or close_mean <= 0.0:
            return None

        past_range = np.float64(past_window["high"].max() - past_window["low"].min())
        past_range_pct = past_range / (close_mean + PRICE_EPSILON) * 100.0

        past_vol = np.float64(past_window["hl_range_pct"].mean())
        historical_vol = df["hl_range_pct"].iloc[max(0, start_idx - lookback * 3) : live_idx].mean()
        vol_contraction = past_vol / historical_vol if historical_vol > 0 else 1.0

        price_positions = self._calculate_price_positions(df, start_idx, end_idx)
        pos_std = np.float64(np.std(price_positions)) if price_positions else 1.0

        hurst_value = HURST_DEFAULT
        if "hurst_20" in df.columns:
            hurst_value = np.float64(df["hurst_20"].iloc[end_idx])

        interaction_score = self._calculate_interaction_score(past_window)
        position_component = (
            0.5 * (1.0 - min(pos_std * self.params.position_sensitivity, 1.0))
            + 0.5 * interaction_score
        )
        attention_context_score = self._calculate_attention_context_score(
            df=df,
            reference_idx=end_idx,
            start_idx=start_idx,
            lookback=lookback,
        )
        attention_context_score_norm = self._normalize_component_score(
            attention_context_score,
            ATTENTION_SCORE_TYPICAL_MIN,
            ATTENTION_SCORE_TYPICAL_MAX,
        )
        periodic_context_score = self._calculate_periodic_context_score(
            df=df,
            reference_idx=end_idx,
            start_idx=start_idx,
            lookback=lookback,
        )
        periodic_context_score_norm = self._normalize_component_score(
            periodic_context_score,
            PERIODIC_SCORE_TYPICAL_MIN,
            PERIODIC_SCORE_TYPICAL_MAX,
        )
        scale_anchor_score = self._calculate_scale_anchor_score(past_window["close"])
        scale_anchor_score_norm = self._normalize_component_score(
            scale_anchor_score,
            SCALE_SCORE_TYPICAL_MIN,
            SCALE_SCORE_TYPICAL_MAX,
        )
        sutte_score_raw = self._calculate_sutte_score(past_window)
        sutte_score_norm = self._normalize_component_score(
            sutte_score_raw,
            SUTTE_SCORE_TYPICAL_MIN,
            SUTTE_SCORE_TYPICAL_MAX,
        )

        consolidation_score = (
            self.params.weight_contraction * (1.0 - min(vol_contraction, 1.0))
            + self.params.weight_range
            * (
                1.0
                - min(
                    past_range_pct / ((past_vol + PRICE_EPSILON) * self.params.range_threshold),
                    1.0,
                )
            )
            + self.params.weight_position * position_component
            + self.params.weight_hurst * self._hurst_non_persistence_score(hurst_value)
            + self.params.weight_attention * attention_context_score_norm
            + self.params.weight_periodic * periodic_context_score_norm
            + self.params.weight_scale * scale_anchor_score_norm
            + self.params.weight_sutte * sutte_score_norm
        )
        sutte_divergence = self.detect_sutte_stacking_divergence(
            past_window,
            consolidation_score,
        )

        return ConsolidationFeatures(
            start_idx=start_idx,
            end_idx=end_idx,
            duration=lookback,
            range_width=past_range,
            range_width_pct=past_range_pct,
            volatility_ratio=vol_contraction,
            mean_reversion_strength=position_component,
            hurst_exponent=hurst_value,
            consolidation_score=consolidation_score,
            attention_context_score_raw=attention_context_score,
            periodic_context_score_raw=periodic_context_score,
            scale_anchor_score_raw=scale_anchor_score,
            attention_context_score=attention_context_score_norm,
            periodic_context_score=periodic_context_score_norm,
            scale_anchor_score=scale_anchor_score_norm,
            sutte_score_raw=sutte_score_raw,
            sutte_score=sutte_score_norm,
            sutte_divergence=np.float64(sutte_divergence["divergence"]),
            sutte_divergence_confidence=np.float64(sutte_divergence["confidence"]),
            sutte_divergence_signal=str(sutte_divergence["signal"]),
            interaction_score=interaction_score,
            timestamp_start=df.index[start_idx] if hasattr(df.index, "__getitem__") else None,
            timestamp_end=df.index[end_idx] if hasattr(df.index, "__getitem__") else None,
        )

    def _build_diagnostic_frame(
        self,
        df: pd.DataFrame,
        sample_size: int = THRESHOLD_DIAGNOSTIC_SAMPLE_SIZE,
    ) -> pd.DataFrame:
        """Collect scored windows for threshold diagnostics."""
        if df.empty:
            return pd.DataFrame()

        safe_sample_size = max(1, int(sample_size))
        step = max(1, len(df) // safe_sample_size)
        records: list[dict[str, float]] = []

        for idx in range(0, len(df), step):
            scored = self._score_window(df, idx)
            if scored is None:
                continue
            records.append(
                {
                    "idx": np.float64(idx),
                    "score": scored.consolidation_score,
                    "attention_raw": scored.attention_context_score_raw,
                    "periodic_raw": scored.periodic_context_score_raw,
                    "scale_raw": scored.scale_anchor_score_raw,
                    "sutte_raw": scored.sutte_score_raw,
                    "attention": scored.attention_context_score,
                    "periodic": scored.periodic_context_score,
                    "scale": scored.scale_anchor_score,
                    "sutte": scored.sutte_score,
                    "interaction": scored.interaction_score,
                    "mean_reversion": scored.mean_reversion_strength,
                    "volatility_ratio": scored.volatility_ratio,
                    "hurst": scored.hurst_exponent,
                }
            )

        return pd.DataFrame.from_records(records)

    def diagnose_thresholds(
        self,
        df: pd.DataFrame,
        sample_size: int = THRESHOLD_DIAGNOSTIC_SAMPLE_SIZE,
    ) -> pd.DataFrame:
        """Log the score distribution so thresholds can be calibrated from data."""
        scores_df = self._build_diagnostic_frame(df=df, sample_size=sample_size)
        if scores_df.empty:
            self.logger.error("No scored windows were generated for threshold diagnostics.")
            return scores_df

        score_summary = scores_df["score"].describe().round(4).to_dict()
        normalized_means = (
            scores_df[["attention", "periodic", "scale", "sutte", "interaction", "mean_reversion"]]
            .mean()
            .round(4)
            .to_dict()
        )
        raw_means = (
            scores_df[["attention_raw", "periodic_raw", "scale_raw", "sutte_raw"]]
            .mean()
            .round(4)
            .to_dict()
        )
        self.logger.info("Consolidation score distribution: %s", score_summary)
        self.logger.info("Normalized component means: %s", normalized_means)
        self.logger.info("Raw context component means: %s", raw_means)

        thresholds = sorted(
            {
                np.float64(self.params.consolidation_threshold),
                *[np.float64(level) for level in THRESHOLD_DIAGNOSTIC_LEVELS],
            }
        )
        total_windows = len(scores_df)
        for threshold in thresholds:
            zone_count = int((scores_df["score"] > threshold).sum())
            zone_ratio = 100.0 * zone_count / total_windows
            self.logger.info(
                "Threshold %.3f yields %d zones (%.1f%% of scored windows)",
                threshold,
                zone_count,
                zone_ratio,
            )

        return scores_df

    def auto_calibrate_threshold(
        self,
        df: pd.DataFrame,
        percentile: float = AUTO_CALIBRATION_PERCENTILE,
        sample_size: int = THRESHOLD_DIAGNOSTIC_SAMPLE_SIZE,
    ) -> float:
        """Lower the threshold toward a data-driven score percentile."""
        scores_df = self._build_diagnostic_frame(df=df, sample_size=sample_size)
        if scores_df.empty:
            return np.float64(self.params.consolidation_threshold)

        bounded_percentile = min(max(np.float64(percentile), 0.0), 100.0)
        calibrated = np.float64(np.percentile(scores_df["score"], bounded_percentile))
        calibrated = min(calibrated, np.float64(self.params.consolidation_threshold))
        calibrated = max(MIN_CONSOLIDATION_THRESHOLD, min(MAX_CONSOLIDATION_THRESHOLD, calibrated))
        return calibrated

    def _evaluate_window(self, df: pd.DataFrame, idx: int) -> ConsolidationFeatures | None:
        """Evaluate if current window is a consolidation zone."""
        scored = self._score_window(df, idx)
        if scored is None:
            return None
        if scored.consolidation_score <= self.params.consolidation_threshold:
            return None
        return scored


# =============================================================================
# BREAKOUT PREDICTOR
# =============================================================================


class BreakoutPredictor:
    """Predict breakout probabilities and directions."""

    def __init__(self, params: OptimizationParams):
        """
        Initialize predictor with optimized parameters.

        Args:
            params: Optimized parameters for prediction
        """
        self.params = params
        self.logger = logging.getLogger(__name__)

    def predict(
        self, df: pd.DataFrame, consolidations: list[ConsolidationFeatures]
    ) -> tuple[pd.Series, pd.Series]:
        """
        Predict breakout probabilities and directions.

        Args:
            df: DataFrame with price data
            consolidations: List of detected consolidations

        Returns:
            Tuple of (probabilities series, directions series)
        """
        self.logger.info(f"Predicting breakouts from {len(consolidations)} consolidations")

        breakout_probs = pd.Series(0.0, index=df.index)
        breakout_dirs = pd.Series(0, index=df.index)

        for cons in consolidations:
            self._evaluate_breakout_potential(df, cons, breakout_probs, breakout_dirs)

        return breakout_probs, breakout_dirs

    def predict_current_breakout(
        self,
        df: pd.DataFrame,
        cons: ConsolidationFeatures,
        idx: int | None = None,
    ) -> tuple[float, int]:
        """Score breakout probability for a single live bar without future lookahead."""
        if df.empty:
            return 0.0, 0

        live_idx = len(df) - 1 if idx is None else int(idx)
        if live_idx <= cons.end_idx or live_idx >= len(df):
            return 0.0, 0

        zone_window = df.iloc[cons.start_idx : cons.end_idx + 1]
        zone_high = np.float64(zone_window["high"].max())
        zone_low = np.float64(zone_window["low"].min())
        zone_range = zone_high - zone_low
        if not np.isfinite(zone_range) or zone_range <= 0.0:
            return 0.0, 0

        current_price = np.float64(df["close"].iloc[live_idx])
        cons_vol = np.float64(df["hl_range_pct"].iloc[cons.start_idx : cons.end_idx + 1].mean())
        if not np.isfinite(cons_vol) or cons_vol <= 0.0:
            cons_vol = PRICE_EPSILON

        current_vol = np.float64(df["hl_range_pct"].iloc[max(0, live_idx - 10) : live_idx].mean())
        if not np.isfinite(current_vol) or current_vol <= 0.0:
            current_vol = cons_vol

        vol_ratio = current_vol / (cons_vol + PRICE_EPSILON)
        context_bias = 1.0 + BREAKOUT_CONTEXT_BONUS * np.mean(
            [
                cons.consolidation_score,
                cons.attention_context_score,
                cons.periodic_context_score,
                cons.scale_anchor_score,
            ]
        )

        direction = 0
        breakout_strength = 0.0
        if current_price > zone_high:
            dist_to_high = (current_price - zone_high) / zone_range
            breakout_strength = (1.0 + dist_to_high) * max(vol_ratio, 0.0) * context_bias
            direction = 1
        elif current_price < zone_low:
            dist_to_low = (zone_low - current_price) / zone_range
            breakout_strength = (1.0 + dist_to_low) * max(vol_ratio, 0.0) * context_bias
            direction = -1
        else:
            return 0.0, 0

        if breakout_strength < self.params.min_expansion:
            return 0.0, 0

        prob = 1 / (
            1
            + np.exp(
                -self.params.logistic_steepness
                * (breakout_strength - self.params.min_expansion - self.params.logistic_threshold)
            )
        )
        prob = np.clip(
            prob
            * self._sutte_confidence(df, live_idx, direction)
            * self._sutte_structure_confidence(cons, direction),
            0.0,
            1.0,
        )
        confirmed_direction = direction if prob > 0.5 else 0
        return np.float64(prob), confirmed_direction

    def _sutte_confidence(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        """Adjust breakout confidence using the current Sutte signal alignment."""
        if (
            idx >= len(df)
            or idx + 1 < SUTTE_MIN_HISTORY
            or "sutte_direction" not in df.columns
            or "sutte_conviction" not in df.columns
        ):
            return 1.0

        sutte_dir = int(df["sutte_direction"].iloc[idx])
        if sutte_dir == 0:
            return 1.0

        conviction = df["sutte_conviction"].iloc[idx]
        if pd.isna(conviction):
            return 1.0

        conviction_value = min(max(np.float64(conviction), 0.0), 1.0)
        if sutte_dir == direction:
            return np.float64(1.0 + conviction_value * SUTTE_CONFIDENCE_BOOST)
        return np.float64(1.0 - conviction_value * SUTTE_CONFIDENCE_PENALTY)

    def _sutte_structure_confidence(
        self,
        cons: ConsolidationFeatures,
        direction: int,
    ) -> float:
        """Adjust breakout confidence using the consolidation window's trust-or-fade bias."""
        if cons.sutte_divergence_signal == "NEUTRAL" or cons.sutte_divergence_confidence <= 0.0:
            return 1.0

        divergence_direction = int(np.sign(cons.sutte_divergence))
        confidence_value = min(max(np.float64(cons.sutte_divergence_confidence), 0.0), 1.0)
        if divergence_direction == direction:
            return np.float64(1.0 + confidence_value * (SUTTE_CONFIDENCE_BOOST * 0.5))
        return np.float64(1.0 - confidence_value * (SUTTE_CONFIDENCE_PENALTY * 0.5))

    def _evaluate_breakout_potential(
        self, df: pd.DataFrame, cons: ConsolidationFeatures, probs: pd.Series, dirs: pd.Series
    ) -> None:
        """Evaluate breakout potential for a single consolidation."""
        hold_period = self._adaptive_hold_period(cons.duration)
        max_idx = min(cons.end_idx + hold_period + 1, len(df))
        context_bias = 1.0 + BREAKOUT_CONTEXT_BONUS * np.mean(
            [
                cons.consolidation_score,
                cons.attention_context_score,
                cons.periodic_context_score,
                cons.scale_anchor_score,
            ]
        )

        for i in range(cons.end_idx + 1, max_idx):
            current_price = df["close"].iloc[i]

            zone_high = df["high"].iloc[cons.start_idx : cons.end_idx + 1].max()
            zone_low = df["low"].iloc[cons.start_idx : cons.end_idx + 1].min()
            zone_range = zone_high - zone_low

            if zone_range <= 0:
                continue

            dist_to_high = (current_price - zone_high) / zone_range
            dist_to_low = (zone_low - current_price) / zone_range

            current_vol = df["hl_range_pct"].iloc[max(0, i - 10) : i].mean()
            cons_vol = df["hl_range_pct"].iloc[cons.start_idx : cons.end_idx + 1].mean()
            vol_ratio = current_vol / (cons_vol + PRICE_EPSILON)

            if current_price > zone_high:
                breakout_strength = (1.0 + dist_to_high) * max(vol_ratio, 0.0) * context_bias
                if breakout_strength < self.params.min_expansion:
                    continue
                prob = 1 / (
                    1
                    + np.exp(
                        -self.params.logistic_steepness
                        * (
                            breakout_strength
                            - self.params.min_expansion
                            - self.params.logistic_threshold
                        )
                    )
                )
                prob = np.clip(
                    prob
                    * self._sutte_confidence(df, i, 1)
                    * self._sutte_structure_confidence(cons, 1),
                    0.0,
                    1.0,
                )
                probs.iloc[i] = max(probs.iloc[i], prob)
                if prob > 0.5:
                    dirs.iloc[i] = 1

            elif current_price < zone_low:
                breakout_strength = (1.0 + dist_to_low) * max(vol_ratio, 0.0) * context_bias
                if breakout_strength < self.params.min_expansion:
                    continue
                prob = 1 / (
                    1
                    + np.exp(
                        -self.params.logistic_steepness
                        * (
                            breakout_strength
                            - self.params.min_expansion
                            - self.params.logistic_threshold
                        )
                    )
                )
                prob = np.clip(
                    prob
                    * self._sutte_confidence(df, i, -1)
                    * self._sutte_structure_confidence(cons, -1),
                    0.0,
                    1.0,
                )
                probs.iloc[i] = max(probs.iloc[i], prob)
                if prob > 0.5:
                    dirs.iloc[i] = -1

    def _adaptive_hold_period(self, consolidation_duration: int) -> int:
        """Scale hold period by consolidation duration with bounded limits."""
        scaled = int(round(consolidation_duration * HOLD_DURATION_MULTIPLIER))
        return max(MIN_HOLD_PERIODS, min(self.params.max_hold, scaled))


# =============================================================================
# PARAMETER OPTIMIZER - FIXED TO PREVENT LOOKAHEAD BIAS
# =============================================================================


class ParameterOptimizer:
    """Optimize consolidation detection parameters using Optuna."""

    def __init__(self, n_trials: int = OPTIMIZATION_TRIALS, n_splits: int = VALIDATION_SPLITS):
        """
        Initialize optimizer.

        Args:
            n_trials: Number of optimization trials
            n_splits: Number of cross-validation splits
        """
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.logger = logging.getLogger(__name__)
        self.study = None
        self.best_params = None

    def optimize(self, df: pd.DataFrame) -> OptimizationParams:
        """
        Find optimal parameters through Bayesian optimization.
        Uses rolling-origin validation to preserve structural-break realism.

        Args:
            df: DataFrame with calculated features

        Returns:
            Optimized parameters
        """
        self.logger.info(f"Starting optimization with {self.n_trials} trials")

        try:
            cv_splits = _build_rolling_origin_splits(len(df), self.n_splits)
        except ValueError as exc:
            raise ValueError(f"Insufficient rows for rolling CV: rows={len(df)}.") from exc

        first_train_idx, first_val_idx = cv_splits[0]
        self.logger.info(
            "Rolling CV layout: splits=%d, train_window=%d, validation_window=%d",
            len(cv_splits),
            len(first_train_idx),
            len(first_val_idx),
        )

        self.study = optuna.create_study(
            direction=OPTIMIZATION_DIRECTION,
            sampler=TPESampler(seed=RANDOM_SEED),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )

        self.study.optimize(
            lambda trial: self._objective(trial, df, cv_splits),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        self.best_params = self._dict_to_params(self.study.best_params)
        best_mean_brier = -np.float64(self.study.best_value)
        self.logger.info(
            "Optimization complete. Best mean validation Brier: %.4f",
            best_mean_brier,
        )
        if self.study.best_trial is not None:
            self.logger.info(
                "Best fold coverage: %.1f%%, answered accuracy: %.1f%%",
                100.0 * np.float64(self.study.best_trial.user_attrs.get("fold_mean_coverage", 0.0)),
                100.0
                * np.float64(
                    self.study.best_trial.user_attrs.get("fold_mean_answered_accuracy", 0.0)
                ),
            )
        return self.best_params

    def _predict_validation_fold(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        params: OptimizationParams,
    ) -> tuple[pd.Series, pd.Series]:
        """Predict the validation window while keeping the preceding train history visible."""
        return _predict_validation_fold_with_history(df, train_idx, val_idx, params)

    def _objective(
        self,
        trial: optuna.Trial,
        df: pd.DataFrame,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Optimize the detector on out-of-fold probability quality."""
        params = self._suggest_params(trial)
        fold_metrics: list[dict[str, float]] = []

        for train_idx, val_idx in cv_splits:
            val_data = df.iloc[val_idx].copy()
            val_probs, val_dirs = self._predict_validation_fold(df, train_idx, val_idx, params)
            metrics = self._calculate_validation_metrics(
                val_data,
                val_probs,
                val_dirs,
                params.max_hold,
            )
            if metrics is not None:
                fold_metrics.append(metrics)

        if len(fold_metrics) < MIN_VALIDATION_FOLDS:
            return -1.0

        mean_brier = np.float64(np.mean([metrics["brier"] for metrics in fold_metrics]))
        mean_coverage = np.float64(np.mean([metrics["coverage"] for metrics in fold_metrics]))
        mean_answered_accuracy = np.float64(
            np.mean([metrics["answered_accuracy"] for metrics in fold_metrics])
        )
        mean_f1 = np.float64(np.mean([metrics["active_f1"] for metrics in fold_metrics]))
        mean_sharpe = np.float64(np.mean([metrics["active_sharpe"] for metrics in fold_metrics]))
        brier_std = np.float64(np.std([metrics["brier"] for metrics in fold_metrics]))

        trial.set_user_attr("fold_mean_brier", mean_brier)
        trial.set_user_attr("fold_brier_std", brier_std)
        trial.set_user_attr("fold_mean_coverage", mean_coverage)
        trial.set_user_attr("fold_mean_answered_accuracy", mean_answered_accuracy)
        trial.set_user_attr("fold_mean_active_f1", mean_f1)
        trial.set_user_attr("fold_mean_active_sharpe", mean_sharpe)
        if mean_coverage <= PRICE_EPSILON:
            return -1.0
        return np.float64(-mean_brier)

    def _suggest_params(self, trial: optuna.Trial) -> OptimizationParams:
        """Suggest parameters for optimization trial."""
        raw_cont = trial.suggest_float("weight_contraction", MIN_WEIGHT, MAX_WEIGHT)
        raw_range = trial.suggest_float("weight_range", MIN_WEIGHT, MAX_WEIGHT)
        raw_pos = trial.suggest_float("weight_position", MIN_WEIGHT, MAX_WEIGHT)
        raw_hurst = trial.suggest_float("weight_hurst", MIN_WEIGHT, MAX_WEIGHT)
        raw_periodic = trial.suggest_float("weight_periodic", MIN_WEIGHT, MAX_WEIGHT)
        raw_attention = trial.suggest_float("weight_attention", MIN_WEIGHT, MAX_WEIGHT)
        raw_scale = trial.suggest_float("weight_scale", MIN_WEIGHT, MAX_WEIGHT)
        raw_sutte = trial.suggest_float("weight_sutte", MIN_WEIGHT, MAX_WEIGHT)

        total = (
            raw_cont
            + raw_range
            + raw_pos
            + raw_hurst
            + raw_periodic
            + raw_attention
            + raw_scale
            + raw_sutte
        )
        norm_cont = raw_cont / total
        norm_range = raw_range / total
        norm_pos = raw_pos / total
        norm_hurst = raw_hurst / total
        norm_periodic = raw_periodic / total
        norm_attention = raw_attention / total
        norm_scale = raw_scale / total
        norm_sutte = raw_sutte / total

        return OptimizationParams(
            lookback_period=trial.suggest_int("lookback_period", MIN_LOOKBACK, MAX_LOOKBACK),
            lookforward_period=trial.suggest_int(
                "lookforward_period", MIN_LOOKFORWARD, MAX_LOOKFORWARD
            ),
            min_bars=trial.suggest_int("min_bars", MIN_CONSOLIDATION_BARS, MAX_CONSOLIDATION_BARS),
            range_threshold=trial.suggest_float("range_threshold", 1.0, 3.0),
            position_sensitivity=trial.suggest_float("position_sensitivity", 0.5, 2.0),
            consolidation_threshold=trial.suggest_float(
                "consolidation_threshold",
                MIN_CONSOLIDATION_THRESHOLD,
                MAX_CONSOLIDATION_THRESHOLD,
            ),
            min_expansion=trial.suggest_float("min_expansion", 1.1, 2.0),
            weight_contraction=norm_cont,
            weight_range=norm_range,
            weight_position=norm_pos,
            weight_hurst=norm_hurst,
            max_hold=trial.suggest_int("max_hold", MIN_HOLD_PERIODS, MAX_HOLD_PERIODS),
            logistic_steepness=trial.suggest_float(
                "logistic_steepness", MIN_LOGISTIC_STEEPNESS, MAX_LOGISTIC_STEEPNESS
            ),
            logistic_threshold=trial.suggest_float(
                "logistic_threshold", MIN_LOGISTIC_THRESHOLD, MAX_LOGISTIC_THRESHOLD
            ),
            weight_periodic=norm_periodic,
            weight_attention=norm_attention,
            weight_scale=norm_scale,
            weight_sutte=norm_sutte,
        )

    def _calculate_validation_metrics(
        self,
        df: pd.DataFrame,
        probabilities: pd.Series,
        directions: pd.Series,
        max_hold: int,
    ) -> dict[str, float] | None:
        """Return interpretable out-of-fold diagnostics for one validation split."""
        actual_dirs = _build_future_direction_targets(df, max_hold)
        common_idx = probabilities.index.intersection(directions.index).intersection(
            actual_dirs.index
        )
        if len(common_idx) == 0:
            return None

        aligned_probs = probabilities.loc[common_idx].astype(np.float64)
        aligned_pred = directions.loc[common_idx].astype(int)
        aligned_actual = actual_dirs.loc[common_idx]

        valid_mask = aligned_actual.isin([-1.0, 1.0])
        if not valid_mask.any():
            return None

        aligned_probs = aligned_probs.loc[valid_mask]
        aligned_pred = aligned_pred.loc[valid_mask]
        aligned_actual = aligned_actual.loc[valid_mask].astype(int)
        up_probability = _map_directional_probabilities_to_up_probability(
            aligned_probs,
            aligned_pred,
        )
        actual_up = (aligned_actual == 1).astype(int)
        brier = np.float64(brier_score_loss(actual_up, up_probability))
        coverage = np.float64((aligned_pred != 0).mean())

        active_mask = aligned_pred != 0
        if not active_mask.any():
            return {
                "brier": brier,
                "coverage": coverage,
                "answered_accuracy": 0.0,
                "active_f1": 0.0,
                "active_sharpe": 0.0,
            }

        active_pred = aligned_pred.loc[active_mask]
        active_actual = aligned_actual.loc[active_mask]
        answered_accuracy = np.float64((active_pred == active_actual).mean())
        f1 = np.float64(
            f1_score(
                active_actual,
                active_pred,
                average="weighted",
                zero_division=0,
            )
        )
        future_returns = df[CLOSE_COLUMN].pct_change(max_hold).shift(-max_hold).loc[common_idx]
        strategy_returns = active_pred * future_returns.loc[active_pred.index]
        sharpe = np.float64(
            strategy_returns.mean() / (strategy_returns.std() + PRICE_EPSILON) * np.sqrt(252)
        )
        return {
            "brier": brier,
            "coverage": coverage,
            "answered_accuracy": answered_accuracy,
            "active_f1": f1,
            "active_sharpe": sharpe,
        }

    def _dict_to_params(self, params_dict: dict[str, Any]) -> OptimizationParams:
        """Convert dictionary to OptimizationParams."""
        params_copy = dict(params_dict)
        weight_keys = (
            "weight_contraction",
            "weight_range",
            "weight_position",
            "weight_hurst",
            "weight_periodic",
            "weight_attention",
            "weight_scale",
            "weight_sutte",
        )
        if any(key in params_copy for key in weight_keys) and "weight_sutte" not in params_copy:
            params_copy["weight_sutte"] = 0.0
        present_weight_keys = tuple(key for key in weight_keys if key in params_copy)
        if present_weight_keys:
            total = sum(np.float64(params_copy[key]) for key in present_weight_keys)
            if total > 0:
                for key in present_weight_keys:
                    params_copy[key] = np.float64(params_copy[key]) / total

        return OptimizationParams(**params_copy)

    def get_feature_importance(self) -> dict[str, float]:
        """Calculate feature importance from optimization study."""
        if self.study is None:
            return {}

        importance = {}
        for param_name in self.study.best_params.keys():
            param_values = [
                t.params[param_name] for t in self.study.trials if param_name in t.params
            ]
            obj_values = [t.value for t in self.study.trials if t.value is not None]

            if len(param_values) > 1 and len(obj_values) > 1:
                min_len = min(len(param_values), len(obj_values))
                x = np.asarray(param_values[:min_len], dtype=np.float64)
                y = np.asarray(obj_values[:min_len], dtype=np.float64)

                # Constant vectors produce undefined correlation.
                if np.std(x) == 0.0 or np.std(y) == 0.0:
                    importance[param_name] = 0.0
                    continue

                corr = np.corrcoef(x, y)[0, 1]
                importance[param_name] = abs(np.float64(corr)) if np.isfinite(corr) else 0.0
            else:
                importance[param_name] = 0.0

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# =============================================================================
# PROBABILITY CALIBRATION
# =============================================================================


@dataclass
class ProbabilityCalibrationModel:
    """Store piecewise-linear anchors for monotonic probability calibration."""

    raw_points: np.ndarray
    calibrated_points: np.ndarray

    def map(self, probabilities: pd.Series) -> pd.Series:
        """Map raw probabilities onto calibrated probabilities."""
        if probabilities.empty:
            return probabilities.astype(np.float64)

        clipped = probabilities.clip(0.0, 1.0).to_numpy(dtype=np.float64, copy=False)
        calibrated = np.interp(
            clipped,
            self.raw_points,
            self.calibrated_points,
        )
        return pd.Series(calibrated, index=probabilities.index, dtype=np.float64)


class ProbabilityCalibrator:
    """Calibrate breakout probabilities from out-of-fold empirical hit rates."""

    def __init__(
        self,
        params: OptimizationParams,
        n_splits: int = VALIDATION_SPLITS,
    ):
        """
        Initialize the probability calibrator.

        Args:
            params: Optimized detector and predictor parameters.
            n_splits: Number of rolling-origin folds used to fit calibration anchors.
        """
        self.params = params
        self.n_splits = n_splits
        self.logger = logging.getLogger(__name__)
        self.model: ProbabilityCalibrationModel | None = None

    def _predict_validation_fold(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> tuple[pd.Series, pd.Series]:
        """Predict validation probabilities with the train window available as history."""
        return _predict_validation_fold_with_history(df, train_idx, val_idx, self.params)

    def fit(self, df: pd.DataFrame) -> ProbabilityCalibrationModel | None:
        """
        Fit a monotonic calibration curve from out-of-fold breakout predictions.

        Args:
            df: Feature-enriched OHLC dataframe.

        Returns:
            Calibration model if enough out-of-fold samples exist, otherwise `None`.
        """
        try:
            cv_splits = _build_rolling_origin_splits(len(df), self.n_splits)
        except ValueError as exc:
            self.logger.warning("Skipping probability calibration: %s", exc)
            return None

        future_returns = df["close"].pct_change(self.params.max_hold).shift(-self.params.max_hold)
        raw_prob_chunks: list[np.ndarray] = []
        outcome_chunks: list[np.ndarray] = []

        for train_idx, val_idx in cv_splits:
            evaluation_index = df.index[val_idx]
            val_probs, val_dirs = self._predict_validation_fold(df, train_idx, val_idx)
            actual_dirs = np.sign(future_returns.reindex(evaluation_index))
            valid_mask = (
                (val_dirs != 0) & actual_dirs.notna() & (actual_dirs != 0) & (val_probs > 0.0)
            )
            if not valid_mask.any():
                continue

            raw_prob_chunks.append(val_probs[valid_mask].to_numpy(dtype=np.float64, copy=False))
            outcomes = (val_dirs[valid_mask] == actual_dirs[valid_mask]).astype(np.float64)
            outcome_chunks.append(outcomes.to_numpy(dtype=np.float64, copy=False))

        if not raw_prob_chunks:
            self.logger.warning("Skipping probability calibration: no out-of-fold signals found.")
            return None

        raw_probs = np.concatenate(raw_prob_chunks)
        outcomes = np.concatenate(outcome_chunks)
        if raw_probs.size < PROBABILITY_CALIBRATION_MIN_SAMPLES:
            self.logger.warning(
                "Skipping probability calibration: sample_count=%d < min_samples=%d",
                raw_probs.size,
                PROBABILITY_CALIBRATION_MIN_SAMPLES,
            )
            return None

        bin_count = min(PROBABILITY_CALIBRATION_BINS, raw_probs.size)
        quantiles = np.linspace(0.0, 1.0, bin_count + 1)
        bin_edges = np.unique(np.quantile(raw_probs, quantiles))
        if bin_edges.size < 2:
            self.logger.warning(
                "Skipping probability calibration: insufficient probability spread."
            )
            return None

        raw_points: list[float] = []
        calibrated_points: list[float] = []
        for edge_idx in range(bin_edges.size - 1):
            low_edge = np.float64(bin_edges[edge_idx])
            high_edge = np.float64(bin_edges[edge_idx + 1])
            if edge_idx == bin_edges.size - 2:
                mask = (raw_probs >= low_edge) & (raw_probs <= high_edge)
            else:
                mask = (raw_probs >= low_edge) & (raw_probs < high_edge)
            if not np.any(mask):
                continue

            raw_center = np.float64(np.mean(raw_probs[mask]))
            empirical_rate = np.float64(np.mean(outcomes[mask]))
            blended_rate = (
                1.0 - PROBABILITY_CALIBRATION_BLEND
            ) * raw_center + PROBABILITY_CALIBRATION_BLEND * empirical_rate
            raw_points.append(raw_center)
            calibrated_points.append(blended_rate)

        if len(raw_points) < 2:
            self.logger.warning(
                "Skipping probability calibration: insufficient calibration anchors."
            )
            return None

        raw_anchor_array = np.maximum.accumulate(
            np.clip(np.asarray(raw_points, dtype=np.float64), 0.0, 1.0)
        )
        calibrated_anchor_array = np.maximum.accumulate(
            np.clip(np.asarray(calibrated_points, dtype=np.float64), 0.0, 1.0)
        )

        if raw_anchor_array[0] > 0.0:
            raw_anchor_array = np.insert(raw_anchor_array, 0, 0.0)
            calibrated_anchor_array = np.insert(calibrated_anchor_array, 0, 0.0)
        if raw_anchor_array[-1] < 1.0:
            raw_anchor_array = np.append(raw_anchor_array, 1.0)
            calibrated_anchor_array = np.append(
                calibrated_anchor_array,
                calibrated_anchor_array[-1],
            )

        dedup_raw_points: list[float] = []
        dedup_calibrated_points: list[float] = []
        for raw_point, calibrated_point in zip(raw_anchor_array, calibrated_anchor_array):
            if dedup_raw_points and abs(raw_point - dedup_raw_points[-1]) <= PRICE_EPSILON:
                dedup_calibrated_points[-1] = max(
                    dedup_calibrated_points[-1],
                    np.float64(calibrated_point),
                )
                continue
            dedup_raw_points.append(np.float64(raw_point))
            dedup_calibrated_points.append(np.float64(calibrated_point))

        if len(dedup_raw_points) < 2:
            self.logger.warning(
                "Skipping probability calibration: calibration anchors collapsed after dedupe."
            )
            return None

        self.model = ProbabilityCalibrationModel(
            raw_points=np.asarray(dedup_raw_points, dtype=np.float64),
            calibrated_points=np.asarray(dedup_calibrated_points, dtype=np.float64),
        )
        self.logger.info(
            "Fitted probability calibration with %d samples and %d anchors.",
            raw_probs.size,
            len(dedup_raw_points),
        )
        return self.model

    def apply(
        self,
        probabilities: pd.Series,
        directions: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Apply the fitted calibration curve to active breakout probabilities.

        Args:
            probabilities: Raw breakout probability series.
            directions: Raw breakout direction series.

        Returns:
            Tuple of calibrated probabilities and threshold-adjusted directions.
        """
        if self.model is None:
            return probabilities.copy(), directions.copy()

        calibrated = probabilities.astype(np.float64).copy()
        active_mask = calibrated > 0.0
        if active_mask.any():
            calibrated.loc[active_mask] = self.model.map(calibrated.loc[active_mask])
        calibrated = calibrated.clip(0.0, 1.0)

        adjusted_dirs = directions.astype(int).copy()
        adjusted_dirs[(adjusted_dirs != 0) & (calibrated <= 0.5)] = 0
        return calibrated, adjusted_dirs


# =============================================================================
# BASELINE BENCHMARK
# =============================================================================


@dataclass
class BenchmarkResult:
    """Aggregate out-of-fold benchmark metrics for a directional model."""

    sample_count: int
    coverage: float
    answered_accuracy: float
    brier_score: float


@dataclass
class RealTimeForecast:
    """Container for the latest-bar consolidation forecast."""

    timestamp: datetime | None
    is_consolidating: bool
    consolidation_score: float
    breakout_probability: float
    breakout_direction: int
    confidence_interval: tuple[float, float]
    regime_warning: bool
    bars_since_last_consolidation: int
    current_zone: tuple[float, float] | None
    zone_position: float | None
    signal_source: str
    model_confidence: str
    similar_historical_setups: int
    similar_outcomes: dict[str, int]
    horizon_bars: int
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable forecast payload."""
        direction_label = "none"
        if self.breakout_direction > 0:
            direction_label = "up"
        elif self.breakout_direction < 0:
            direction_label = "down"

        zone_value: list[float] | None = None
        if self.current_zone is not None:
            zone_value = [float(self.current_zone[0]), float(self.current_zone[1])]

        forecast_block: dict[str, Any] | None = None
        if self.is_consolidating:
            forecast_block = {
                "direction": direction_label,
                "probability": float(self.breakout_probability),
                "horizon_bars": int(self.horizon_bars),
                "confidence_interval": [
                    float(self.confidence_interval[0]),
                    float(self.confidence_interval[1]),
                ],
                "source": self.signal_source,
            }

        return {
            "now": self.timestamp.isoformat() if self.timestamp is not None else None,
            "consolidation": {
                "active": bool(self.is_consolidating),
                "score": float(self.consolidation_score),
                "zone": zone_value,
                "bars_since_last": int(self.bars_since_last_consolidation),
                "position_in_zone": (
                    None if self.zone_position is None else float(self.zone_position)
                ),
            },
            "forecast": forecast_block,
            "warnings": list(self.warnings),
            "context": {
                "regime_shift_detected": bool(self.regime_warning),
                "model_confidence": self.model_confidence,
                "similar_historical_setups": int(self.similar_historical_setups),
                "similar_outcomes": dict(self.similar_outcomes),
            },
        }


class BaselineBenchmark:
    """Compare the optimized detector against a simple logistic baseline."""

    def __init__(
        self,
        params: OptimizationParams,
        n_splits: int = VALIDATION_SPLITS,
        feature_columns: tuple[str, ...] = BASELINE_FEATURE_COLUMNS,
    ):
        """
        Initialize the baseline benchmark runner.

        Args:
            params: Optimized detector and predictor parameters.
            n_splits: Number of rolling-origin splits used for benchmarking.
            feature_columns: Feature columns exposed to the logistic baseline.
        """
        self.params = params
        self.n_splits = n_splits
        self.feature_columns = feature_columns
        self.logger = logging.getLogger(__name__)

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Benchmark the optimized model against a regularized logistic baseline."""
        try:
            cv_splits = _build_rolling_origin_splits(len(df), self.n_splits)
        except ValueError as exc:
            self.logger.warning("Skipping baseline benchmark: %s", exc)
            return {}

        model_actual_chunks: list[pd.Series] = []
        model_dir_chunks: list[pd.Series] = []
        model_prob_chunks: list[pd.Series] = []
        baseline_actual_chunks: list[pd.Series] = []
        baseline_dir_chunks: list[pd.Series] = []
        baseline_prob_chunks: list[pd.Series] = []

        for train_idx, val_idx in cv_splits:
            fold_result = self._evaluate_fold(df, train_idx, val_idx)
            if fold_result is None:
                continue

            (
                model_actual,
                model_dirs,
                model_up_prob,
                baseline_actual,
                baseline_dirs,
                baseline_up_prob,
            ) = fold_result
            model_actual_chunks.append(model_actual)
            model_dir_chunks.append(model_dirs)
            model_prob_chunks.append(model_up_prob)
            baseline_actual_chunks.append(baseline_actual)
            baseline_dir_chunks.append(baseline_dirs)
            baseline_prob_chunks.append(baseline_up_prob)

        if not model_actual_chunks or not baseline_actual_chunks:
            self.logger.warning("Skipping baseline benchmark: no labeled validation rows found.")
            return {}

        model_result = self._aggregate_result(
            actual_dirs=pd.concat(model_actual_chunks),
            predicted_dirs=pd.concat(model_dir_chunks),
            up_probabilities=pd.concat(model_prob_chunks),
        )
        baseline_result = self._aggregate_result(
            actual_dirs=pd.concat(baseline_actual_chunks),
            predicted_dirs=pd.concat(baseline_dir_chunks),
            up_probabilities=pd.concat(baseline_prob_chunks),
        )

        metrics = {
            "benchmark_sample_count": float(model_result.sample_count),
            "benchmark_model_coverage": model_result.coverage,
            "benchmark_model_answered_accuracy": model_result.answered_accuracy,
            "benchmark_model_brier": model_result.brier_score,
            "benchmark_baseline_coverage": baseline_result.coverage,
            "benchmark_baseline_answered_accuracy": baseline_result.answered_accuracy,
            "benchmark_baseline_brier": baseline_result.brier_score,
            "benchmark_accuracy_delta": (
                model_result.answered_accuracy - baseline_result.answered_accuracy
            ),
            "benchmark_brier_improvement": (baseline_result.brier_score - model_result.brier_score),
            "benchmark_model_beats_baseline": float(
                model_result.brier_score + PRICE_EPSILON < baseline_result.brier_score
            ),
        }
        self.logger.info(
            "Benchmark summary: model brier=%.4f, logistic brier=%.4f, improvement=%.4f",
            metrics["benchmark_model_brier"],
            metrics["benchmark_baseline_brier"],
            metrics["benchmark_brier_improvement"],
        )
        self.logger.info(
            "Benchmark coverage: model=%.1f%%, logistic=%.1f%%",
            100.0 * metrics["benchmark_model_coverage"],
            100.0 * metrics["benchmark_baseline_coverage"],
        )
        return metrics

    def _evaluate_fold(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series] | None:
        """Return aligned fold outputs for the optimized model and logistic baseline."""
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        actual_dirs = _build_future_direction_targets(val_df, self.params.max_hold)
        valid_mask = actual_dirs.isin([-1.0, 1.0])
        if not valid_mask.any():
            return None

        model_probs, model_dirs = _predict_validation_fold_with_history(
            df,
            train_idx,
            val_idx,
            self.params,
        )
        model_actual = actual_dirs.loc[valid_mask].astype(int)
        model_dirs = model_dirs.loc[valid_mask].astype(int)
        model_up_prob = _map_directional_probabilities_to_up_probability(
            model_probs.loc[valid_mask],
            model_dirs,
        )

        baseline_up_prob, baseline_dirs = self._predict_logistic_baseline(train_df, val_df)
        baseline_actual = actual_dirs.loc[valid_mask].astype(int)
        baseline_dirs = baseline_dirs.loc[valid_mask].astype(int)
        baseline_up_prob = baseline_up_prob.loc[valid_mask]

        return (
            model_actual,
            model_dirs,
            model_up_prob,
            baseline_actual,
            baseline_dirs,
            baseline_up_prob,
        )

    def _predict_logistic_baseline(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        """Predict breakout direction from a compact regularized logistic baseline."""
        feature_columns = [col for col in self.feature_columns if col in train_df.columns]
        if not feature_columns:
            neutral = pd.Series(0.5, index=val_df.index, dtype=np.float64)
            directions = pd.Series(0, index=val_df.index, dtype=int)
            return neutral, directions

        train_targets = _build_future_direction_targets(train_df, self.params.max_hold)
        train_mask = train_targets.isin([-1.0, 1.0])
        if not train_mask.any():
            neutral = pd.Series(0.5, index=val_df.index, dtype=np.float64)
            directions = pd.Series(0, index=val_df.index, dtype=int)
            return neutral, directions

        train_x = train_df.loc[train_mask, feature_columns]
        train_y = (train_targets.loc[train_mask] == 1.0).astype(int)
        if train_y.nunique() < 2:
            constant_up_probability = np.float64(train_y.iloc[0])
            probabilities = pd.Series(constant_up_probability, index=val_df.index, dtype=np.float64)
            directions = pd.Series(
                np.where(probabilities >= 0.5, 1, -1),
                index=val_df.index,
                dtype=int,
            )
            return probabilities, directions

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=BASELINE_LOGISTIC_C,
                        class_weight="balanced",
                        max_iter=BASELINE_LOGISTIC_MAX_ITER,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        pipeline.fit(train_x, train_y)
        val_x = val_df.loc[:, feature_columns]
        up_probability = pipeline.predict_proba(val_x)[:, 1]
        probabilities = pd.Series(up_probability, index=val_df.index, dtype=np.float64).clip(
            0.0, 1.0
        )
        directions = pd.Series(
            np.where(probabilities >= 0.5, 1, -1),
            index=val_df.index,
            dtype=int,
        )
        return probabilities, directions

    def _aggregate_result(
        self,
        actual_dirs: pd.Series,
        predicted_dirs: pd.Series,
        up_probabilities: pd.Series,
    ) -> BenchmarkResult:
        """Collapse aligned out-of-fold predictions into benchmark metrics."""
        valid_mask = actual_dirs.isin([-1, 1]) & up_probabilities.notna()
        if not valid_mask.any():
            return BenchmarkResult(
                sample_count=0, coverage=0.0, answered_accuracy=0.0, brier_score=1.0
            )

        actual = actual_dirs.loc[valid_mask].astype(int)
        predicted = predicted_dirs.loc[valid_mask].astype(int)
        probabilities = up_probabilities.loc[valid_mask].astype(np.float64).clip(0.0, 1.0)
        coverage = np.float64((predicted != 0).mean())

        answered_mask = predicted != 0
        if answered_mask.any():
            answered_accuracy = np.float64(
                (predicted.loc[answered_mask] == actual.loc[answered_mask]).mean()
            )
        else:
            answered_accuracy = 0.0

        actual_up = (actual == 1).astype(int)
        brier = np.float64(brier_score_loss(actual_up, probabilities))
        return BenchmarkResult(
            sample_count=int(valid_mask.sum()),
            coverage=coverage,
            answered_accuracy=answered_accuracy,
            brier_score=brier,
        )


# =============================================================================
# DATA LOADER
# =============================================================================


class DataLoader:
    """Load and validate OHLC data from backtest prices."""

    def __init__(self, file_path: str | None = DATA_FILE_PATH):
        """
        Initialize data loader.

        Args:
            file_path: Optional path to CSV file
        """
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def load(self) -> pd.DataFrame:
        """
        Load and validate OHLC data.

        Returns:
            Validated DataFrame with datetime index
        """
        source_path = self._resolve_csv_path()
        self.logger.info("Loading data from %s", source_path)

        # Avoid the upstream stale-file rejection path when local config explicitly
        # allows loading older snapshots for analysis and dashboard generation.
        if not REJECT_STALE_BACKTEST_DATA:
            df = self._load_csv_without_stale_guard(source_path)
            df = df.set_index(TIMESTAMP_COLUMN).sort_index()
            self.logger.info("Loaded %d rows from %s to %s", len(df), df.index[0], df.index[-1])
            return df

        load_kwargs: dict[str, Any] = {
            "csv_path": self.file_path,
            "required_columns": (
                OPEN_COLUMN,
                HIGH_COLUMN,
                LOW_COLUMN,
                CLOSE_COLUMN,
                TIMESTAMP_COLUMN,
            ),
            "datetime_column": TIMESTAMP_COLUMN,
            "datetime_format": DATETIME_FORMAT,
        }
        signature = inspect.signature(load_backtest_prices)
        if "reject_stale" in signature.parameters:
            load_kwargs["reject_stale"] = REJECT_STALE_BACKTEST_DATA
        if "max_csv_age_hours" in signature.parameters:
            load_kwargs["max_csv_age_hours"] = MAX_BACKTEST_CSV_AGE_HOURS

        try:
            df, summary = load_backtest_prices(**load_kwargs)
        except TypeError as exc:
            # Support older loader signatures when local modules are not in sync.
            self.logger.warning(
                "Backtest loader signature mismatch (%s). Retrying with baseline arguments.",
                exc,
            )
            load_kwargs.pop("reject_stale", None)
            load_kwargs.pop("max_csv_age_hours", None)
            try:
                df, summary = load_backtest_prices(**load_kwargs)
            except (FileNotFoundError, TypeError, ValueError) as inner_exc:
                self.logger.error("Failed to load backtest data from %s", source_path)
                raise RuntimeError(f"Unable to load data from {source_path}") from inner_exc
        except (FileNotFoundError, ValueError) as exc:
            self.logger.error("Failed to load backtest data from %s", source_path)
            raise RuntimeError(f"Unable to load data from {source_path}") from exc

        if df.empty:
            if not REJECT_STALE_BACKTEST_DATA:
                self.logger.warning(
                    "Backtest loader returned empty data while stale rejection is disabled. "
                    "Attempting direct CSV load without stale guard."
                )
                summary_path = getattr(summary, "file_path", None)
                csv_path = summary_path if summary_path is not None else source_path
                df = self._load_csv_without_stale_guard(csv_path)
            else:
                raise ValueError(
                    f"Backtest data is empty for {summary.file_path}. "
                    "Refresh backtest_prices.csv and retry."
                )

        if TIMESTAMP_COLUMN not in df.columns:
            available = ", ".join(df.columns.astype(str))
            raise ValueError(
                f"Expected datetime column '{TIMESTAMP_COLUMN}' in loaded data. "
                f"Available columns: {available}"
            )

        df = df.set_index(TIMESTAMP_COLUMN).sort_index()
        self.logger.info("Loaded %d rows from %s to %s", len(df), df.index[0], df.index[-1])
        return df

    def _resolve_csv_path(self) -> Path:
        """Resolve the effective CSV path for local snapshot loading."""
        if self.file_path is not None:
            return Path(self.file_path).expanduser().resolve()
        return Path(__file__).with_name("backtest_prices.csv").resolve()

    def _load_csv_without_stale_guard(self, csv_path: str | Path) -> pd.DataFrame:
        """Fallback CSV loader used when stale filtering is hardcoded upstream."""
        path = Path(csv_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path, engine="c", low_memory=False)
        if TIMESTAMP_COLUMN not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": TIMESTAMP_COLUMN})

        required_cols = [TIMESTAMP_COLUMN, OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, CLOSE_COLUMN]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in fallback load: {missing}")

        parsed = pd.to_datetime(df[TIMESTAMP_COLUMN], format=DATETIME_FORMAT, errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")
        df[TIMESTAMP_COLUMN] = parsed

        for col in (OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, CLOSE_COLUMN):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required_cols).copy()
        if df.empty:
            raise ValueError(f"Fallback load produced empty data from {path}")

        return df


# =============================================================================
# PERFORMANCE ANALYZER
# =============================================================================


class PerformanceAnalyzer:
    """Calculate and analyze performance metrics."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(
        self,
        df: pd.DataFrame,
        consolidations: list[ConsolidationFeatures],
        probabilities: pd.Series,
        directions: pd.Series,
    ) -> dict[str, float]:
        """Calculate comprehensive performance metrics."""
        self.logger.info("Calculating performance metrics")

        metrics = {}

        metrics["total_consolidations"] = len(consolidations)
        metrics["breakout_signals"] = len(directions[directions != 0])
        metrics["high_prob_signals"] = len(probabilities[probabilities > 0.7])
        if consolidations:
            metrics["mean_consolidation_score"] = np.float64(
                np.mean([cons.consolidation_score for cons in consolidations])
            )
            metrics["mean_attention_context_score"] = np.float64(
                np.mean([cons.attention_context_score for cons in consolidations])
            )
            metrics["mean_periodic_context_score"] = np.float64(
                np.mean([cons.periodic_context_score for cons in consolidations])
            )
            metrics["mean_scale_anchor_score"] = np.float64(
                np.mean([cons.scale_anchor_score for cons in consolidations])
            )
            metrics["mean_sutte_score"] = np.float64(
                np.mean([cons.sutte_score for cons in consolidations])
            )
            metrics["mean_interaction_score"] = np.float64(
                np.mean([cons.interaction_score for cons in consolidations])
            )
            metrics["mean_sutte_divergence_confidence"] = np.float64(
                np.mean([cons.sutte_divergence_confidence for cons in consolidations])
            )
        else:
            metrics["mean_consolidation_score"] = 0.0
            metrics["mean_attention_context_score"] = 0.0
            metrics["mean_periodic_context_score"] = 0.0
            metrics["mean_scale_anchor_score"] = 0.0
            metrics["mean_sutte_score"] = 0.0
            metrics["mean_interaction_score"] = 0.0
            metrics["mean_sutte_divergence_confidence"] = 0.0

        up_signals = len(directions[directions == 1])
        down_signals = len(directions[directions == -1])
        total_signals = up_signals + down_signals
        metrics["up_breakout_signals"] = up_signals
        metrics["down_breakout_signals"] = down_signals
        metrics["neutral_signal_bars"] = len(directions) - total_signals
        if total_signals > 0:
            metrics["signal_bias"] = (up_signals - down_signals) / total_signals
        else:
            metrics["signal_bias"] = 0.0

        non_zero_probs = probabilities[probabilities > 0]
        if len(non_zero_probs) > 0:
            metrics["mean_probability"] = non_zero_probs.mean()
            metrics["prob_std"] = non_zero_probs.std()
            metrics["max_probability"] = non_zero_probs.max()
            metrics["min_probability"] = non_zero_probs.min()
            prob_ci_low, prob_ci_median, prob_ci_high = self._bootstrap_probability_interval(
                non_zero_probs
            )
            metrics["probability_ci_p05"] = prob_ci_low
            metrics["probability_ci_p50"] = prob_ci_median
            metrics["probability_ci_p95"] = prob_ci_high
        else:
            metrics["mean_probability"] = 0.0
            metrics["prob_std"] = 0.0
            metrics["max_probability"] = 0.0
            metrics["min_probability"] = 0.0
            metrics["probability_ci_p05"] = 0.0
            metrics["probability_ci_p50"] = 0.0
            metrics["probability_ci_p95"] = 0.0

        future_returns = df["close"].pct_change(10).shift(-10)
        simulated_returns = directions * future_returns

        valid_returns = simulated_returns.dropna()
        if len(valid_returns) > 0:
            metrics["simulated_mean_return"] = valid_returns.mean()
            metrics["simulated_std_return"] = valid_returns.std()
            metrics["simulated_sharpe"] = (
                valid_returns.mean() / (valid_returns.std() + PRICE_EPSILON) * np.sqrt(252)
            )
            metrics["simulated_win_rate"] = (valid_returns > 0).mean()
            metrics["simulated_total_return"] = (1 + valid_returns).prod() - 1
        else:
            metrics["simulated_sharpe"] = 0.0
            metrics["simulated_win_rate"] = 0.0

        metrics.update(self._calculate_regime_metrics(df))
        metrics.update(self._calculate_feature_instability(df))
        return metrics

    def _bootstrap_probability_interval(
        self, probabilities: pd.Series
    ) -> tuple[float, float, float]:
        """Estimate uncertainty bands for signal probabilities via bootstrap."""
        clean = probabilities.dropna().to_numpy(dtype=np.float64)
        if clean.size == 0:
            return 0.0, 0.0, 0.0

        rng = np.random.default_rng(RANDOM_SEED)
        bootstrap_means = np.empty(PROBABILITY_CI_BOOTSTRAP_SAMPLES, dtype=np.float64)
        for idx in range(PROBABILITY_CI_BOOTSTRAP_SAMPLES):
            sample = rng.choice(clean, size=clean.size, replace=True)
            bootstrap_means[idx] = np.float64(np.mean(sample))

        p_low, p_median, p_high = np.percentile(
            bootstrap_means,
            [PROBABILITY_CI_LOW, PROBABILITY_CI_MEDIAN, PROBABILITY_CI_HIGH],
        )
        return np.float64(p_low), np.float64(p_median), np.float64(p_high)

    def _calculate_regime_metrics(self, df: pd.DataFrame) -> dict[str, float]:
        """Return volatility-regime diagnostics from recent vs baseline windows."""
        if "close" not in df.columns:
            return {
                "regime_recent_vol": 0.0,
                "regime_baseline_vol": 0.0,
                "regime_vol_ratio": 1.0,
                "regime_recent_ewm_vol": 0.0,
                "regime_baseline_ewm_vol": 0.0,
                "regime_ewm_vol_ratio": 1.0,
                "regime_shift_flag": 0.0,
            }

        returns = df["close"].pct_change().dropna().to_numpy(dtype=np.float64)
        if returns.size == 0:
            return {
                "regime_recent_vol": 0.0,
                "regime_baseline_vol": 0.0,
                "regime_vol_ratio": 1.0,
                "regime_recent_ewm_vol": 0.0,
                "regime_baseline_ewm_vol": 0.0,
                "regime_ewm_vol_ratio": 1.0,
                "regime_shift_flag": 0.0,
            }

        recent_window = min(REGIME_RECENT_WINDOW, returns.size)
        baseline_window = min(REGIME_BASELINE_WINDOW, returns.size)
        recent_vol = np.float64(np.std(returns[-recent_window:]))
        baseline_vol = np.float64(np.std(returns[-baseline_window:]))
        recent_ewm_vol = self._exponential_weighted_volatility(returns[-recent_window:])
        baseline_ewm_vol = self._exponential_weighted_volatility(returns[-baseline_window:])
        vol_ratio = recent_vol / (baseline_vol + PRICE_EPSILON)
        ewm_vol_ratio = recent_ewm_vol / (baseline_ewm_vol + PRICE_EPSILON)
        regime_shift = np.float64(
            vol_ratio >= REGIME_SHIFT_HIGH_THRESHOLD
            or vol_ratio <= REGIME_SHIFT_LOW_THRESHOLD
            or ewm_vol_ratio >= REGIME_SHIFT_HIGH_THRESHOLD
            or ewm_vol_ratio <= REGIME_SHIFT_LOW_THRESHOLD
        )

        return {
            "regime_recent_vol": recent_vol,
            "regime_baseline_vol": baseline_vol,
            "regime_vol_ratio": np.float64(vol_ratio),
            "regime_recent_ewm_vol": recent_ewm_vol,
            "regime_baseline_ewm_vol": baseline_ewm_vol,
            "regime_ewm_vol_ratio": np.float64(ewm_vol_ratio),
            "regime_shift_flag": regime_shift,
        }

    def _exponential_weighted_volatility(
        self,
        returns: np.ndarray,
        half_life: int = REGIME_EW_HALF_LIFE,
    ) -> float:
        """Estimate volatility with exponential decay toward recent observations."""
        if returns.size == 0:
            return 0.0

        safe_half_life = max(1, int(half_life))
        ages = np.arange(returns.size - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-np.log(2.0) * ages / safe_half_life)
        weighted_second_moment = np.average(np.square(returns), weights=weights)
        return np.float64(np.sqrt(max(weighted_second_moment, 0.0)))

    def _calculate_feature_instability(self, df: pd.DataFrame) -> dict[str, float]:
        """Measure instability of core features as rolling standard deviation."""
        metrics: dict[str, float] = {}
        candidate_cols = ("hurst_20", "vol_ratio_20", "rsi_20", "price_position_20")

        for col in candidate_cols:
            if col not in df.columns:
                metrics[f"{col}_instability"] = 0.0
                continue

            series = df[col].dropna()
            if series.empty:
                metrics[f"{col}_instability"] = 0.0
                continue

            rolling_std = series.rolling(REGIME_RECENT_WINDOW, min_periods=5).std()
            tail_mean = rolling_std.dropna().tail(REGIME_RECENT_WINDOW).mean()
            metrics[f"{col}_instability"] = np.float64(tail_mean) if pd.notna(tail_mean) else 0.0

        return metrics


# =============================================================================
# DASHBOARD GENERATION
# =============================================================================


class ConsolidationDashboard:
    """Render interactive dashboard artifacts for consolidation analysis."""

    def __init__(
        self,
        df: pd.DataFrame,
        consolidations: list[ConsolidationFeatures],
        probabilities: pd.Series,
        directions: pd.Series,
        params: OptimizationParams,
        metrics: dict[str, float],
        feature_importance: dict[str, float] | None = None,
    ):
        """
        Initialize the dashboard generator.

        Args:
            df: Feature-enriched OHLC dataframe.
            consolidations: Detected consolidation windows.
            probabilities: Breakout probability series.
            directions: Breakout direction series.
            params: Optimized detector parameters.
            metrics: Calculated performance metrics.
            feature_importance: Optional parameter importance map.
        """
        self.df = df.copy()
        self.consolidations = consolidations
        self.probabilities = probabilities.copy()
        self.directions = directions.copy()
        self.params = params
        self.metrics = metrics
        self.feature_importance = feature_importance or {}
        self.logger = logging.getLogger(__name__)

    def _require_plotly(self) -> tuple[Any, Any]:
        """Import Plotly lazily so the analytics pipeline stays usable without UI extras."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as exc:
            raise RuntimeError(
                "Plotly is required to generate the dashboard. Install 'plotly' and retry."
            ) from exc
        return go, make_subplots

    def _slice_plot_frame(self, days_back: int | None) -> pd.DataFrame:
        """Return the plotting window, defaulting to the full loaded dataset."""
        if self.df.empty:
            raise ValueError("Cannot build dashboard from an empty dataframe.")

        if days_back is None:
            return self.df.copy()

        safe_days_back = max(1, int(days_back))
        if not isinstance(self.df.index, pd.DatetimeIndex):
            return self.df.tail(min(len(self.df), safe_days_back)).copy()

        end_timestamp = self.df.index[-1]
        start_timestamp = end_timestamp - timedelta(days=safe_days_back)
        plot_df = self.df[self.df.index >= start_timestamp].copy()
        if plot_df.empty:
            plot_df = self.df.tail(min(len(self.df), safe_days_back)).copy()
        return plot_df

    def _format_timespan(self, span: pd.Timedelta) -> str:
        """Format a timedelta into a compact human-readable span."""
        total_seconds = max(int(span.total_seconds()), 0)
        days, remainder = divmod(total_seconds, 24 * 60 * 60)
        hours, remainder = divmod(remainder, 60 * 60)
        minutes, _ = divmod(remainder, 60)

        parts: list[str] = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 and days == 0:
            parts.append(f"{minutes}m")
        return " ".join(parts) if parts else "single bar"

    def _describe_plot_window(self, frame: pd.DataFrame) -> str:
        """Describe the number of bars and span rendered in a dashboard panel."""
        bar_count = len(frame)
        bar_label = "bar" if bar_count == 1 else "bars"

        if bar_count < 2 or not isinstance(frame.index, pd.DatetimeIndex):
            return f"{bar_count:,} {bar_label} from the loaded dataset"

        time_span = pd.Timestamp(frame.index[-1]) - pd.Timestamp(frame.index[0])
        return f"{bar_count:,} {bar_label} across {self._format_timespan(time_span)}"

    def _select_heatmap_dates(
        self,
        all_dates: list[pd.Timestamp],
        days_back: int | None,
    ) -> list[pd.Timestamp]:
        """Return heatmap dates using the full dataset unless explicitly constrained."""
        if days_back is None:
            return all_dates
        return all_dates[-max(1, int(days_back)) :]

    def _describe_forecast_horizon(self, lookahead: int | None) -> str:
        """Describe how much post-consolidation history the forecast panel uses."""
        latest_with_time = [cons for cons in self.consolidations if cons.timestamp_end is not None]
        if not latest_with_time:
            return "No detected consolidation zones"

        latest_cons = latest_with_time[-1]
        available_bars = max(0, len(self.df) - latest_cons.end_idx - 1)
        if lookahead is None:
            return f"All available post-zone bars ({available_bars})"

        effective_bars = min(available_bars, max(1, int(lookahead)))
        return f"{effective_bars} post-zone bars"

    def _build_range_selector_buttons(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Build range buttons that match the actual dataset span."""
        if len(frame) < 2 or not isinstance(frame.index, pd.DatetimeIndex):
            return []

        total_hours = (
            pd.Timestamp(frame.index[-1]) - pd.Timestamp(frame.index[0])
        ).total_seconds() / 3600.0
        buttons: list[dict[str, Any]] = []

        if total_hours >= 6.0:
            buttons.append(dict(count=6, label="6h", step="hour", stepmode="backward"))
        if total_hours >= 24.0:
            buttons.append(dict(count=1, label="1d", step="day", stepmode="backward"))
        if total_hours >= 24.0 * 7.0:
            buttons.append(dict(count=7, label="1w", step="day", stepmode="backward"))
        if total_hours >= 24.0 * 30.0:
            buttons.append(dict(count=1, label="1m", step="month", stepmode="backward"))
        if total_hours >= 24.0 * 90.0:
            buttons.append(dict(count=3, label="3m", step="month", stepmode="backward"))
        if total_hours >= 24.0 * 365.0:
            buttons.append(dict(count=1, label="1y", step="year", stepmode="backward"))

        buttons.append(dict(step="all"))
        return buttons

    def _infer_bar_width_ms(self, index: pd.Index) -> int | None:
        """Infer a readable bar width from the median timestamp spacing."""
        if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
            return None

        deltas = index.to_series().diff().dropna()
        if deltas.empty:
            return None

        median_seconds = deltas.dt.total_seconds().median()
        if pd.isna(median_seconds) or median_seconds <= 0:
            return None
        return int(median_seconds * 1000 * 0.8)

    def _filter_consolidations(
        self,
        start_timestamp: pd.Timestamp,
        limit: int | None = DASHBOARD_MAX_ZONES,
    ) -> list[ConsolidationFeatures]:
        """Return recent consolidations that overlap the current plot window."""
        recent = [
            cons
            for cons in self.consolidations
            if cons.timestamp_start is not None
            and cons.timestamp_end is not None
            and pd.Timestamp(cons.timestamp_end) >= start_timestamp
        ]
        if limit is None:
            return recent
        return recent[-max(1, int(limit)) :]

    def _build_consolidation_event_frame(self, start_timestamp: pd.Timestamp) -> pd.DataFrame:
        """Build a time-indexed frame of consolidation scores and component values."""
        rows: list[dict[str, float | datetime]] = []

        for cons in self.consolidations:
            if cons.timestamp_end is None:
                continue
            if pd.Timestamp(cons.timestamp_end) < start_timestamp:
                continue
            rows.append(
                {
                    "timestamp": cons.timestamp_end,
                    "score": cons.consolidation_score,
                    "attention": cons.attention_context_score,
                    "periodic": cons.periodic_context_score,
                    "scale": cons.scale_anchor_score,
                    "sutte": cons.sutte_score,
                    "interaction": cons.interaction_score,
                }
            )

        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame.from_records(rows)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        if frame.empty:
            return frame

        frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        return frame.set_index("timestamp")

    def _format_breakout_label(self, direction: int) -> str:
        """Return a readable label for breakout direction."""
        if direction > 0:
            return "Up breakout"
        if direction < 0:
            return "Down breakout"
        return "Standby"

    def _signal_tone(self, direction: int, visibility: str) -> str:
        """Map breakout direction and visibility to a dashboard tone."""
        if visibility == "subthreshold":
            return "watch"
        if direction > 0:
            return "support"
        if direction < 0:
            return "risk"
        return "neutral"

    def _extract_latest_signal_record(
        self,
        probabilities: pd.Series,
        directions: pd.Series,
        mask: pd.Series,
    ) -> dict[str, Any]:
        """Return the latest signal selected by a boolean mask."""
        aligned_mask = mask.reindex(probabilities.index).fillna(False).astype(bool)
        if not aligned_mask.any():
            return {
                "label": "Standby",
                "probability": 0.0,
                "timestamp": "No breakout signal",
                "timestamp_value": None,
                "direction": 0,
            }

        latest_timestamp = directions[aligned_mask].index[-1]
        latest_direction = int(directions.loc[latest_timestamp])
        latest_probability = np.float64(probabilities.loc[latest_timestamp])
        return {
            "label": self._format_breakout_label(latest_direction),
            "probability": latest_probability,
            "timestamp": pd.Timestamp(latest_timestamp).strftime("%Y-%m-%d %H:%M"),
            "timestamp_value": pd.Timestamp(latest_timestamp),
            "direction": latest_direction,
        }

    def _latest_signal_summary(
        self,
        probability_threshold: float = DASHBOARD_SIGNAL_PROBABILITY_THRESHOLD,
    ) -> dict[str, Any]:
        """Return active and raw breakout direction state for dashboard rendering."""
        threshold_value = np.float64(min(max(probability_threshold, 0.0), 1.0))
        probabilities = self.probabilities.reindex(self.df.index).fillna(0.0).astype(np.float64)
        directions = self.directions.reindex(self.df.index).fillna(0).astype(int)

        raw_mask = directions != 0
        active_mask = raw_mask & (probabilities >= threshold_value)
        active_up_count = int(((directions == 1) & active_mask).sum())
        active_down_count = int(((directions == -1) & active_mask).sum())
        raw_up_count = int((directions == 1).sum())
        raw_down_count = int((directions == -1).sum())

        active_record = self._extract_latest_signal_record(
            probabilities=probabilities,
            directions=directions,
            mask=active_mask,
        )
        raw_record = self._extract_latest_signal_record(
            probabilities=probabilities,
            directions=directions,
            mask=raw_mask,
        )

        if active_record["direction"] != 0:
            visibility = "active"
            label = str(active_record["label"])
            probability = np.float64(active_record["probability"])
            timestamp = str(active_record["timestamp"])
            timestamp_value = active_record["timestamp_value"]
            direction = int(active_record["direction"])
        elif raw_record["direction"] != 0:
            visibility = "subthreshold"
            label = "Standby"
            probability = 0.0
            timestamp = "No active breakout signal"
            timestamp_value = None
            direction = 0
        else:
            visibility = "none"
            label = "Standby"
            probability = 0.0
            timestamp = "No active breakout signal"
            timestamp_value = None
            direction = 0

        return {
            "label": label,
            "probability": probability,
            "timestamp": timestamp,
            "timestamp_value": timestamp_value,
            "direction": direction,
            "tone": self._signal_tone(direction, visibility),
            "visibility": visibility,
            "threshold": threshold_value,
            "active_up_count": active_up_count,
            "active_down_count": active_down_count,
            "active_signal_count": active_up_count + active_down_count,
            "raw_up_count": raw_up_count,
            "raw_down_count": raw_down_count,
            "raw_signal_count": raw_up_count + raw_down_count,
            "latest_raw_label": str(raw_record["label"]),
            "latest_raw_probability": np.float64(raw_record["probability"]),
            "latest_raw_timestamp": str(raw_record["timestamp"]),
            "latest_raw_timestamp_value": raw_record["timestamp_value"],
            "latest_raw_direction": int(raw_record["direction"]),
        }

    def create_full_dashboard(
        self,
        days_back: int | None = DASHBOARD_DAYS_BACK,
        height: int = DASHBOARD_HEIGHT,
    ) -> Any:
        """Create the main multi-panel dashboard figure.

        Panel order (top to bottom):
          1. Price + accepted zones + breakout signals  (primary decision surface)
          2. Directional conviction                     (the answer: which way, how strong)
          3. Acceptance score vs threshold              (why the model fired)
          4. Volatility and regime context              (environmental conditions)
          5. Detector component mix                     (internals, for diagnostics)
        """
        go, make_subplots = self._require_plotly()
        plot_df = self._slice_plot_frame(days_back)
        start_timestamp = pd.Timestamp(plot_df.index[0])
        recent_cons = self._filter_consolidations(start_timestamp)
        score_frame = self._build_consolidation_event_frame(start_timestamp)
        bar_width_ms = self._infer_bar_width_ms(plot_df.index)
        range_buttons = self._build_range_selector_buttons(plot_df)

        # Dark, neutral template so the plot background matches the HTML card surface.
        _TEMPLATE = "plotly_dark"
        _PAPER_BG = "#111827"
        _PLOT_BG = "#131c2e"
        _GRID = "rgba(255,255,255,0.06)"
        _TEXT = "#cbd5e1"
        _MUTED = "#64748b"

        fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.022,
            # Conviction panel gets slightly more room than score/regime panels.
            row_heights=[0.40, 0.15, 0.13, 0.15, 0.17],
            specs=[
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": True}],
                [{"secondary_y": False}],
            ],
        )

        # ── Panel 1: Price + accepted zones + breakout signals ─────────────────────
        # Primary decision surface. Zones rendered behind candles; signals on top.
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["open"],
                high=plot_df["high"],
                low=plot_df["low"],
                close=plot_df["close"],
                name="Price",
                increasing_line_color="#34d399",
                decreasing_line_color="#f87171",
                increasing_fillcolor="#064e3b",
                decreasing_fillcolor="#7f1d1d",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        for cons in recent_cons:
            # Zone opacity scales with score so high-conviction windows stand out.
            opacity = min(max(cons.consolidation_score, 0.05) * 0.30, 0.35)
            fig.add_vrect(
                x0=cons.timestamp_start,
                x1=cons.timestamp_end,
                fillcolor="#f59e0b",
                opacity=opacity,
                layer="below",
                line_width=0,
                row=1,
                col=1,
            )

        latest_cons = recent_cons[-3:]
        for cons in latest_cons:
            zone_window = self.df.iloc[cons.start_idx : cons.end_idx + 1]
            zone_high = np.float64(zone_window["high"].max())
            zone_low = np.float64(zone_window["low"].min())
            fig.add_trace(
                go.Scatter(
                    x=[cons.timestamp_start, cons.timestamp_end],
                    y=[zone_high, zone_high],
                    mode="lines",
                    line=dict(color="#fbbf24", width=1.4, dash="dash"),
                    name="Resistance",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[cons.timestamp_start, cons.timestamp_end],
                    y=[zone_low, zone_low],
                    mode="lines",
                    line=dict(color="#34d399", width=1.4, dash="dash"),
                    name="Support",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        plot_probs = self.probabilities.reindex(plot_df.index).fillna(0.0)
        plot_dirs = self.directions.reindex(plot_df.index).fillna(0).astype(int)
        signal_mask = (plot_probs >= DASHBOARD_SIGNAL_PROBABILITY_THRESHOLD) & (plot_dirs != 0)
        up_signals = plot_probs[signal_mask & (plot_dirs == 1)]
        down_signals = plot_probs[signal_mask & (plot_dirs == -1)]
        latest_signal = self._latest_signal_summary()

        if not up_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=up_signals.index,
                    y=plot_df.loc[up_signals.index, "high"] * 1.003,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=9 + up_signals.to_numpy(dtype=np.float64) * 10,
                        color="#34d399",
                        line=dict(color="#064e3b", width=1),
                    ),
                    name="Up signal",
                    text=[
                        f"Up breakout<br>Probability: {value:.1%}"
                        for value in up_signals.to_numpy(dtype=np.float64)
                    ],
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if not down_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=down_signals.index,
                    y=plot_df.loc[down_signals.index, "low"] * 0.997,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=9 + down_signals.to_numpy(dtype=np.float64) * 10,
                        color="#f87171",
                        line=dict(color="#7f1d1d", width=1),
                    ),
                    name="Down signal",
                    text=[
                        f"Down breakout<br>Probability: {value:.1%}"
                        for value in down_signals.to_numpy(dtype=np.float64)
                    ],
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if latest_signal["visibility"] == "active" and latest_signal["timestamp_value"] is not None:
            latest_timestamp = pd.Timestamp(latest_signal["timestamp_value"])
            if latest_timestamp in plot_df.index:
                latest_direction = int(latest_signal["direction"])
                latest_color = "#34d399" if latest_direction > 0 else "#f87171"
                latest_label = str(latest_signal["label"]).lower()
                latest_probability = np.float64(latest_signal["probability"])
                latest_row = plot_df.loc[plot_df.index == latest_timestamp].iloc[-1]
                latest_y = (
                    np.float64(latest_row["high"]) * 1.006
                    if latest_direction > 0
                    else np.float64(latest_row["low"]) * 0.994
                )
                fig.add_vline(
                    x=latest_timestamp,
                    line_dash="dot",
                    line_color=latest_color,
                    line_width=1.4,
                    opacity=0.85,
                    row=1,
                    col=1,
                )
                fig.add_annotation(
                    x=latest_timestamp,
                    y=latest_y,
                    text=f"Latest {latest_label} {latest_probability:.0%}",
                    showarrow=True,
                    arrowcolor=latest_color,
                    arrowhead=2,
                    ax=30,
                    ay=-40 if latest_direction > 0 else 40,
                    bgcolor="rgba(17, 24, 39, 0.92)",
                    bordercolor=latest_color,
                    borderpad=6,
                    font=dict(color=latest_color, size=11),
                    row=1,
                    col=1,
                )

        # ── Panel 2: Directional conviction ─────────────────────────────────────────
        # Most actionable read. Up bars above zero, down bars below. Threshold lines
        # show where the model considers a signal confirmed.
        up_bar_kwargs: dict[str, Any] = {
            "x": plot_probs[plot_dirs == 1].index,
            "y": plot_probs[plot_dirs == 1],
            "name": "Up conviction",
            "marker_color": "#34d399",
            "opacity": 0.82,
        }
        if bar_width_ms is not None:
            up_bar_kwargs["width"] = bar_width_ms
        if len(up_bar_kwargs["x"]) > 0:
            fig.add_trace(go.Bar(**up_bar_kwargs), row=2, col=1)

        down_bar_kwargs: dict[str, Any] = {
            "x": plot_probs[plot_dirs == -1].index,
            "y": -plot_probs[plot_dirs == -1],
            "name": "Down conviction",
            "marker_color": "#f87171",
            "opacity": 0.82,
        }
        if bar_width_ms is not None:
            down_bar_kwargs["width"] = bar_width_ms
        if len(down_bar_kwargs["x"]) > 0:
            fig.add_trace(go.Bar(**down_bar_kwargs), row=2, col=1)

        fig.add_hline(y=0.0, line_color="rgba(255,255,255,0.20)", line_width=1, row=2, col=1)
        fig.add_hline(
            y=DASHBOARD_SIGNAL_PROBABILITY_THRESHOLD,
            line_dash="dot",
            line_color="rgba(251,191,36,0.55)",
            annotation_text=f"threshold {DASHBOARD_SIGNAL_PROBABILITY_THRESHOLD:.0%}",
            annotation_font_color="#fbbf24",
            annotation_font_size=10,
            row=2,
            col=1,
        )
        fig.add_hline(
            y=-DASHBOARD_SIGNAL_PROBABILITY_THRESHOLD,
            line_dash="dot",
            line_color="rgba(251,191,36,0.55)",
            row=2,
            col=1,
        )

        # ── Panel 3: Acceptance score vs threshold ───────────────────────────────────
        # Explains which windows passed the detector gate and by how much.
        if not score_frame.empty:
            fig.add_trace(
                go.Scatter(
                    x=score_frame.index,
                    y=score_frame["score"],
                    mode="lines+markers",
                    line=dict(color="#fbbf24", width=2.0),
                    marker=dict(size=5, color="#f59e0b", line=dict(color="#78350f", width=0.8)),
                    fill="tozeroy",
                    fillcolor="rgba(245, 158, 11, 0.10)",
                    name="Acceptance score",
                ),
                row=3,
                col=1,
            )
            fig.add_hline(
                y=self.params.consolidation_threshold,
                line_dash="dash",
                line_color="#f87171",
                opacity=0.75,
                annotation_text=f"gate {self.params.consolidation_threshold:.3f}",
                annotation_font_color="#f87171",
                annotation_font_size=10,
                row=3,
                col=1,
            )

        # ── Panel 4: Volatility and regime context ───────────────────────────────────
        # Environmental conditions. ATR on primary axis, vol ratio / Hurst / Sutte on
        # secondary so the scales don't fight each other.
        atr_like = (
            plot_df["hl_range"].rolling(20, min_periods=5).mean()
            if "hl_range" in plot_df.columns
            else plot_df["high"].sub(plot_df["low"]).rolling(20, min_periods=5).mean()
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=atr_like,
                mode="lines",
                line=dict(color="#60a5fa", width=2.0),
                name="ATR-20",
            ),
            row=4,
            col=1,
            secondary_y=False,
        )

        if "vol_ratio_20" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df["vol_ratio_20"],
                    mode="lines",
                    line=dict(color="#fbbf24", width=1.6, dash="dash"),
                    name="Vol ratio 20",
                ),
                row=4,
                col=1,
                secondary_y=True,
            )

        if "hurst_20" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df["hurst_20"],
                    mode="lines",
                    line=dict(color="#94a3b8", width=1.4, dash="dot"),
                    name="Hurst 20",
                ),
                row=4,
                col=1,
                secondary_y=True,
            )

        if "sutte_signal" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df["sutte_signal"],
                    mode="lines",
                    line=dict(color="#c084fc", width=1.4, dash="dot"),
                    name="Sutte signal",
                ),
                row=4,
                col=1,
                secondary_y=True,
            )

        if self.metrics.get("regime_shift_flag", 0.0) > 0 and len(plot_df) > 2:
            regime_window = max(3, min(12, len(plot_df) // 8))
            fig.add_vrect(
                x0=plot_df.index[-regime_window],
                x1=plot_df.index[-1],
                fillcolor="#f87171",
                opacity=0.10,
                layer="below",
                line_width=0,
                row=4,
                col=1,
            )

        # ── Panel 5: Detector component mix ─────────────────────────────────────────
        # Deep internals. Rendered last so dominant panels stay at the top.
        if not score_frame.empty:
            component_frame = (
                score_frame
                if DASHBOARD_MAX_COMPONENT_POINTS is None
                else score_frame.tail(max(1, int(DASHBOARD_MAX_COMPONENT_POINTS)))
            )
            component_specs = (
                ("attention", "#34d399", "Similarity"),
                ("periodic", "#60a5fa", "Periodic"),
                ("scale", "#fbbf24", "Scale"),
                ("sutte", "#c084fc", "Sutte"),
                ("interaction", "#fb923c", "Interaction"),
            )
            for column, color, label in component_specs:
                fig.add_trace(
                    go.Scatter(
                        x=component_frame.index,
                        y=component_frame[column],
                        mode="lines+markers",
                        line=dict(color=color, width=1.8),
                        marker=dict(size=4),
                        name=label,
                    ),
                    row=5,
                    col=1,
                )

        # ── Layout ───────────────────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(
                text=(
                    "Consolidation structure dashboard"
                    f"<br><sup>{plot_df.index[0].strftime('%Y-%m-%d %H:%M')} to "
                    f"{plot_df.index[-1].strftime('%Y-%m-%d %H:%M')} | "
                    f"Accepted windows: {len(self.consolidations)} | "
                    f"Triggers: {int(self.metrics.get('breakout_signals', 0))} | "
                    f"Simulated Sharpe: {np.float64(self.metrics.get('simulated_sharpe', 0.0)):.2f}</sup>"
                ),
                font=dict(size=16, color="#e2e8f0", family="'IBM Plex Mono', monospace"),
            ),
            height=height,
            template=_TEMPLATE,
            # Keep unified hover only on the price panel; sub-panels use closest
            # to avoid a 5-row label stack that obscures the chart.
            hovermode="x unified",
            barmode="relative",
            paper_bgcolor=_PAPER_BG,
            plot_bgcolor=_PLOT_BG,
            font=dict(color=_TEXT, family="'IBM Plex Mono', 'Courier New', monospace"),
            margin=dict(l=68, r=68, t=90, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="right",
                x=1.0,
                bgcolor="rgba(17, 24, 39, 0.72)",
                bordercolor="rgba(255,255,255,0.08)",
                borderwidth=1,
                font=dict(size=11),
            ),
        )

        # Range selector lives on the price panel where the trader's eye is.
        _xaxis_common = dict(showgrid=True, gridcolor=_GRID, rangeslider=dict(visible=False))
        fig.update_xaxes(**_xaxis_common, row=1, col=1)
        fig.update_xaxes(**_xaxis_common, row=2, col=1)
        fig.update_xaxes(**_xaxis_common, row=3, col=1)
        fig.update_xaxes(**_xaxis_common, row=4, col=1)
        fig.update_xaxes(**_xaxis_common, row=5, col=1)

        if range_buttons:
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=range_buttons,
                    bgcolor="rgba(255,255,255,0.06)",
                    activecolor="rgba(251,191,36,0.22)",
                    font=dict(size=11, color=_TEXT),
                ),
                row=1,
                col=1,
            )

        _yaxis_common = dict(showgrid=True, gridcolor=_GRID)
        fig.update_yaxes(title_text="Price", **_yaxis_common, row=1, col=1)
        fig.update_yaxes(
            title_text="Conviction",
            **_yaxis_common,
            range=[-1.05, 1.05],
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.18)",
            zerolinewidth=1,
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Score", **_yaxis_common, row=3, col=1)
        fig.update_yaxes(
            title_text="ATR-20",
            **_yaxis_common,
            row=4,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Ratio / Hurst",
            showgrid=False,
            row=4,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Component", **_yaxis_common, row=5, col=1)
        return fig

    def create_consolidation_heatmap(self, days_back: int | None = DASHBOARD_HEATMAP_DAYS) -> Any:
        """Create a heatmap of consolidation strength by hour and date."""
        go, _ = self._require_plotly()
        hourly_scores: dict[tuple[pd.Timestamp, int], list[float]] = {}

        for cons in self.consolidations:
            if cons.timestamp_end is None:
                continue
            ts = pd.Timestamp(cons.timestamp_end)
            key = (ts.normalize(), int(ts.hour))
            hourly_scores.setdefault(key, []).append(np.float64(cons.consolidation_score))

        if not hourly_scores:
            fig = go.Figure()
            fig.add_annotation(
                text="No consolidations available for heatmap rendering.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=15),
            )
            fig.update_layout(
                title="Accepted-window strength heatmap",
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#131c2e",
                font=dict(color="#cbd5e1", family="'IBM Plex Mono', monospace"),
                height=430,
            )
            return fig

        all_dates = sorted({day for day, _ in hourly_scores})
        dates = self._select_heatmap_dates(all_dates, days_back)
        hours = list(range(24))
        heatmap_values = []

        for date_key in dates:
            row: list[float] = []
            for hour in hours:
                scores = hourly_scores.get((date_key, hour))
                row.append(np.float64(np.mean(scores)) if scores else np.nan)
            heatmap_values.append(row)

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_values,
                x=[f"{hour:02d}:00" for hour in hours],
                y=[pd.Timestamp(day).strftime("%Y-%m-%d") for day in dates],
                colorscale=[
                    [0.0, "#0f172a"],
                    [0.45, "#b45309"],
                    [1.0, "#34d399"],
                ],
                zmin=0.0,
                zmax=1.0,
                hoverongaps=False,
                colorbar=dict(
                    title=dict(text="Score", font=dict(color="#94a3b8")),
                    tickfont=dict(color="#cbd5e1"),
                ),
            )
        )
        fig.update_layout(
            title="Accepted-window strength by hour",
            xaxis_title="Hour of day",
            yaxis_title="Date",
            yaxis_type="category",
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#131c2e",
            font=dict(color="#cbd5e1", family="'IBM Plex Mono', monospace"),
            margin=dict(l=60, r=40, t=60, b=50),
            height=430,
        )
        return fig

    def create_forecast_chart(self, lookahead: int | None = DASHBOARD_FORECAST_BARS) -> Any:
        """Create a focused view of the latest consolidation and subsequent price path."""
        go, _ = self._require_plotly()
        fig = go.Figure()

        latest_with_time = [cons for cons in self.consolidations if cons.timestamp_end is not None]
        if not latest_with_time:
            fig.add_annotation(
                text="No consolidation windows were detected.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=15),
            )
            fig.update_layout(
                title="Latest accepted-window follow-through",
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#131c2e",
                font=dict(color="#cbd5e1", family="'IBM Plex Mono', monospace"),
                height=430,
            )
            return fig

        latest_cons = latest_with_time[-1]
        context_start = max(0, latest_cons.start_idx - latest_cons.duration)
        if lookahead is None:
            future_end = len(self.df) - 1
        else:
            future_end = min(len(self.df) - 1, latest_cons.end_idx + max(1, int(lookahead)))

        history_window = self.df.iloc[context_start : latest_cons.end_idx + 1]
        history_window = history_window[history_window["close"] > 0]
        post_window = self.df.iloc[latest_cons.end_idx + 1 : future_end + 1]
        post_window = post_window[post_window["close"] > 0]
        zone_window = self.df.iloc[latest_cons.start_idx : latest_cons.end_idx + 1]
        zone_high = np.float64(zone_window["high"].max())
        zone_low = np.float64(zone_window["low"].min())
        zone_mid = (zone_high + zone_low) / 2.0

        fig.add_trace(
            go.Scatter(
                x=history_window.index,
                y=history_window["close"],
                mode="lines",
                line=dict(color="#94a3b8", width=1.8),
                name="Historical close",
            )
        )
        fig.add_vrect(
            x0=self.df.index[latest_cons.start_idx],
            x1=self.df.index[latest_cons.end_idx],
            fillcolor="#f59e0b",
            opacity=0.16,
            layer="below",
            line_width=0,
        )
        fig.add_hline(
            y=zone_high, line_dash="dash", line_color="#b45309", annotation_text="Resistance"
        )
        fig.add_hline(y=zone_low, line_dash="dash", line_color="#0f766e", annotation_text="Support")
        fig.add_hline(y=zone_mid, line_dash="dot", line_color="#94a3b8", annotation_text="Midline")

        if not post_window.empty:
            fig.add_trace(
                go.Scatter(
                    x=post_window.index,
                    y=post_window["close"],
                    mode="lines+markers",
                    line=dict(color="#34d399", width=2.4),
                    marker=dict(size=5, color="#34d399"),
                    name="Post-zone path",
                )
            )

            post_probs = self.probabilities.iloc[latest_cons.end_idx + 1 : future_end + 1]
            post_dirs = self.directions.iloc[latest_cons.end_idx + 1 : future_end + 1]
            active_mask = (post_probs >= DASHBOARD_SIGNAL_PROBABILITY_THRESHOLD) & (post_dirs != 0)

            if active_mask.any():
                active_probs = post_probs[active_mask]
                active_dirs = post_dirs[active_mask]
                up_mask = active_dirs == 1
                down_mask = active_dirs == -1

                if up_mask.any():
                    up_index = active_probs[up_mask].index
                    fig.add_trace(
                        go.Scatter(
                            x=up_index,
                            y=self.df.loc[up_index, "close"],
                            mode="markers",
                            marker=dict(
                                symbol="triangle-up",
                                size=10 + active_probs[up_mask].to_numpy(dtype=np.float64) * 10,
                                color="#10b981",
                                line=dict(color="#064e3b", width=1),
                            ),
                            name="Up breakout",
                            hovertemplate="Probability: %{text}<extra></extra>",
                            text=[f"{value:.1%}" for value in active_probs[up_mask]],
                        )
                    )

                if down_mask.any():
                    down_index = active_probs[down_mask].index
                    fig.add_trace(
                        go.Scatter(
                            x=down_index,
                            y=self.df.loc[down_index, "close"],
                            mode="markers",
                            marker=dict(
                                symbol="triangle-down",
                                size=10 + active_probs[down_mask].to_numpy(dtype=np.float64) * 10,
                                color="#ef4444",
                                line=dict(color="#7f1d1d", width=1),
                            ),
                            name="Down breakout",
                            hovertemplate="Probability: %{text}<extra></extra>",
                            text=[f"{value:.1%}" for value in active_probs[down_mask]],
                        )
                    )
        else:
            fig.add_annotation(
                text="No bars are available after the latest consolidation yet.",
                x=0.5,
                y=0.1,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=13, color="#475569"),
            )

        fig.update_layout(
            title=f"Latest accepted-window follow-through ({latest_cons.duration} bars)",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#131c2e",
            font=dict(color="#cbd5e1", family="'IBM Plex Mono', monospace"),
            margin=dict(l=60, r=40, t=60, b=50),
            height=430,
            hovermode="x unified",
        )
        return fig

    def _latest_zone_snapshot(self) -> dict[str, str | float | int]:
        """Return timing and maturity context for the latest accepted window."""
        latest_with_time = [cons for cons in self.consolidations if cons.timestamp_end is not None]
        if not latest_with_time:
            return {
                "available": "no",
                "window": "No completed accepted window",
                "duration": 0,
                "post_bars": 0,
                "score": 0.0,
                "maturity": "None",
            }

        latest_cons = latest_with_time[-1]
        post_bars = max(0, len(self.df) - latest_cons.end_idx - 1)
        start_label = (
            pd.Timestamp(latest_cons.timestamp_start).strftime("%Y-%m-%d %H:%M")
            if latest_cons.timestamp_start is not None
            else "Unknown start"
        )
        end_label = pd.Timestamp(latest_cons.timestamp_end).strftime("%Y-%m-%d %H:%M")

        if post_bars >= 8:
            maturity = "Developed"
        elif post_bars >= 4:
            maturity = "Forming"
        elif post_bars >= 1:
            maturity = "Early"
        else:
            maturity = "Unconfirmed"

        return {
            "available": "yes",
            "window": f"{start_label} to {end_label}",
            "duration": int(latest_cons.duration),
            "post_bars": post_bars,
            "score": np.float64(latest_cons.consolidation_score),
            "maturity": maturity,
        }

    def _benchmark_verdict_label(self) -> str:
        """Summarize whether the current model clears the logistic baseline."""
        if "benchmark_brier_improvement" not in self.metrics:
            return "Unavailable"

        improvement = np.float64(self.metrics.get("benchmark_brier_improvement", 0.0))
        if improvement > 0.0:
            return f"Better by {improvement:.4f} Brier"
        if improvement < 0.0:
            return f"Worse by {abs(improvement):.4f} Brier"
        return "Tied on Brier"

    def _build_dashboard_brief(
        self,
        latest_signal: dict[str, Any],
        latest_zone: dict[str, str | float | int],
    ) -> dict[str, Any]:
        """Build a direct answer to the latest consolidation and breakout question."""
        sharpe = np.float64(self.metrics.get("simulated_sharpe", 0.0))
        latest_visibility = str(latest_signal["visibility"])
        latest_probability = np.float64(latest_signal["probability"])
        latest_label = str(latest_signal["label"])
        latest_timestamp = str(latest_signal["timestamp"])
        latest_direction = int(latest_signal["direction"])
        latest_raw_label = str(latest_signal["latest_raw_label"])
        latest_raw_probability = np.float64(latest_signal["latest_raw_probability"])
        latest_raw_timestamp = str(latest_signal["latest_raw_timestamp"])
        latest_raw_direction = int(latest_signal["latest_raw_direction"])
        threshold = np.float64(latest_signal["threshold"])
        active_up_count = int(latest_signal["active_up_count"])
        active_down_count = int(latest_signal["active_down_count"])
        raw_up_count = int(latest_signal["raw_up_count"])
        raw_down_count = int(latest_signal["raw_down_count"])

        has_zone = str(latest_zone["available"]) == "yes"
        zone_duration = int(latest_zone["duration"])
        post_bars = int(latest_zone["post_bars"])
        latest_zone_score = np.float64(latest_zone["score"])
        latest_zone_window = str(latest_zone["window"])
        latest_zone_maturity = str(latest_zone["maturity"])

        if latest_direction > 0:
            confirmed_side = "Up"
        elif latest_direction < 0:
            confirmed_side = "Down"
        else:
            confirmed_side = "None"

        if latest_raw_direction > 0:
            raw_side = "Up lean"
        elif latest_raw_direction < 0:
            raw_side = "Down lean"
        else:
            raw_side = "Undecided"

        if not has_zone:
            setup_state = "No active setup"
            stance_tone = "neutral"
            trust_label = "No zone"
            consolidation_answer = "No"
            breakout_side = "None"
            confidence_label = "n/a"
            latest_signal_time = "n/a"
            headline = "No accepted consolidation is active right now."
            summary = "The detector does not currently have a live zone to trade from."
        elif latest_visibility == "active":
            setup_state = "Breaking up" if latest_direction > 0 else "Breaking down"
            stance_tone = "support" if latest_direction > 0 else "risk"
            trust_label = "Confirmed"
            consolidation_answer = "Yes"
            breakout_side = confirmed_side
            confidence_label = f"{latest_probability:.1%}"
            latest_signal_time = latest_timestamp
            headline = (
                f"Yes. The latest accepted consolidation is breaking {confirmed_side.lower()}."
            )
            summary = (
                f"Confirmed {latest_label.lower()} at {latest_probability:.1%} on "
                f"{latest_timestamp}. Price has moved beyond the latest accepted zone."
            )
        elif post_bars == 0:
            setup_state = "Still consolidating"
            stance_tone = "watch"
            trust_label = "Inside zone"
            consolidation_answer = "Yes"
            breakout_side = raw_side
            confidence_label = (
                f"{latest_raw_probability:.1%} raw" if latest_raw_direction != 0 else "n/a"
            )
            latest_signal_time = latest_raw_timestamp if latest_raw_direction != 0 else "n/a"
            headline = "Yes. Price is still inside the latest accepted consolidation."
            if latest_raw_direction != 0:
                summary = (
                    f"The current raw lean is {latest_raw_label.lower()} at "
                    f"{latest_raw_probability:.1%}, but it is still below the {threshold:.0%} "
                    "confirmation threshold."
                )
            else:
                summary = "No directional breakout signal is active yet."
        elif latest_visibility == "subthreshold":
            setup_state = "Waiting for confirmation"
            stance_tone = "watch"
            trust_label = "Below trigger"
            consolidation_answer = "Recent"
            breakout_side = raw_side
            confidence_label = (
                f"{latest_raw_probability:.1%} raw" if latest_raw_direction != 0 else "n/a"
            )
            latest_signal_time = latest_raw_timestamp if latest_raw_direction != 0 else "n/a"
            headline = (
                f"The latest consolidation has ended, but the breakout is not confirmed above "
                f"{threshold:.0%}."
            )
            if latest_raw_direction != 0:
                summary = (
                    f"The latest raw signal is {latest_raw_label.lower()} at "
                    f"{latest_raw_probability:.1%} on {latest_raw_timestamp}."
                )
            else:
                summary = "The latest zone has ended without a current directional trigger."
        else:
            setup_state = "Recent zone, no trigger"
            stance_tone = "neutral"
            trust_label = "No trigger"
            consolidation_answer = "Recent"
            breakout_side = "None"
            confidence_label = "n/a"
            latest_signal_time = "n/a"
            headline = "A recent consolidation exists, but there is no current breakout signal."
            summary = "The model has no active directional read from the latest accepted zone."

        summary_rows = [
            ("Is this a consolidation?", consolidation_answer),
            ("Setup state", setup_state),
            ("Breakout side", breakout_side),
            ("Confidence", confidence_label),
            ("Latest signal time", latest_signal_time),
            ("Zone score", "n/a" if not has_zone else f"{latest_zone_score:.3f}"),
            ("Zone duration", "n/a" if not has_zone else f"{zone_duration} bars"),
            ("Post-zone bars", "n/a" if not has_zone else str(post_bars)),
            ("Active up / down", f"{active_up_count} / {active_down_count}"),
            ("Raw up / down", f"{raw_up_count} / {raw_down_count}"),
            ("Display threshold", f"{threshold:.0%}"),
            ("Simulated Sharpe", f"{sharpe:.2f}"),
        ]
        benchmark_verdict = self._benchmark_verdict_label()
        if benchmark_verdict != "Unavailable":
            summary_rows.append(("Benchmark vs baseline", benchmark_verdict))

        return {
            "setup_state": setup_state,
            "stance_tone": stance_tone,
            "trust_label": trust_label,
            "headline": headline,
            "summary": summary,
            "summary_rows": summary_rows,
            "consolidation_answer": consolidation_answer,
            "breakout_side": breakout_side,
            "confidence_label": confidence_label,
            "latest_signal_time": latest_signal_time,
            "zone_score": "n/a" if not has_zone else f"{latest_zone_score:.3f}",
            "zone_duration": "n/a" if not has_zone else f"{zone_duration} bars",
            "post_zone_bars": "n/a" if not has_zone else str(post_bars),
            "active_breakout_mix": f"{active_up_count} / {active_down_count}",
            "raw_breakout_mix": f"{raw_up_count} / {raw_down_count}",
            "display_threshold": f"{threshold:.0%}",
            "benchmark_verdict": benchmark_verdict,
            "latest_zone_window": latest_zone_window,
            "latest_zone_maturity": latest_zone_maturity,
        }

    def _render_executive_panel(self, brief: dict[str, Any]) -> str:
        """Render the direct-answer panel for the current run."""
        rendered_rows = []
        for label, value in brief["summary_rows"]:
            rendered_rows.append(
                f"""
                <div class="summary-metric">
                    <dt>{html.escape(str(label))}</dt>
                    <dd>{html.escape(str(value))}</dd>
                </div>
                """
            )

        return f"""
        <article class="summary-card summary-card-primary tone-{brief["stance_tone"]}">
            <div class="summary-header">
                <div>
                    <div class="panel-kicker">Direct Answer</div>
                    <h2 class="summary-title">Is this a consolidation, and which way is it breaking?</h2>
                </div>
                <div class="status-chip status-{brief["stance_tone"]}">
                    {html.escape(str(brief["trust_label"]))}
                </div>
            </div>
            <p class="summary-copy">{html.escape(str(brief["headline"]))}</p>
            <p class="summary-copy summary-copy-muted">{html.escape(str(brief["summary"]))}</p>
            <dl class="summary-metrics">
                {"".join(rendered_rows)}
            </dl>
            <div class="summary-footer">
                <span>Latest window: {html.escape(str(brief["latest_zone_window"]))}</span>
                <span>Maturity: {html.escape(str(brief["latest_zone_maturity"]))}</span>
                <span>Display threshold: {html.escape(str(brief["display_threshold"]))}</span>
            </div>
        </article>
        """

    def _render_caution_panel(self, brief: dict[str, Any]) -> str:
        """Render the highest-priority caveats for the run."""
        rendered_points = []
        for point in brief["regret_points"]:
            rendered_points.append(f"<li>{html.escape(str(point))}</li>")

        return f"""
        <article class="summary-card summary-card-secondary tone-{brief["stance_tone"]}">
            <div class="panel-kicker">What you'd regret not knowing</div>
            <h2 class="summary-title">The contradiction summary</h2>
            <ul class="summary-list">
                {"".join(rendered_points)}
            </ul>
        </article>
        """

    def _render_direction_panel(self, brief: dict[str, Any]) -> str:
        """Render the top-line setup answer in the hero panel."""
        return f"""
        <aside class="hero-status tone-{html.escape(str(brief["stance_tone"]))}">
            <div class="hero-status-label">Current setup</div>
            <div class="hero-status-value">{html.escape(str(brief["setup_state"]))}</div>
            <p class="hero-status-copy">{html.escape(str(brief["headline"]))}</p>
            <div class="hero-status-grid">
                <div class="hero-status-metric">
                    <span>Consolidation</span>
                    <strong>{html.escape(str(brief["consolidation_answer"]))}</strong>
                </div>
                <div class="hero-status-metric">
                    <span>Breakout side</span>
                    <strong>{html.escape(str(brief["breakout_side"]))}</strong>
                </div>
                <div class="hero-status-metric">
                    <span>Confidence</span>
                    <strong>{html.escape(str(brief["confidence_label"]))}</strong>
                </div>
                <div class="hero-status-metric">
                    <span>Latest signal time</span>
                    <strong>{html.escape(str(brief["latest_signal_time"]))}</strong>
                </div>
            </div>
        </aside>
        """

    def _render_snapshot_cards(self, brief: dict[str, Any]) -> str:
        """Render compact cards that support the top-line answer."""
        cards = [
            (
                "Breakout side",
                str(brief["breakout_side"]),
                "Confirmed side or best raw lean from the latest setup",
            ),
            (
                "Confidence",
                str(brief["confidence_label"]),
                "Confirmed probability or raw lean strength",
            ),
            (
                "Zone score",
                str(brief["zone_score"]),
                "Latest accepted consolidation score",
            ),
            (
                "Post-zone bars",
                str(brief["post_zone_bars"]),
                "0 means price is still inside the latest zone",
            ),
            (
                "Active up / down",
                str(brief["active_breakout_mix"]),
                f"Signals above the {brief['display_threshold']} display threshold",
            ),
            (
                "Raw up / down",
                str(brief["raw_breakout_mix"]),
                "All non-zero model directions in the loaded window",
            ),
        ]

        rendered_cards = []
        for label, value, subtitle in cards:
            rendered_cards.append(
                f"""
                <article class="metric-card tone-neutral">
                    <div class="metric-label">{html.escape(label)}</div>
                    <div class="metric-value">{html.escape(value)}</div>
                    <div class="metric-subtitle">{html.escape(subtitle)}</div>
                </article>
                """
            )
        return "".join(rendered_cards)

    def _render_component_table(self) -> str:
        """Render component averages alongside learned parameter weights."""
        component_rows = [
            (
                "Similarity context",
                np.float64(self.metrics.get("mean_attention_context_score", 0.0)),
                np.float64(self.params.weight_attention),
                "Similarity to prior compression windows",
            ),
            (
                "Periodic context",
                np.float64(self.metrics.get("mean_periodic_context_score", 0.0)),
                np.float64(self.params.weight_periodic),
                "Time-of-day and weekly structure",
            ),
            (
                "Scale anchor",
                np.float64(self.metrics.get("mean_scale_anchor_score", 0.0)),
                np.float64(self.params.weight_scale),
                "Local autoregressive fit quality",
            ),
            (
                "Sutte structure",
                np.float64(self.metrics.get("mean_sutte_score", 0.0)),
                np.float64(self.params.weight_sutte),
                "Internal candle balance and directional conviction",
            ),
            (
                "Interaction",
                np.float64(self.metrics.get("mean_interaction_score", 0.0)),
                np.float64(self.params.weight_position) * 0.5,
                "Candle compression inside the zone",
            ),
        ]

        rendered_rows = []
        for label, mean_score, weight, description in component_rows:
            contribution = mean_score * weight * 100.0
            rendered_rows.append(
                f"""
                <tr>
                    <td>{label}</td>
                    <td>{mean_score:.3f}</td>
                    <td>{weight:.3f}</td>
                    <td>{contribution:.1f}%</td>
                    <td>{description}</td>
                </tr>
                """
            )

        return (
            """
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Mean Score</th>
                        <th>Weight</th>
                        <th>Weighted Blend</th>
                        <th>Intent</th>
                    </tr>
                </thead>
                <tbody>
            """
            + "".join(rendered_rows)
            + """
                </tbody>
            </table>
            """
        )

    def _render_benchmark_table(self) -> str:
        """Render the simple baseline comparison used to falsify the complex model."""
        if "benchmark_model_brier" not in self.metrics:
            return """
            <div class="empty-state">
                Baseline benchmarking is unavailable for this run.
            </div>
            """

        rows = [
            (
                "Current detector",
                np.float64(self.metrics.get("benchmark_model_brier", 0.0)),
                np.float64(self.metrics.get("benchmark_model_answered_accuracy", 0.0)),
                np.float64(self.metrics.get("benchmark_model_coverage", 0.0)),
                "Consolidation detector plus breakout scorer",
            ),
            (
                "Logistic baseline",
                np.float64(self.metrics.get("benchmark_baseline_brier", 0.0)),
                np.float64(self.metrics.get("benchmark_baseline_answered_accuracy", 0.0)),
                np.float64(self.metrics.get("benchmark_baseline_coverage", 0.0)),
                "Regularized logistic regression on compact price features",
            ),
        ]

        rendered_rows = []
        for label, brier_score, answered_accuracy, coverage, description in rows:
            rendered_rows.append(
                f"""
                <tr>
                    <td>{html.escape(label)}</td>
                    <td>{brier_score:.4f}</td>
                    <td>{answered_accuracy:.1%}</td>
                    <td>{coverage:.1%}</td>
                    <td>{description}</td>
                </tr>
                """
            )

        verdict = html.escape(self._benchmark_verdict_label())
        sample_count = int(np.float64(self.metrics.get("benchmark_sample_count", 0.0)))
        return (
            f"""
            <div class="table-note">
                Lower Brier is better. Answered accuracy is measured only on bars where the model
                emitted a direction. Sample count: {sample_count}. Verdict: {verdict}.
            </div>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Brier</th>
                        <th>Answered Accuracy</th>
                        <th>Coverage</th>
                        <th>Intent</th>
                    </tr>
                </thead>
                <tbody>
            """
            + "".join(rendered_rows)
            + """
                </tbody>
            </table>
            """
        )

    def generate_dashboard_html(
        self,
        output_path: str | Path | None = None,
        days_back: int | None = DASHBOARD_DAYS_BACK,
        height: int = DASHBOARD_HEIGHT,
    ) -> Path:
        """Generate a styled HTML dashboard and return its filesystem path."""
        plot_df = self._slice_plot_frame(days_back)
        plot_window_label = self._describe_plot_window(plot_df)
        heatmap_window_label = (
            "Entire loaded dataset"
            if DASHBOARD_HEATMAP_DAYS is None
            else f"Last {int(DASHBOARD_HEATMAP_DAYS)} dates with consolidations"
        )
        forecast_horizon_label = self._describe_forecast_horizon(DASHBOARD_FORECAST_BARS)

        main_fig = self.create_full_dashboard(days_back=days_back, height=height)
        heatmap_fig = self.create_consolidation_heatmap(days_back=DASHBOARD_HEATMAP_DAYS)
        forecast_fig = self.create_forecast_chart(lookahead=DASHBOARD_FORECAST_BARS)
        latest_signal = self._latest_signal_summary()
        latest_zone = self._latest_zone_snapshot()
        dashboard_brief = self._build_dashboard_brief(latest_signal, latest_zone)
        direction_panel_html = self._render_direction_panel(dashboard_brief)
        executive_panel_html = self._render_executive_panel(dashboard_brief)
        snapshot_cards_html = self._render_snapshot_cards(dashboard_brief)
        component_table_html = self._render_component_table()
        benchmark_table_html = self._render_benchmark_table()
        caution_panel_html = (
            self._render_caution_panel(dashboard_brief)
            if "regret_points" in dashboard_brief
            else ""
        )

        if output_path is None:
            output_path = Path(DASHBOARD_OUTPUT_DIR) / DASHBOARD_OUTPUT_NAME

        output = Path(output_path).expanduser()
        if not output.is_absolute():
            output = Path.cwd() / output
        output.parent.mkdir(parents=True, exist_ok=True)

        plot_config = {
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"],
        }

        main_chart_html = main_fig.to_html(
            include_plotlyjs=False,
            full_html=False,
            config=plot_config,
            div_id="main_dashboard_chart",
        )
        heatmap_chart_html = heatmap_fig.to_html(
            include_plotlyjs=False,
            full_html=False,
            config=plot_config,
            div_id="heatmap_dashboard_chart",
        )
        forecast_chart_html = forecast_fig.to_html(
            include_plotlyjs=False,
            full_html=False,
            config=plot_config,
            div_id="forecast_dashboard_chart",
        )

        title_start = plot_df.index[0].strftime("%Y-%m-%d %H:%M")
        title_end = plot_df.index[-1].strftime("%Y-%m-%d %H:%M")
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Consolidation Detection Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --bg-top: #0a0f1e;
            --bg-bottom: #0f172a;
            --panel: rgba(17, 24, 39, 0.88);
            --panel-strong: rgba(30, 41, 59, 0.96);
            --border: rgba(148, 163, 184, 0.12);
            --shadow: 0 24px 60px rgba(0, 0, 0, 0.45);
            --text: #e2e8f0;
            --muted: #94a3b8;
            --teal: #34d399;
            --amber: #fbbf24;
            --red: #f87171;
            --blue: #60a5fa;
            --slate: #94a3b8;
            --radius: 26px;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            min-height: 100vh;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(52, 211, 153, 0.06), transparent 34%),
                radial-gradient(circle at top right, rgba(96, 165, 250, 0.05), transparent 30%),
                linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
            font-family: "IBM Plex Mono", "Courier New", monospace;
        }}

        .page {{
            width: min(1440px, calc(100vw - 28px));
            margin: 20px auto 40px;
        }}

        .hero {{
            position: relative;
            overflow: hidden;
            padding: 32px;
            border-radius: 32px;
            border: 1px solid var(--border);
            background:
                linear-gradient(140deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 42, 0.90));
            box-shadow: var(--shadow);
        }}

        .hero::after {{
            content: "";
            position: absolute;
            inset: auto -80px -100px auto;
            width: 260px;
            height: 260px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(52, 211, 153, 0.08), transparent 70%);
            pointer-events: none;
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: 1.6fr 0.9fr;
            gap: 20px;
            align-items: end;
        }}

        .eyebrow,
        .metric-label,
        .panel-kicker {{
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
        }}

        .hero-title {{
            margin: 0;
            max-width: 11ch;
            font-size: clamp(2rem, 4vw, 3.6rem);
            line-height: 0.96;
            letter-spacing: -0.04em;
        }}

        .hero-copy {{
            margin: 16px 0 0;
            max-width: 760px;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
        }}

        .hero-status,
        .summary-card,
        .metric-card,
        .panel,
        .question-card,
        .details-shell {{
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--panel);
            backdrop-filter: blur(10px);
            box-shadow: var(--shadow);
        }}

        .hero-status {{
            padding: 22px;
            background: rgba(30, 41, 59, 0.72);
        }}

        .hero-status-label {{
            color: var(--muted);
            font-size: 0.80rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
        }}

        .hero-status-value {{
            margin-top: 12px;
            font-family: "IBM Plex Mono", monospace;
            font-size: clamp(1.5rem, 2vw, 2.1rem);
            line-height: 1.1;
        }}

        .hero-status-copy {{
            margin: 12px 0 0;
            color: var(--muted);
            line-height: 1.6;
        }}

        .hero-status-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 18px;
        }}

        .hero-status-metric {{
            padding: 12px 14px;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(30, 41, 59, 0.64);
        }}

        .hero-status-metric span {{
            display: block;
            color: var(--muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .hero-status-metric strong {{
            display: block;
            margin-top: 8px;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.98rem;
            font-weight: 600;
        }}

        .pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 22px;
        }}

        .pill {{
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(30, 41, 59, 0.68);
            color: var(--text);
            font-size: 0.92rem;
        }}

        .summary-grid,
        .question-grid,
        .snapshot-grid,
        .panel-grid,
        .details-grid {{
            display: grid;
            gap: 18px;
            margin-top: 18px;
        }}

        .summary-grid {{
            grid-template-columns: 1fr;
        }}

        .question-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}

        .snapshot-grid {{
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }}

        .panel-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}

        .details-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}

        .tone-risk {{
            border-color: rgba(248, 113, 113, 0.28);
        }}

        .tone-watch {{
            border-color: rgba(251, 191, 36, 0.28);
        }}

        .tone-support {{
            border-color: rgba(52, 211, 153, 0.28);
        }}

        .tone-neutral {{
            border-color: rgba(148, 163, 184, 0.18);
        }}

        .summary-card {{
            padding: 24px;
        }}

        .summary-card-primary {{
            background: linear-gradient(160deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 42, 0.88));
        }}

        .summary-card-secondary {{
            background: linear-gradient(160deg, rgba(20, 28, 46, 0.96), rgba(17, 24, 39, 0.86));
        }}

        .summary-header {{
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: start;
        }}

        .summary-title {{
            margin: 8px 0 0;
            font-size: clamp(1.45rem, 2.1vw, 2rem);
            letter-spacing: -0.03em;
        }}

        .status-chip {{
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid transparent;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            white-space: nowrap;
        }}

        .status-risk {{
            color: #f87171;
            background: rgba(248, 113, 113, 0.10);
            border-color: rgba(248, 113, 113, 0.22);
        }}

        .status-watch {{
            color: #fbbf24;
            background: rgba(251, 191, 36, 0.10);
            border-color: rgba(251, 191, 36, 0.22);
        }}

        .status-support {{
            color: #34d399;
            background: rgba(52, 211, 153, 0.10);
            border-color: rgba(52, 211, 153, 0.20);
        }}

        .status-neutral {{
            color: var(--slate);
            background: rgba(148, 163, 184, 0.08);
            border-color: rgba(148, 163, 184, 0.18);
        }}

        .summary-copy {{
            margin: 16px 0 0;
            line-height: 1.65;
            font-size: 1rem;
        }}

        .summary-copy-muted {{
            color: var(--muted);
        }}

        .summary-metrics {{
            margin: 18px 0 0;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
        }}

        .summary-metric {{
            margin: 0;
            padding: 14px 16px;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(30, 41, 59, 0.64);
        }}

        .summary-metric dt {{
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .summary-metric dd {{
            margin: 10px 0 0;
            font-family: "IBM Plex Mono", monospace;
            font-size: 1.05rem;
        }}

        .summary-footer {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px 18px;
            margin-top: 18px;
            color: var(--muted);
            font-size: 0.92rem;
        }}

        .summary-list {{
            margin: 18px 0 0;
            padding-left: 18px;
        }}

        .summary-list li {{
            margin-top: 12px;
            line-height: 1.6;
        }}

        .summary-list li::marker {{
            color: var(--amber);
        }}

        .question-card {{
            padding: 22px;
        }}

        .question-title {{
            margin: 8px 0 0;
            font-size: 1.18rem;
            letter-spacing: -0.02em;
        }}

        .question-copy {{
            margin: 14px 0 0;
            color: var(--muted);
            line-height: 1.65;
            font-size: 0.98rem;
        }}

        .metric-card {{
            padding: 20px;
            min-height: 156px;
            transform: translateY(10px);
            opacity: 0;
            animation: rise 0.55s ease forwards;
        }}

        .metric-card:nth-child(2) {{ animation-delay: 0.04s; }}
        .metric-card:nth-child(3) {{ animation-delay: 0.08s; }}
        .metric-card:nth-child(4) {{ animation-delay: 0.12s; }}

        .metric-value {{
            margin-top: 14px;
            font-family: "IBM Plex Mono", monospace;
            font-size: clamp(1.35rem, 1.9vw, 2rem);
            line-height: 1.1;
        }}

        .metric-subtitle {{
            margin-top: 12px;
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.55;
        }}

        .panel {{
            padding: 22px;
        }}

        .panel-wide {{
            margin-top: 18px;
        }}

        .panel-head {{
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: baseline;
            margin-bottom: 12px;
        }}

        .panel-title {{
            margin: 6px 0 0;
            font-size: 1.24rem;
            letter-spacing: -0.02em;
        }}

        .panel-meta {{
            color: var(--muted);
            font-size: 0.92rem;
        }}

        .plot-shell {{
            overflow: hidden;
        }}

        .plot-shell > div {{
            width: 100%;
        }}

        .details-shell {{
            margin-top: 18px;
            padding: 20px;
        }}

        .details-shell[open] {{
            background: rgba(17, 24, 39, 0.92);
        }}

        .details-summary {{
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: baseline;
            cursor: pointer;
            list-style: none;
        }}

        .details-summary::-webkit-details-marker {{
            display: none;
        }}

        .details-title {{
            margin: 6px 0 0;
            font-size: 1.16rem;
            letter-spacing: -0.02em;
        }}

        .details-meta {{
            color: var(--muted);
            font-size: 0.92rem;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
        }}

        .data-table th,
        .data-table td {{
            padding: 14px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(148, 163, 184, 0.16);
            font-size: 0.95rem;
        }}

        .data-table thead th {{
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .empty-state {{
            padding: 24px 0;
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.6;
        }}

        .table-note {{
            margin-bottom: 14px;
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.55;
        }}

        @keyframes rise {{
            from {{
                opacity: 0;
                transform: translateY(14px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @media (max-width: 1180px) {{
            .hero-grid,
            .summary-grid,
            .panel-grid,
            .details-grid {{
                grid-template-columns: 1fr;
            }}

            .question-grid {{
                grid-template-columns: 1fr;
            }}

            .snapshot-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}

            .summary-metrics {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}

        @media (max-width: 760px) {{
            .page {{
                width: min(100vw - 16px, 100%);
                margin: 8px auto 24px;
            }}

            .hero {{
                padding: 22px;
                border-radius: 24px;
            }}

            .summary-card,
            .question-card,
            .metric-card,
            .panel,
            .details-shell {{
                padding: 18px;
            }}

            .snapshot-grid,
            .summary-metrics {{
                grid-template-columns: 1fr;
            }}

            .hero-status-grid {{
                grid-template-columns: 1fr;
            }}

            .panel-head,
            .details-summary,
            .summary-header {{
                flex-direction: column;
                align-items: flex-start;
            }}

            .pill {{
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <main class="page">
        <section class="hero">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">Consolidation detection dashboard</div>
                    <h1 class="hero-title">Is it consolidating, and which way is it breaking?</h1>
                    <p class="hero-copy">
                        The dashboard answers the setup question first: is there a current
                        consolidation, what is the breakout side, and how strong is that call.
                        The charts and appendix underneath are supporting evidence, not the main point.
                    </p>
                    <div class="pill-row">
                        <div class="pill">Window: {title_start} to {title_end}</div>
                        <div class="pill">Plot span: {html.escape(plot_window_label)}</div>
                        <div class="pill">Parameter hash: {self.params.get_hash()}</div>
                        <div class="pill">Consolidation threshold: {self.params.consolidation_threshold:.3f}</div>
                        <div class="pill">Signal display threshold: {np.float64(latest_signal["threshold"]):.0%}</div>
                        <div class="pill">Generated: {generated_at}</div>
                    </div>
                </div>
                {direction_panel_html}
            </div>
        </section>

        <section class="summary-grid">
            {executive_panel_html}
            {caution_panel_html}
        </section>

        <section class="snapshot-grid">
            {snapshot_cards_html}
        </section>

        <section class="panel panel-wide">
            <div class="panel-head">
                <div>
                    <div class="panel-kicker">Main Timeline</div>
                    <h2 class="panel-title">Structural price view</h2>
                </div>
                <div class="panel-meta">{html.escape(plot_window_label)}</div>
            </div>
            <div class="plot-shell">{main_chart_html}</div>
        </section>

        <section class="panel-grid">
            <article class="panel">
                <div class="panel-head">
                    <div>
                        <div class="panel-kicker">Heatmap</div>
                        <h2 class="panel-title">When accepted windows cluster</h2>
                    </div>
                    <div class="panel-meta">{html.escape(heatmap_window_label)}</div>
                </div>
                <div class="plot-shell">{heatmap_chart_html}</div>
            </article>
            <article class="panel">
                <div class="panel-head">
                    <div>
                        <div class="panel-kicker">Latest Window</div>
                        <h2 class="panel-title">Follow-through after the latest accepted window</h2>
                    </div>
                    <div class="panel-meta">{html.escape(forecast_horizon_label)}</div>
                </div>
                <div class="plot-shell">{forecast_chart_html}</div>
            </article>
        </section>

        <details class="details-shell">
            <summary class="details-summary">
                <div>
                    <div class="panel-kicker">Technical appendix</div>
                    <h2 class="details-title">Model internals and attribution</h2>
                </div>
                <div class="details-meta">
                    Open when you need the component blend and optimization trace.
                </div>
            </summary>
            <div class="details-grid">
                <article class="panel">
                    <div class="panel-head">
                        <div>
                            <div class="panel-kicker">Component Blend</div>
                            <h2 class="panel-title">How the detector is leaning</h2>
                        </div>
                        <div class="panel-meta">Mean scores with effective weights</div>
                    </div>
                    {component_table_html}
                </article>
                <article class="panel">
                    <div class="panel-head">
                        <div>
                            <div class="panel-kicker">Baseline Check</div>
                            <h2 class="panel-title">Does the complex model beat a simple one?</h2>
                        </div>
                        <div class="panel-meta">Same rolling splits, same horizon, lower Brier wins</div>
                    </div>
                    {benchmark_table_html}
                </article>
            </div>
        </details>
    </main>
</body>
</html>
"""

        output.write_text(html_content, encoding="utf-8")
        self.logger.info("Dashboard saved to %s", output)
        return output


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> None:
    """Main execution function."""
    logger.info("Starting autonomous consolidation detection system")
    setup_random_seed()

    loader = DataLoader()
    df_raw = loader.load()

    engineer = FeatureEngineer(df_raw)
    df = engineer.calculate_all_features()

    optimizer = ParameterOptimizer(n_trials=OPTIMIZATION_TRIALS, n_splits=VALIDATION_SPLITS)
    best_params = optimizer.optimize(df)

    detector = ConsolidationDetector(best_params)
    consolidations = detector.detect(df)
    if not consolidations:
        logger.warning(
            "No consolidations detected with threshold %.4f. Running score diagnostics.",
            best_params.consolidation_threshold,
        )
        detector.diagnose_thresholds(df)
        if AUTO_CALIBRATE_THRESHOLD_ON_EMPTY:
            calibrated_threshold = detector.auto_calibrate_threshold(df)
            if calibrated_threshold + PRICE_EPSILON < best_params.consolidation_threshold:
                logger.warning(
                    "Auto-calibrating consolidation threshold from %.4f to %.4f",
                    best_params.consolidation_threshold,
                    calibrated_threshold,
                )
                best_params = replace(
                    best_params,
                    consolidation_threshold=calibrated_threshold,
                )
                detector = ConsolidationDetector(best_params)
                consolidations = detector.detect(df)
        if not consolidations:
            logger.warning("Consolidation detector remained inactive after threshold calibration.")

    predictor = BreakoutPredictor(best_params)
    probabilities, directions = predictor.predict(df, consolidations)
    if PROBABILITY_CALIBRATION_ENABLED:
        calibrator = ProbabilityCalibrator(best_params, n_splits=VALIDATION_SPLITS)
        calibration_model = calibrator.fit(df)
        if calibration_model is not None:
            probabilities, directions = calibrator.apply(probabilities, directions)
            logger.info(
                "Applied probability calibration with %d anchors.",
                len(calibration_model.raw_points),
            )

    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(df, consolidations, probabilities, directions)
    if BASELINE_BENCHMARK_ENABLED:
        benchmark = BaselineBenchmark(best_params, n_splits=VALIDATION_SPLITS)
        metrics.update(benchmark.evaluate(df))

    feature_importance = optimizer.get_feature_importance()

    logger.info("Analysis complete")
    logger.info(f"Total consolidations: {metrics['total_consolidations']}")
    logger.info(f"Breakout signals: {metrics['breakout_signals']}")
    logger.info(f"Simulated Sharpe: {metrics['simulated_sharpe']:.4f}")
    logger.info("Mean consolidation score: %.4f", metrics["mean_consolidation_score"])
    logger.info(
        "Mean context scores (similarity/periodic/scale/sutte): %.4f / %.4f / %.4f / %.4f",
        metrics["mean_attention_context_score"],
        metrics["mean_periodic_context_score"],
        metrics["mean_scale_anchor_score"],
        metrics["mean_sutte_score"],
    )
    logger.info(
        "Probability CI (p05/p50/p95): %.4f / %.4f / %.4f",
        metrics["probability_ci_p05"],
        metrics["probability_ci_p50"],
        metrics["probability_ci_p95"],
    )
    logger.info(
        "Signal distribution (up/down/neutral): %d / %d / %d",
        int(metrics["up_breakout_signals"]),
        int(metrics["down_breakout_signals"]),
        int(metrics["neutral_signal_bars"]),
    )
    visible_up_signals = int(
        ((directions == 1) & (probabilities >= DASHBOARD_SIGNAL_PROBABILITY_THRESHOLD)).sum()
    )
    visible_down_signals = int(
        ((directions == -1) & (probabilities >= DASHBOARD_SIGNAL_PROBABILITY_THRESHOLD)).sum()
    )
    logger.info(
        "Dashboard-visible signals at %.2f probability (up/down): %d / %d",
        DASHBOARD_SIGNAL_PROBABILITY_THRESHOLD,
        visible_up_signals,
        visible_down_signals,
    )
    logger.info("Regime volatility ratio: %.4f", metrics["regime_vol_ratio"])
    logger.info("Regime EW volatility ratio: %.4f", metrics["regime_ewm_vol_ratio"])
    if "benchmark_model_brier" in metrics:
        logger.info(
            "Benchmark Brier (model/logistic): %.4f / %.4f",
            metrics["benchmark_model_brier"],
            metrics["benchmark_baseline_brier"],
        )
        logger.info(
            "Benchmark answered accuracy (model/logistic): %.1f%% / %.1f%%",
            100.0 * metrics["benchmark_model_answered_accuracy"],
            100.0 * metrics["benchmark_baseline_answered_accuracy"],
        )
        logger.info(
            "Benchmark coverage (model/logistic): %.1f%% / %.1f%%",
            100.0 * metrics["benchmark_model_coverage"],
            100.0 * metrics["benchmark_baseline_coverage"],
        )
        logger.info(
            "Benchmark verdict: %s",
            "model beats logistic baseline"
            if metrics["benchmark_model_beats_baseline"] > 0
            else "logistic baseline matches or beats model",
        )
    logger.info("Optimized parameter hash: %s", best_params.get_hash())
    logger.info("Final consolidation threshold: %.4f", best_params.consolidation_threshold)
    logger.debug("Feature importance summary: %s", feature_importance)

    if DASHBOARD_ENABLED:
        dashboard = ConsolidationDashboard(
            df=df,
            consolidations=consolidations,
            probabilities=probabilities,
            directions=directions,
            params=best_params,
            metrics=metrics,
            feature_importance=feature_importance,
        )
        try:
            dashboard_path = dashboard.generate_dashboard_html()
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("Dashboard generation failed: %s", exc)
        else:
            logger.info("Dashboard generated: %s", dashboard_path)

    if metrics["breakout_signals"] == 0:
        logger.warning("No breakout signals generated; model may be inactive in current regime.")
    if metrics["regime_shift_flag"] > 0:
        logger.warning(
            "Regime shift detected (volatility ratio %.4f). "
            "Treat model outputs as lower confidence and consider re-optimization.",
            metrics["regime_vol_ratio"],
        )


if __name__ == "__main__":
    main()
