"""
Forecasting model classes for the demand forecasting tournament.

Each model exposes a single public method:
    fit_predict(series: pd.Series) -> float

The method trains on the entire provided series and returns a single float
representing the forecast for the next period.
"""

import numpy as np
import pandas as pd
from config import CONFIG


class NaiveMean:
    """Baseline model: returns the arithmetic mean of the series."""

    name = "Naive"

    def fit_predict(self, series: pd.Series) -> float:
        return float(series.mean())


class WeightedMovingAverage:
    """
    Weighted moving average with linearly increasing weights.

    The oldest observation receives weight 1, the most recent receives weight N,
    so recent demand is emphasised without discarding older history entirely.
    """

    name = "WMA"

    def fit_predict(self, series: pd.Series) -> float:
        n = len(series)
        weights = np.arange(1, n + 1, dtype=float)
        return float(np.average(series.values, weights=weights))


class SimpleExponentialSmoothing:
    """
    Simple Exponential Smoothing with alpha tuned via grid search.

    For short series (fewer than 4 points) alpha defaults to 0.3.
    Otherwise the alpha that minimises leave-one-out MAE from index 3 onward
    is selected from CONFIG['ses_alpha_grid'].
    """

    name = "SES"
    best_alpha: float = 0.3

    def _smooth(self, series: pd.Series, alpha: float) -> float:
        """Return the final smoothed value after applying SES to the series."""
        s = float(series.iloc[0])
        for x in series.iloc[1:]:
            s = alpha * float(x) + (1.0 - alpha) * s
        return s

    def fit_predict(self, series: pd.Series) -> float:
        if len(series) < 4:
            self.best_alpha = 0.3
            return self._smooth(series, self.best_alpha)

        best_alpha = 0.3
        best_mae = float("inf")

        for alpha in CONFIG["ses_alpha_grid"]:
            errors = []
            for k in range(3, len(series)):
                train = series.iloc[:k]
                actual = float(series.iloc[k])
                pred = self._smooth(train, alpha)
                errors.append(abs(pred - actual))
            mae = float(np.mean(errors)) if errors else float("inf")
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha

        self.best_alpha = best_alpha
        return self._smooth(series, self.best_alpha)


class HoltsLinearTrend:
    """
    Holt's Linear Trend (double exponential smoothing).

    Both the level parameter (alpha) and trend parameter (beta) are tuned via
    grid search over CONFIG['holt_grid'] x CONFIG['holt_grid'], minimising MAE
    on leave-one-out predictions from index 3 onward.

    Falls back to series mean for very short series (fewer than 4 points).
    """

    name = "Holt"
    best_alpha: float = 0.3
    best_beta: float = 0.1

    def _holt(self, series: pd.Series, alpha: float, beta: float) -> float:
        """Apply Holt's method and return the one-step-ahead forecast."""
        l = float(series.iloc[0])
        b = float(series.iloc[1]) - float(series.iloc[0]) if len(series) > 1 else 0.0

        for x in series.iloc[1:]:
            l_prev = l
            l = alpha * float(x) + (1.0 - alpha) * (l + b)
            b = beta * (l - l_prev) + (1.0 - beta) * b

        return l + b

    def fit_predict(self, series: pd.Series) -> float:
        if len(series) < 4:
            return float(series.mean())

        best_alpha = 0.3
        best_beta = 0.1
        best_mae = float("inf")

        for alpha in CONFIG["holt_grid"]:
            for beta in CONFIG["holt_grid"]:
                errors = []
                for k in range(3, len(series)):
                    train = series.iloc[:k]
                    actual = float(series.iloc[k])
                    try:
                        pred = self._holt(train, alpha, beta)
                    except Exception:
                        pred = float("nan")
                    if not np.isnan(pred):
                        errors.append(abs(pred - actual))
                mae = float(np.mean(errors)) if errors else float("inf")
                if mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha
                    best_beta = beta

        self.best_alpha = best_alpha
        self.best_beta = best_beta
        return self._holt(series, self.best_alpha, self.best_beta)


class LinearRegressionModel:
    """
    Ordinary least-squares linear regression over time.

    The independent variable is days elapsed since the first observation.
    The model projects to today's date (or the last date if the index has no
    datetime information).  Predictions are floored at zero.

    Implemented without scikit-learn; uses closed-form OLS formulae.
    """

    name = "LinReg"

    def fit_predict(self, series: pd.Series) -> float:
        if len(series) < 2:
            return float(series.mean())

        # Build x as days since the first observation.
        # If the index is datetime-like use it; otherwise use integer positions.
        try:
            first_date = pd.Timestamp(series.index[0])
            x = np.array(
                [(pd.Timestamp(d) - first_date).days for d in series.index],
                dtype=float,
            )
            today = pd.Timestamp.today().normalize()
            x_pred = float((today - first_date).days)
        except Exception:
            x = np.arange(len(series), dtype=float)
            x_pred = float(len(series))

        y = series.values.astype(float)

        x_mean = x.mean()
        y_mean = y.mean()
        ss_xx = np.sum((x - x_mean) ** 2)

        if ss_xx == 0.0:
            # All observations on the same date — return mean.
            return max(0.0, y_mean)

        slope = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
        intercept = y_mean - slope * x_mean

        forecast = slope * x_pred + intercept
        return max(0.0, float(forecast))
