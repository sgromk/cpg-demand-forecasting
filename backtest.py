"""
Tournament backtesting engine.

Runs all five forecasting models against a demand time series using an
expanding-window backtest, then selects the winner by lowest MAE over the
most recent backtest periods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from models import (
    NaiveMean,
    WeightedMovingAverage,
    SimpleExponentialSmoothing,
    HoltsLinearTrend,
    LinearRegressionModel,
)
from config import CONFIG
from data_loader import (
    get_all_store_ids,
    get_store_demand_series,
    get_store_number,
    get_aggregate_demand,
)

# Canonical ordered list of models used throughout the tournament.
_MODEL_CLASSES = [
    NaiveMean,
    WeightedMovingAverage,
    SimpleExponentialSmoothing,
    HoltsLinearTrend,
    LinearRegressionModel,
]


def run_tournament(
    series: pd.Series, n_backtest: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Run the five-model forecasting tournament on a single demand series.

    Uses an expanding-window backtest: for each evaluation step k the model
    is trained on series.iloc[:k] and evaluated against series.iloc[k].
    The backtest window covers the last `n_backtest` observations.

    Parameters
    ----------
    series : pd.Series
        Clean, sorted demand series (non-negative, no NaN).
    n_backtest : int, optional
        Number of trailing observations to use as the backtest window.
        Defaults to CONFIG['n_backtest'].

    Returns
    -------
    dict
        Keys are model names.  Each value is a dict with:
            'forecast' : float  – point forecast for the next period
            'mae'      : float  – mean absolute error over backtest window
            'mape'     : float  – mean absolute percentage error (NaN if no
                                  actual > 0 in backtest window)
            'is_winner': bool   – True for the model with the lowest MAE
    """
    if n_backtest is None:
        n_backtest = CONFIG["n_backtest"]

    min_pts = CONFIG["min_data_points"]
    results: Dict[str, Dict] = {}

    # --- Fallback: too few data points to run a meaningful tournament ----------
    if len(series) < min_pts:
        naive_forecast = float(series.mean()) if len(series) > 0 else 0.0
        for cls in _MODEL_CLASSES:
            results[cls.name] = {
                "forecast": naive_forecast,
                "mae": float("nan"),
                "mape": float("nan"),
                "is_winner": cls is NaiveMean,
            }
        return results

    # --- Expanding-window backtest --------------------------------------------
    # The first training slice ends at index max(3, len-n_backtest) so that
    # every model has at least 3 training points for its first prediction.
    start_k = max(3, len(series) - n_backtest)

    for cls in _MODEL_CLASSES:
        model = cls()
        errors: List[float] = []
        actuals: List[float] = []

        for k in range(start_k, len(series)):
            train = series.iloc[:k]
            actual = float(series.iloc[k])
            try:
                pred = model.fit_predict(train)
                if np.isnan(pred) or np.isinf(pred):
                    pred = float(train.mean())
            except Exception:
                pred = float(train.mean())

            errors.append(abs(pred - actual))
            actuals.append(actual)

        # Final forecast trained on the full series.
        try:
            forecast = model.fit_predict(series)
            if np.isnan(forecast) or np.isinf(forecast):
                forecast = float(series.mean())
        except Exception:
            forecast = float(series.mean())

        mae = float(np.mean(errors)) if errors else float("inf")

        # MAPE: only computed for backtest steps where the actual > 0.
        mape_terms = [
            abs(e) / a
            for e, a in zip(errors, actuals)
            if a > 0
        ]
        mape = float(np.mean(mape_terms)) * 100.0 if mape_terms else float("nan")

        results[cls.name] = {
            "forecast": forecast,
            "mae": mae,
            "mape": mape,
            "is_winner": False,
        }

    # --- Select winner by lowest MAE -----------------------------------------
    best_name = min(
        results,
        key=lambda name: results[name]["mae"]
        if not np.isnan(results[name]["mae"])
        else float("inf"),
    )
    results[best_name]["is_winner"] = True

    return results


def run_all_stores(
    demand_df: pd.DataFrame, n_backtest: Optional[int] = None
) -> List[Dict]:
    """
    Run the tournament for every store in the demand DataFrame.

    Parameters
    ----------
    demand_df : pd.DataFrame
        Weekly Demand sheet as returned by a data_loader function.
    n_backtest : int, optional
        Forwarded to run_tournament.

    Returns
    -------
    list of dict
        One dict per store with keys:
            'store_id'   : store identifier
            'store_num'  : store number (string)
            'n_obs'      : number of valid demand observations
            'results'    : the dict returned by run_tournament
    """
    store_ids = get_all_store_ids(demand_df)
    output = []

    for store_id in store_ids:
        series = get_store_demand_series(demand_df, store_id)
        store_num = get_store_number(demand_df, store_id)
        tournament_results = run_tournament(series, n_backtest=n_backtest)
        output.append(
            {
                "store_id": store_id,
                "store_num": store_num,
                "n_obs": len(series),
                "results": tournament_results,
            }
        )

    return output


def run_aggregate(
    demand_df: pd.DataFrame, n_backtest: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Run the tournament on the aggregate (sum-across-stores) demand series.

    Parameters
    ----------
    demand_df : pd.DataFrame
    n_backtest : int, optional

    Returns
    -------
    dict
        The dict returned by run_tournament for the aggregate series.
    """
    min_stores = CONFIG["min_stores_for_agg"]
    agg_series = get_aggregate_demand(demand_df, min_stores=min_stores)
    return run_tournament(agg_series, n_backtest=n_backtest)
