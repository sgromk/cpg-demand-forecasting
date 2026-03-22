"""
Plot generation for the demand forecasting tournament.

For each store (and for the aggregate), produces a PNG showing:
  - Actual demand observations (scatter)
  - Winning model's one-step-ahead backtest predictions (solid line)
  - Winning model's forecast extended N weeks into the future (dashed line)

Output goes to a run-specific subdirectory passed in by the caller.
"""

import os
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import CONFIG
from models import (
    NaiveMean,
    WeightedMovingAverage,
    SimpleExponentialSmoothing,
    HoltsLinearTrend,
    LinearRegressionModel,
)

_MODEL_CLASS_MAP = {
    "Naive": NaiveMean,
    "WMA": WeightedMovingAverage,
    "SES": SimpleExponentialSmoothing,
    "Holt": HoltsLinearTrend,
    "LinReg": LinearRegressionModel,
}

_ACTUAL_COLOR = "#2E86AB"
_PRED_COLOR = "#E84855"
_FUTURE_COLOR = "#E84855"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _backtest_predictions(series: pd.Series, model_class, n_backtest: int) -> pd.Series:
    """
    Re-run the expanding-window backtest for a single model and return the
    one-step-ahead predictions as a Series indexed by the same dates as series.
    """
    start_k = max(3, len(series) - n_backtest)
    preds = {}
    model = model_class()
    for k in range(start_k, len(series)):
        train = series.iloc[:k]
        try:
            pred = model.fit_predict(train)
            if math.isnan(pred) or math.isinf(pred):
                pred = float(train.mean())
        except Exception:
            pred = float(train.mean())
        preds[series.index[k]] = pred
    return pd.Series(preds)


def _future_extension(
    series: pd.Series,
    winner_name: str,
    forecast_value: float,
    weeks: int,
) -> tuple:
    """
    Build future dates and forecast values for the dashed extension.

    For LinearRegression the extension follows the fitted trend line.
    For all other models it is a flat line at forecast_value.

    Returns (dates_list, values_list) where dates_list[0] is the last
    observed date (anchor point so the dashed line connects smoothly).
    """
    last_date = pd.Timestamp(series.index[-1])
    future_dates = [last_date + pd.Timedelta(weeks=i + 1) for i in range(weeks)]

    if winner_name == "LinReg" and len(series) >= 2:
        first_date = pd.Timestamp(series.index[0])
        x = np.array(
            [(pd.Timestamp(d) - first_date).days for d in series.index], dtype=float
        )
        y = series.values.astype(float)
        x_mean, y_mean = x.mean(), y.mean()
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx > 0:
            slope = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
            intercept = y_mean - slope * x_mean
            anchor_y = max(0.0, intercept + slope * (last_date - first_date).days)
            future_values = [
                max(0.0, intercept + slope * (d - first_date).days)
                for d in future_dates
            ]
        else:
            anchor_y = forecast_value
            future_values = [forecast_value] * weeks
    else:
        anchor_y = forecast_value
        future_values = [forecast_value] * weeks

    all_dates = [last_date] + future_dates
    all_values = [anchor_y] + future_values
    return all_dates, all_values


def _draw_plot(
    ax,
    series: pd.Series,
    backtest_preds: pd.Series,
    future_dates,
    future_values,
    title: str,
) -> None:
    """Render all plot elements onto the given Axes."""
    dates = [pd.Timestamp(d) for d in series.index]

    # Actual observations
    ax.plot(dates, series.values, color=_ACTUAL_COLOR, alpha=0.35, linewidth=1, zorder=2)
    ax.scatter(dates, series.values, color=_ACTUAL_COLOR, s=55, zorder=4, label="Actual")

    # Backtest one-step-ahead predictions
    if not backtest_preds.empty:
        bp_dates = [pd.Timestamp(d) for d in backtest_preds.index]
        ax.plot(
            bp_dates,
            backtest_preds.values,
            color=_PRED_COLOR,
            linewidth=1.8,
            marker="o",
            markersize=4,
            zorder=5,
            label="Predicted (backtest)",
        )

    # Future extension (dashed)
    fut_ts = [pd.Timestamp(d) for d in future_dates]
    ax.plot(
        fut_ts,
        future_values,
        color=_FUTURE_COLOR,
        linewidth=1.8,
        linestyle="--",
        zorder=5,
        label="Forecast (future)",
    )
    # Diamond markers on the projected forecast points only (skip the anchor)
    ax.scatter(
        fut_ts[1:], future_values[1:], color=_FUTURE_COLOR, s=45, marker="D", zorder=6
    )

    # Vertical line at today / last observation boundary
    ax.axvline(
        pd.Timestamp(series.index[-1]),
        color="gray",
        linewidth=0.8,
        linestyle=":",
        alpha=0.6,
    )

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Demand (packs / wk)", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.25)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_store(
    store_id,
    store_num,
    series: pd.Series,
    tournament_results: dict,
    output_dir: str = None,
    n_backtest: int = None,
) -> str:
    """
    Generate and save a forecast PNG for a single store.

    Returns the filepath of the saved image.
    """
    if output_dir is None:
        output_dir = os.path.join(CONFIG["outputs_root"], "plots")
    if n_backtest is None:
        n_backtest = CONFIG["n_backtest"]
    os.makedirs(output_dir, exist_ok=True)

    winner_name = next(
        (name for name, info in tournament_results.items() if info.get("is_winner")),
        "Naive",
    )
    winner_info = tournament_results[winner_name]
    forecast_value = winner_info["forecast"]
    mae = winner_info["mae"]
    mae_str = f"{mae:.2f}" if not (math.isnan(mae) or math.isinf(mae)) else "N/A"

    model_class = _MODEL_CLASS_MAP.get(winner_name, NaiveMean)

    if len(series) >= CONFIG["min_data_points"]:
        backtest_preds = _backtest_predictions(series, model_class, n_backtest)
    else:
        backtest_preds = pd.Series(dtype=float)

    future_dates, future_values = _future_extension(
        series, winner_name, forecast_value, CONFIG["forecast_weeks"]
    )

    store_label = f"Store {store_num}" if store_num else str(store_id)
    title = (
        f"{store_label} ({store_id})   Winner: {winner_name}   "
        f"Backtest MAE: {mae_str} packs/wk"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    _draw_plot(ax, series, backtest_preds, future_dates, future_values, title)
    plt.tight_layout()

    safe_id = str(store_id).replace("/", "_").replace(" ", "_")
    filepath = os.path.join(output_dir, f"store_{safe_id}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_aggregate(
    series: pd.Series,
    tournament_results: dict,
    output_dir: str = None,
    n_backtest: int = None,
) -> str:
    """
    Generate and save a forecast PNG for the aggregate demand series.

    Returns the filepath of the saved image.
    """
    if output_dir is None:
        output_dir = os.path.join(CONFIG["outputs_root"], "plots")
    if n_backtest is None:
        n_backtest = CONFIG["n_backtest"]
    os.makedirs(output_dir, exist_ok=True)

    winner_name = next(
        (name for name, info in tournament_results.items() if info.get("is_winner")),
        "Naive",
    )
    winner_info = tournament_results[winner_name]
    forecast_value = winner_info["forecast"]
    mae = winner_info["mae"]
    mae_str = f"{mae:.2f}" if not (math.isnan(mae) or math.isinf(mae)) else "N/A"

    model_class = _MODEL_CLASS_MAP.get(winner_name, NaiveMean)

    if len(series) >= CONFIG["min_data_points"]:
        backtest_preds = _backtest_predictions(series, model_class, n_backtest)
    else:
        backtest_preds = pd.Series(dtype=float)

    future_dates, future_values = _future_extension(
        series, winner_name, forecast_value, CONFIG["forecast_weeks"]
    )

    title = f"Aggregate Demand   Winner: {winner_name}   Backtest MAE: {mae_str} packs/wk"

    fig, ax = plt.subplots(figsize=(12, 5))
    _draw_plot(ax, series, backtest_preds, future_dates, future_values, title)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "aggregate.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


_MODEL_COLORS = {
    "Naive":  "#888888",
    "WMA":    "#2E86AB",
    "SES":    "#48A999",
    "Holt":   "#E88C2E",
    "LinReg": "#9B59B6",
}
_MODEL_ORDER = ["Naive", "WMA", "SES", "Holt", "LinReg"]


def plot_store_summary(store_results: list, output_dir: str = None) -> str:
    """
    Generate a summary dashboard PNG across all stores.

    Three panels:
      Top    — horizontal bar chart of winner forecast per store, colored by winning model
      Bottom left  — bar chart of how many stores each model won
      Bottom right — MAE heatmap (stores x models); greener = lower error

    Returns the filepath of the saved image.
    """
    if output_dir is None:
        output_dir = os.path.join(CONFIG["outputs_root"], "plots")
    os.makedirs(output_dir, exist_ok=True)

    # ---- collect data -------------------------------------------------------
    rows = []
    for store in store_results:
        res = store["results"]
        winner = next((n for n, r in res.items() if r.get("is_winner")), "Naive")
        winner_fc = res[winner]["forecast"]
        mae_vals = {m: res[m]["mae"] for m in _MODEL_ORDER}
        rows.append({
            "label": f"Store {store['store_num']}" if store["store_num"] else str(store["store_id"]),
            "winner": winner,
            "forecast": winner_fc,
            "mae": mae_vals,
        })

    # sort stores by forecast descending for the bar chart
    rows.sort(key=lambda r: r["forecast"], reverse=True)
    labels   = [r["label"]   for r in rows]
    forecasts = [r["forecast"] for r in rows]
    winners  = [r["winner"]  for r in rows]
    bar_colors = [_MODEL_COLORS.get(w, "#888888") for w in winners]

    # MAE matrix: rows = stores (same order), cols = models
    mae_matrix = np.array(
        [[r["mae"][m] if not (math.isnan(r["mae"][m]) or math.isinf(r["mae"][m])) else np.nan
          for m in _MODEL_ORDER]
         for r in rows],
        dtype=float,
    )

    # winner counts
    from collections import Counter
    win_counts = Counter(winners)

    # ---- layout -------------------------------------------------------------
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.1, 1.2],
        hspace=0.42,
        wspace=0.32,
    )
    ax_bars = fig.add_subplot(gs[0, :])   # top: full width
    ax_wins = fig.add_subplot(gs[1, 0])   # bottom left
    ax_heat = fig.add_subplot(gs[1, 1])   # bottom right

    # ---- panel 1: forecast bars ---------------------------------------------
    y_pos = np.arange(len(labels))
    ax_bars.barh(y_pos, forecasts, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(labels, fontsize=8)
    ax_bars.invert_yaxis()
    ax_bars.set_xlabel("Forecast (packs / wk)", fontsize=9)
    ax_bars.set_title("Winner Forecast per Store", fontsize=11, pad=6)
    ax_bars.grid(axis="x", alpha=0.25)

    # add forecast value labels on bars
    for i, (fc, winner) in enumerate(zip(forecasts, winners)):
        ax_bars.text(
            fc + max(forecasts) * 0.01, i,
            f"{fc:.1f}  [{winner}]",
            va="center", fontsize=7, color="#333333",
        )

    # legend for model colors
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_MODEL_COLORS[m], label=m) for m in _MODEL_ORDER
    ]
    ax_bars.legend(
        handles=legend_handles, title="Winning model",
        loc="lower right", fontsize=7, title_fontsize=8,
    )

    # ---- panel 2: win counts ------------------------------------------------
    model_wins = [win_counts.get(m, 0) for m in _MODEL_ORDER]
    ax_wins.bar(
        _MODEL_ORDER, model_wins,
        color=[_MODEL_COLORS[m] for m in _MODEL_ORDER],
        edgecolor="white", linewidth=0.5,
    )
    ax_wins.set_ylabel("Stores won", fontsize=9)
    ax_wins.set_title("Model Win Counts", fontsize=11, pad=6)
    ax_wins.grid(axis="y", alpha=0.25)
    for i, v in enumerate(model_wins):
        if v > 0:
            ax_wins.text(i, v + 0.1, str(v), ha="center", fontsize=9, fontweight="bold")

    # ---- panel 3: MAE heatmap -----------------------------------------------
    vmin = np.nanmin(mae_matrix)
    vmax = np.nanpercentile(mae_matrix, 90)  # cap colorscale at 90th pct to reduce outlier skew
    im = ax_heat.imshow(
        mae_matrix, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax
    )
    ax_heat.set_xticks(np.arange(len(_MODEL_ORDER)))
    ax_heat.set_xticklabels(_MODEL_ORDER, fontsize=8)
    ax_heat.set_yticks(np.arange(len(labels)))
    ax_heat.set_yticklabels(labels, fontsize=7)
    ax_heat.set_title("MAE by Store and Model\n(green = lower error)", fontsize=11, pad=6)

    # annotate cells with MAE values
    for i in range(len(rows)):
        for j, model in enumerate(_MODEL_ORDER):
            val = mae_matrix[i, j]
            text = f"{val:.1f}" if not np.isnan(val) else "—"
            brightness = im.cmap(im.norm(val))[:3]
            text_color = "white" if sum(brightness) / 3 < 0.5 else "#222222"
            ax_heat.text(j, i, text, ha="center", va="center", fontsize=6.5, color=text_color)

    fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.04, label="MAE (packs/wk)")

    # ---- save ---------------------------------------------------------------
    filepath = os.path.join(output_dir, "summary.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_all_stores(
    store_results: list,
    output_dir: str = None,
    n_backtest: int = None,
) -> list:
    """
    Generate PNGs for every store in store_results.

    store_results is the list returned by backtest.run_all_stores().
    Returns a list of saved filepaths.
    """
    from data_loader import get_store_demand_series  # avoid circular at module level

    paths = []
    for store in store_results:
        series = store.get("series")
        if series is None or len(series) == 0:
            continue
        path = plot_store(
            store_id=store["store_id"],
            store_num=store["store_num"],
            series=series,
            tournament_results=store["results"],
            output_dir=output_dir,
            n_backtest=n_backtest,
        )
        paths.append(path)
    return paths
