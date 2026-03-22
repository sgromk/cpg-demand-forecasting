# CPG Demand Forecasting Tournament

## What This Is

A demand forecasting system built for CPG (Consumer Packaged Goods) brands distributing products across a network of retail stores. The system reads weekly sell-through data from a Google Sheet or Excel tracker, runs five competing forecasting models against each store's historical demand, backtests them, and selects a winner per store based on recent accuracy. It also produces an aggregate forecast across all stores to support production planning.

## Business Context

Early-stage CPG brands operating in retail face a common challenge: demand data is sparse, shipment timing is irregular, and stores vary widely in velocity. Manual forecasting from a spreadsheet is error-prone and slow to update as new sell-through data comes in. This system automates that process, turning raw sell-through history into a defensible production planning number that can be handed directly to a co-manufacturer or operations team.

The per-store granularity matters because aggregate demand can mask significant variance across locations. High-volume flagship stores and slower neighborhood stores behave differently, and a single blended forecast would lead to misallocation of inventory. This system handles both levels: per-store forecasts for store-level replenishment decisions, and an independent aggregate forecast for overall production runs.

## Technical Approach

Five models compete in a tournament for each store:

1. **Naive Mean** — simple average of all historical demand. A baseline that is hard to beat with sparse data.
2. **Weighted Moving Average** — linearly increasing weights so recent weeks matter more than older ones.
3. **Simple Exponential Smoothing (SES)** — exponentially decaying weights with an alpha parameter tuned per store via grid search to minimize historical error.
4. **Holt's Linear Trend** — extends SES with a trend component, useful for stores with clear growth or decline trajectories. Both the level (alpha) and trend (beta) parameters are grid-searched.
5. **Linear Regression** — fits a line through the time series using days since first observation as the independent variable and projects forward to today. Floored at zero.

Model selection uses expanding-window backtesting. For each backtest step, the model is trained on all data up to that point and evaluated against the next observed value. This mirrors real-world conditions where the model only sees data available at prediction time. The winner is the model with the lowest Mean Absolute Error (MAE) over the most recent backtest periods.

## Key Engineering Decisions

**Irregular time intervals.** Sell-through data does not arrive on a fixed weekly cadence. The data loader extracts date columns by parsing column headers with `pd.to_datetime`, skipping metadata columns (store ID, store number, summary totals) that cannot be parsed as dates.

**Sparse data handling.** Many stores have fewer observations than ideal for fitting trend-aware models. The system applies a minimum data threshold before running a full tournament. Stores below the threshold fall back to naive mean.

**Negative demand filtering.** Inventory adjustments and missing delivery records can produce negative demand entries in the tracker. These are filtered before model fitting so they do not distort forecasts.

**Auto-generated summary columns.** Excel trackers commonly include auto-calculated summary columns alongside date columns. The date-detection logic handles these gracefully without requiring manual configuration.

**Two levels of aggregation.** Aggregate demand is computed by summing across stores per date, restricted to dates where a minimum number of stores reported data. This avoids understating total demand on weeks with incomplete reporting.

## Output

The system prints structured summary tables to the console:

- **Per-store forecasts** — one row per store showing each model's forecast, the winning model, and the winner's forecast value
- **Model accuracy (MAE)** — one row per store showing each model's backtest error, making it easy to see which models consistently outperform
- **Aggregate forecast** — all five models' forecasts and accuracy metrics for the summed demand series, with the winner marked

Optionally writes `store_forecasts.csv` and `aggregate_forecast.csv` to the working directory for downstream use in planning spreadsheets.

## Tech Stack

- **Python** with **pandas** and **NumPy** for data manipulation and numerical computation
- **openpyxl** for reading Excel demand trackers
- **gspread** and **google-auth** for reading directly from a live Google Sheet via a service account
- All forecasting models are implemented from scratch with no scikit-learn or statsmodels dependency
