# CPG Demand Forecasting Tournament

## What This Is

A demand forecasting system built for Tozi Superfoods, a CPG (Consumer Packaged Goods) startup distributing a retail product line to 21 HEB grocery stores across Texas. The system reads weekly sell-through data from a Google Sheet (or an Excel tracker), runs five competing forecasting models against each store's historical demand, evaluates them via backtesting, and selects a winner per store based on recent accuracy. It also produces an aggregate forecast across all stores to support production planning decisions.

## Business Context

Tozi Superfoods operates with the tight margins and short planning horizons typical of early-stage CPG. Weekly demand data is sparse — stores don't always sell every week, and shipments are irregular. Manual forecasting from a spreadsheet is error-prone and time-consuming. This system automates that process, turning raw sell-through history into a defensible production planning number that can be handed directly to a co-manufacturer.

The 21 HEB stores vary significantly in velocity. Some are high-volume flagship locations; others are slower neighborhood stores. A single aggregate forecast would mask this variance and lead to misallocation of inventory. This system handles both granularities: per-store forecasts for store-level replenishment decisions, and an aggregate forecast for overall production runs.

## Technical Approach

Five models compete in a tournament for each store:

1. **Naive Mean** — simple average of all historical demand. A baseline that's hard to beat with sparse data.
2. **Weighted Moving Average** — linearly increasing weights so recent weeks matter more than older ones.
3. **Simple Exponential Smoothing (SES)** — exponentially decaying weights with an alpha parameter tuned per store via grid search to minimize historical error.
4. **Holt's Linear Trend** — extends SES with a trend component, useful for stores with clear growth or decline trajectories. Both level (alpha) and trend (beta) parameters are grid-searched.
5. **Linear Regression** — fits a line through the time series using days-since-first-observation as the independent variable and projects forward to today. Floored at zero.

Model selection uses **expanding-window backtesting**: for each backtest step, the model is trained on all data up to that point and evaluated against the next observed value. This mirrors real-world conditions where the model only sees data available at the time of prediction. The winner is the model with the lowest Mean Absolute Error (MAE) over the most recent backtest periods.

## Key Engineering Decisions

- **Irregular time intervals**: Demand data doesn't arrive on a fixed weekly cadence — HEB sell-through reports can have gaps. The data loader extracts date columns by parsing column headers with `pd.to_datetime`, skipping metadata columns (store ID, store number, etc.) that can't be parsed as dates.
- **Sparse data handling**: Many stores have fewer observations than ideal. The system applies a minimum data threshold before running a full tournament; stores below the threshold fall back to naive mean.
- **Negative demand filtering**: Inventory adjustments and returns can produce negative demand entries in the tracker. These are filtered before model fitting.
- **Auto-generated columns**: The Excel tracker includes auto-calculated summary columns (e.g., BA–BD) that are not date columns. The date-detection logic handles these gracefully.
- **Per-store and aggregate forecasts**: Aggregate demand is computed by summing across stores per date, restricted to dates with readings from a minimum number of stores to avoid understating demand on weeks with incomplete reporting.

## Output

The system prints structured summary tables to the console:

- **Per-store forecasts**: one row per store showing each model's forecast, the winning model, and the winner's forecast value.
- **Model accuracy (MAE)**: one row per store showing each model's backtest MAE, making it easy to see which models are consistently outperforming.
- **Aggregate forecast**: a single table showing all five models' forecasts and accuracy metrics for the summed demand series, with the winner marked.

If CSV output is enabled, `store_forecasts.csv` and `aggregate_forecast.csv` are written to the working directory for downstream use in planning spreadsheets.

## Tech Stack

- **Python** with **pandas** and **NumPy** for data manipulation and numerical computation
- **openpyxl** for reading the Excel demand tracker
- **gspread** and **google-auth** for reading directly from the live Google Sheet via a service account
- All forecasting models are implemented from scratch — no sklearn or statsmodels dependency
