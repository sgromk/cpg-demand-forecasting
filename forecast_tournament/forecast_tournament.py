"""
Demand Forecasting Tournament – main entry point.

Usage
-----
    # From an Excel tracker:
    python -m forecast_tournament.forecast_tournament --input tracker.xlsx

    # From a Google Sheet:
    python -m forecast_tournament.forecast_tournament \\
        --gsheet "Tozi Demand Tracker" \\
        --credentials creds.json

    # Override backtest window and skip CSV output:
    python -m forecast_tournament.forecast_tournament \\
        --input tracker.xlsx --n-backtest 3 --no-csv
"""

import argparse
import sys
import os
import math
from typing import List, Dict, Optional

import pandas as pd

from .config import CONFIG
from .data_loader import load_from_xlsx, load_from_gsheet
from .backtest import run_all_stores, run_aggregate

# ---------------------------------------------------------------------------
# Model display order and short names
# ---------------------------------------------------------------------------
_MODEL_NAMES = ["Naive", "WMA", "SES", "Holt", "LinReg"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value, decimals: int = 1, na_str: str = "—") -> str:
    """Format a float to a fixed number of decimal places, or return na_str."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return na_str
    if isinstance(value, float) and math.isinf(value):
        return na_str
    return f"{value:.{decimals}f}"


def _pct(value, na_str: str = "—") -> str:
    """Format a float as a percentage string, e.g. 18.3%."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return na_str
    return f"{value:.1f}%"


def _winner_name(results: Dict[str, Dict]) -> str:
    """Return the name of the winning model in a results dict."""
    for name, info in results.items():
        if info.get("is_winner"):
            return name
    return "—"


# ---------------------------------------------------------------------------
# Console table printers
# ---------------------------------------------------------------------------

def _print_per_store_forecasts(store_results: List[Dict]) -> None:
    """Print the per-store forecast table to stdout."""
    col_widths = {
        "Store ID": 10,
        "Store #": 8,
        "Naive": 7,
        "WMA": 7,
        "SES": 7,
        "Holt": 7,
        "LinReg": 8,
        "Winner": 8,
        "Winner Forecast": 16,
    }
    header = (
        f"{'Store ID':<{col_widths['Store ID']}} "
        f"{'Store #':<{col_widths['Store #']}} "
        f"{'Naive':>{col_widths['Naive']}} "
        f"{'WMA':>{col_widths['WMA']}} "
        f"{'SES':>{col_widths['SES']}} "
        f"{'Holt':>{col_widths['Holt']}} "
        f"{'LinReg':>{col_widths['LinReg']}} "
        f"{'Winner':<{col_widths['Winner']}} "
        f"{'Winner Forecast':>{col_widths['Winner Forecast']}}"
    )
    separator = "-" * len(header)

    print("\n=== PER-STORE FORECASTS ===")
    print(header)
    print(separator)

    for store in store_results:
        res = store["results"]
        winner = _winner_name(res)
        winner_fc = _fmt(res[winner]["forecast"]) if winner != "—" else "—"
        row = (
            f"{str(store['store_id']):<{col_widths['Store ID']}} "
            f"{str(store['store_num'] or ''):<{col_widths['Store #']}} "
            f"{_fmt(res['Naive']['forecast']):>{col_widths['Naive']}} "
            f"{_fmt(res['WMA']['forecast']):>{col_widths['WMA']}} "
            f"{_fmt(res['SES']['forecast']):>{col_widths['SES']}} "
            f"{_fmt(res['Holt']['forecast']):>{col_widths['Holt']}} "
            f"{_fmt(res['LinReg']['forecast']):>{col_widths['LinReg']}} "
            f"{winner:<{col_widths['Winner']}} "
            f"{winner_fc:>{col_widths['Winner Forecast']}}"
        )
        print(row)


def _print_mae_table(store_results: List[Dict]) -> None:
    """Print the per-store model MAE comparison table to stdout."""
    col_widths = {
        "Store ID": 10,
        "Naive MAE": 10,
        "WMA MAE": 8,
        "SES MAE": 8,
        "Holt MAE": 9,
        "LinReg MAE": 11,
    }
    header = (
        f"{'Store ID':<{col_widths['Store ID']}} "
        f"{'Naive MAE':>{col_widths['Naive MAE']}} "
        f"{'WMA MAE':>{col_widths['WMA MAE']}} "
        f"{'SES MAE':>{col_widths['SES MAE']}} "
        f"{'Holt MAE':>{col_widths['Holt MAE']}} "
        f"{'LinReg MAE':>{col_widths['LinReg MAE']}}"
    )
    separator = "-" * len(header)

    print("\n=== MODEL ACCURACY (MAE) ===")
    print(header)
    print(separator)

    for store in store_results:
        res = store["results"]
        row = (
            f"{str(store['store_id']):<{col_widths['Store ID']}} "
            f"{_fmt(res['Naive']['mae'], 2):>{col_widths['Naive MAE']}} "
            f"{_fmt(res['WMA']['mae'], 2):>{col_widths['WMA MAE']}} "
            f"{_fmt(res['SES']['mae'], 2):>{col_widths['SES MAE']}} "
            f"{_fmt(res['Holt']['mae'], 2):>{col_widths['Holt MAE']}} "
            f"{_fmt(res['LinReg']['mae'], 2):>{col_widths['LinReg MAE']}}"
        )
        print(row)


def _print_aggregate(agg_results: Dict[str, Dict]) -> None:
    """Print the aggregate forecast table to stdout."""
    col_widths = {
        "Model": 8,
        "Forecast": 10,
        "MAE": 8,
        "MAPE": 8,
        "Winner": 7,
    }
    header = (
        f"{'Model':<{col_widths['Model']}} "
        f"{'Forecast':>{col_widths['Forecast']}} "
        f"{'MAE':>{col_widths['MAE']}} "
        f"{'MAPE':>{col_widths['MAPE']}} "
        f"{'Winner':<{col_widths['Winner']}}"
    )
    separator = "-" * len(header)

    print("\n=== AGGREGATE FORECAST ===")
    print(header)
    print(separator)

    for name in _MODEL_NAMES:
        info = agg_results.get(name, {})
        winner_marker = "*" if info.get("is_winner") else ""
        row = (
            f"{name:<{col_widths['Model']}} "
            f"{_fmt(info.get('forecast')):>{col_widths['Forecast']}} "
            f"{_fmt(info.get('mae'), 2):>{col_widths['MAE']}} "
            f"{_pct(info.get('mape')):>{col_widths['MAPE']}} "
            f"{winner_marker:<{col_widths['Winner']}}"
        )
        print(row)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_store_csv(store_results: List[Dict], filepath: str = "store_forecasts.csv") -> None:
    """Write per-store forecast results to a CSV file."""
    rows = []
    for store in store_results:
        res = store["results"]
        winner = _winner_name(res)
        row = {
            "store_id": store["store_id"],
            "store_num": store["store_num"],
            "n_obs": store["n_obs"],
            "naive_forecast": res["Naive"]["forecast"],
            "wma_forecast": res["WMA"]["forecast"],
            "ses_forecast": res["SES"]["forecast"],
            "holt_forecast": res["Holt"]["forecast"],
            "linreg_forecast": res["LinReg"]["forecast"],
            "naive_mae": res["Naive"]["mae"],
            "wma_mae": res["WMA"]["mae"],
            "ses_mae": res["SES"]["mae"],
            "holt_mae": res["Holt"]["mae"],
            "linreg_mae": res["LinReg"]["mae"],
            "winner": winner,
            "winner_forecast": res[winner]["forecast"] if winner != "—" else None,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"\n[CSV] Store forecasts written to: {os.path.abspath(filepath)}")


def _write_aggregate_csv(agg_results: Dict[str, Dict], filepath: str = "aggregate_forecast.csv") -> None:
    """Write aggregate forecast results to a CSV file."""
    rows = []
    for name in _MODEL_NAMES:
        info = agg_results.get(name, {})
        rows.append(
            {
                "model": name,
                "forecast": info.get("forecast"),
                "mae": info.get("mae"),
                "mape_pct": info.get("mape"),
                "is_winner": info.get("is_winner", False),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"[CSV] Aggregate forecast written to: {os.path.abspath(filepath)}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast_tournament",
        description="Run a demand forecasting model tournament across HEB store sell-through data.",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        metavar="FILEPATH",
        help="Path to the .xlsx demand tracker file.",
    )
    source_group.add_argument(
        "--gsheet",
        metavar="SHEET_NAME",
        help="Name of the Google Spreadsheet (alternative to --input).",
    )

    parser.add_argument(
        "--credentials",
        metavar="FILE",
        default=None,
        help="Path to a service-account credentials JSON file (required with --gsheet).",
    )
    parser.add_argument(
        "--n-backtest",
        type=int,
        default=None,
        metavar="N",
        help=f"Override the backtest window size (default: {CONFIG['n_backtest']}).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        default=False,
        help="Skip writing CSV output files.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Load data -----------------------------------------------------------
    if args.input:
        print(f"Loading data from Excel: {args.input}")
        try:
            _uoh_df, demand_df, _deliveries_df = load_from_xlsx(args.input)
        except FileNotFoundError:
            print(f"ERROR: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"ERROR loading Excel file: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.credentials:
            print(
                "ERROR: --credentials is required when using --gsheet.", file=sys.stderr
            )
            sys.exit(1)
        print(f"Loading data from Google Sheet: '{args.gsheet}'")
        try:
            _uoh_df, demand_df, _deliveries_df = load_from_gsheet(
                args.gsheet, args.credentials
            )
        except Exception as exc:
            print(f"ERROR loading Google Sheet: {exc}", file=sys.stderr)
            sys.exit(1)

    # --- Run tournaments -----------------------------------------------------
    print("Running per-store forecasting tournament…")
    store_results = run_all_stores(demand_df, n_backtest=args.n_backtest)

    print("Running aggregate forecasting tournament…")
    agg_results = run_aggregate(demand_df, n_backtest=args.n_backtest)

    # --- Print console output ------------------------------------------------
    _print_per_store_forecasts(store_results)
    _print_mae_table(store_results)
    _print_aggregate(agg_results)

    # --- Write CSV -----------------------------------------------------------
    write_csv = CONFIG["output_csv"] and not args.no_csv
    if write_csv:
        _write_store_csv(store_results)
        _write_aggregate_csv(agg_results)


if __name__ == "__main__":
    main()
