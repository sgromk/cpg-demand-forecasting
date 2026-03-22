"""
Data loading utilities for the demand forecasting tournament.

Supports two sources:
  - Excel (.xlsx) with sheets 'UOH History', 'Weekly Demand', 'Delivery History'
  - Google Sheets via a service-account credentials JSON (gspread)

Column convention in Weekly Demand sheet
-----------------------------------------
Column 0 (A): Store ID       – string identifier, e.g. "HEB-592"
Column 1 (B): Store Number   – numeric store number, e.g. 592
Columns 2+:   Date columns   – column headers are dates; values are weekly demand units.
              Some trailing columns (e.g. BA–BD) are auto-calculated summaries and are
              NOT dates; these are detected and skipped via pd.to_datetime(..., errors='coerce').
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Excel loader
# ---------------------------------------------------------------------------

def load_from_xlsx(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three main sheets from an Excel demand tracker.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .xlsx file.

    Returns
    -------
    (uoh_df, demand_df, deliveries_df)
        Three DataFrames corresponding to UOH History, Weekly Demand, and
        Delivery History sheets.  Row 3 is expected to be the header row
        (header=2 in zero-based indexing).
    """
    uoh = pd.read_excel(filepath, sheet_name="UOH History", header=2)
    demand = pd.read_excel(filepath, sheet_name="Weekly Demand", header=2)
    deliveries = pd.read_excel(filepath, sheet_name="Delivery History", header=2)
    return uoh, demand, deliveries


# ---------------------------------------------------------------------------
# Google Sheets loader
# ---------------------------------------------------------------------------

def load_from_gsheet(
    sheet_name: str, credentials_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three main worksheets from a Google Sheet.

    Parameters
    ----------
    sheet_name : str
        The name of the Google Spreadsheet (as it appears in Google Drive).
    credentials_file : str
        Path to a service-account credentials JSON file with access to the
        spreadsheet.

    Returns
    -------
    (uoh_df, demand_df, deliveries_df)
        DataFrames matching the structure produced by load_from_xlsx.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as exc:
        raise ImportError(
            "gspread and google-auth are required for Google Sheets support. "
            "Install them with: pip install gspread google-auth"
        ) from exc

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
    client = gspread.authorize(creds)
    spreadsheet = client.open(sheet_name)

    def _worksheet_to_df(ws_name: str) -> pd.DataFrame:
        worksheet = spreadsheet.worksheet(ws_name)
        records = worksheet.get_all_values()
        if len(records) < 3:
            return pd.DataFrame()
        # Row index 2 (third row) is the header.
        header = records[2]
        data_rows = records[3:]
        df = pd.DataFrame(data_rows, columns=header)
        # Attempt to coerce numeric-looking columns.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        return df

    uoh = _worksheet_to_df("UOH History")
    demand = _worksheet_to_df("Weekly Demand")
    deliveries = _worksheet_to_df("Delivery History")
    return uoh, demand, deliveries


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------

def _get_date_columns(demand_df: pd.DataFrame) -> List:
    """
    Return the list of column labels that can be parsed as dates.

    This filters out metadata columns (Store ID, Store Number) and any
    auto-generated summary columns (e.g. BA–BD) whose headers are plain
    strings or numbers rather than dates.
    """
    date_cols = []
    for col in demand_df.columns:
        parsed = pd.to_datetime(col, errors="coerce")
        if parsed is not pd.NaT and not pd.isnull(parsed):
            date_cols.append(col)
    return date_cols


def _store_id_column(demand_df: pd.DataFrame) -> str:
    """Return the name of the Store ID column (first column)."""
    return demand_df.columns[0]


def _store_number_column(demand_df: pd.DataFrame) -> str:
    """Return the name of the Store Number column (second column)."""
    return demand_df.columns[1]


# ---------------------------------------------------------------------------
# Public accessor functions
# ---------------------------------------------------------------------------

def get_all_store_ids(demand_df: pd.DataFrame) -> List:
    """
    Return a list of all store IDs from the demand DataFrame.

    Parameters
    ----------
    demand_df : pd.DataFrame
        The Weekly Demand sheet as loaded by load_from_xlsx or load_from_gsheet.

    Returns
    -------
    list
        Store IDs (values from the first column), with NaN rows dropped.
    """
    id_col = _store_id_column(demand_df)
    return demand_df[id_col].dropna().tolist()


def get_store_number(demand_df: pd.DataFrame, store_id) -> Optional[str]:
    """
    Return the store number (column B) for a given store ID.

    Parameters
    ----------
    demand_df : pd.DataFrame
    store_id : any
        Value matching an entry in the Store ID column.

    Returns
    -------
    str or None
        The store number, or None if the store_id is not found.
    """
    id_col = _store_id_column(demand_df)
    num_col = _store_number_column(demand_df)
    rows = demand_df[demand_df[id_col] == store_id]
    if rows.empty:
        return None
    return str(rows.iloc[0][num_col])


def get_store_demand_series(demand_df: pd.DataFrame, store_id) -> pd.Series:
    """
    Extract a clean demand time series for a single store.

    Steps:
    1. Find the row(s) whose Store ID matches store_id.
    2. Restrict to date columns (detected via pd.to_datetime).
    3. Transpose to a Series indexed by date.
    4. Drop NaN values.
    5. Drop non-negative values (demand < 0 is an inventory adjustment artefact).
    6. Sort by date ascending.

    Parameters
    ----------
    demand_df : pd.DataFrame
    store_id : any

    Returns
    -------
    pd.Series
        Index: pd.Timestamp dates, Values: float demand units.
        Returns an empty Series if the store is not found or has no valid data.
    """
    id_col = _store_id_column(demand_df)
    date_cols = _get_date_columns(demand_df)

    rows = demand_df[demand_df[id_col] == store_id]
    if rows.empty or not date_cols:
        return pd.Series(dtype=float)

    row = rows.iloc[0]
    raw = row[date_cols]

    # Build Series with proper datetime index.
    series = pd.Series(
        data=pd.to_numeric(raw.values, errors="coerce"),
        index=pd.to_datetime(date_cols, errors="coerce"),
        dtype=float,
    )

    # Remove NaT index entries (shouldn't happen given _get_date_columns, but defensive).
    series = series[series.index.notna()]
    # Drop NaN demand.
    series = series.dropna()
    # Drop negative demand.
    series = series[series >= 0]
    # Sort chronologically.
    series = series.sort_index()

    return series


def get_aggregate_demand(demand_df: pd.DataFrame, min_stores: int = 3) -> pd.Series:
    """
    Compute total demand across all stores for each date.

    Only dates with readings from at least `min_stores` stores are included.
    This prevents under-counting on weeks where only a subset of stores
    reported, which would artificially deflate the aggregate forecast.

    Parameters
    ----------
    demand_df : pd.DataFrame
    min_stores : int
        Minimum number of stores that must have a non-NaN, non-negative demand
        entry for a date to be included in the aggregate.

    Returns
    -------
    pd.Series
        Index: pd.Timestamp dates, Values: float total demand.
    """
    date_cols = _get_date_columns(demand_df)
    if not date_cols:
        return pd.Series(dtype=float)

    # Extract the demand sub-matrix (stores x dates).
    demand_matrix = demand_df[date_cols].apply(pd.to_numeric, errors="coerce")
    # Negative values treated as missing for aggregation purposes.
    demand_matrix = demand_matrix.where(demand_matrix >= 0, other=np.nan)

    # Count how many stores have valid data per date.
    store_counts = demand_matrix.notna().sum(axis=0)
    # Sum across stores per date.
    date_totals = demand_matrix.sum(axis=0, skipna=True)

    # Filter to dates meeting the minimum-store threshold.
    valid_mask = store_counts >= min_stores
    date_totals = date_totals[valid_mask]

    # Build a properly-indexed Series.
    agg_series = pd.Series(
        data=date_totals.values,
        index=pd.to_datetime(date_cols, errors="coerce")[valid_mask],
        dtype=float,
    )
    agg_series = agg_series[agg_series.index.notna()].sort_index()

    return agg_series
