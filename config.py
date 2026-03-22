CONFIG = {
    'n_backtest': 5,
    'min_data_points': 4,
    'min_stores_for_agg': 3,
    'ses_alpha_grid': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'holt_grid': [0.1, 0.3, 0.5, 0.7, 0.9],
    'output_csv': True,
    'output_gsheet': False,
    'output_plots': True,
    'outputs_root': 'outputs',  # parent directory; each run gets its own subfolder
    'forecast_weeks': 4,        # how many weeks ahead to extend the forecast line
}
