# src/features.py

import numpy as np

rolling_sensors = [
    'sensor_measurement_2',
    'sensor_measurement_3',
    'sensor_measurement_4',
    'sensor_measurement_7',
    'sensor_measurement_11'
]

window_size = 5

def rolling_slope(series):
    x = np.arange(len(series))
    if len(series) < 2:
        return 0.0
    return np.polyfit(x, series, 1)[0]


def add_temporal_features(df):
    df = df.sort_values(['unit_number', 'time_in_cycles'])

    for sensor in rolling_sensors:
        df[f'{sensor}_roll_mean'] = (
            df.groupby('unit_number')[sensor]
            .rolling(window_size)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f'{sensor}_roll_std'] = (
            df.groupby('unit_number')[sensor]
            .rolling(window_size)
            .std()
            .reset_index(level=0, drop=True)
        )

    for sensor in rolling_sensors:
        df[f'{sensor}_roll_slope'] = (
            df.groupby('unit_number')[sensor]
            .rolling(window_size)
            .apply(rolling_slope, raw=False)
            .reset_index(level=0, drop=True)
        )

    for sensor in rolling_sensors:
        df[f'{sensor}_delta'] = (
            df.groupby('unit_number')[sensor]
            .diff()
        )

    df = df.dropna().reset_index(drop=True)
    return df
