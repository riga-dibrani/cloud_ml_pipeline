# src/preprocess.py

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from features import rolling_sensors
from paths import X_TRAIN_PATH, Y_TRAIN_PATH

def preprocess_training_data(df_train):
    feature_cols = (
        ['time_in_cycles',
         'operational_setting_1',
         'operational_setting_2',
         'operational_setting_3']
        + [f'sensor_measurement_{i}' for i in range(1, 22)]
        + [f'{s}_roll_mean' for s in rolling_sensors]
        + [f'{s}_roll_std' for s in rolling_sensors]
        + [f'{s}_roll_slope' for s in rolling_sensors]
        + [f'{s}_delta' for s in rolling_sensors]
    )

    X = df_train[feature_cols]

    max_rul = 125
    df_train['RUL_capped'] = df_train['RUL'].clip(upper=max_rul)
    y = df_train['RUL_capped']

    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    plt.figure(figsize=(10,4))
    plt.plot(X_scaled[:100, 0], label='Operational Setting 1 (scaled)')
    plt.plot(X_scaled[:100, 3], label='Sensor 3 (scaled)')
    plt.legend()
    plt.title("Sample Scaled Features (First 100 cycles)")
    plt.show()

    np.save(X_TRAIN_PATH, X_scaled)
    np.save(Y_TRAIN_PATH, y.values)

    return X_scaled, y.values, scaler, feature_cols
