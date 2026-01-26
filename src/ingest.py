# src/ingest.py

import pandas as pd
from paths import TRAIN_FILE, TEST_FILE, RUL_FILE

def load_train_data():
    df_train = pd.read_csv(TRAIN_FILE, sep=r"\s+", header=None)

    columns = ['unit_number', 'time_in_cycles',
               'operational_setting_1',
               'operational_setting_2',
               'operational_setting_3']
    sensor_cols = [f'sensor_measurement_{i}' for i in range(1, 22)]
    columns += sensor_cols
    df_train.columns = columns

    print("Train shape:", df_train.shape)
    print(df_train.head())
    print(df_train['unit_number'].nunique(), "engines in train set")

    max_cycles = (
        df_train.groupby('unit_number')['time_in_cycles']
        .max()
        .reset_index()
    )
    max_cycles.columns = ['unit_number', 'max_cycle']

    df_train = df_train.merge(max_cycles, on='unit_number', how='left')
    df_train['RUL'] = df_train['max_cycle'] - df_train['time_in_cycles']

    print(df_train[['unit_number', 'time_in_cycles', 'RUL']].head())
    print(df_train['RUL'].describe())

    return df_train, columns


def load_test_data(columns):
    df_test = pd.read_csv(TEST_FILE, sep=r"\s+", header=None)
    df_test.columns = columns

    print("Test shape:", df_test.shape)
    print(df_test.head())
    print(df_test['unit_number'].nunique(), "engines in test set")

    return df_test


def load_true_rul():
    return pd.read_csv(RUL_FILE, header=None)[0].values
