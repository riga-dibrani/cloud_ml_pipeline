# src/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_validation(df_train, X_scaled, y):
    groups = df_train['unit_number'].values

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X_scaled, y, groups))

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("Baseline Model Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.2f}")

    # Error vs True RUL
    error = y_val - y_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_val, np.abs(error), alpha=0.5)
    plt.xlabel("True RUL")
    plt.ylabel("Absolute Error")
    plt.title("Error vs True RUL")
    plt.show()

    # Pick one validation engine
    val_units = df_train.iloc[val_idx]['unit_number']
    engine_id = val_units.unique()[0]

    engine_mask = (df_train['unit_number'] == engine_id)
    engine_mask &= np.isin(df_train.index, val_idx)

    engine_data = df_train[engine_mask].sort_values('time_in_cycles')
    engine_X = X_scaled[engine_data.index]
    engine_y_true = engine_data['RUL'].values
    engine_y_pred = rf.predict(engine_X)

    plt.figure(figsize=(8,4))
    plt.plot(engine_data['time_in_cycles'], engine_y_true, label='True RUL')
    plt.plot(engine_data['time_in_cycles'], engine_y_pred, label='Predicted RUL')
    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.title(f"RUL Prediction – Engine {engine_id}")
    plt.legend()
    plt.show()

    # Per-engine MAE
    val_df = df_train.iloc[val_idx].copy()
    val_df['predicted_RUL'] = y_pred
    val_df['abs_error'] = np.abs(val_df['RUL'] - val_df['predicted_RUL'])

    engine_mae = val_df.groupby('unit_number')['abs_error'].mean()
    print(engine_mae.describe())

    engine_mae.sort_values().plot(kind='bar', figsize=(10,3))
    plt.ylabel("MAE")
    plt.title("Per-Engine MAE")
    plt.show()

    # Monotonicity check
    def count_increases(pred):
        return np.sum(np.diff(pred) > 0)

    engine_id = val_df['unit_number'].unique()[0]
    engine_mask = val_df['unit_number'] == engine_id
    engine_y_pred_single = val_df.loc[engine_mask, 'predicted_RUL'].values

    print(
        f"Prediction increases for Engine {engine_id}:",
        count_increases(engine_y_pred_single)
    )

    # Residual distribution
    plt.figure(figsize=(5,3))
    plt.hist(y_val - y_pred, bins=40)
    plt.xlabel("Residual")
    plt.title("Residual Distribution")
    plt.show()

    return rf, val_idx, y_val, y_pred
