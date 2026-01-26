# src/infer.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_test_inference(df_test, feature_cols, scaler, rf, y_test_true):

    # retrieve last cycle per engine
    df_test_last = df_test.groupby('unit_number').last().reset_index()

    print("Test last-cycle shape:", df_test_last.shape)
    print(df_test_last.head())

    X_test = df_test_last[feature_cols]
    X_test_scaled = scaler.transform(X_test)

    print("Scaled test features shape:", X_test_scaled.shape)

    y_test_pred = rf.predict(X_test_scaled)

    print("Predicted RUL (first 10):")
    print(y_test_pred[:10])

    print("True RUL (first 10):")
    print(y_test_true[:10])

    rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mae_test = mean_absolute_error(y_test_true, y_test_pred)
    r2_test = r2_score(y_test_true, y_test_pred)

    print("Test Set Evaluation:")
    print(f"RMSE: {rmse_test:.2f}")
    print(f"MAE:  {mae_test:.2f}")
    print(f"R²:   {r2_test:.2f}")

    test_error = y_test_pred - y_test_true
    print(test_error)

    # Error distribution
    plt.figure(figsize=(6,4))
    plt.hist(test_error, bins=30, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Zero error')
    plt.xlabel("Prediction Error (Predicted − True RUL)")
    plt.ylabel("Frequency")
    plt.title("Test Set Error Distribution")
    plt.legend()
    plt.show()

    # Error vs True RUL
    plt.figure(figsize=(6,4))
    plt.scatter(y_test_true, test_error, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("True RUL")
    plt.ylabel("Prediction Error")
    plt.title("Prediction Error vs True RUL (Test Set)")
    plt.show()

    # Predicted vs True
    plt.figure(figsize=(5,5))
    plt.scatter(y_test_true, y_test_pred, alpha=0.6)
    plt.plot([0, 150], [0, 150], 'r--')
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs True RUL (Test)")
    plt.show()

    # Over / under estimation
    overestimation_rate = np.mean(test_error > 0)
    underestimation_rate = np.mean(test_error < 0)

    print(f"Overestimation rate: {overestimation_rate:.2f}")
    print(f"Underestimation rate: {underestimation_rate:.2f}")

    # Error by RUL regime
    early_mask = y_test_true > 80
    mid_mask = (y_test_true <= 80) & (y_test_true > 30)
    late_mask = y_test_true <= 30

    print("MAE by RUL regime:")
    print("Early-life:", mean_absolute_error(
        y_test_true[early_mask], y_test_pred[early_mask]
    ))
    print("Mid-life:", mean_absolute_error(
        y_test_true[mid_mask], y_test_pred[mid_mask]
    ))
    print("Late-life:", mean_absolute_error(
        y_test_true[late_mask], y_test_pred[late_mask]
    ))
