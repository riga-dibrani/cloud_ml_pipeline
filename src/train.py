# src/train.py

from ingest import load_train_data, load_test_data, load_true_rul
from features import add_temporal_features
from preprocess import preprocess_training_data
from evaluate import evaluate_validation
from infer import run_test_inference

def main():
    df_train, columns = load_train_data()
    df_train = add_temporal_features(df_train)

    X_scaled, y, scaler, feature_cols = preprocess_training_data(df_train)

    rf, val_idx, y_val, y_pred = evaluate_validation(
        df_train, X_scaled, y
    )

    df_test = load_test_data(columns)
    df_test = add_temporal_features(df_test)

    y_test_true = load_true_rul()
    run_test_inference(df_test, feature_cols, scaler, rf, y_test_true)

if __name__ == "__main__":
    main()
