# src/paths.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(BASE_DIR, "data", "raw", "train_FD001.txt")
TEST_FILE = os.path.join(BASE_DIR, "data", "raw", "test_FD001.txt")
RUL_FILE  = os.path.join(BASE_DIR, "data", "raw", "RUL_FD001.txt")

X_TRAIN_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "X_train_scaled.npy")
Y_TRAIN_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "y_train.npy")
