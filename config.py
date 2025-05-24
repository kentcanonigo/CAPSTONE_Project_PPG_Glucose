# This file will store all configurations, making it easy to change paths or parameters

import os

# --- Project Root Path ---
# Assuming your script is run from the CAPSTONE-... directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of config.py

# --- Data Paths ---
BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "PPG_Dataset")
RAW_DATA_FOLDER = os.path.join(BASE_DATA_PATH, "RawData")
LABELS_FILE = os.path.join(BASE_DATA_PATH, "Labels", "Total.csv")

# --- Signal Processing Parameters ---
ORIGINAL_FS = 2175  # Hz (from Mendeley dataset description for the raw signals)
TARGET_FS = 50      # Hz (target sampling rate for processing, as per your thesis)

# Butterworth Bandpass Filter parameters (Thesis 3.6.1)
FILTER_LOWCUT = 0.5
FILTER_HIGHCUT = 5.0
FILTER_ORDER = 2

# Savitzky-Golay Filter parameters (Thesis 3.6.1)
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 3

# Segmentation parameters (Thesis 3.5.2, 3.6.2)
WINDOW_DURATION_SEC = 5
SAMPLES_PER_WINDOW = int(WINDOW_DURATION_SEC * TARGET_FS) # Should be 250

# --- Model Training Parameters ---
TEST_SET_SIZE = 0.3  # 30% for testing, 70% for training
RANDOM_STATE = 42    # For reproducible splits

# LightGBM parameters (Thesis 3.6.5)
LGBM_PARAMS = {
    'objective': 'regression', # For MSE, as per your thesis L_total
    'metric': 'rmse',          # Root Mean Squared Error
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1,
    'seed': RANDOM_STATE
}
LGBM_NUM_BOOST_ROUND = 200
LGBM_EARLY_STOPPING_ROUNDS = 20

# --- Output Paths ---
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models") # Directory to save trained models
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)

SAVED_MODEL_NAME = "lgbm_glucose_model.txt"
SAVED_FEATURES_NAME = "model_features.json"