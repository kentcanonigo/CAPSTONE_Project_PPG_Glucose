# Configuration for the PPG-Glucose Estimation Project

import os

# --- Data Paths ---
# Base directory for the Mendeley PPG Dataset
# Adjust this path based on where you store your 'PPG_Dataset' folder
BASE_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../PPG_Dataset')

RAW_DATA_FOLDER = os.path.join(BASE_DATASET_DIR, 'RawData')
LABELS_FILE = os.path.join(BASE_DATASET_DIR, 'Labels', 'Total.csv') # Based on your data_loader.py


# --- Signal Processing Parameters ---
TARGET_FS = 125 # Target sampling frequency after resampling (Hz)
ORIGINAL_FS = 256 # Original sampling frequency of the raw data (Hz)
USE_MENDELEY_FS = True # Flag to use Mendeley's sampling frequency if applicable
# You might need to adjust these based on your preprocessing.py logic
# Example values, adjust as needed based on your dataset characteristics
SEGMENT_LENGTH_SEC = 10 # Length of each PPG segment in seconds
OVERLAP_RATIO = 0.5     # Overlap between segments (e.g., 0.5 for 50% overlap)
FILTER_LOWCUT = 0.5   # Low-pass filter cutoff frequency (Hz)
FILTER_HIGHCUT = 8    # High-pass filter cutoff frequency (Hz)
FILTER_ORDER = 4        # Order of the Butterworth filter

# Savitzky-Golay Filter parameters - ADDED THESE LINES
SAVGOL_WINDOW = 5       # Window length for Savitzky-Golay filter (must be odd)
SAVGOL_POLYORDER = 3    # Polynomial order for Savitzky-Golay filter

# Calculated based on other parameters
SAMPLES_PER_WINDOW = int(SEGMENT_LENGTH_SEC * TARGET_FS) # Number of samples in each segment


# --- Model Training Parameters ---
TEST_SET_SIZE = 0.3 # 30% for test set, 70% for training (subject-level split)
RANDOM_STATE = 42   # For reproducibility of splits and model training

# LightGBM Model Parameters
LGBM_PARAMS = {
    'objective': 'regression_l1', # MAE objective, good for robustness to outliers
    'metric': 'mae',              # Metric to optimize
    'n_estimators': 1000,         # Number of boosting rounds
    'learning_rate': 0.05,
    'feature_fraction': 0.8,      # Fraction of features to consider at each split
    'bagging_fraction': 0.8,      # Fraction of data to use for each boosting round
    'bagging_freq': 1,
    'lambda_l1': 0.1,             # L1 regularization
    'lambda_l2': 0.1,             # L2 regularization
    'num_leaves': 31,             # Maximum number of leaves in one tree
    'verbose': -1,                # Suppress verbose output during training
    'n_jobs': -1,                 # Use all available CPU cores
    'seed': RANDOM_STATE,         # Random seed for reproducibility
    'boosting_type': 'gbdt',      # Traditional Gradient Boosting Decision Tree
}
LGBM_NUM_BOOST_ROUND = 1000       # Maximum number of boosting rounds
LGBM_EARLY_STOPPING_ROUNDS = 50   # Stop if no improvement for this many rounds

# K-Fold Cross-Validation splits
KFOLD_SPLITS = 5 # Number of folds for subject-level cross-validation

# --- Model Output Paths ---
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
SAVED_MODEL_NAME = "lgbm_glucose_model_retrained_v1.txt"
SAVED_SCALER_NAME = "mendeley_feature_scaler_retrained_v1.pkl"
SAVED_FEATURES_NAME = "model_features_retrained_v1.json"