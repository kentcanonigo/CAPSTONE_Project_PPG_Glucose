# 02_Machine_Learning_Mendeley/src/config.py

import os

# --- Project Root Path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 

# --- Data Paths ---
# Corrected path to go one level up from src to find the parent directory, then into PPG_Dataset
# PLEASE VERIFY THIS PATH IS CORRECT FOR YOUR FOLDER STRUCTURE
BASE_DATA_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "PPG_Dataset"))

if not os.path.exists(BASE_DATA_PATH):
    print(f"FATAL ERROR: Mendeley data path not found: {BASE_DATA_PATH}")
    print("Please ensure the 'PPG_Dataset' folder is located alongside the 'src' folder in '02_Machine_Learning_Mendeley'.")
    sys.exit(1) # Exit if data path is wrong

RAW_DATA_FOLDER = os.path.join(BASE_DATA_PATH, "RawData")
LABELS_FILE = os.path.join(BASE_DATA_PATH, "Labels", "Total.csv")

# --- Signal Processing Parameters ---
ORIGINAL_FS = 2175
TARGET_FS = 100
FILTER_LOWCUT = 0.5
FILTER_HIGHCUT = 5.0
FILTER_ORDER = 2
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 3
WINDOW_DURATION_SEC = 5
SAMPLES_PER_WINDOW = int(WINDOW_DURATION_SEC * TARGET_FS)

# --- Model Training Parameters ---
TEST_SET_SIZE = 0.3
RANDOM_STATE = 42

# --- MODIFIED LightGBM parameters for potentially better sensitivity/generalization ---
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 41,          # Increased from 31: allows more complex trees
    'learning_rate': 0.02,     # Decreased from 0.05: smaller steps, needs more rounds
    'feature_fraction': 0.8,   # Use a subset of features per tree
    'bagging_fraction': 0.8,   # Use a subset of data per tree (row sampling)
    'bagging_freq': 5,         # Perform bagging every 5 iterations
    'min_child_samples': 15,   # Decreased from default 20: allows splits on smaller groups
    'lambda_l1': 0.1,          # L1 regularization
    'lambda_l2': 0.1,          # L2 regularization
    'verbose': -1,             
    'seed': RANDOM_STATE,
    'n_jobs': -1               # Use all available cores for training
}
LGBM_NUM_BOOST_ROUND = 1500 # Increased from 200
LGBM_EARLY_STOPPING_ROUNDS = 50 # Increased from 20

# --- Output Paths ---
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models") 
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)

# Give new names to distinguish from the original model
SAVED_MODEL_NAME = "lgbm_glucose_model_retrained_v1.txt"
SAVED_SCALER_NAME = "mendeley_feature_scaler_retrained_v1.pkl"
SAVED_FEATURES_NAME = "model_features_retrained_v1.json"
