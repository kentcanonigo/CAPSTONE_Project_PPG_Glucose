# =============================================================================
# tune_hyperparameters.py
#
# Description:
# This script performs a systematic hyperparameter search for the LightGBM model
# using GridSearchCV. It loads the pre-computed features and corresponding labels,
# splits the data, and then searches through a defined grid of parameters to find
# the combination that yields the best performance (lowest RMSE) through
# cross-validation.
#
# Author: [Your Name]
# Date: June 8, 2025
# =============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import sys
import os

# Important: Add the src directory to the Python path to import config
# This allows the script to find and import your config.py file
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config
except ImportError:
    print("FATAL ERROR: config.py not found. Make sure this script is in the same 'src' directory as config.py.")
    sys.exit(1)

print("--- Starting Hyperparameter Tuning ---")

# --- 1. Load BOTH features and labels files ---
try:
    # Construct paths to the files using paths from config.py for consistency
    features_csv_path = os.path.abspath(os.path.join(config.PROJECT_ROOT, "..", "ppg_features_master.csv"))
    labels_csv_path = config.LABELS_FILE
    
    print(f"Loading features from: {features_csv_path}")
    print(f"Loading labels from: {labels_csv_path}")
    
    features_df = pd.read_csv(features_csv_path)
    labels_df = pd.read_csv(labels_csv_path)

except FileNotFoundError as e:
    print(f"FATAL ERROR: A required file was not found.")
    print(f"Missing file details: {e}")
    print("Please ensure 'ppg_features_master.csv' is in the project root and 'Total.csv' is in the Labels folder.")
    sys.exit(1)


# --- 2. Create a common 'ID' column in the features DataFrame for merging ---
# This extracts the number (ID) from the 'file' column.
# For example, from a filename like 'record_1.txt', it extracts '1'.
print("Creating common ID for merging...")
features_df['ID'] = features_df['file'].str.extract('(\d+)').astype(int)


# --- 3. Merge the two DataFrames into one master DataFrame ---
print("Merging features and labels...")
master_df = pd.merge(features_df, labels_df, on='ID')


# --- 4. Prepare X (features) and y (labels) from the MERGED DataFrame ---
print("Preparing final X and y for training...")
# The label is 'Glucose' from the labels file
y = master_df['Glucose']
# The features are all columns EXCEPT the ones we don't need for training
# This includes metadata from both original files
X = master_df.drop(columns=['Glucose', 'ID', 'Gender', 'Age', 'Height', 'Weight', 'file', 'segment'])


# --- 5. Split and Scale Data ---
# Use the same split parameters from config.py to ensure consistency
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SET_SIZE, random_state=config.RANDOM_STATE
)

# Scaling is crucial for many ML models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(f"Data successfully loaded and prepared. Training set shape: {X_train_scaled.shape}")


# --- 6. Define the Parameter Grid for Grid Search ---
# This is the "menu" of hyperparameters you want to test.
# The search will try every single combination.
param_grid = {
    'num_leaves': [31, 41, 51],
    'learning_rate': [0.01, 0.02, 0.05],
    'n_estimators': [1000, 1500, 2000], # n_estimators is equivalent to num_boost_round
    'lambda_l1': [0.1, 0.5, 1.0],      # L1 Regularization
    'lambda_l2': [0.1, 0.5, 1.0]       # L2 Regularization
}


# --- 7. Set Up and Run GridSearchCV ---
# Use the base model parameters from config.py that are NOT being tuned
estimator = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    feature_fraction=config.LGBM_PARAMS.get('feature_fraction', 0.8),
    bagging_fraction=config.LGBM_PARAMS.get('bagging_fraction', 0.8),
    bagging_freq=config.LGBM_PARAMS.get('bagging_freq', 5),
    seed=config.RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

# Configure the grid search object
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error', # This corresponds to RMSE
    cv=5,       # 5-fold cross-validation
    n_jobs=-1,  # Use all available CPU cores
    verbose=2   # Shows detailed progress during the search
)

print("\nStarting Grid Search... This may take a while.")
grid_search.fit(X_train_scaled, y_train)
print("Grid Search Complete.")


# --- 8. Analyze and Print the Results ---
print("\n" + "="*50)
print("---              TUNING RESULTS              ---")
print("="*50)

print("\n--- Best Parameters Found ---")
print(grid_search.best_params_)

print("\n--- Best Cross-Validation Score (RMSE) ---")
# The score is negative, so we multiply by -1 to get the actual RMSE
best_rmse = -1 * grid_search.best_score_
print(f"Best CV RMSE: {best_rmse:.4f}")
print("\nRecommendation: Update your config.py with these best parameters and re-run your main training pipeline.")
print("="*50)