# model_trainer.py

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json
import os
import config

def calculate_mard(y_true, y_pred_vals):
    """Calculates Mean Absolute Relative Difference (mARD)."""
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred_vals)
    # Ensure y_true_np is not zero to avoid division by zero
    mask = y_true_np != 0
    if np.sum(mask) == 0:
        print("Warning: All true values are zero in mARD calculation. Returning NaN.")
        return np.nan
    if len(y_true_np[mask]) != len(y_pred_np[mask]):
        print("Warning: Length mismatch after masking zero true values in mARD. Returning NaN.")
        return np.nan

    mard_val = np.mean(np.abs(y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask]) * 100
    return mard_val

def train_evaluate_model(feature_df, labels_series):
    """Trains the LightGBM model and evaluates it."""
    if feature_df.empty or len(feature_df) != len(labels_series):
        print("Error: Feature DataFrame is empty or its length does not match labels. Aborting training.")
        return None, {}

    # Handle any remaining NaNs in features (e.g., fill with median or mean)
    # This should ideally be done before splitting, or ensure test set doesn't use train set info for filling
    if feature_df.isnull().values.any():
        print("Warning: NaNs found in feature DataFrame. Filling with median.")
        feature_df = feature_df.fillna(feature_df.median()) # Fill NaNs based on the whole dataset before split

    X = feature_df
    y = labels_series

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SET_SIZE, random_state=config.RANDOM_STATE
    )

    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    lgb_train_data = lgb.Dataset(X_train, label=y_train)
    lgb_test_data = lgb.Dataset(X_test, label=y_test, reference=lgb_train_data)

    print("Starting LightGBM model training...")
    model = lgb.train(
        config.LGBM_PARAMS,
        lgb_train_data,
        num_boost_round=config.LGBM_NUM_BOOST_ROUND,
        valid_sets=[lgb_train_data, lgb_test_data],
        callbacks=[
            lgb.early_stopping(config.LGBM_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=50) # Logs every 50 rounds
        ]
    )
    print("Model training finished.")

    # Evaluate on the test set
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    mard_test = calculate_mard(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(f"\n--- Model Evaluation on Test Set ---")
    print(f"  mARD: {mard_test:.2f}%")
    print(f"  RMSE: {rmse_test:.2f}")
    print(f"  MAE:  {mae_test:.2f}")

    evaluation_results = {
        "mARD": mard_test,
        "RMSE": rmse_test,
        "MAE": mae_test,
        "num_train_samples": X_train.shape[0],
        "num_test_samples": X_test.shape[0],
        "best_iteration": model.best_iteration
    }
    
    # Save feature names with the model
    feature_names = list(X_train.columns) # Get feature names from the training DataFrame
    model.feature_name_ = feature_names # Store feature names in the model object if not already there

    return model, evaluation_results, feature_names

def save_model_and_features(model, feature_names, model_dir, model_filename, features_filename):
    """Saves the trained LightGBM model and its feature names."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_filename)
    features_path = os.path.join(model_dir, features_filename)

    try:
        model.save_model(model_path) # LightGBM's built-in save
        print(f"Trained model saved to: {model_path}")

        with open(features_path, 'w') as f:
            json.dump(feature_names, f)
        print(f"Model feature names saved to: {features_path}")
    except Exception as e:
        print(f"Error saving model or features: {e}")


def load_model_and_features(model_dir, model_filename, features_filename):
    """Loads a trained LightGBM model and its feature names."""
    model_path = os.path.join(model_dir, model_filename)
    features_path = os.path.join(model_dir, features_filename)

    try:
        loaded_model = lgb.Booster(model_file=model_path) # LightGBM's built-in load
        print(f"Model loaded from: {model_path}")

        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        print(f"Model feature names loaded from: {features_path}")
        loaded_model.feature_name_ = feature_names # Important for predict if input is DataFrame
        return loaded_model, feature_names
    except Exception as e:
        print(f"Error loading model or features: {e}")
        return None, None

if __name__ == '__main__':
    # Example usage: Create dummy data for testing
    num_samples = 100
    num_features = 10 # Must match number of features from feature_extraction.py
    dummy_features = pd.DataFrame(np.random.rand(num_samples, num_features),
                                  columns=[f'feat_{i}' for i in range(num_features)])
    dummy_labels = pd.Series(np.random.uniform(70, 180, num_samples)) # Glucose-like values

    print("Testing model training with dummy data...")
    trained_model, eval_results, f_names = train_evaluate_model(dummy_features, dummy_labels)

    if trained_model:
        print("\nDummy model training and evaluation successful.")
        print(f"Evaluation results: {eval_results}")
        save_model_and_features(trained_model, f_names, config.MODEL_OUTPUT_DIR, "dummy_model.txt", "dummy_features.json")
        loaded_m, loaded_f = load_model_and_features(config.MODEL_OUTPUT_DIR, "dummy_model.txt", "dummy_features.json")
        if loaded_m:
            print("Dummy model loaded successfully.")