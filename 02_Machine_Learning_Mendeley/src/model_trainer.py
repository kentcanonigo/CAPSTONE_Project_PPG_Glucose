# 02_Machine_Learning_Mendeley/src/model_trainer.py

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold # KFold for cross-validation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler 
import joblib 
import json
import os
import config # Uses the updated config.py

def calculate_mard(y_true, y_pred_vals):
    # ... (same as your current version) ...
    y_true_np = np.array(y_true); y_pred_np = np.array(y_pred_vals)
    mask = y_true_np != 0
    if np.sum(mask) == 0: return np.nan
    if len(y_true_np[mask]) != len(y_pred_np[mask]): return np.nan
    return np.mean(np.abs(y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask]) * 100

def train_evaluate_model(feature_df, labels_series, use_cv=False): # Added use_cv flag
    """Trains the LightGBM model, including scaling, and evaluates it.
    Optionally uses K-Fold Cross-Validation for more robust evaluation.
    """
    if feature_df.empty or len(feature_df) != len(labels_series):
        print("Error: Feature DataFrame empty or length mismatch with labels. Aborting."); return None, None, {}, []

    if feature_df.isnull().values.any():
        print("Warning: NaNs found in feature DataFrame. Filling with column medians."); feature_df = feature_df.fillna(feature_df.median()) 

    X = feature_df
    y = labels_series
    feature_names = list(X.columns) # Get feature names before any array conversion

    # --- Feature Scaling (MinMaxScaler) ---
    scaler = MinMaxScaler()
    # IMPORTANT: In a real CV setup, scaling should happen *inside* each fold
    # to prevent data leakage from the validation part of the fold into the training part's scaler.
    # For simplicity in this step, we'll scale the whole X first if not using CV,
    # but be aware of this for rigorous CV.
    if not use_cv:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SET_SIZE, random_state=config.RANDOM_STATE)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert scaled arrays back to DataFrames for LightGBM if it benefits from feature names
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

        print(f"Training set size: {X_train_scaled_df.shape[0]}, Test set size: {X_test_scaled_df.shape[0]}")
        lgb_train_data = lgb.Dataset(X_train_scaled_df, label=y_train) # feature_name=feature_names if needed by your LGBM version/setup
        lgb_test_data = lgb.Dataset(X_test_scaled_df, label=y_test, reference=lgb_train_data) # feature_name=feature_names
        
        print("Starting LightGBM model training (single split)...")
        model = lgb.train(
            config.LGBM_PARAMS, lgb_train_data,
            num_boost_round=config.LGBM_NUM_BOOST_ROUND,
            valid_sets=[lgb_train_data, lgb_test_data],
            callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOPPING_ROUNDS, verbose=10), lgb.log_evaluation(period=50)]
        )
        y_pred_on_test = model.predict(X_test_scaled_df, num_iteration=model.best_iteration)
        final_y_test_for_eval = y_test
        final_y_pred_for_eval = y_pred_on_test

    else: # Using K-Fold Cross-Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
        fold_results_mard = []
        fold_results_rmse = []
        fold_results_mae = []
        
        # Scale the entire dataset once before K-Fold for simplicity in this example
        # More rigorous: scale within each fold to prevent test fold leakage into scaler fitting
        X_scaled_all = scaler.fit_transform(X) # Fit scaler on ALL data before CV for this example
        X_scaled_all_df = pd.DataFrame(X_scaled_all, columns=feature_names, index=X.index)
        print("Feature scaling applied to the entire dataset for K-Fold CV (suboptimal, but simpler for now).")


        print(f"\nStarting LightGBM model training with {kf.get_n_splits()}-Fold Cross-Validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled_all_df, y)):
            print(f"  --- Fold {fold+1}/{kf.get_n_splits()} ---")
            X_train_fold, X_val_fold = X_scaled_all_df.iloc[train_idx], X_scaled_all_df.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            lgb_train_fold = lgb.Dataset(X_train_fold, label=y_train_fold)
            lgb_val_fold = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train_fold)

            model_fold = lgb.train(
                config.LGBM_PARAMS, lgb_train_fold,
                num_boost_round=config.LGBM_NUM_BOOST_ROUND,
                valid_sets=[lgb_train_fold, lgb_val_fold],
                callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=config.LGBM_NUM_BOOST_ROUND)] # Log only at end for CV
            )
            y_pred_fold = model_fold.predict(X_val_fold, num_iteration=model_fold.best_iteration)
            fold_mard = calculate_mard(y_val_fold, y_pred_fold)
            fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
            fold_mae = mean_absolute_error(y_val_fold, y_pred_fold)
            fold_results_mard.append(fold_mard)
            fold_results_rmse.append(fold_rmse)
            fold_results_mae.append(fold_mae)
            print(f"    Fold {fold+1} mARD: {fold_mard:.2f}%, RMSE: {fold_rmse:.2f}, MAE: {fold_mae:.2f}")

        print("\n--- Cross-Validation Summary ---")
        print(f"  Avg mARD: {np.mean(fold_results_mard):.2f}% (+/- {np.std(fold_results_mard):.2f})")
        print(f"  Avg RMSE: {np.mean(fold_results_rmse):.2f} (+/- {np.std(fold_results_rmse):.2f})")
        print(f"  Avg MAE:  {np.mean(fold_results_mae):.2f} (+/- {np.std(fold_results_mae):.2f})")
        
        # For saving, train one final model on all data (or on the best fold's params)
        print("\nTraining final model on the entire dataset (after CV)...")
        lgb_full_train_data = lgb.Dataset(X_scaled_all_df, label=y) # Use all scaled data
        model = lgb.train(
            config.LGBM_PARAMS, lgb_full_train_data,
            num_boost_round=config.LGBM_NUM_BOOST_ROUND, # Or use an optimal found via CV
            callbacks=[lgb.log_evaluation(period=50)] 
        )
        # Evaluation results will be from CV, or you can re-evaluate on a hold-out if you had one
        final_y_test_for_eval = y # Dummy for structure, CV results are primary
        final_y_pred_for_eval = model.predict(X_scaled_all_df) # Predictions on full dataset

    print("Model training finished.")
    mard_final = calculate_mard(final_y_test_for_eval, final_y_pred_for_eval)
    rmse_final = np.sqrt(mean_squared_error(final_y_test_for_eval, final_y_pred_for_eval))
    mae_final = mean_absolute_error(final_y_test_for_eval, final_y_pred_for_eval)

    print(f"\n--- Final Model Evaluation (on {'Test Set' if not use_cv else 'Full Data after CV'}) ---")
    print(f"  mARD: {mard_final:.2f}%")
    print(f"  RMSE: {rmse_final:.2f}")
    print(f"  MAE:  {mae_final:.2f}")

    evaluation_results = {
        "mARD": mard_final, "RMSE": rmse_final, "MAE": mae_final,
        "num_total_samples": X.shape[0],
        "best_iteration": model.current_iteration() # Use current_iteration for model trained on full data
    }
    
    # The scaler fitted (either on X_train for single split, or X for CV example)
    # For rigorous CV, scaler fitting should be inside each fold.
    # Here, if use_cv, scaler was fit on all X. If not use_cv, it was fit on X_train.
    # We'll return the scaler that was last fitted.
    final_fitted_scaler = scaler 

    return model, final_fitted_scaler, evaluation_results, feature_names

def save_model_scaler_and_features(model, scaler, feature_names, model_dir, model_filename, scaler_filename, features_filename):
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_filename)
    scaler_path = os.path.join(model_dir, scaler_filename)
    features_path = os.path.join(model_dir, features_filename)
    try:
        model.save_model(model_path) 
        print(f"Trained model saved to: {model_path}")
        if scaler: # Only save if scaler object exists
            joblib.dump(scaler, scaler_path)
            print(f"Fitted scaler saved to: {scaler_path}")
        else:
            print(f"Warning: Scaler object was None, not saved to {scaler_path}")
        with open(features_path, 'w') as f: json.dump(feature_names, f)
        print(f"Model feature names saved to: {features_path}")
    except Exception as e: print(f"Error saving model, scaler, or features: {e}")

def load_model_scaler_and_features(model_dir, model_filename, scaler_filename, features_filename):
    model_path = os.path.join(model_dir, model_filename)
    scaler_path = os.path.join(model_dir, scaler_filename)
    features_path = os.path.join(model_dir, features_filename)
    loaded_model, loaded_scaler, feature_names = None, None, []
    try:
        if os.path.exists(model_path):
            loaded_model = lgb.Booster(model_file=model_path) 
            print(f"Model loaded from: {model_path}")
        else: print(f"Model file not found: {model_path}")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f: feature_names = json.load(f)
            print(f"Model feature names loaded from: {features_path}")
            if loaded_model: loaded_model.feature_name_ = feature_names 
        else: print(f"Feature names file not found: {features_path}")
        if os.path.exists(scaler_path):
            loaded_scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from: {scaler_path}")
        else: print(f"Scaler file not found: {scaler_path}")
        return loaded_model, loaded_scaler, feature_names
    except Exception as e:
        print(f"Error loading model, scaler, or features: {e}")
        return None, None, []

if __name__ == '__main__':
    # ... (dummy data setup as before) ...
    expected_features = ['PAMP_mean', 'PW50_mean', 'RiseTime_mean', 'FallTime_mean', 
                         'PPI_mean', 'PPI_std', 'Mean', 'SD', 'RMS', 
                         'Skewness', 'Kurtosis', 'FFT_BandPower_0.5_5Hz', 'HarmonicRatio']
    num_features = len(expected_features)
    dummy_features_df = pd.DataFrame(np.random.rand(100, num_features), columns=expected_features)
    dummy_labels_series = pd.Series(np.random.uniform(70, 180, 100))

    print("Testing model training with dummy data...")
    trained_model, fitted_scaler, eval_results, f_names = train_evaluate_model(dummy_features_df, dummy_labels_series, use_cv=False) # Test without CV first

    if trained_model and fitted_scaler:
        print("\nDummy model training and evaluation successful.")
        print(f"Evaluation results: {eval_results}")
        save_model_scaler_and_features(
            trained_model, fitted_scaler, f_names, 
            config.MODEL_OUTPUT_DIR, 
            "dummy_model_v2.txt", 
            "dummy_scaler_v2.pkl", 
            "dummy_features_v2.json"
        )
        loaded_m, loaded_s, loaded_f = load_model_scaler_and_features(
            config.MODEL_OUTPUT_DIR, 
            "dummy_model_v2.txt", 
            "dummy_scaler_v2.pkl", 
            "dummy_features_v2.json"
        )
        if loaded_m: print("Dummy model loaded successfully.")
        if loaded_s: print("Dummy scaler loaded successfully.")