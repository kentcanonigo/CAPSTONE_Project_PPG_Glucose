import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import GroupShuffleSplit

# --- Add paths to import necessary modules ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.insert(0, current_dir)

project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
mendeley_src_path = os.path.join(project_root, "02_Machine_Learning_Mendeley", "src")
if mendeley_src_path not in sys.path: sys.path.insert(0, mendeley_src_path)

# --- Import all required modules ---
import config_custom as config
import data_loader_custom
import preprocessing as preprocessing_mendeley
import feature_extraction_new as feature_extraction_mendeley
from model_trainer import load_model_scaler_and_features, calculate_mard # Re-use these from your Mendeley trainer

def run_fine_tuning():
    """Orchestrates the entire fine-tuning pipeline."""
    
    # 1. Load your 30-respondent custom dataset
    print("--- Phase 1: Loading Custom 30-Respondent Dataset ---")
    custom_data = data_loader_custom.load_all_custom_data()
    if not custom_data: return

    # 2. Load the Pre-trained Mendeley Model, Scaler, and Feature Order
    print("\n--- Phase 2: Loading Pre-trained Mendeley Model and Scaler ---")
    pretrained_model, mendeley_scaler, expected_feature_order = load_model_scaler_and_features(
        config.MENDELEY_MODEL_DIR,
        config.PRE_TRAINED_MODEL_FILENAME,
        config.PRE_TRAINED_SCALER_FILENAME,
        config.PRE_TRAINED_FEATURES_FILENAME
    )
    if pretrained_model is None or mendeley_scaler is None or not expected_feature_order:
        print("Failed to load pre-trained model, scaler, or features. Exiting.")
        return
        
    # 3. Preprocess, Extract Features, and Scale ALL Custom Data
    print("\n--- Phase 3: Preparing Custom Data for Fine-Tuning ---")
    all_feature_vectors = []
    
    for sample in custom_data:
        signals = [sample['ppg_finger1'], sample['ppg_finger2'], sample['ppg_finger3']]
        for finger_idx, raw_signal in enumerate(signals):
            # Assumes a custom 'custom_fs' parameter in your Mendeley preprocessing script
            # to handle your device's 100 Hz rate instead of Mendeley's 2175 Hz.
            segments = preprocessing_mendeley.full_preprocess_pipeline(
                raw_signal, 
                use_mendeley_fs=False, 
                custom_fs=100 
            )
            
            if not segments: continue
                
            for segment in segments:
                features_dict = feature_extraction_mendeley.extract_all_features_from_segment(
                    segment, config_mendeley.TARGET_FS # Use target FS for features
                )
                ordered_features = [features_dict.get(feat, np.nan) for feat in expected_feature_order]
                all_feature_vectors.append({
                    "subject_id": sample['subject_id'],
                    "features": ordered_features,
                    "glucose": sample['glucose']
                })
    
    if not all_feature_vectors: print("No feature vectors extracted from custom data. Exiting."); return
        
    features_df = pd.DataFrame([item['features'] for item in all_feature_vectors], columns=expected_feature_order)
    labels = pd.Series([item['glucose'] for item in all_feature_vectors])
    groups = pd.Series([item['subject_id'] for item in all_feature_vectors])

    if features_df.isnull().values.any():
        print(f"NaNs found in custom features. Filling with median of custom data.")
        features_df = features_df.fillna(features_df.median())

    features_scaled = mendeley_scaler.transform(features_df)
    features_scaled_df = pd.DataFrame(features_scaled, columns=feature_names)
    print(f"Custom data preprocessed and scaled. Total feature sets: {len(features_scaled_df)}")

    # 4. Split Custom Data into Fine-tuning and Hold-out Test Sets (by participant)
    print("\n--- Phase 4: Splitting Custom Data ---")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
    ft_indices, test_indices = next(splitter.split(features_scaled_df, labels, groups=groups))

    X_ft, y_ft = features_scaled_df.iloc[ft_indices], labels.iloc[ft_indices]
    X_test_custom, y_test_custom = features_scaled_df.iloc[test_indices], labels.iloc[test_indices]
    
    ft_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_STATE)
    ft_train_indices, ft_valid_indices = next(ft_splitter.split(X_ft, y_ft, groups=groups.iloc[ft_indices]))
    
    X_ft_train, y_ft_train = X_ft.iloc[ft_train_indices], y_ft.iloc[ft_train_indices]
    X_ft_valid, y_ft_valid = X_ft.iloc[ft_valid_indices], y_ft.iloc[ft_valid_indices]
    
    print(f"Custom data split complete:\n - Fine-tuning training set size: {len(X_ft_train)}\n - Fine-tuning validation set size: {len(X_ft_valid)}\n - Final hold-out test set size: {len(X_test_custom)}")

    # 5. Perform Fine-Tuning
    print("\n--- Phase 5: Fine-tuning the Model ---")
    lgb_ft_train = lgb.Dataset(X_ft_train, label=y_ft_train)
    lgb_ft_valid = lgb.Dataset(X_ft_valid, label=y_ft_valid)
    
    ft_params = pretrained_model.params.copy()
    ft_params.update(config.FT_PARAMS) # Override with fine-tuning specific params
    if 'n_estimators' in ft_params: del ft_params['n_estimators']
    if 'num_iterations' in ft_params: del ft_params['num_iterations']

    fine_tuned_model = lgb.train(
        ft_params, lgb_ft_train,
        num_boost_round=config.FT_NUM_BOOST_ROUND,
        valid_sets=[lgb_ft_train, lgb_ft_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=config.FT_EARLY_STOPPING_ROUNDS, verbose=True), lgb.log_evaluation(period=50)],
        init_model=pretrained_model # The key to fine-tuning!
    )
    print("Fine-tuning finished.")

    # 6. Save the Fine-Tuned Model
    print("\n--- Phase 6: Saving the Fine-Tuned Model ---")
    fine_tuned_model_name = f"lgbm_model_finetuned_on_{len(custom_data)}subjects.txt"
    fine_tuned_model_path = os.path.join(config.CUSTOM_MODEL_OUTPUT_DIR, fine_tuned_model_name)
    fine_tuned_model.save_model(fine_tuned_model_path)
    print(f"Fine-tuned model saved to: {fine_tuned_model_path}")
    
    # 7. Evaluate Performance on Hold-out Custom Test Set
    print("\n--- Phase 7: Evaluating on Hold-out Custom Test Set ---")
    
    y_pred_finetuned = fine_tuned_model.predict(X_test_custom, num_iteration=fine_tuned_model.best_iteration)
    mard_finetuned = calculate_mard(y_test_custom, y_pred_finetuned)
    rmse_finetuned = np.sqrt(mean_squared_error(y_test_custom, y_pred_finetuned))
    print(f"  **Fine-Tuned Model** on Custom Test Set -> mARD: {mard_finetuned:.2f}%, RMSE: {rmse_finetuned:.2f}")

    y_pred_original = pretrained_model.predict(X_test_custom)
    mard_original = calculate_mard(y_test_custom, y_pred_original)
    rmse_original = np.sqrt(mean_squared_error(y_test_custom, y_pred_original))
    print(f"  **Original Mendeley Model** on Custom Test Set -> mARD: {mard_original:.2f}%, RMSE: {rmse_original:.2f}")
    
    improvement = mard_original - mard_finetuned
    print(f"\nImprovement in mARD due to fine-tuning: {improvement:.2f}%")


if __name__ == '__main__':
    run_fine_tuning()
