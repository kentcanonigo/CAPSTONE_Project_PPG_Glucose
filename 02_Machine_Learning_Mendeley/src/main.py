import os
import pandas as pd
import numpy as np
import re # Import regular expressions module

import config
import data_loader
import preprocessing
import feature_extraction_new
import model_trainer

def run_training_pipeline():
    """Runs the full data processing and model training pipeline."""
    print("--- Starting Glucose Estimation Model Training Pipeline ---")

    # 1. Load Labels
    print("\n--- Phase 1: Loading Data ---")
    label_map = data_loader.load_labels()
    if not label_map:
        print("Failed to load labels. Exiting.")
        return

    if label_map:
        sample_keys = list(label_map.keys())[:5]
        print(f"  Sample keys from label_map (Total: {len(label_map)}): {sample_keys}")
        if sample_keys:
            print(f"    Details of first key: '{sample_keys[0]}' (Type: {type(sample_keys[0])}, Repr: {repr(sample_keys[0])})")

    all_raw_data_files = data_loader.get_all_data_files()
    if not all_raw_data_files:
        print("No data files found. Exiting.")
        return

    print("\n--- Phase 2: Preprocessing and Segmentation ---")
    # Change: Store segments, their labels, and subject IDs together
    all_data_for_features = [] # List of (segment, glucose_val, subject_id) tuples
    files_processed_count = 0
    
    for file_idx, ppg_file_name in enumerate(all_raw_data_files):
        # ppg_file_name is like "signal_01_0001.csv"
        
        # MODIFIED PART: Extract subject ID from filename
        match = re.search(r"signal_(\d+)_(\d+)\.csv", ppg_file_name)
        if not match:
            print(f"  Warning: Filename {ppg_file_name} does not match expected pattern 'signal_XX_YYYY.csv'. Skipping.")
            continue
            
        subject_num_str_from_file = match.group(1) # This is 'XX', e.g., '01', '09', '10'
        
        # Convert subject number to integer then back to string to match label_map keys like '1', '10'
        # This handles potential leading zeros in filename vs. no leading zeros in label_map keys
        try:
            lookup_key_for_label_map = str(int(subject_num_str_from_file)) # "01" -> 1 -> "1"
        except ValueError:
            print(f"  Warning: Could not parse subject number from {subject_num_str_from_file} in {ppg_file_name}. Skipping.")
            continue

        if file_idx < 5 or (file_idx + 1) % 10 == 0 :
             print(f"  File: {ppg_file_name} -> Extracted Subject Key for lookup: '{lookup_key_for_label_map}'")

        if lookup_key_for_label_map not in label_map:
            print(f"  Warning: No label for subject_id '{lookup_key_for_label_map}' (from file {ppg_file_name}). Skipping file.")
            continue
            
        glucose_val = label_map[lookup_key_for_label_map]
        raw_signal = data_loader.load_raw_ppg_signal(os.path.join(config.RAW_DATA_FOLDER, ppg_file_name))

        if raw_signal is None:
            print(f"  Warning: Could not load signal from {ppg_file_name}. Skipping.")
            continue
        
        segments_from_file = preprocessing.full_preprocess_pipeline(raw_signal)
        
        if segments_from_file:
            # Store segments, their glucose_val, and the subject_id
            for segment in segments_from_file:
                all_data_for_features.append((segment, glucose_val, lookup_key_for_label_map))
            files_processed_count +=1
            if file_idx < 5 or (file_idx + 1) % 10 == 0 : # Also print if segments were generated
                print(f"    Generated {len(segments_from_file)} segments for {ppg_file_name} with glucose {glucose_val}")
        else:
            # This print might be too verbose if many files yield no segments after preprocessing
            # print(f"  No valid segments generated from {ppg_file_name}.")
            pass


    if not all_data_for_features: # Check this new list
        print("No valid data (segments with labels and subject IDs) found after preprocessing all files. Exiting.")
        return
        
    if files_processed_count == 0 :
        print("No files were successfully processed to generate segments. Exiting.")
        return

    print(f"Total entries for feature extraction: {len(all_data_for_features)} from {files_processed_count} successfully processed files.")

    print("\n--- Phase 3: Feature Extraction ---")
    feature_data_list = []
    labels_for_df = [] # Will be used to create labels_series
    subject_ids_for_df = [] # New list to store subject IDs for each feature vector

    for idx, (segment, glucose_val, subject_id) in enumerate(all_data_for_features):
        if (idx + 1) % 100 == 0: 
            print(f"  Extracting features for entry {idx + 1}/{len(all_data_for_features)}")
        features = feature_extraction_new.extract_all_features_from_segment(segment, config.TARGET_FS)
        feature_data_list.append(features)
        labels_for_df.append(glucose_val)
        subject_ids_for_df.append(subject_id)
    
    feature_df = pd.DataFrame(feature_data_list)
    labels_series = pd.Series(labels_for_df)
    subject_ids_series = pd.Series(subject_ids_for_df) # New series for subject IDs


    if feature_df.isnull().values.any():
        print(f"  NaNs found in features (count: {feature_df.isnull().sum().sum()}). Example features with NaNs:")
        print(feature_df.isnull().sum()[feature_df.isnull().sum() > 0])
    
    print(f"Feature extraction complete. Feature matrix shape: {feature_df.shape}, Labels series length: {len(labels_series)}, Subject IDs series length: {len(subject_ids_series)}")
    if feature_df.empty:
        print("Feature matrix is empty. Cannot proceed to training. Check feature extraction logic.")
        return
    if len(feature_df) != len(labels_series) or len(feature_df) != len(subject_ids_series):
        print(f"CRITICAL ERROR: Mismatch between number of feature sets ({len(feature_df)}), labels ({len(labels_series)}), or subject IDs ({len(subject_ids_series)}). Exiting.")
        return

    print("\n--- Phase 4: Model Training and Evaluation ---")
    # Pass the new subject_ids_series to the model_trainer
    trained_model, fitted_scaler, eval_results, feature_names = model_trainer.train_evaluate_model(
        feature_df, 
        labels_series,
        subject_ids_series, # Pass the subject IDs
        use_cv=False # Set to True to try K-Fold CV, ensure scaler logic is robust for CV if you do
    )

    if trained_model and fitted_scaler: # Check for scaler too
        print("\n--- Pipeline Completed Successfully ---")
        
        # Use filenames from config.py for consistency
        model_trainer.save_model_scaler_and_features( 
            trained_model,
            fitted_scaler, # Pass the fitted scaler
            feature_names, 
            config.MODEL_OUTPUT_DIR,
            config.SAVED_MODEL_NAME,     # e.g., "lgbm_glucose_model_v2.txt"
            config.SAVED_SCALER_NAME,    # e.g., "mendeley_feature_scaler_v2.pkl"
            config.SAVED_FEATURES_NAME   # e.g., "model_features_v2.json"
        )
    else:
        print("\n--- Pipeline Failed: Model or Scaler not trained/created ---")


if __name__ == '__main__':
    run_training_pipeline()