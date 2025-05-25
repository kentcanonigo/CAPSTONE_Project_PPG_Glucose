# main.py

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
    all_processed_segments = []
    all_corresponding_labels = []
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
            all_processed_segments.extend(segments_from_file)
            all_corresponding_labels.extend([glucose_val] * len(segments_from_file))
            files_processed_count +=1
            if file_idx < 5 or (file_idx + 1) % 10 == 0 : # Also print if segments were generated
                print(f"    Generated {len(segments_from_file)} segments for {ppg_file_name} with glucose {glucose_val}")
        else:
            # This print might be too verbose if many files yield no segments after preprocessing
            # print(f"  No valid segments generated from {ppg_file_name}.")
            pass


    if not all_processed_segments:
        print("No valid segments found after preprocessing all files. Exiting.")
        return
        
    if files_processed_count == 0 :
        print("No files were successfully processed to generate segments. Exiting.")
        return

    print(f"Total segments for feature extraction: {len(all_processed_segments)} from {files_processed_count} successfully processed files.")

    # ... (rest of the main.py file, ensure it handles all_processed_segments and all_corresponding_labels)

    print("\n--- Phase 3: Feature Extraction ---")
    # ... (feature extraction logic remains the same, using all_processed_segments and all_corresponding_labels)
    feature_data_list = []
    # Ensure that all_corresponding_labels is correctly populated and has same length potential as feature_data_list
    valid_labels_for_features = []

    for seg_idx, segment in enumerate(all_processed_segments):
        if (seg_idx + 1) % 100 == 0: 
            print(f"  Extracting features for segment {seg_idx + 1}/{len(all_processed_segments)}")
        features = feature_extraction_new.extract_all_features_from_segment(segment, config.TARGET_FS)
        feature_data_list.append(features)
        # This assumes all_corresponding_labels was populated correctly in sync with all_processed_segments
        valid_labels_for_features.append(all_corresponding_labels[seg_idx]) 
    
    feature_df = pd.DataFrame(feature_data_list)
    # Use the newly created valid_labels_for_features
    labels_series = pd.Series(valid_labels_for_features)


    if feature_df.isnull().values.any():
        print(f"  NaNs found in features (count: {feature_df.isnull().sum().sum()}). Example features with NaNs:")
        print(feature_df.isnull().sum()[feature_df.isnull().sum() > 0])
    
    print(f"Feature extraction complete. Feature matrix shape: {feature_df.shape}, Labels series length: {len(labels_series)}")
    if feature_df.empty:
        print("Feature matrix is empty. Cannot proceed to training. Check feature extraction logic.")
        return
    if len(feature_df) != len(labels_series):
        print(f"CRITICAL ERROR: Mismatch between number of feature sets ({len(feature_df)}) and labels ({len(labels_series)}). Exiting.")
        return

    print("\n--- Phase 4: Model Training and Evaluation ---")
    # ... (model training logic)
    trained_model, eval_results, feature_names = model_trainer.train_evaluate_model(feature_df, labels_series)

    if trained_model:
        print("\n--- Pipeline Completed Successfully ---")
        model_trainer.save_model_and_features(
            trained_model,
            feature_names, 
            config.MODEL_OUTPUT_DIR,
            config.SAVED_MODEL_NAME,
            config.SAVED_FEATURES_NAME
        )
    else:
        print("\n--- Pipeline Failed: Model not trained ---")


if __name__ == '__main__':
    run_training_pipeline()

# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GroupKFold

# # Paths to data folders
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# RAW_DATA_DIR = os.path.join(SCRIPT_DIR, 'PPG_Dataset', 'RawData')
# LABELS_DIR = os.path.join(SCRIPT_DIR, 'PPG_Dataset', 'Labels')

# def extract_person_id(signal_file):
#     """Extracts the person ID from a signal filename like 'signal_01_0001.csv' -> '01'"""
#     return signal_file.split('_')[1]

# def group_aware_split(matched_pairs, n_splits=5):
#     """Performs group-aware splitting using GroupKFold."""
#     groups = [extract_person_id(signal_file) for signal_file, _ in matched_pairs]
#     X = np.arange(len(matched_pairs))  # Dummy X, just indices
#     y = np.zeros(len(matched_pairs))   # Dummy y, not used
#     gkf = GroupKFold(n_splits=n_splits)
#     for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
#         print(f"Fold {fold+1}:")
#         print(f"  Train participants: {sorted(set([groups[i] for i in train_idx]))}")
#         print(f"  Test participants: {sorted(set([groups[i] for i in test_idx]))}\n")
#     return

# def main():
#     # List all signal and label CSV files
#     signal_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')])
#     label_files = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith('.csv')])

#     # Match signal and label files by name pattern
#     matched_pairs = []
#     for signal_file in signal_files:
#         base_name = signal_file.replace('signal', 'label')
#         if base_name in label_files:
#             matched_pairs.append((signal_file, base_name))

#     print(f"Found {len(matched_pairs)} matched signal-label pairs.")

#     # Group-aware split demonstration
#     group_aware_split(matched_pairs, n_splits=5)

#     SAMPLING_FREQ = 2175  # Hz

#     # Print duration for each matched signal file
#     last_person_id = None
#     for signal_file, label_file in matched_pairs:
#         # Extract person ID from filename (e.g., signal_01_0001.csv -> 01)
#         person_id = signal_file.split('_')[1]
#         if last_person_id is not None and person_id != last_person_id:
#             print()  # Print a newline when moving to a new person
#         last_person_id = person_id
#         signal_path = os.path.join(RAW_DATA_DIR, signal_file)
#         label_path = os.path.join(LABELS_DIR, label_file)
#         signal_df = pd.read_csv(signal_path, header=None)
#         label_df = pd.read_csv(label_path, header=None)
#         num_samples = len(signal_df)
#         duration_sec = num_samples / SAMPLING_FREQ
#         # Try to get the blood glucose value (skip header if present)
#         if label_df.shape[0] > 1:
#             blood_glucose = label_df.iloc[1, 3]
#         else:
#             blood_glucose = label_df.iloc[0, 3]
#         print(f"{signal_file}: {num_samples} samples, {duration_sec:.2f} seconds, Blood Glucose: {blood_glucose} mg/dL")

#     # Load one pair as a test
#     if matched_pairs:
#         test_signal_file, test_label_file = matched_pairs[0]
#         signal_path = os.path.join(RAW_DATA_DIR, test_signal_file)
#         label_path = os.path.join(LABELS_DIR, test_label_file)
#         signal_df = pd.read_csv(signal_path, header=None)
#         label_df = pd.read_csv(label_path, header=None)
#         print(f"Loaded signal shape: {signal_df.shape}")
#         print(f"Loaded label: {label_df.values}")
#     else:
#         print("No matched pairs found.")

# if __name__ == "__main__":
#     main()
