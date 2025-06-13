import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# --- Dynamic Path and Module Setup ---
# Ensures the script can find your custom modules from the Mendeley project.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    mendeley_src_path = os.path.join(project_root, "02_Machine_Learning_Mendeley", "src")
    if mendeley_src_path not in sys.path:
        sys.path.insert(0, mendeley_src_path)

    import config as config_mendeley
    import preprocessing as preprocessing_mendeley
    import feature_extraction_new as feature_extraction_mendeley
    print(f"Successfully imported processing modules from: {mendeley_src_path}")
except ImportError as e:
    print(f"ERROR: Could not import required modules. Ensure you are running from the '04_Collected_Data_Analysis' directory. Details: {e}")
    sys.exit(1)

def generate_feature_matrix(data_folder, output_path):
    """
    Loads raw custom PPG data, processes it, extracts features, and saves the
    resulting data matrix to a CSV file.
    """
    labels_path = os.path.join(data_folder, "Labels", "collected_labels.csv")
    raw_data_path = os.path.join(data_folder, "RawData")

    if not os.path.exists(labels_path):
        print(f"ERROR: Labels file not found at {labels_path}")
        return

    labels_df = pd.read_csv(labels_path)
    print(f"Found {len(labels_df)} participant entries to process.")

    all_feature_rows = []

    for index, row in labels_df.iterrows():
        sample_id_str = f"{row['ID']}_{row['Sample_Num']}"
        ppg_filepath = os.path.join(raw_data_path, f"{sample_id_str}_ppg.csv")

        if not os.path.exists(ppg_filepath):
            print(f"Warning: Skipping {sample_id_str} - Raw PPG file not found.")
            continue

        print(f"Processing sample: {sample_id_str}...")
        actual_glucose = row['Glucose_mgdL']
        ppg_df = pd.read_csv(ppg_filepath)
        
        # Process each of the three finger signals
        for i in range(1, 4):
            finger_name = ["Index", "Middle", "Ring"][i-1]
            signal_column = f'ppg_finger{i}'
            
            if signal_column not in ppg_df.columns:
                continue

            raw_signal = pd.to_numeric(ppg_df[signal_column], errors='coerce').dropna().to_numpy()

            # 1. Preprocess the signal to get segments
            # Your thesis mentions 20s windows, your code generates segments from the whole signal.
            # This code follows your script's logic.
            segments = preprocessing_mendeley.full_preprocess_pipeline(raw_signal, use_mendeley_fs=False, custom_fs=100)
            
            if not segments:
                continue
                
            # 2. Extract features from each segment
            features_for_this_finger = [feature_extraction_mendeley.extract_all_features_from_segment(s, config_mendeley.TARGET_FS) for s in segments]
            
            # 3. Add metadata and append to the master list
            for segment_idx, feature_dict in enumerate(features_for_this_finger):
                feature_dict['ID'] = row['ID']
                feature_dict['Sample_Num'] = row['Sample_Num']
                # Add a segment index to distinguish rows from the same 20s recording
                feature_dict['Segment_Index'] = segment_idx 
                feature_dict['Finger'] = finger_name
                feature_dict['Glucose_mgdL'] = actual_glucose
                all_feature_rows.append(feature_dict)

    if not all_feature_rows:
        print("No features were extracted. Please check the raw data.")
        return
        
    # 4. Create and save the final DataFrame
    feature_matrix_df = pd.DataFrame(all_feature_rows)
    
    # Reorder columns to be more like the example in your thesis
    first_cols = ['ID', 'Sample_Num', 'Segment_Index', 'Finger', 'Glucose_mgdL']
    other_cols = [col for col in feature_matrix_df.columns if col not in first_cols]
    feature_matrix_df = feature_matrix_df[first_cols + other_cols]

    feature_matrix_df.to_csv(output_path, index=False)
    print(f"\nSuccess! Feature matrix saved to:\n{output_path}")

if __name__ == '__main__':
    # Define the path to your collected data
    collected_data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Collected_Data')
    
    # Define the path for the output file
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation_results')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_csv_path = os.path.join(output_folder, 'custom_samples_feature_matrix.csv')
    
    # Run the generation process
    generate_feature_matrix(collected_data_folder, output_csv_path)