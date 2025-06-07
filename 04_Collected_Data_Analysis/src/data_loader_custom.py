import pandas as pd
import numpy as np
import os
import config_custom as config

def load_all_custom_data():
    """Loads all custom data samples and their labels from the collection directory."""
    try:
        labels_df = pd.read_csv(config.CUSTOM_LABELS_FILE)
    except FileNotFoundError:
        print(f"ERROR: Custom labels file not found at {config.CUSTOM_LABELS_FILE}")
        return []

    all_data = []
    print(f"Found {len(labels_df)} entries in labels file. Loading raw PPG data files...")
    for index, row in labels_df.iterrows():
        subject_id = row['ID']
        sample_num = row['Sample_Num']
        glucose_val = row['Glucose_mgdL']
        
        ppg_filename = f"{subject_id}_{sample_num}_ppg.csv"
        ppg_filepath = os.path.join(config.CUSTOM_RAW_DATA_DIR, ppg_filename)

        if os.path.exists(ppg_filepath):
            try:
                ppg_df = pd.read_csv(ppg_filepath)
                if all(col in ppg_df.columns for col in ['ppg_finger1', 'ppg_finger2', 'ppg_finger3']):
                    all_data.append({
                        "subject_id": subject_id,
                        "sample_num": sample_num,
                        "glucose": glucose_val,
                        "ppg_finger1": ppg_df['ppg_finger1'].to_numpy(),
                        "ppg_finger2": ppg_df['ppg_finger2'].to_numpy(),
                        "ppg_finger3": ppg_df['ppg_finger3'].to_numpy(),
                    })
                else:
                    print(f"Warning: Skipping {ppg_filename} due to missing one or more PPG columns.")
            except Exception as e:
                print(f"Warning: Could not read or process {ppg_filename}: {e}")
        else:
            print(f"Warning: PPG file not found for subject {subject_id}, sample {sample_num}: {ppg_filepath}")
            
    print(f"Successfully loaded data for {len(all_data)} samples.")
    return all_data
