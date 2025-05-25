# This handles loading the raw PPG signals and corresponding glucose labels

import pandas as pd
import os
import numpy as np
import config # Import from config.py

def load_labels():
    """Loads glucose labels from the CSV file."""
    try:
        labels_df = pd.read_csv(config.LABELS_FILE)
        
        # Available columns from error: ['ID', 'Gender', 'Age', 'Glucose', 'Height', 'Weight']

        if 'ID' not in labels_df.columns:
            print(f"ERROR: 'ID' column not found in {config.LABELS_FILE}. This column is needed to match raw data files.")
            print(f"Available columns are: {labels_df.columns.tolist()}")
            return None
        
        # MODIFIED LINE: Change 'glucose' to 'Glucose' (uppercase G)
        target_glucose_column = 'Glucose' # Use the correct column name from your file
        
        if target_glucose_column not in labels_df.columns:
            print(f"ERROR: '{target_glucose_column}' column not found in {config.LABELS_FILE}. This column is needed as the target variable.")
            print(f"Available columns are: {labels_df.columns.tolist()}")
            return None

        labels_df['File_base_name'] = labels_df['ID'].astype(str).str.strip()
        
        # MODIFIED LINE: Use target_glucose_column variable
        label_map = labels_df.set_index('File_base_name')[target_glucose_column].to_dict() 
        
        print(f"Successfully loaded labels for {len(label_map)} entries from {config.LABELS_FILE}")
        return label_map
        
    except FileNotFoundError:
        print(f"ERROR: Labels file not found at {config.LABELS_FILE}. Please check the path in config.py.")
        return None
    except KeyError as e:
        print(f"ERROR: Expected column not found in labels file: {e}. Verify column names like 'ID' and '{target_glucose_column}' in {config.LABELS_FILE}.")
        return None
    except Exception as e:
        print(f"ERROR loading labels from {config.LABELS_FILE}: {e}")
        return None

def load_raw_ppg_signal(file_path):
    """Loads a raw PPG signal from a CSV file."""
    try:
        # Assuming the CSV file contains a single column of PPG data without a header
        signal_df = pd.read_csv(file_path, header=None)
        if signal_df.shape[1] != 1:
            print(f"Warning: File {file_path} has more than one column. Using the first column.")
        return signal_df.iloc[:, 0].values.flatten() # Get first column as numpy array
    except pd.errors.EmptyDataError:
        print(f"Warning: File {file_path} is empty. Skipping.")
        return None
    except Exception as e:
        print(f"Error reading PPG signal from {file_path}: {e}")
        return None

def get_all_data_files():
    """Gets a list of all .csv data files from the RawData folder."""
    try:
        raw_files = [f for f in os.listdir(config.RAW_DATA_FOLDER) if f.endswith(".csv")]
        if not raw_files:
            print(f"Warning: No .csv files found in {config.RAW_DATA_FOLDER}.")
        return raw_files
    except FileNotFoundError:
        print(f"ERROR: RawData folder not found at {config.RAW_DATA_FOLDER}. Please check path in config.py.")
        return []

if __name__ == '__main__':
    # Example usage (for testing this module)
    labels = load_labels()
    if labels:
        print(f"First 5 labels: {dict(list(labels.items())[:5])}")

    files = get_all_data_files()
    if files:
        print(f"Found {len(files)} CSV files. First 5: {files[:5]}")
        if len(files) > 0:
            sample_signal_path = os.path.join(config.RAW_DATA_FOLDER, files[0])
            sample_signal = load_raw_ppg_signal(sample_signal_path)
            if sample_signal is not None:
                print(f"Successfully loaded sample signal from {files[0]}, length: {len(sample_signal)}")