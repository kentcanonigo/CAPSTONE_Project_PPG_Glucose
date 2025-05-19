import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

# Paths to data folders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, 'PPG_Dataset', 'RawData')
LABELS_DIR = os.path.join(SCRIPT_DIR, 'PPG_Dataset', 'Labels')

def extract_person_id(signal_file):
    """Extracts the person ID from a signal filename like 'signal_01_0001.csv' -> '01'"""
    return signal_file.split('_')[1]

def group_aware_split(matched_pairs, n_splits=5):
    """Performs group-aware splitting using GroupKFold."""
    groups = [extract_person_id(signal_file) for signal_file, _ in matched_pairs]
    X = np.arange(len(matched_pairs))  # Dummy X, just indices
    y = np.zeros(len(matched_pairs))   # Dummy y, not used
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"Fold {fold+1}:")
        print(f"  Train participants: {sorted(set([groups[i] for i in train_idx]))}")
        print(f"  Test participants: {sorted(set([groups[i] for i in test_idx]))}\n")
    return

def main():
    # List all signal and label CSV files
    signal_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')])
    label_files = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith('.csv')])

    # Match signal and label files by name pattern
    matched_pairs = []
    for signal_file in signal_files:
        base_name = signal_file.replace('signal', 'label')
        if base_name in label_files:
            matched_pairs.append((signal_file, base_name))

    print(f"Found {len(matched_pairs)} matched signal-label pairs.")

    # Group-aware split demonstration
    group_aware_split(matched_pairs, n_splits=5)

    SAMPLING_FREQ = 2175  # Hz

    # Print duration for each matched signal file
    last_person_id = None
    for signal_file, label_file in matched_pairs:
        # Extract person ID from filename (e.g., signal_01_0001.csv -> 01)
        person_id = signal_file.split('_')[1]
        if last_person_id is not None and person_id != last_person_id:
            print()  # Print a newline when moving to a new person
        last_person_id = person_id
        signal_path = os.path.join(RAW_DATA_DIR, signal_file)
        label_path = os.path.join(LABELS_DIR, label_file)
        signal_df = pd.read_csv(signal_path, header=None)
        label_df = pd.read_csv(label_path, header=None)
        num_samples = len(signal_df)
        duration_sec = num_samples / SAMPLING_FREQ
        # Try to get the blood glucose value (skip header if present)
        if label_df.shape[0] > 1:
            blood_glucose = label_df.iloc[1, 3]
        else:
            blood_glucose = label_df.iloc[0, 3]
        print(f"{signal_file}: {num_samples} samples, {duration_sec:.2f} seconds, Blood Glucose: {blood_glucose} mg/dL")

    # Load one pair as a test
    if matched_pairs:
        test_signal_file, test_label_file = matched_pairs[0]
        signal_path = os.path.join(RAW_DATA_DIR, test_signal_file)
        label_path = os.path.join(LABELS_DIR, test_label_file)
        signal_df = pd.read_csv(signal_path, header=None)
        label_df = pd.read_csv(label_path, header=None)
        print(f"Loaded signal shape: {signal_df.shape}")
        print(f"Loaded label: {label_df.values}")
    else:
        print("No matched pairs found.")

if __name__ == "__main__":
    main()
