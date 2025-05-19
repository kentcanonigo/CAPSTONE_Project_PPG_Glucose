import os
import pandas as pd

# Paths to data folders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, 'PPG_Dataset', 'RawData')
LABELS_DIR = os.path.join(SCRIPT_DIR, 'PPG_Dataset', 'Labels')

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
