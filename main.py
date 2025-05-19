import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# Set paths to the dataset directories
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "PPG_Dataset")  # Now always relative to script
signal_dir = os.path.join(data_dir, "RawData")
label_dir = os.path.join(data_dir, "Labels")


def list_paired_files():
    """List and pair signal and label files based on matching IDs"""
    signal_files = sorted([f for f in os.listdir(signal_dir) if f.startswith("signal_")])
    label_files = sorted([f for f in os.listdir(label_dir) if f.startswith("label_")])

    file_pairs = []
    for signal_file in signal_files:
        # Extract ID (e.g., "01_0001" from "signal_01_0001.csv")
        file_id = re.search(r"signal_(.+)\.csv", signal_file).group(1)
        matching_label = f"label_{file_id}.csv"

        if matching_label in label_files:
            file_pairs.append((
                os.path.join(signal_dir, signal_file),
                os.path.join(label_dir, matching_label)
            ))

    print(f"Found {len(file_pairs)} matching signal-label pairs")
    return file_pairs


def inspect_sample_data(file_pairs, num_samples=2):
    """Load and inspect sample files to understand data structure"""
    for i in range(min(num_samples, len(file_pairs))):
        signal_file, label_file = file_pairs[i]

        signal_df = pd.read_csv(signal_file)
        label_df = pd.read_csv(label_file)

        print(f"\nSample {i + 1}:")
        print(f"Signal file: {os.path.basename(signal_file)}")
        print(f"Signal shape: {signal_df.shape}")
        print(f"Signal columns: {signal_df.columns.tolist()}")
        print(f"Signal first 5 rows:\n{signal_df.head()}")

        print(f"\nLabel file: {os.path.basename(label_file)}")
        print(f"Label shape: {label_df.shape}")
        print(f"Label columns: {label_df.columns.tolist()}")
        print(f"Label data:\n{label_df}")


def extract_ppg_features(signal_df):
    """Extract features from PPG signal"""
    features = {}

    # Identify the PPG column (adjust based on your data inspection)
    signal_col = signal_df.columns[0]  # Assuming first column contains PPG signal
    ppg_signal = signal_df[signal_col].values

    # Time domain features
    features['mean'] = np.mean(ppg_signal)
    features['std'] = np.std(ppg_signal)
    features['min'] = np.min(ppg_signal)
    features['max'] = np.max(ppg_signal)
    features['range'] = features['max'] - features['min']
    features['median'] = np.median(ppg_signal)
    features['q25'] = np.percentile(ppg_signal, 25)
    features['q75'] = np.percentile(ppg_signal, 75)
    features['iqr'] = features['q75'] - features['q25']
    features['skew'] = pd.Series(ppg_signal).skew()
    features['kurtosis'] = pd.Series(ppg_signal).kurtosis()
    features['rms'] = np.sqrt(np.mean(np.square(ppg_signal)))

    # Frequency domain features
    fft_vals = np.abs(np.fft.rfft(ppg_signal))
    fft_freq = np.fft.rfftfreq(len(ppg_signal))

    features['dominant_freq'] = fft_freq[np.argmax(fft_vals)]
    features['power_dominant_freq'] = np.max(fft_vals)
    features['total_power'] = np.sum(fft_vals)

    # Derivative-based features
    diff1 = np.diff(ppg_signal)
    features['mean_diff1'] = np.mean(diff1)
    features['std_diff1'] = np.std(diff1)
    features['max_diff1'] = np.max(diff1)
    features['min_diff1'] = np.min(diff1)

    return features


def build_dataset(file_pairs):
    """Build feature dataset from all signal-label pairs"""
    features_list = []
    labels = []

    for signal_file, label_file in tqdm(file_pairs, desc="Processing files"):
        # Load data
        signal_df = pd.read_csv(signal_file)
        label_df = pd.read_csv(label_file)

        # Extract features
        features = extract_ppg_features(signal_df)

        # Get glucose label (adjust column name based on your inspection)
        glucose_value = label_df.iloc[0, 0]  # Assuming first column, first row contains glucose value

        features_list.append(features)
        labels.append(glucose_value)

    X = pd.DataFrame(features_list)
    y = pd.Series(labels, name="glucose")

    return X, y


def train_lightgbm_model(X, y):
    """Train and evaluate LightGBM regression model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        early_stopping_rounds=50
    )

    # Evaluate
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=20)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    return model, X_test, y_test, y_pred


def main():
    """Main workflow"""
    # Get paired signal and label files
    file_pairs = list_paired_files()

    if not file_pairs:
        print("No matching file pairs found. Check your data directory.")
        return

    # Inspect sample data to understand structure
    inspect_sample_data(file_pairs)

    # Build feature dataset
    print("\nExtracting features and building dataset...")
    X, y = build_dataset(file_pairs)
    print(f"Dataset created with {X.shape[0]} samples and {X.shape[1]} features")

    # Save feature dataset to CSV for inspection
    feature_dataset = X.copy()
    feature_dataset['glucose'] = y
    feature_dataset.to_csv("feature_dataset.csv", index=False)
    print("Feature dataset saved to 'feature_dataset.csv'")

    # Commented out model training and saving for now
    # print("\nTraining LightGBM model...")
    # model, X_test, y_test, y_pred = train_lightgbm_model(X, y)
    # model.save_model("ppg_glucose_model.txt")
    # print("Model saved to 'ppg_glucose_model.txt'")

    # Plot actual vs predicted values (also commented out)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test, y_pred, alpha=0.6)
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    # plt.xlabel("Actual Glucose")
    # plt.ylabel("Predicted Glucose")
    # plt.title("Actual vs Predicted Glucose Levels")
    # plt.tight_layout()
    # plt.savefig('actual_vs_predicted.png')


if __name__ == "__main__":
    main()
