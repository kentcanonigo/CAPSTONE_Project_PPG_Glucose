import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_widths, welch
from scipy.stats import skew, kurtosis
from numpy.fft import fft
from scipy.signal import savgol_filter

# Folder containing raw PPG .csv files
DATA_DIR = "PPG_Dataset\RawDatainCSV"
SAMPLES_PER_SEGMENT = 2175 * 5  # 5 seconds x 2175 Hz = 10,875 samples per segment

# Placeholder for all feature rows
all_feature_rows = []

def extract_features(signal_segment, fs=2175):
    features = {}
    # Peak detection
    peaks, _ = find_peaks(signal_segment, distance=fs//2)
    if len(peaks) > 1:
        pp_intervals = np.diff(peaks) / fs  # Convert to seconds
        features['delta_T'] = np.mean(pp_intervals)
    else:
        features['delta_T'] = np.nan

    # PAMP: Peak-to-trough amplitude
    pamp = np.max(signal_segment) - np.min(signal_segment)
    features['PAMP'] = pamp

    # PW50 (Pulse Width at 50% amplitude)
    if len(peaks) > 0:
        results_half = peak_widths(signal_segment, peaks, rel_height=0.5)
        features['PW50'] = np.mean(results_half[0] / fs)
    else:
        features['PW50'] = np.nan

    # Statistical features
    features['mean'] = np.mean(signal_segment)
    features['std'] = np.std(signal_segment)
    features['rms'] = np.sqrt(np.mean(signal_segment**2))
    features['skewness'] = skew(signal_segment)
    features['kurtosis'] = kurtosis(signal_segment)

    # FFT band power (0.5-5 Hz)
    freqs, psd = welch(signal_segment, fs=fs)
    band_mask = (freqs >= 0.5) & (freqs <= 5)
    features['band_power_0.5_5Hz'] = np.trapezoid(psd[band_mask], freqs[band_mask])

    # Harmonic ratio (2nd / 1st)
    fft_vals = np.abs(fft(signal_segment))
    fft_vals = fft_vals[:len(fft_vals) // 2]  # Keep positive frequencies
    fundamental = np.argmax(fft_vals[1:]) + 1  # Avoid DC
    if 2 * fundamental < len(fft_vals):
        features['harmonic_ratio'] = fft_vals[2 * fundamental] / fft_vals[fundamental]
    else:
        features['harmonic_ratio'] = np.nan

    return features

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".csv"):
        file_path = os.path.join(DATA_DIR, filename)
        raw_signal = pd.read_csv(file_path, header=None).squeeze("columns").values

        # Optional: smooth the signal to reduce noise before feature extraction
        smoothed = savgol_filter(raw_signal, window_length=51, polyorder=3)

        # Each .csv contains a 10-second PPG recording at 2175 Hz ~21,750 samples
        # Since we use 5-second analysis windows (10,875 samples), we divide into 2 segments
        num_segments = len(smoothed) // SAMPLES_PER_SEGMENT  # Expected: 2 segments per file

        for i in range(num_segments):
            start = i * SAMPLES_PER_SEGMENT
            end = start + SAMPLES_PER_SEGMENT
            segment = smoothed[start:end]

            # Extract features from the 5-second segment
            features = extract_features(segment)
            features['file'] = filename
            features['segment'] = i + 1  # Track which 5s chunk it came from

            all_feature_rows.append(features)

# Combine all extracted features into a single DataFrame
feature_df = pd.DataFrame(all_feature_rows)
feature_df.to_csv("ppg_features_master2.csv", index=False)
