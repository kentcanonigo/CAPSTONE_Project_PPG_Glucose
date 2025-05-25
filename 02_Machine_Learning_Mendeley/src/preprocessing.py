# preprocessing.py

import numpy as np
from scipy.signal import resample, butter, filtfilt, savgol_filter
import config

def downsample_signal(raw_signal, original_fs, target_fs):
    """Downsamples a signal from original_fs to target_fs."""
    num_samples_original = len(raw_signal)
    num_samples_target = int(num_samples_original * target_fs / original_fs)
    if num_samples_target < 1 : # Ensure at least 1 sample after downsampling
        print(f"Warning: Signal too short ({num_samples_original} samples) to downsample from {original_fs}Hz to {target_fs}Hz. Returning original.")
        return raw_signal # Or handle as an error
    downsampled_signal = resample(raw_signal, num_samples_target)
    return downsampled_signal

def apply_bandpass_filter(signal_data, lowcut, highcut, fs, order):
    """Applies a Butterworth bandpass filter to the signal."""
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    # Ensure low and high are valid (0 < low < high < 1)
    if not (0 < low < 1 and 0 < high < 1 and low < high):
        print(f"Warning: Invalid filter critical frequencies ({low}, {high}) for Nyquist {nyquist_freq}. Signal length: {len(signal_data)}. Lowcut: {lowcut}, Highcut: {highcut}, FS: {fs}")
        # Check if signal is too short for filtering or if cutoffs are bad
        if len(signal_data) <= order * 3: # Heuristic, filtfilt needs signal longer than 3 * (max(len(a), len(b)) -1)
             print("Signal too short for filtering. Returning original.")
             return signal_data
        # If frequencies are bad but signal okay, maybe skip filtering or use defaults
        # For now, returning original if frequencies are bad to avoid crash.
        return signal_data

    b, a = butter(order, [low, high], btype='band')
    if len(signal_data) <= max(len(b), len(a)) * 3 : # A more robust check for filtfilt
        print(f"Warning: Signal length {len(signal_data)} is too short for designed filter (order {order}). Returning original.")
        return signal_data
    filtered_signal = filtfilt(b, a, signal_data)
    return filtered_signal

def apply_savgol_smoothing(signal_data, window_length, poly_order):
    """Applies Savitzky-Golay smoothing to the signal."""
    if window_length % 2 == 0: # window_length must be odd
        window_length +=1
    if len(signal_data) < window_length:
        print(f"Warning: Signal length {len(signal_data)} is less than SavGol window {window_length}. Adjusting window or returning original.")
        # Option: reduce window_length if possible, or return original
        if len(signal_data) > poly_order : # Ensure window is at least poly_order + 1
            window_length = max(poly_order + 1 + (poly_order+1)%2,3) # Smallest odd window > poly_order
            if len(signal_data) < window_length:
                 return signal_data # Still too short
        else:
            return signal_data # Too short for even smallest window

    smoothed_signal = savgol_filter(signal_data, window_length, poly_order)
    return smoothed_signal

def segment_signal(processed_signal, samples_per_window):
    """Segments the signal into fixed-length windows."""
    num_segments = len(processed_signal) // samples_per_window
    segments = []
    for i in range(num_segments):
        segment = processed_signal[i * samples_per_window : (i + 1) * samples_per_window]
        if len(segment) == samples_per_window: # Ensure full segment
            segments.append(segment)
    return segments

def full_preprocess_pipeline(raw_signal):
    """Runs the full preprocessing pipeline on a raw signal."""
    if raw_signal is None or len(raw_signal) == 0:
        return [] # Return empty list if signal is bad

    # 1. Downsample
    downsampled = downsample_signal(raw_signal, config.ORIGINAL_FS, config.TARGET_FS)
    if len(downsampled) < config.SAMPLES_PER_WINDOW : # Check if signal is too short after downsampling
        print(f"Signal too short after downsampling ({len(downsampled)} samples) for segmentation. Skipping.")
        return []

    # 2. Bandpass Filter
    bandpassed = apply_bandpass_filter(downsampled, config.FILTER_LOWCUT, config.FILTER_HIGHCUT,
                                       config.TARGET_FS, config.FILTER_ORDER)

    # 3. Savitzky-Golay Smoothing
    smoothed = apply_savgol_smoothing(bandpassed, config.SAVGOL_WINDOW, config.SAVGOL_POLYORDER)

    # 4. Segmentation
    signal_segments = segment_signal(smoothed, config.SAMPLES_PER_WINDOW)

    return signal_segments


if __name__ == '__main__':
    # Example usage (for testing this module)
    # Create a dummy raw signal similar to what load_raw_ppg_signal would return
    dummy_raw_signal = np.random.rand(config.ORIGINAL_FS * 10) # 10 seconds of data at original FS
    print(f"Dummy raw signal length: {len(dummy_raw_signal)} at {config.ORIGINAL_FS} Hz")

    segments = full_preprocess_pipeline(dummy_raw_signal)
    if segments:
        print(f"Successfully preprocessed and segmented. Number of segments: {len(segments)}")
        print(f"Length of first segment: {len(segments[0])} (should be {config.SAMPLES_PER_WINDOW}) at {config.TARGET_FS} Hz")
    else:
        print("Preprocessing or segmentation failed for dummy signal.")