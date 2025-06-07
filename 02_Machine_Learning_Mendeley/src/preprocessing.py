# 02_Machine_Learning_Mendeley/src/preprocessing.py

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, resample
import config # Imports parameters from your config.py

def apply_bandpass_filter(signal, lowcut, highcut, fs, order=2):
    """
    Applies a Butterworth bandpass filter to the signal.
    
    Args:
        signal (np.ndarray): The input signal array.
        lowcut (float): The low cut-off frequency.
        highcut (float): The high cut-off frequency.
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.
        
    Returns:
        np.ndarray: The filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def apply_savgol_smoothing(signal, window_length, polyorder):
    """
    Applies a Savitzky-Golay filter to smooth the signal.
    
    Args:
        signal (np.ndarray): The input signal array.
        window_length (int): The length of the filter window (must be odd).
        polyorder (int): The order of the polynomial used to fit the samples.
        
    Returns:
        np.ndarray: The smoothed signal.
    """
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(signal, window_length, polyorder)

def segment_signal(signal, samples_per_window):
    """
    Segments the signal into non-overlapping windows of a fixed size.
    
    Args:
        signal (np.ndarray): The input signal array.
        samples_per_window (int): The number of samples in each segment.
        
    Returns:
        list: A list of signal segments (each segment is a np.ndarray).
    """
    if signal is None or samples_per_window <= 0:
        return []
        
    num_segments = len(signal) // samples_per_window
    if num_segments == 0:
        return []
        
    # Use a list comprehension to create the list of segments
    segments = [
        signal[i * samples_per_window : (i + 1) * samples_per_window]
        for i in range(num_segments)
    ]
    return segments

def full_preprocess_pipeline(raw_signal, use_mendeley_fs=True, custom_fs=100):
    """
    Complete preprocessing pipeline for a raw PPG signal. This function is designed
    to be flexible for both the original Mendeley dataset and custom datasets with
    different sampling rates.
    
    Args:
        raw_signal (np.ndarray): The input raw PPG signal array.
        use_mendeley_fs (bool): If True, uses ORIGINAL_FS from config for downsampling.
                                If False, uses the provided custom_fs.
        custom_fs (int): The sampling rate of the custom device (e.g., 100 Hz).
    
    Returns:
        list: A list of processed and segmented signal windows.
    """
    if raw_signal is None or raw_signal.size < 2:
        return []

    # --- Step 1: Determine Input Sampling Rate and Downsample ---
    # This logic handles both the Mendeley data (2175 Hz) and your custom data (e.g., 100 Hz)
    input_fs = config.ORIGINAL_FS if use_mendeley_fs else custom_fs
    
    # Downsample only if the input sampling rate is different from the target
    if input_fs != config.TARGET_FS:
        num_target_samples = int(len(raw_signal) * (config.TARGET_FS / float(input_fs)))
        
        # Check if the signal is long enough to produce at least one full segment after downsampling
        if num_target_samples < config.SAMPLES_PER_WINDOW:
            print(f"Warning: Signal too short to process. Original len: {len(raw_signal)}, Target len after downsample: {num_target_samples}")
            return []
            
        signal_to_process = resample(raw_signal, num_target_samples)
    else:
        signal_to_process = raw_signal
    
    current_fs = config.TARGET_FS

    # --- Step 2: Apply Filters ---
    # Apply bandpass filter to remove baseline drift and high-frequency noise
    filtered_signal = apply_bandpass_filter(
        signal_to_process, 
        config.FILTER_LOWCUT, 
        config.FILTER_HIGHCUT, 
        current_fs, 
        config.FILTER_ORDER
    )
    
    # Apply Savitzky-Golay filter for smoothing
    smoothed_signal = apply_savgol_smoothing(
        filtered_signal, 
        config.SAVGOL_WINDOW, 
        config.SAVGOL_POLYORDER
    )
    
    # --- Step 3: Segment the Signal into windows ---
    segments = segment_signal(smoothed_signal, config.SAMPLES_PER_WINDOW)
    
    return segments

# This block allows you to test the script directly if needed
if __name__ == '__main__':
    # Create a dummy signal for testing purposes
    print("--- Running preprocessing.py in test mode ---")
    fs_test = 1000  # A test sampling rate
    duration_test = 30 # seconds
    num_samples_test = fs_test * duration_test
    
    # Create a signal with a 1.5 Hz sine wave (representing heart rate) and some noise
    time_vector = np.linspace(0, duration_test, num_samples_test, endpoint=False)
    # A low-frequency drift component
    drift = 0.5 * np.sin(2 * np.pi * 0.1 * time_vector)
    # A pulsatile component
    pulse = 1.0 * np.sin(2 * np.pi * 1.5 * time_vector)
    # High-frequency noise
    noise = np.random.normal(0, 0.1, num_samples_test)
    
    dummy_raw_signal = drift + pulse + noise
    
    print(f"Created a dummy raw signal with {len(dummy_raw_signal)} samples at {fs_test} Hz.")
    
    # Test the full pipeline
    # We will pretend this is custom data with a 1000 Hz sampling rate
    processed_segments = full_preprocess_pipeline(dummy_raw_signal, use_mendeley_fs=False, custom_fs=fs_test)
    
    if processed_segments:
        print(f"Successfully processed the signal into {len(processed_segments)} segments.")
        print(f"Each segment has {len(processed_segments[0])} samples.")
        
        # Optional: Plot for visual confirmation if you have matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.title("Original vs. Smoothed Signal (First 5 seconds)")
            plt.plot(time_vector[:fs_test*5], dummy_raw_signal[:fs_test*5], label="Original Raw Signal")
            
            # Recreate a smoothed version for plotting comparison (downsampled)
            num_target_samples_plot = int(len(dummy_raw_signal) * (config.TARGET_FS / float(fs_test)))
            resampled_for_plot = resample(dummy_raw_signal, num_target_samples_plot)
            filtered_for_plot = apply_bandpass_filter(resampled_for_plot, config.FILTER_LOWCUT, config.FILTER_HIGHCUT, config.TARGET_FS, config.FILTER_ORDER)
            smoothed_for_plot = apply_savgol_smoothing(filtered_for_plot, config.SAVGOL_WINDOW, config.SAVGOL_POLYORDER)
            time_vector_resampled = np.linspace(0, duration_test, num_target_samples_plot, endpoint=False)
            
            plt.plot(time_vector_resampled[:config.TARGET_FS*5], smoothed_for_plot[:config.TARGET_FS*5], label=f"Processed Signal (at {config.TARGET_FS} Hz)", alpha=0.8)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.title("First Processed Segment")
            plt.plot(processed_segments[0])
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not found. Skipping plot test.")
    else:
        print("Processing failed to produce segments.")
