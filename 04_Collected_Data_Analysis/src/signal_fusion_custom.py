# 04_Collected_Data_Analysis/src/signal_fusion_custom.py

import numpy as np
from scipy.signal import find_peaks # For SQI calculation if needed

# You might need to import your config_mendeley or a new config_custom for parameters
# For example, if fs is needed for SQI and not passed directly.
# import config_custom # or import config as config_mendeley

def calculate_snr_for_segment(segment_array, fs):
    """
    Calculates Signal-to-Noise Ratio for a given PPG segment.
    This is a basic example; you may need a more sophisticated approach.
    """
    if segment_array is None or len(segment_array) < 2:
        return 0.0 # Or a very small number to avoid division by zero if segment is bad

    try:
        # Example: Signal power as variance of the signal (AC component)
        signal_power = np.var(segment_array)
        
        # Example: Noise power - could be estimated from a high-frequency band
        # or difference between raw and smoothed, or std of derivative.
        # For simplicity, let's use a proxy: 1 / (1 + std of 2nd derivative) to penalize jerkiness
        # This is a very heuristic SNR, replace with a more robust one from literature or your design.
        if len(segment_array) > 5: # Need enough points for 2nd derivative
            noise_proxy_inv = 1.0 / (1.0 + np.std(np.diff(np.diff(segment_array))))
            snr = signal_power * noise_proxy_inv # Higher signal_power and higher noise_proxy_inv (lower noise) is better
        else:
            snr = signal_power # Fallback if too short for derivative based noise proxy

        return max(0.01, snr) # Ensure SNR is not zero to avoid issues with weights
    except Exception as e:
        print(f"Error calculating SNR: {e}")
        return 0.01 # Return a small default value

def calculate_sqi_for_segment(segment_array, fs):
    """
    Calculates a Signal Quality Index for a PPG segment.
    Based on your thesis description (peak-to-peak amplitude, std, PPI consistency).
    """
    if segment_array is None or len(segment_array) < fs: # Need at least 1s for meaningful PPI
        return 0.0

    try:
        std_dev = np.std(segment_array)
        if std_dev == 0: # Flat line or constant
            return 0.0

        peak_to_peak_amplitude = np.ptp(segment_array)
        if peak_to_peak_amplitude == 0:
            return 0.0

        # Find peaks for PPI consistency (parameters might need tuning)
        # Distance should be related to minimum expected heart rate (e.g., 40bpm -> 1.5s -> 1.5*fs samples)
        # and maximum (e.g., 200bpm -> 0.3s -> 0.3*fs samples)
        min_peak_dist = int(fs * 0.3) # Min distance for HR up to 200bpm
        peaks, _ = find_peaks(segment_array, distance=min_peak_dist, prominence=0.1 * peak_to_peak_amplitude)

        if len(peaks) < 2:
            ppi_std_norm = 1.0 # High penalty if not enough peaks for PPI
        else:
            ppi_values = np.diff(peaks) / fs
            ppi_mean = np.mean(ppi_values)
            ppi_std = np.std(ppi_values)
            ppi_std_norm = ppi_std / ppi_mean if ppi_mean > 0 else 1.0 # Normalized PPI std

        # Thesis SQI: mean_amp / (std * (1 + ppi_consistency)) -- Higher is better
        # Here, mean_amp is peak_to_peak_amplitude
        # ppi_consistency is represented by ppi_std_norm (lower is better, so 1+ppi_std_norm in denom)
        sqi = peak_to_peak_amplitude / (std_dev * (1 + ppi_std_norm))
        return max(0.0, sqi) # Ensure non-negative
    except Exception as e:
        print(f"Error calculating SQI: {e}")
        return 0.0


def fuse_features_snr_weighted(features_f1_scaled, features_f2_scaled, features_f3_scaled, 
                               segments_f1, segments_f2, segments_f3, fs):
    """
    Performs SNR-weighted feature fusion.
    - features_fX_scaled: List of scaled feature vectors for finger X.
    - segments_fX: List of corresponding original preprocessed signal segments for finger X (used for SNR calc).
    - fs: Sampling frequency of the segments.
    Returns a list of fused feature vectors.
    """
    fused_features_list = []
    num_segments = min(len(features_f1_scaled), len(features_f2_scaled), len(features_f3_scaled))

    if num_segments == 0:
        return []

    for i in range(num_segments):
        snr1 = calculate_snr_for_segment(segments_f1[i], fs)
        snr2 = calculate_snr_for_segment(segments_f2[i], fs)
        snr3 = calculate_snr_for_segment(segments_f3[i], fs)

        total_snr = snr1 + snr2 + snr3
        if total_snr == 0: # Avoid division by zero if all SNRs are 0
            # Default to simple average if no SNR info, or handle as error/skip segment
            w1, w2, w3 = 1/3, 1/3, 1/3
        else:
            w1 = snr1 / total_snr
            w2 = snr2 / total_snr
            w3 = snr3 / total_snr
        
        # Assuming features are numpy arrays or lists of numbers
        fused_vector = (w1 * np.array(features_f1_scaled[i]) +
                        w2 * np.array(features_f2_scaled[i]) +
                        w3 * np.array(features_f3_scaled[i]))
        fused_features_list.append(fused_vector.tolist())
        
    print(f"Applied SNR-Weighted Fusion to {num_segments} segment sets.")
    return fused_features_list


def fuse_features_sqi_selected(features_f1_scaled, features_f2_scaled, features_f3_scaled,
                               segments_f1, segments_f2, segments_f3, fs):
    """
    Performs SQI-based feature selection.
    - features_fX_scaled: List of scaled feature vectors for finger X.
    - segments_fX: List of corresponding original preprocessed signal segments for finger X (used for SQI calc).
    - fs: Sampling frequency of the segments.
    Returns a list of feature vectors from the finger with the highest SQI for each segment window.
    """
    selected_features_list = []
    num_segments = min(len(features_f1_scaled), len(features_f2_scaled), len(features_f3_scaled))

    if num_segments == 0:
        return []

    for i in range(num_segments):
        sqi1 = calculate_sqi_for_segment(segments_f1[i], fs)
        sqi2 = calculate_sqi_for_segment(segments_f2[i], fs)
        sqi3 = calculate_sqi_for_segment(segments_f3[i], fs)

        sqis = [sqi1, sqi2, sqi3]
        features_all_fingers_current_segment = [
            features_f1_scaled[i], 
            features_f2_scaled[i], 
            features_f3_scaled[i]
        ]
        
        best_finger_index = np.argmax(sqis) # Index of the finger with the highest SQI
        
        # If all SQIs are 0 (e.g., bad signals), argmax might pick the first one.
        # You might want to add a threshold: if max(sqis) < threshold, return NaN features or skip.
        if max(sqis) == 0:
            print(f"Warning: All SQIs are 0 for segment set {i}. Using features from finger 1 as fallback.")
            # Or append a vector of NaNs: selected_features_list.append([np.nan] * len(features_f1_scaled[i]))
            # For now, just picking finger 1's features in this edge case if they exist
            selected_features_list.append(features_all_fingers_current_segment[0])

        else:
            selected_features_list.append(features_all_fingers_current_segment[best_finger_index])
            
    print(f"Applied SQI-Based Feature Selection to {num_segments} segment sets.")
    return selected_features_list