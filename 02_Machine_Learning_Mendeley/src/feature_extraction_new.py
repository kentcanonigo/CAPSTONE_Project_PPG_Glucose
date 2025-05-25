# feature_extraction.py

import numpy as np
from scipy.signal import find_peaks, peak_widths, welch
from scipy.fft import fft
from scipy.stats import skew, kurtosis
import config # Assuming config.py is in the same directory or accessible via PYTHONPATH

# --- Enhanced Helper for robust peak/trough finding ---
def get_beats(segment, fs):
    """
    Identifies individual heartbeats by finding systolic peaks and their
    corresponding preceding diastolic troughs.
    Returns a list of tuples, where each tuple is (trough_idx, peak_idx).
    """
    beats = []
    if segment is None or len(segment) < 10: # Need a minimum length for any meaningful analysis
        return beats

    # Estimate typical peak distance based on physiological heart rates
    min_peak_distance = int(fs / 3.5) # Approx max HR of 210 bpm (fs/ (210/60))
    if min_peak_distance < 1:
        min_peak_distance = 1 # Ensure distance is at least 1

    segment_amplitude = np.ptp(segment)
    if segment_amplitude == 0: # Flat line segment
        return beats
    
    # Prominence: A peak must stick out by at least this much from its surroundings
    min_prominence = 0.1 * segment_amplitude # Example: 10% of segment's peak-to-peak amplitude
    if min_prominence <= 0: # If amplitude is tiny or negative (shouldn't happen with ptp)
        min_prominence = np.std(segment) * 0.5 # Fallback to a fraction of std dev
        if min_prominence <=0: # If still zero or negative (e.g. flat line, std is 0)
             min_prominence = None # Let find_peaks use its default or no prominence if segment is truly flat


    # Find systolic peaks
    peaks, _ = find_peaks(segment, distance=min_peak_distance, prominence=min_prominence)

    if len(peaks) == 0:
        return beats

    # Find diastolic troughs (look for peaks in the inverted signal)
    # Use similar prominence for troughs; can be tuned separately if needed
    troughs, _ = find_peaks(-segment, distance=min_peak_distance, prominence=min_prominence)

    if len(troughs) == 0:
        return beats

    # Pair peaks with their immediately preceding troughs to form beats
    # This is a common heuristic.
    processed_troughs = set() # To avoid using the same trough for multiple close peaks
    for peak_idx in peaks:
        potential_troughs = troughs[troughs < peak_idx]
        if len(potential_troughs) > 0:
            trough_idx = potential_troughs[-1] # The last trough before the peak
            
            # Ensure this trough hasn't been used with a very close previous peak
            # and that the peak value is indeed greater than the trough value
            if trough_idx not in processed_troughs and segment[peak_idx] > segment[trough_idx]:
                beats.append((trough_idx, peak_idx))
                processed_troughs.add(trough_idx)
    
    return beats # List of (trough_idx, peak_idx) tuples


# --- Feature Extraction Functions ---
def extract_morphological_features(segment, fs):
    """Extracts PAMP, PW50, Rise Time, Fall Time on a beat-by-beat basis."""
    features = {}
    identified_beats = get_beats(segment, fs)

    pamp_values = []
    rise_time_values = []
    fall_time_values = [] # Simplified fall time

    if not identified_beats:
        features['PAMP_mean'] = np.nan
        features['RiseTime_mean'] = np.nan
        features['FallTime_mean'] = np.nan
    else:
        for i, (trough_idx, peak_idx) in enumerate(identified_beats):
            # PAMP (Pulse Amplitude)
            pamp = segment[peak_idx] - segment[trough_idx]
            pamp_values.append(pamp)

            # Rise Time (Time from trough to peak)
            rise_time = (peak_idx - trough_idx) / fs # in seconds
            rise_time_values.append(rise_time)

            # Fall Time (Simplified: Time from current peak to the start of the next beat's trough)
            if i + 1 < len(identified_beats): # If there is a next beat
                next_beat_trough_idx = identified_beats[i+1][0]
                if next_beat_trough_idx > peak_idx: # Ensure next trough is after current peak
                    fall_time = (next_beat_trough_idx - peak_idx) / fs
                    fall_time_values.append(fall_time)
            # else: Could try to find a final trough if it's the last identified beat, but might be less reliable.

        features['PAMP_mean'] = np.mean(pamp_values) if pamp_values else np.nan
        features['RiseTime_mean'] = np.mean(rise_time_values) if rise_time_values else np.nan
        features['FallTime_mean'] = np.mean(fall_time_values) if fall_time_values else np.nan
    
    # PW50: uses all distinct peaks in the segment, not just those successfully paired into beats by get_beats
    # This is because PW50 is a characteristic of the pulse shape itself.
    segment_amplitude_for_pw50 = np.ptp(segment)
    min_prominence_for_pw50 = 0.1 * segment_amplitude_for_pw50 if segment_amplitude_for_pw50 > 0 else None
    min_peak_distance_for_pw50 = int(fs / 3.5)
    if min_peak_distance_for_pw50 < 1: min_peak_distance_for_pw50 = 1

    peaks_for_pw50, _ = find_peaks(segment, distance=min_peak_distance_for_pw50, prominence=min_prominence_for_pw50)
    if len(peaks_for_pw50) > 0:
        try:
            widths, _, _, _ = peak_widths(segment, peaks_for_pw50, rel_height=0.5) # rel_height=0.5 for 50%
            features['PW50_mean'] = np.mean(widths) / fs if len(widths) > 0 else np.nan # in seconds
        except ValueError: # Can happen if issues with peaks or widths (e.g. all peaks at edges)
            features['PW50_mean'] = np.nan
    else:
        features['PW50_mean'] = np.nan
        
    return features


def extract_interval_features(segment, fs):
    """Extracts Peak-to-Peak Interval (PPI / Delta T)."""
    features = {}
    segment_amplitude = np.ptp(segment)
    if segment_amplitude == 0: # Flat line
        features['PPI_mean'] = np.nan
        features['PPI_std'] = np.nan
        return features

    min_prominence_for_ppi = 0.1 * segment_amplitude
    min_peak_distance_for_ppi = int(fs / 3.5)
    if min_peak_distance_for_ppi < 1: min_peak_distance_for_ppi = 1

    peaks, _ = find_peaks(segment, distance=min_peak_distance_for_ppi, prominence=min_prominence_for_ppi)

    if len(peaks) < 2: # Need at least 2 peaks to calculate an interval
        features['PPI_mean'] = np.nan
        features['PPI_std'] = np.nan
    else:
        ppi_values = np.diff(peaks) / fs # in seconds
        features['PPI_mean'] = np.mean(ppi_values)
        features['PPI_std'] = np.std(ppi_values)
    return features


def extract_statistical_features(segment):
    """Extracts Mean, SD, RMS, Skewness, Kurtosis."""
    features = {}
    if segment is None or len(segment) == 0:
        features.update({'Mean': np.nan, 'SD': np.nan, 'RMS': np.nan, 'Skewness': np.nan, 'Kurtosis': np.nan})
        return features

    features['Mean'] = np.mean(segment)
    features['SD'] = np.std(segment)
    features['RMS'] = np.sqrt(np.mean(segment**2))
    
    if len(np.unique(segment)) > 1 and features['SD'] > 1e-9 : # Avoid issues with flat or near-flat signals for skew/kurt
        features['Skewness'] = skew(segment)
        features['Kurtosis'] = kurtosis(segment)
    else:
        features['Skewness'] = 0.0 # For a constant signal, skewness is often defined as 0 (or NaN)
        features['Kurtosis'] = -3.0 # For a constant signal, excess kurtosis is -3 (or NaN for sample kurtosis)
                                   # scipy.stats.kurtosis calculates excess kurtosis (normal is 0)
                                   # If you prefer regular kurtosis, add 3 or use a different definition.
                                   # Using 0 for simplicity if you prefer to avoid -3.
    return features


def extract_frequency_domain_features(segment, fs):
    """Extracts FFT Band Power and Harmonic Ratio."""
    features = {}
    n_samples = len(segment)

    if segment is None or n_samples < 10: # Need some minimum length for FFT
        features['FFT_BandPower_0.5_5Hz'] = np.nan
        features['HarmonicRatio'] = np.nan
        return features
    
    segment_amplitude = np.ptp(segment)
    if segment_amplitude == 0: # Flat line, no meaningful spectral content
        features['FFT_BandPower_0.5_5Hz'] = 0.0 # Or np.nan
        features['HarmonicRatio'] = np.nan
        return features

    # FFT Band Power (0.5-5 Hz)
    nperseg_val = min(n_samples, 256) 
    if n_samples < fs / config.FILTER_HIGHCUT : # If segment is shorter than period of highest freq of interest
        nperseg_val = n_samples 
    
    if nperseg_val > 0:
        try:
            f_welch, Pxx_welch = welch(segment, fs=fs, nperseg=nperseg_val, scaling='density')
            band_mask = (f_welch >= config.FILTER_LOWCUT) & (f_welch <= config.FILTER_HIGHCUT) # Use config values
            if np.any(band_mask) and len(Pxx_welch[band_mask]) > 0 :
                # Integrate power: sum of PSD values in band * frequency resolution
                df = f_welch[1] - f_welch[0] if len(f_welch) > 1 else 1.0/fs # Frequency resolution
                features['FFT_BandPower_0.5_5Hz'] = np.sum(Pxx_welch[band_mask]) * df
            else:
                 features['FFT_BandPower_0.5_5Hz'] = 0.0 # No power in band (or np.nan)
        except ValueError: 
            features['FFT_BandPower_0.5_5Hz'] = np.nan
    else:
        features['FFT_BandPower_0.5_5Hz'] = np.nan

    # Harmonic Ratio (A2/A1)
    ppi_mean_for_f0 = extract_interval_features(segment, fs).get('PPI_mean', np.nan)
    
    if not np.isnan(ppi_mean_for_f0) and ppi_mean_for_f0 > (1.0 / (fs/2)) and n_samples > 0: # PPI must be valid and correspond to freq < Nyquist
        F0 = 1.0 / ppi_mean_for_f0 # Fundamental frequency from heart rate
        # Physiological HR limits (e.g., 0.5 Hz or 30bpm to 3.5 Hz or 210bpm)
        if F0 < 0.5 or F0 > 4.0: # Check if F0 is in a reasonable physiological range
            features['HarmonicRatio'] = np.nan
        else:
            fft_magnitudes = np.abs(fft(segment))[:n_samples//2] # One-sided spectrum
            fft_frequencies = np.fft.fftfreq(n_samples, d=1/fs)[:n_samples//2]

            if len(fft_frequencies) == 0:
                 features['HarmonicRatio'] = np.nan
            else:
                # Find indices for F0 and 2*F0
                # Create a small window around F0 and 2*F0 to find the peak magnitude
                # This is more robust than taking the exact fft_frequencies[idx]
                window_hz = 0.2 # Look in a +/- 0.2 Hz window around F0 and 2*F0
                
                mask_F0 = (fft_frequencies >= F0 - window_hz) & (fft_frequencies <= F0 + window_hz)
                mask_2F0 = (fft_frequencies >= 2*F0 - window_hz) & (fft_frequencies <= 2*F0 + window_hz)

                A1 = np.max(fft_magnitudes[mask_F0]) if np.any(mask_F0) else 0
                A2 = np.max(fft_magnitudes[mask_2F0]) if np.any(mask_2F0) else 0
                
                features['HarmonicRatio'] = A2 / A1 if A1 > 1e-9 else np.nan
    else:
        features['HarmonicRatio'] = np.nan
        
    return features


# --- Main Orchestrator for Feature Extraction from a Segment ---
def extract_all_features_from_segment(segment, fs):
    """Extracts all defined features from a single PPG segment."""
    # This is the list of features your model will expect, in order.
    # Make sure the keys match exactly what you use to create the DataFrame.
    feature_keys_in_order = [
        'PAMP_mean', 'PW50_mean', 'RiseTime_mean', 'FallTime_mean',
        'PPI_mean', 'PPI_std', 'Mean', 'SD', 'RMS',
        'Skewness', 'Kurtosis', 'FFT_BandPower_0.5_5Hz', 'HarmonicRatio'
    ]
    
    # Initialize all features to NaN to ensure DataFrame has all columns even if some calculations fail
    all_segment_features = {key: np.nan for key in feature_keys_in_order}

    if segment is None or len(segment) < int(fs * 0.5): # Require at least 0.5s of data for meaningful features
        # print(f"Segment too short ({len(segment)} samples) or None for feature extraction.")
        return all_segment_features # Return dict of NaNs

    # Calculate features from different categories
    morph_feats = extract_morphological_features(segment, fs)
    interval_feats = extract_interval_features(segment, fs)
    stat_feats = extract_statistical_features(segment)
    freq_feats = extract_frequency_domain_features(segment, fs)

    # Update the initialized dictionary with calculated values
    all_segment_features.update(morph_feats)
    all_segment_features.update(interval_feats)
    all_segment_features.update(stat_feats)
    all_segment_features.update(freq_feats)
    
    return all_segment_features

if __name__ == '__main__':
    # Example usage (for testing this module directly)
    print(f"Testing feature extraction with dummy segment at TARGET_FS={config.TARGET_FS} Hz...")
    # Create a more realistic-looking dummy PPG segment
    fs_test = config.TARGET_FS
    t_test = np.linspace(0, config.WINDOW_DURATION_SEC, config.SAMPLES_PER_WINDOW, endpoint=False)
    # A sum of sines to mimic a basic PPG shape + some noise
    hr_hz = 1.2 # Heart rate of 1.2 Hz (72 bpm)
    dummy_segment = (np.sin(2 * np.pi * hr_hz * t_test) + 
                     0.5 * np.sin(2 * np.pi * 2 * hr_hz * t_test + np.pi/4) + 
                     0.2 * np.random.randn(len(t_test)))
    
    dummy_segment_flat = np.zeros(config.SAMPLES_PER_WINDOW) # Test flat segment
    dummy_segment_short = np.random.rand(10) # Test short segment

    print("\n--- Testing with normal dummy segment ---")
    features = extract_all_features_from_segment(dummy_segment, fs_test)
    for key, value in features.items():
        print(f"  {key}: {value}")

    print("\n--- Testing with flat dummy segment ---")
    features_flat = extract_all_features_from_segment(dummy_segment_flat, fs_test)
    for key, value in features_flat.items():
        print(f"  {key}: {value}")

    print("\n--- Testing with short dummy segment ---")
    features_short = extract_all_features_from_segment(dummy_segment_short, fs_test)
    for key, value in features_short.items():
        print(f"  {key}: {value}")