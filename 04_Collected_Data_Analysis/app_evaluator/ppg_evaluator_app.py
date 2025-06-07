import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ttkthemes import ThemedTk
import pandas as pd
import numpy as np
import os
import sys 
import threading
import time 
import random 
import math   
from datetime import datetime 
import joblib 
from scipy.signal import resample, find_peaks 
import lightgbm as lgb 

# --- Dynamically add the path to your Mendeley ML scripts ---
SCRIPTS_LOADED_SUCCESSFULLY = False
config_mendeley = None
preprocessing_mendeley = None
feature_extraction_mendeley = None 
model_trainer_mendeley = None
USE_DUMMY_FUNCTIONS_FOR_PROCESSING = True 

try:
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    app_evaluator_dir = current_script_path
    collected_data_analysis_dir = os.path.dirname(app_evaluator_dir) 
    project_root_for_evaluator = os.path.dirname(collected_data_analysis_dir) 
    
    mendeley_src_path = os.path.join(project_root_for_evaluator, "02_Machine_Learning_Mendeley", "src")
    if not os.path.isdir(mendeley_src_path): raise ImportError(f"Mendeley src path does not exist: {mendeley_src_path}")
    if mendeley_src_path not in sys.path: sys.path.insert(0, mendeley_src_path)
    
    import config as config_mendeley 
    import preprocessing as preprocessing_mendeley
    import feature_extraction_new as feature_extraction_mendeley 
    import model_trainer as model_trainer_mendeley 
    print(f"Successfully imported processing modules from: {mendeley_src_path}")
    SCRIPTS_LOADED_SUCCESSFULLY = True
    USE_DUMMY_FUNCTIONS_FOR_PROCESSING = False 
except ImportError as e:
    print(f"ERROR: Could not import processing scripts: {e}")
except Exception as e_path:
    print(f"Error setting up path for Mendeley scripts: {e_path}")

# --- Integrated Fusion Functions (from your signal_fusion_custom.py content) ---
def calculate_snr_for_segment(segment_array, fs):
    if segment_array is None or len(segment_array) < 2: return 0.01
    try:
        signal_power = np.var(segment_array)
        if len(segment_array) > 5:
            std_diff2 = np.std(np.diff(np.diff(segment_array)))
            noise_proxy_inv = 1.0 / (1.0 + std_diff2) if std_diff2 > 1e-9 else 1.0
            snr = signal_power * noise_proxy_inv
        else: snr = signal_power 
        return max(0.01, snr) 
    except Exception as e: print(f"Error calculating SNR for segment: {e}"); return 0.01

def calculate_sqi_for_segment(segment_array, fs):
    if segment_array is None or len(segment_array) < fs : return 0.0
    try:
        std_dev = np.std(segment_array)
        if std_dev < 1e-9: return 0.0
        peak_to_peak_amplitude = np.ptp(segment_array)
        if peak_to_peak_amplitude < 1e-6: return 0.0
        min_peak_dist = int(fs * 0.3) 
        prominence_threshold = 0.1 * peak_to_peak_amplitude 
        if prominence_threshold < 1e-3 : prominence_threshold = None 
        peaks, _ = find_peaks(segment_array, distance=min_peak_dist, prominence=prominence_threshold)
        if len(peaks) < 2: ppi_std_norm = 1.0 
        else:
            ppi_values = np.diff(peaks) / fs
            ppi_mean = np.mean(ppi_values)
            ppi_std = np.std(ppi_values)
            ppi_std_norm = ppi_std / ppi_mean if ppi_mean > 1e-9 else 1.0
        sqi = peak_to_peak_amplitude / (std_dev * (1 + ppi_std_norm))
        return max(0.0, sqi)
    except Exception as e: print(f"Error calculating SQI for segment: {e}"); return 0.0

def fuse_features_snr_weighted(features_f1_scaled, features_f2_scaled, features_f3_scaled, 
                               segments_f1, segments_f2, segments_f3, fs):
    fused_features_list = []
    if not (features_f1_scaled and features_f2_scaled and features_f3_scaled and \
            segments_f1 and segments_f2 and segments_f3 and \
            len(features_f1_scaled) == len(segments_f1) and \
            len(features_f2_scaled) == len(segments_f2) and \
            len(features_f3_scaled) == len(segments_f3) and \
            len(features_f1_scaled) == len(features_f2_scaled) == len(features_f3_scaled) ):
        if 'app' in globals() and hasattr(app, '_log_message'): app._log_message("SNR Fusion: Input feature/segment list(s) empty or length mismatch.")
        return []
    num_segments = len(features_f1_scaled)
    if num_segments == 0: return []
    if 'app' in globals() and hasattr(app, '_log_message'): app._log_message("Applying ACTUAL SNR-Weighted Fusion logic...")
    for i in range(num_segments):
        snr1 = calculate_snr_for_segment(segments_f1[i], fs); snr2 = calculate_snr_for_segment(segments_f2[i], fs); snr3 = calculate_snr_for_segment(segments_f3[i], fs)
        if 'app' in globals() and hasattr(app, '_log_message'): app._log_message(f"Seg {i} SNRs: F1={snr1:.2f}, F2={snr2:.2f}, F3={snr3:.2f}")
        total_snr = snr1 + snr2 + snr3
        w1, w2, w3 = (1/3, 1/3, 1/3) if total_snr < 1e-6 else (snr1/total_snr, snr2/total_snr, snr3/total_snr)
        if 'app' in globals() and hasattr(app, '_log_message'): app._log_message(f"Seg {i} Weights: W1={w1:.2f}, W2={w2:.2f}, W3={w3:.2f}")
        f1_arr=np.array(features_f1_scaled[i]); f2_arr=np.array(features_f2_scaled[i]); f3_arr=np.array(features_f3_scaled[i])
        if not (f1_arr.shape == f2_arr.shape == f3_arr.shape and f1_arr.size > 0):
            n_feat = len(app.expected_feature_order) if 'app' in globals() and app.expected_feature_order else 13
            fused_features_list.append([np.nan] * n_feat); continue
        fused_features_list.append((w1*f1_arr + w2*f2_arr + w3*f3_arr).tolist())
    return fused_features_list

def fuse_features_sqi_selected(features_f1_scaled, features_f2_scaled, features_f3_scaled,
                               segments_f1, segments_f2, segments_f3, fs):
    selected_features_list = []
    if not (features_f1_scaled and features_f2_scaled and features_f3_scaled and \
            segments_f1 and segments_f2 and segments_f3 and \
            len(features_f1_scaled) == len(segments_f1) and \
            len(features_f2_scaled) == len(segments_f2) and \
            len(features_f3_scaled) == len(segments_f3) and \
            len(features_f1_scaled) == len(features_f2_scaled) == len(features_f3_scaled)): # Added check for equal feature list lengths
        if 'app' in globals() and hasattr(app, '_log_message'): app._log_message("SQI Fusion: Input feature/segment list(s) empty or length mismatch.")
        return []
    num_segments = len(features_f1_scaled)
    if num_segments == 0: return []
    if 'app' in globals() and hasattr(app, '_log_message'): app._log_message("Applying ACTUAL SQI-Based Feature Selection logic...")
    for i in range(num_segments):
        sqi1=calculate_sqi_for_segment(segments_f1[i],fs); sqi2=calculate_sqi_for_segment(segments_f2[i],fs); sqi3=calculate_sqi_for_segment(segments_f3[i],fs)
        sqis = [sqi1, sqi2, sqi3]; features_current_segment = [features_f1_scaled[i], features_f2_scaled[i], features_f3_scaled[i]]
        best_finger_index = np.argmax(sqis)
        if 'app' in globals() and hasattr(app, '_log_message'): app._log_message(f"Seg {i} SQIs: F1={sqi1:.2f}, F2={sqi2:.2f}, F3={sqi3:.2f}. Best Idx: {best_finger_index}")
        selected_features_list.append(features_current_segment[best_finger_index])
    return selected_features_list

# --- Dummy functions for non-fusion processing (only used if SCRIPTS_LOADED_SUCCESSFULLY is False for those parts) ---
if USE_DUMMY_FUNCTIONS_FOR_PROCESSING:
    # ... (All dummy functions: dummy_load_ppg_signals_from_file, dummy_preprocess_signal_and_segment, etc. defined here)
    def dummy_load_ppg_signals_from_file(ppg_filepath):
        print(f"DUMMY: Loading raw signals from {os.path.basename(ppg_filepath)}")
        s1 = np.random.randint(1000,3000,size=30000).astype(float) 
        s2 = s1 + np.random.randint(-50,50, size=30000).astype(float)
        s3 = s1 + np.random.randint(-70,70, size=30000).astype(float)
        time.sleep(0.3); return [s1, np.clip(s2,0,4095), np.clip(s3,0,4095)]
    def dummy_preprocess_signal_and_segment(raw_signal, input_fs=100, target_fs_out=50, window_samples_out=250, filter_params=None, savgol_params=None):
        print(f"DUMMY: Preprocessing one signal (InFS:{input_fs}, OutFS:{target_fs_out}, WinSamples:{window_samples_out})...")
        time.sleep(0.2); 
        if len(raw_signal) == 0: return []
        num_downsampled_samples = int(len(raw_signal) * (target_fs_out / float(input_fs)))
        if num_downsampled_samples < window_samples_out : return []
        num_segments = num_downsampled_samples // window_samples_out
        return [np.random.rand(window_samples_out) + (np.mean(raw_signal[:100])*0.00001 if len(raw_signal)>100 else 0) for _ in range(max(1, num_segments))]
    def dummy_extract_features_for_finger(segments_one_finger, fs=50, expected_feature_order=None):
        print(f"DUMMY: Extracting features for {len(segments_one_finger)} segments...")
        time.sleep(0.1); num_expected_features = len(expected_feature_order) if expected_feature_order else 13
        return [(np.random.rand(num_expected_features) + (np.mean(seg)*0.0001 if len(seg)>0 else 0) ).tolist() for seg in segments_one_finger]
    def dummy_apply_scaler_to_features(feature_vectors_list, scaler_object=None):
        print(f"DUMMY: Applying feature scaler (scaler object: {'Provided' if scaler_object else 'None'})..."); return feature_vectors_list 
    class DummyModel:
        def __init__(self, model_path=None): print(f"DUMMY MODEL: Init '{model_path}'")
        def predict(self, X):
            if not isinstance(X, np.ndarray): X = np.array(X)
            if X.ndim == 1: X = X.reshape(1, -1) 
            if X.size == 0: return np.array([])
            print(f"DUMMY MODEL: Predicting for {X.shape[0]} samples...")
            return np.array([random.uniform(80,150) + (np.sum(X[i,:]) if X.shape[1] > 0 else 0)*0.001 for i in range(X.shape[0])])
    def dummy_load_model_scaler_and_features(model_dir, model_filename, scaler_filename, features_filename): 
        print(f"DUMMY: Loading model from {os.path.join(model_dir, model_filename)}") 
        dummy_feature_order = ['PAMP_mean', 'PW50_mean', 'RiseTime_mean', 'FallTime_mean', 'PPI_mean', 'PPI_std', 'Mean', 'SD', 'RMS', 'Skewness', 'Kurtosis', 'FFT_BandPower_0.5_5Hz', 'HarmonicRatio']
        return DummyModel(os.path.join(model_dir, model_filename)), None, dummy_feature_order 

# --- Function Wrappers ---
def load_collected_ppg_data_from_file(ppg_filepath):
    if USE_DUMMY_FUNCTIONS_FOR_PROCESSING: return dummy_load_ppg_signals_from_file(ppg_filepath)
    try:
        df = pd.read_csv(ppg_filepath)
        sig1 = df['ppg_finger1'].astype(float).to_numpy()
        sig2 = df['ppg_finger2'].astype(float).to_numpy()
        sig3 = df['ppg_finger3'].astype(float).to_numpy()
        return [sig1, sig2, sig3]
    except Exception as e:
        log_target = app if 'app' in globals() and hasattr(app, '_log_message') else None
        msg = f"Error loading actual PPG data from {ppg_filepath}: {e}"
        if log_target: log_target._log_message(msg)
        else: print(msg)
        return [np.array([]), np.array([]), np.array([])]

def preprocess_one_finger_signal(raw_signal_array, input_fs, target_fs_for_features, window_duration_sec):
    log_target = app if 'app' in globals() and hasattr(app, '_log_message') else None
    if USE_DUMMY_FUNCTIONS_FOR_PROCESSING or not SCRIPTS_LOADED_SUCCESSFULLY or raw_signal_array.size == 0:
        if log_target: log_target._log_message("Preproc: Using dummy or empty signal.")
        else: print("Preproc: Using dummy or empty signal.")
        return dummy_preprocess_signal_and_segment(raw_signal_array, input_fs, target_fs_for_features, int(window_duration_sec * target_fs_for_features))
    try:
        signal_to_process = raw_signal_array
        if input_fs != target_fs_for_features:
            num_samples_target = int(len(raw_signal_array) * (target_fs_for_features / float(input_fs)))
            if num_samples_target < 1: return []
            signal_to_process = resample(raw_signal_array, num_samples_target)
        current_fs = target_fs_for_features
        filtered_butter = preprocessing_mendeley.apply_bandpass_filter(signal_to_process, config_mendeley.FILTER_LOWCUT, config_mendeley.FILTER_HIGHCUT, current_fs, config_mendeley.FILTER_ORDER)
        smoothed_signal = preprocessing_mendeley.apply_savgol_smoothing(filtered_butter, config_mendeley.SAVGOL_WINDOW, config_mendeley.SAVGOL_POLYORDER)
        samples_per_window = int(window_duration_sec * current_fs)
        segments = preprocessing_mendeley.segment_signal(smoothed_signal, samples_per_window)
        return segments
    except Exception as e:
        if log_target: log_target._log_message(f"Error preprocessing signal: {e}")
        else: print(f"Error preprocessing signal: {e}")
        return []

def extract_features_for_finger_segments(list_of_segments, fs, expected_feature_order):
    log_target = app if 'app' in globals() and hasattr(app, '_log_message') else None
    if USE_DUMMY_FUNCTIONS_FOR_PROCESSING or not SCRIPTS_LOADED_SUCCESSFULLY or not list_of_segments:
        if log_target: log_target._log_message("FeatExtract: Using dummy or no segments.")
        else: print("FeatExtract: Using dummy or no segments.")
        return dummy_extract_features_for_finger(list_of_segments, fs, expected_feature_order)
    all_ordered_features = []
    if not expected_feature_order: 
        if log_target: log_target._log_message("Warning: Expected feature order not available for extraction.")
        else: print("Warning: Expected feature order not available for extraction.")
        num_feats = 13 
        if SCRIPTS_LOADED_SUCCESSFULLY and config_mendeley and hasattr(config_mendeley, 'LGBM_PARAMS') and 'feature_name' in config_mendeley.LGBM_PARAMS:
            num_feats = len(config_mendeley.LGBM_PARAMS['feature_name'])
        return [[np.nan]*num_feats for _ in list_of_segments] 
    try:
        for segment_array in list_of_segments:
            features_dict = feature_extraction_mendeley.extract_all_features_from_segment(segment_array, fs)
            ordered_feature_vector = [features_dict.get(feat_name, np.nan) for feat_name in expected_feature_order]
            all_ordered_features.append(ordered_feature_vector)
    except Exception as e:
        if log_target: log_target._log_message(f"Error extracting features: {e}")
        else: print(f"Error extracting features: {e}")
    return all_ordered_features

def apply_scaler_to_features(feature_vectors_list, scaler_object):
    log_target = app if 'app' in globals() and hasattr(app, '_log_message') else None
    if USE_DUMMY_FUNCTIONS_FOR_PROCESSING or scaler_object is None or not feature_vectors_list:
        if log_target: log_target._log_message("Scaler: No scaler or no features, returning as is.")
        else: print("Scaler: No scaler or no features, returning as is.")
        return dummy_apply_scaler_to_features(feature_vectors_list, scaler_object) 
    try:
        np_features = np.array(feature_vectors_list)
        if np_features.ndim == 1: np_features = np_features.reshape(1, -1)
        if np_features.size == 0 : return []
        scaled_features = scaler_object.transform(np_features)
        return scaled_features.tolist()
    except Exception as e:
        if log_target: log_target._log_message(f"Error applying scaler: {e}. Using unscaled features.")
        else: print(f"Error applying scaler: {e}. Using unscaled features.")
        return feature_vectors_list


class PPGDataEvaluatorApp:
    def __init__(self, root_window, initial_geometry="750x700"):
        self.root = root_window
        self.root.title("PPG Data Evaluator") # Updated version
        self.root.withdraw()
        self.root.geometry(initial_geometry)

        try:
            self.app_dir = os.path.dirname(os.path.abspath(__file__))
            app_evaluator_dir = self.app_dir
            collected_data_analysis_dir = os.path.dirname(app_evaluator_dir) 
            project_root_dir = os.path.dirname(collected_data_analysis_dir)
        except NameError: 
            self.app_dir = os.getcwd()
            project_root_dir = os.path.abspath(os.path.join(self.app_dir, "..", ".."))

        self.collected_labels_path = os.path.join(project_root_dir, "04_Collected_Data_Analysis", "Collected_Data", "Labels", "collected_labels.csv")
        self.collected_raw_data_dir = os.path.join(project_root_dir, "04_Collected_Data_Analysis", "Collected_Data", "RawData")
        self.mendeley_model_dir = os.path.join(project_root_for_evaluator, "02_Machine_Learning_Mendeley", "src", "models")

        model_filename = config_mendeley.SAVED_MODEL_NAME if SCRIPTS_LOADED_SUCCESSFULLY and hasattr(config_mendeley, 'SAVED_MODEL_NAME') else "lgbm_glucose_model_retrained_v1.txt" 
        features_json_filename = config_mendeley.SAVED_FEATURES_NAME if SCRIPTS_LOADED_SUCCESSFULLY and hasattr(config_mendeley, 'SAVED_FEATURES_NAME') else "model_features_retrained_v1.json"
        scaler_filename = config_mendeley.SAVED_SCALER_NAME if SCRIPTS_LOADED_SUCCESSFULLY and hasattr(config_mendeley, 'SAVED_SCALER_NAME') else "mendeley_feature_scaler_retrained_v1.pkl"

        if SCRIPTS_LOADED_SUCCESSFULLY and hasattr(config_mendeley, 'SAVED_MODEL_NAME'):
            model_filename = config_mendeley.SAVED_MODEL_NAME
        if SCRIPTS_LOADED_SUCCESSFULLY and hasattr(config_mendeley, 'SAVED_FEATURES_NAME'):
            features_json_filename = config_mendeley.SAVED_FEATURES_NAME

        self.model_path = os.path.join(self.mendeley_model_dir, model_filename) 
        self.feature_names_path = os.path.join(self.mendeley_model_dir, features_json_filename)
        self.scaler_path = os.path.join(self.mendeley_model_dir, scaler_filename) 

        self.selected_ppg_file = tk.StringVar()
        self.reference_glucose = tk.StringVar(value="N/A")
        self.processing_in_progress = False
        self.model = None
        self.scaler = None
        self.expected_feature_order = []

        self._setup_gui() 
        self._load_model_and_scaler() 
        self._center_window() 
        self.root.deiconify()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

    def _load_model_and_scaler(self):
        if not SCRIPTS_LOADED_SUCCESSFULLY:
            self._log_message("Using DUMMY model, scaler, and features as script imports failed.")
            self.model, self.scaler, self.expected_feature_order = dummy_load_model_scaler_and_features(
                self.mendeley_model_dir, 
                os.path.basename(self.model_path), 
                os.path.basename(self.scaler_path), 
                os.path.basename(self.feature_names_path)
            )
            return

        try:
            self.model, self.scaler, self.expected_feature_order = model_trainer_mendeley.load_model_scaler_and_features(
                self.mendeley_model_dir, 
                os.path.basename(self.model_path),
                os.path.basename(self.scaler_path), 
                os.path.basename(self.feature_names_path)
            )
            if self.model and self.expected_feature_order:
                self._log_message(f"Pre-trained LightGBM model and feature order loaded from {self.mendeley_model_dir}.")
            else: 
                self._log_message(f"Failed to load model/features from {self.mendeley_model_dir} using actual scripts. Using DUMMY model.")
                self.model, self.scaler, self.expected_feature_order = dummy_load_model_scaler_and_features(
                    self.mendeley_model_dir, "lgbm_glucose_model.txt", "mendeley_feature_scaler.pkl", "model_features.json")

            if self.scaler:
                self._log_message(f"Feature scaler loaded successfully.")
            elif os.path.exists(self.scaler_path) and not USE_DUMMY_FUNCTIONS_FOR_PROCESSING : 
                 self._log_message(f"Scaler file exists at {self.scaler_path} but was not loaded by model_trainer. Check load_model_scaler_and_features in model_trainer.py.")
                 self.scaler = None 
            elif not os.path.exists(self.scaler_path) and not USE_DUMMY_FUNCTIONS_FOR_PROCESSING:
                self._log_message(f"Scaler file not found at {self.scaler_path}. Processing will proceed without feature scaling.")
                self.scaler = None
        except AttributeError: 
            self._log_message("ERROR: 'model_trainer_mendeley' module does not have 'load_model_scaler_and_features'. Using DUMMY.")
            self.model, self.scaler, self.expected_feature_order = dummy_load_model_scaler_and_features(
                self.mendeley_model_dir, "lgbm_glucose_model.txt", "mendeley_feature_scaler.pkl", "model_features.json")
        except Exception as e:
            self._log_message(f"Critical error loading model/scaler: {e}. Using DUMMY model.")
            self.model, self.scaler, self.expected_feature_order = dummy_load_model_scaler_and_features(
                self.mendeley_model_dir, "lgbm_glucose_model.txt", "mendeley_feature_scaler.pkl", "model_features.json")

    def _center_window(self):
        # (Same as before)
        self.root.update_idletasks() 
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        if width <= 1 or height <= 1:
            try:
                geom_parts = self.root.geometry().split('+')[0].split('x')
                width = int(geom_parts[0]); height = int(geom_parts[1])
            except: width = 750; height = 700 
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_coordinate = int((screen_width / 2) - (width / 2))
        y_coordinate = int((screen_height / 2) - (height / 2))
        self.root.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}")

    def _setup_gui(self):
        # (Same as before)
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True) 
        row_file_select = 0; row_info_action = 1; row_results = 2; row_log = 3
        file_frame = ttk.LabelFrame(main_frame, text="Select Data Sample", padding="10")
        file_frame.grid(row=row_file_select, column=0, sticky=(tk.W, tk.E), padx=5, pady=(5,10))
        main_frame.columnconfigure(0, weight=1)
        ttk.Button(file_frame, text="Browse Raw PPG File (*_ppg.csv)...", command=self._select_ppg_file).pack(side=tk.LEFT, padx=5)
        self.selected_file_label = ttk.Label(file_frame, text="No file selected.", width=70, anchor="w", relief=tk.SUNKEN, padding=2)
        self.selected_file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        action_frame = ttk.Frame(main_frame, padding="5")
        action_frame.grid(row=row_info_action, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(action_frame, text="Reference Glucose (mg/dL):").pack(side=tk.LEFT, padx=(0,2))
        self.ref_glucose_label = ttk.Label(action_frame, textvariable=self.reference_glucose, font=('Helvetica', 10, 'bold'), width=10)
        self.ref_glucose_label.pack(side=tk.LEFT, padx=(0,10))
        self.process_button = ttk.Button(action_frame, text="Process & Evaluate Sample", command=self._start_processing_thread, state=tk.DISABLED)
        self.process_button.pack(side=tk.RIGHT, padx=5) 
        results_frame = ttk.LabelFrame(main_frame, text="Evaluation Results", padding="10")
        results_frame.grid(row=row_results, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        results_frame.columnconfigure(0, weight=1); results_frame.rowconfigure(0, weight=1) 
        self.results_cols = ["Approach", "Predicted Glucose (mg/dL)", "ARD (%)", "Accuracy (%)"]
        self.results_tree = ttk.Treeview(results_frame, columns=self.results_cols, show="headings", height=7)

        col_widths = {
            "Approach": 220,
            "Predicted Glucose (mg/dL)": 160,
            "ARD (%)": 90,
            "Accuracy (%)": 90
        }

        for col in self.results_cols:
            self.results_tree.heading(col, text=col)
            # Set text alignment for each column
            anchor = 'e' if col != "Approach" else 'w'
            # Use the defined widths and a suitable minwidth
            self.results_tree.column(
                col,
                width=col_widths.get(col, 100),
                anchor=anchor,
                minwidth=80
            )
        results_scrollbar_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar_y.set)
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) 
        results_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S)) 
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=row_log, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=(10,5))
        log_frame.columnconfigure(0, weight=1); log_frame.rowconfigure(0, weight=1) 
        self.log_text = tk.Text(log_frame, height=10, width=80, state=tk.DISABLED, relief=tk.SUNKEN, borderwidth=1)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) 
        log_scrollbar_y_log = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar_y_log.grid(row=0, column=1, sticky=(tk.N, tk.S)) 
        self.log_text['yscrollcommand'] = log_scrollbar_y_log.set
        main_frame.rowconfigure(row_results, weight=1); main_frame.rowconfigure(row_log, weight=1)   
        self._log_message("Evaluator App Initialized. Select a Raw PPG file.")

    def _log_message(self, message):
        if hasattr(self, 'log_text') and self.log_text:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
            if self.root and self.root.winfo_exists(): self.root.update_idletasks()
        else: print(f"LOG_FALLBACK ({datetime.now().strftime('%H:%M:%S')}): {message}")

    def _select_ppg_file(self):
        # (Same as before)
        filepath = filedialog.askopenfilename(
            initialdir=self.collected_raw_data_dir,
            title="Select Raw PPG Data File (*_ppg.csv)",
            filetypes=(("CSV files", "*_ppg.csv"), ("All files", "*.*"))
        )
        if filepath:
            self.selected_ppg_file.set(filepath)
            self.selected_file_label.config(text=os.path.basename(filepath))
            self._log_message(f"Selected PPG file: {os.path.basename(filepath)}")
            self._load_label_for_file(filepath)
            self.process_button.config(state=tk.NORMAL)
            for i in self.results_tree.get_children(): self.results_tree.delete(i)
        else:
            self.selected_file_label.config(text="No file selected.")
            self.reference_glucose.set("N/A")
            self.process_button.config(state=tk.DISABLED)

    def _load_label_for_file(self, ppg_filepath):
        # (Same as before)
        try:
            filename = os.path.basename(ppg_filepath)
            base = filename.replace("_ppg.csv", "")
            parts = base.split("_")
            if len(parts) < 2: 
                self._log_message(f"Filename {filename} doesn't match SubjectID_SampleNum format.")
                self.reference_glucose.set("Error: Filename"); return
            subject_id_from_file = parts[0]; sample_num_from_file = parts[1]
            if not os.path.exists(self.collected_labels_path):
                self._log_message(f"Labels file not found: {self.collected_labels_path}")
                self.reference_glucose.set("Error: No Labels CSV"); return
            labels_df = pd.read_csv(self.collected_labels_path)
            match = labels_df[(labels_df['ID'].astype(str) == str(subject_id_from_file)) & 
                              (labels_df['Sample_Num'].astype(str) == str(sample_num_from_file))]
            if not match.empty:
                ref_glucose_val = match['Glucose_mgdL'].iloc[0]
                self.reference_glucose.set(str(ref_glucose_val))
                self._log_message(f"Ref. glucose for {subject_id_from_file}_{sample_num_from_file}: {ref_glucose_val} mg/dL")
            else:
                self._log_message(f"No label found for {subject_id_from_file}_{sample_num_from_file} in {self.collected_labels_path}")
                self.reference_glucose.set("N/A - Not Found")
        except Exception as e:
            self._log_message(f"Error loading label: {e}")
            self.reference_glucose.set("Error Loading Label")
            messagebox.showerror("Label Error", f"Could not load label: {e}")

    def _start_processing_thread(self):
        # (Same as before)
        if self.processing_in_progress: messagebox.showwarning("Busy", "Processing in progress."); return
        if not self.selected_ppg_file.get(): messagebox.showwarning("No File", "Select PPG file."); return
        ref_glucose_str = self.reference_glucose.get()
        if ref_glucose_str == "N/A" or "Error" in ref_glucose_str or not ref_glucose_str:
            messagebox.showwarning("No Ref. Glucose", "Reference glucose not loaded. Cannot evaluate."); return
        self.processing_in_progress = True
        self.process_button.config(state=tk.DISABLED)
        for i in self.results_tree.get_children(): self.results_tree.delete(i)
        self._log_message("Starting processing & evaluation thread...")
        thread = threading.Thread(target=self._process_and_evaluate, daemon=True)
        thread.start()

    def _process_and_evaluate(self):
        try:
            ppg_file = self.selected_ppg_file.get()
            actual_glucose = float(self.reference_glucose.get())
            self.root.after(0, lambda: self._log_message(f"Processing: {os.path.basename(ppg_file)}"))

            # --- TEMPORARY MODEL SENSITIVITY TEST ---
            if self.model and self.expected_feature_order:
                num_feats = len(self.expected_feature_order)
                self.root.after(0, lambda: self._log_message(f"--- Starting Model Sensitivity Test (Num feats: {num_feats}) ---"))
                
                test_vec_low = np.array([[0.1] * num_feats]) # All features at 0.1
                test_vec_mid = np.array([[0.5] * num_feats]) # All features at 0.5
                test_vec_high = np.array([[0.9] * num_feats])# All features at 0.9
                
                # Create a feature vector with more variation
                test_vec_varied = np.random.rand(1, num_feats) 

                if self.model.feature_name_ is None and self.expected_feature_order:
                    # LightGBM Booster.predict can take feature_name via a parameter if not set on model object
                    # However, it's better if model_trainer.py sets model.feature_name_ or uses lgb.Dataset with feature_name
                    # For now, we assume the model will use the order if feature_name isn't explicitly part of the Booster object.
                    pass

                pred_low = self.model.predict(test_vec_low)
                pred_mid = self.model.predict(test_vec_mid)
                pred_high = self.model.predict(test_vec_high)
                pred_varied = self.model.predict(test_vec_varied)

                self.root.after(0, lambda p=pred_low: self._log_message(f"MANUAL MODEL TEST - Low Feats Pred: {p}"))
                self.root.after(0, lambda p=pred_mid: self._log_message(f"MANUAL MODEL TEST - Mid Feats Pred: {p}"))
                self.root.after(0, lambda p=pred_high: self._log_message(f"MANUAL MODEL TEST - High Feats Pred: {p}"))
                self.root.after(0, lambda p=pred_varied: self._log_message(f"MANUAL MODEL TEST - Varied Feats Pred: {p}"))
                self.root.after(0, lambda: self._log_message(f"--- End of Model Sensitivity Test ---"))
            # --- END OF TEMPORARY MODEL SENSITIVITY TEST ---

            raw_signals_list = load_collected_ppg_data_from_file(ppg_file)
            if not all(isinstance(sig, np.ndarray) and sig.size > 0 for sig in raw_signals_list):
                 self.root.after(0, lambda: self._log_message("Error: Empty or invalid signal array(s) loaded."))
                 raise ValueError("Loaded signal data is empty or invalid.")
            self.root.after(0, lambda: self._log_message(f"Raw PPG signals loaded ({len(raw_signals_list[0])} samples per finger)."))
            
            if raw_signals_list[0].size > 5 : self.root.after(0, lambda: self._log_message(f"Raw Sig 1 (first 5): {raw_signals_list[0][:5]}"))
            if raw_signals_list[1].size > 5 : self.root.after(0, lambda: self._log_message(f"Raw Sig 2 (first 5): {raw_signals_list[1][:5]}"))
            if raw_signals_list[2].size > 5 : self.root.after(0, lambda: self._log_message(f"Raw Sig 3 (first 5): {raw_signals_list[2][:5]}"))

            all_finger_segments_for_features = [] # This will store UNscaled features
            all_finger_original_segments_for_sqi = [] # This will store preprocessed signal segments

            input_fs = 100 
            target_fs_feat = config_mendeley.TARGET_FS if SCRIPTS_LOADED_SUCCESSFULLY and config_mendeley else 50
            win_dur_sec = config_mendeley.WINDOW_DURATION_SEC if SCRIPTS_LOADED_SUCCESSFULLY and config_mendeley else 5
            
            for i in range(3): 
                self.root.after(0, lambda finger=i+1: self._log_message(f"Processing Finger {finger}..."))
                segments_one_finger = preprocess_one_finger_signal(
                    raw_signals_list[i], input_fs=input_fs, 
                    target_fs_for_features=target_fs_feat, window_duration_sec=win_dur_sec
                )
                all_finger_original_segments_for_sqi.append(segments_one_finger) # Store for SQI/SNR
                self.root.after(0, lambda finger=i+1, num_seg=len(segments_one_finger): self._log_message(f"  Finger {finger}: {num_seg} segments created."))
                if not segments_one_finger: 
                    all_finger_segments_for_features.append([]) # Keep placeholder for this finger
                    continue
                
                features_one_finger = extract_features_for_finger_segments(segments_one_finger, fs=target_fs_feat, expected_feature_order=self.expected_feature_order)
                self.root.after(0, lambda finger=i+1, num_feat=len(features_one_finger): self._log_message(f"  Finger {finger}: {num_feat} feature sets extracted."))
                if not features_one_finger: 
                    all_finger_segments_for_features.append([])
                    continue
                all_finger_segments_for_features.append(features_one_finger) # Store unscaled features
            
            if all_finger_segments_for_features[0] and len(all_finger_segments_for_features[0]) > 0: self.root.after(0, lambda: self._log_message(f"Unscaled F1 Feats (Shape {np.array(all_finger_segments_for_features[0]).shape}): {all_finger_segments_for_features[0][0]}"))
            if len(all_finger_segments_for_features) > 1 and all_finger_segments_for_features[1] and len(all_finger_segments_for_features[1]) > 0: self.root.after(0, lambda: self._log_message(f"Unscaled F2 Feats (Shape {np.array(all_finger_segments_for_features[1]).shape}): {all_finger_segments_for_features[1][0]}"))
            if len(all_finger_segments_for_features) > 2 and all_finger_segments_for_features[2] and len(all_finger_segments_for_features[2]) > 0: self.root.after(0, lambda: self._log_message(f"Unscaled F3 Feats (Shape {np.array(all_finger_segments_for_features[2]).shape}): {all_finger_segments_for_features[2][0]}"))

            # --- ENSURE SCALING IS APPLIED ---
            features_f1_s = apply_scaler_to_features(all_finger_segments_for_features[0], self.scaler) if len(all_finger_segments_for_features) > 0 and all_finger_segments_for_features[0] else []
            features_f2_s = apply_scaler_to_features(all_finger_segments_for_features[1], self.scaler) if len(all_finger_segments_for_features) > 1 and all_finger_segments_for_features[1] else []
            features_f3_s = apply_scaler_to_features(all_finger_segments_for_features[2], self.scaler) if len(all_finger_segments_for_features) > 2 and all_finger_segments_for_features[2] else []
            self.root.after(0, lambda: self._log_message("Features scaled (if scaler was loaded)."))
            # --- END OF SCALING FIX ---

            if features_f1_s and len(features_f1_s) > 0: self.root.after(0, lambda: self._log_message(f"SCALED F1 Feats (Shape {np.array(features_f1_s).shape}): {features_f1_s[0]}"))
            if features_f2_s and len(features_f2_s) > 0: self.root.after(0, lambda: self._log_message(f"SCALED F2 Feats (Shape {np.array(features_f2_s).shape}): {features_f2_s[0]}"))
            if features_f3_s and len(features_f3_s) > 0: self.root.after(0, lambda: self._log_message(f"SCALED F3 Feats (Shape {np.array(features_f3_s).shape}): {features_f3_s[0]}"))
            
            fused_features_snr = fuse_features_snr_weighted( 
                features_f1_s, features_f2_s, features_f3_s,
                segments_f1=all_finger_original_segments_for_sqi[0] if len(all_finger_original_segments_for_sqi) > 0 else [], 
                segments_f2=all_finger_original_segments_for_sqi[1] if len(all_finger_original_segments_for_sqi) > 1 else [], 
                segments_f3=all_finger_original_segments_for_sqi[2] if len(all_finger_original_segments_for_sqi) > 2 else [],
                fs=target_fs_feat
            )
            
            fused_features_sqi = fuse_features_sqi_selected( 
                features_f1_s, features_f2_s, features_f3_s,
                segments_f1=all_finger_original_segments_for_sqi[0] if len(all_finger_original_segments_for_sqi) > 0 else [], 
                segments_f2=all_finger_original_segments_for_sqi[1] if len(all_finger_original_segments_for_sqi) > 1 else [], 
                segments_f3=all_finger_original_segments_for_sqi[2] if len(all_finger_original_segments_for_sqi) > 2 else [],
                fs=target_fs_feat
            )

            if fused_features_snr and len(fused_features_snr) > 0: self.root.after(0, lambda: self._log_message(f"SNR Fused Feats (1st seg): {fused_features_snr[0]}"))
            if fused_features_sqi and len(fused_features_sqi) > 0: self.root.after(0, lambda: self._log_message(f"SQI Fused Feats (1st seg): {fused_features_sqi[0]}"))

            if self.model is None: 
                self.root.after(0, lambda: self._log_message("ERROR: Model not loaded! Cannot make predictions."))
                raise ValueError("Model not loaded.")
            
            def predict_and_average(features_list, approach_name="Unknown"):
                if features_list and len(features_list) > 0:
                    np_features_list = np.array(features_list)
                    if np_features_list.ndim == 1: np_features_list = np_features_list.reshape(1, -1)
                    if np_features_list.size == 0: 
                        self.root.after(0, lambda: self._log_message(f"Predict for {approach_name}: No features to predict."))
                        return np.nan
                    self.root.after(0, lambda app_name=approach_name, f_vec_shape=np_features_list.shape, f_vec_sample=np_features_list[0] if len(np_features_list)>0 else []: self._log_message(f"Features for {app_name} PREDICTION (Shape {f_vec_shape}, 1st seg): {f_vec_sample}"))
                    predictions = self.model.predict(np_features_list)
                    self.root.after(0, lambda app_name=approach_name, preds=predictions.tolist() if isinstance(predictions, np.ndarray) else predictions: self._log_message(f"Raw Preds for {app_name}: {preds}"))
                    avg_prediction = np.mean(predictions) if predictions.size > 0 else np.nan
                    self.root.after(0, lambda app_name=approach_name, avg_p=avg_prediction: self._log_message(f"Avg Pred for {app_name}: {avg_p}"))
                    return avg_prediction
                self.root.after(0, lambda app_name=approach_name: self._log_message(f"Predict for {approach_name}: Feature list empty."))
                return np.nan

            pred_f1 = predict_and_average(features_f1_s, "Finger 1")
            pred_f2 = predict_and_average(features_f2_s, "Finger 2")
            pred_f3 = predict_and_average(features_f3_s, "Finger 3")
            pred_snr = predict_and_average(fused_features_snr, "SNR Fusion")
            pred_sqi = predict_and_average(fused_features_sqi, "SQI Fusion")

            results_to_display = []
            approaches = {
                "Finger 1 (Individual)": pred_f1, "Finger 2 (Individual)": pred_f2, 
                "Finger 3 (Individual)": pred_f3, "SNR-Weighted Fusion": pred_snr, 
                "SQI-Selected Fusion": pred_sqi
            }
            for name, pred_glucose_val in approaches.items():
                if pd.isna(pred_glucose_val) or actual_glucose == 0 : 
                    ard_p_str, acc_p_str = "N/A", "N/A"
                    pred_g_str = "N/A" if pd.isna(pred_glucose_val) else f"{pred_glucose_val:.2f}"
                else:
                    ard = abs(actual_glucose - pred_glucose_val) / actual_glucose
                    ard_p_str = f"{ard * 100:.2f}"
                    acc_val = (1 - ard) * 100
                    acc_p_str = f"{max(0, acc_val):.2f}" 
                    pred_g_str = f"{pred_glucose_val:.2f}"
                results_to_display.append([name, pred_g_str, ard_p_str, acc_p_str])
            
            self.root.after(0, lambda r=results_to_display: self._update_results_display(r))

        except Exception as e:
            self.root.after(0, lambda err=e: self._log_message(f"ERROR during processing: {err}"))
            self.root.after(0, lambda err=e: messagebox.showerror("Processing Error", f"An error occurred: {err}\nCheck console for details."))
        finally:
            self.processing_in_progress = False
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))

    def _update_results_display(self, results_list):
        for i in self.results_tree.get_children():
            self.results_tree.delete(i)
        for row_data in results_list:
            self.results_tree.insert("", tk.END, values=row_data)
        self._log_message("Results display updated.")

if __name__ == '__main__':
    global app 
    main_window = ThemedTk(theme="arc") 
    app = PPGDataEvaluatorApp(main_window, initial_geometry="650x700")
    
    def on_closing():
        main_window.destroy()
    main_window.protocol("WM_DELETE_WINDOW", on_closing)
    main_window.mainloop()
