import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ttkthemes import ThemedTk
import pandas as pd
import numpy as np
import os
import sys
import threading
from datetime import datetime
import joblib
from scipy.signal import resample, find_peaks
import lightgbm as lgb
import json

# --- For Standalone Execution: Dynamically find the Mendeley scripts ---
try:
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_path, '..', '..'))
    mendeley_src_path = os.path.join(project_root, "02_Machine_Learning_Mendeley", "src")
    if mendeley_src_path not in sys.path:
        sys.path.insert(0, mendeley_src_path)
    
    # Import necessary functions from your project
    import config as config_mendeley
    import preprocessing as preprocessing_mendeley
    import feature_extraction_new as feature_extraction_mendeley
    SCRIPTS_LOADED = True
    print("Successfully loaded processing modules.")
except ImportError as e:
    SCRIPTS_LOADED = False
    print(f"Could not load processing modules, using dummy functions: {e}")


# --- Helper Functions (Adapted from your evaluator) ---
# NOTE: These functions are simplified for the demo app.

def calculate_snr_for_segment(segment_array, fs):
    if segment_array is None or len(segment_array) < 2: return 0.01
    signal_power = np.var(segment_array)
    if len(segment_array) > 5:
        std_diff2 = np.std(np.diff(np.diff(segment_array)))
        noise_proxy_inv = 1.0 / (1.0 + std_diff2) if std_diff2 > 1e-9 else 1.0
        snr = signal_power * noise_proxy_inv
    else: snr = signal_power
    return max(0.01, snr)

def calculate_sqi_for_segment(segment_array, fs):
    if segment_array is None or len(segment_array) < fs: return 0.0
    try:
        std_dev = np.std(segment_array)
        if std_dev < 1e-9: return 0.0
        peak_to_peak_amplitude = np.ptp(segment_array)
        min_peak_dist = int(fs * 0.3)
        prominence_threshold = 0.1 * peak_to_peak_amplitude
        peaks, _ = find_peaks(segment_array, distance=min_peak_dist, prominence=prominence_threshold)
        if len(peaks) < 2: ppi_std_norm = 1.0
        else:
            ppi_values = np.diff(peaks) / fs
            ppi_mean = np.mean(ppi_values)
            ppi_std = np.std(ppi_values)
            ppi_std_norm = ppi_std / ppi_mean if ppi_mean > 1e-9 else 1.0
        sqi = peak_to_peak_amplitude / (std_dev * (1 + ppi_std_norm))
        return max(0.0, sqi)
    except Exception: return 0.0

def fuse_features_sqi_selected(features_scaled, segments, fs):
    """Simplified SQI fusion for the demo."""
    selected_features_list = []
    num_segments = len(features_scaled[0])
    for i in range(num_segments):
        sqis = [calculate_sqi_for_segment(seg[i], fs) for seg in segments]
        best_finger_idx = np.argmax(sqis)
        selected_features_list.append(features_scaled[best_finger_idx][i])
    return selected_features_list


class LiveDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Glucose Prediction Demo")
        self.root.geometry("600x550")

        # --- 1. UPDATE THESE PATHS ---
        # Update these paths to point to your fine-tuned model and scaler
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.abspath(os.path.join(self.app_dir, '..', '..'))
        
        # This should point to where your fine-tuned models are saved
        self.models_dir = os.path.join(project_root_dir, "04_Collected_Data_Analysis", "models_finetuned")
        self.mendeley_models_dir = os.path.join(project_root_dir, "02_Machine_Learning_Mendeley", "src", "models")

        self.model_path = os.path.join(self.models_dir, "lgbm_model_finetuned_on_16subjects.txt") # <-- YOUR BEST MODEL
        self.scaler_path = os.path.join(self.mendeley_models_dir, "mendeley_feature_scaler_retrained_v1.pkl") # <-- YOUR SCALER
        self.features_path = os.path.join(self.mendeley_models_dir, "model_features_retrained_v1.json") # <-- YOUR FEATURES
        
        # This is the directory where you store the PPG sample from the demo
        self.demo_data_dir = os.path.join(project_root_dir, "04_Collected_Data_Analysis", "Collected_Data", "RawData")
        # --- END OF PATHS SECTION ---

        self.selected_ppg_file = tk.StringVar()
        self.actual_glucose = tk.StringVar()
        self.processing_in_progress = False

        self.model = None
        self.scaler = None
        self.feature_order = []

        self._setup_gui()
        self._load_model_assets()
        self._center_window()
        self.root.deiconify()

    def _load_model_assets(self):
        try:
            self._log("Loading model...")
            self.model = lgb.Booster(model_file=self.model_path)
            self._log(f"-> Model loaded from {os.path.basename(self.model_path)}")
            
            self._log("Loading scaler...")
            self.scaler = joblib.load(self.scaler_path)
            self._log(f"-> Scaler loaded from {os.path.basename(self.scaler_path)}")
            
            self._log("Loading feature list...")
            with open(self.features_path, 'r') as f:
                data = json.load(f)
                # Check if the loaded data is a dictionary with a 'features' key
                if isinstance(data, dict) and 'features' in data:
                    self.feature_order = data['features']
                # Otherwise, assume the data is the list itself
                else:
                    self.feature_order = data
            self._log(f"-> {len(self.feature_order)} features loaded.")
            
            self._log("\nAll assets loaded successfully. Ready for demo.")
        except Exception as e:
            self._log(f"ERROR loading assets: {e}")
            messagebox.showerror("Error", f"Could not load model assets: {e}")

    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(main_frame, text="Step 1: Provide Inputs", padding="10")
        input_frame.pack(fill=tk.X, expand=True, pady=5)

        # PPG File Selection
        ttk.Button(input_frame, text="Browse for PPG File...", command=self._select_file).grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.file_label = ttk.Label(input_frame, text="No PPG file selected.", relief=tk.SUNKEN, padding=5)
        self.file_label.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # Actual Glucose Entry
        ttk.Label(input_frame, text="Enter Actual Glucose (mg/dL):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.glucose_entry = ttk.Entry(input_frame, textvariable=self.actual_glucose, width=25)
        self.glucose_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        input_frame.columnconfigure(1, weight=1)

        # --- Action Button ---
        self.run_button = ttk.Button(main_frame, text="Step 2: Run Prediction", command=self._start_processing, state=tk.DISABLED)
        self.run_button.pack(fill=tk.X, expand=True, pady=10, ipady=5)

        # --- Results Frame ---
        results_frame = ttk.LabelFrame(main_frame, text="Step 3: View Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.result_labels = {}
        results_info = {
            "Predicted Glucose (mg/dL):": "---",
            "Actual Glucose (mg/dL):": "---",
            "MARD (%):": "---",
            "Accuracy (%):": "---"
        }
        for i, (text, val) in enumerate(results_info.items()):
            ttk.Label(results_frame, text=text, font=('Helvetica', 10)).grid(row=i, column=0, sticky="w", padx=5, pady=3)
            self.result_labels[text] = ttk.Label(results_frame, text=val, font=('Helvetica', 12, 'bold'), foreground="navy")
            self.result_labels[text].grid(row=i, column=1, sticky="w", padx=5, pady=3)

        # --- Log Frame ---
        log_frame = ttk.LabelFrame(main_frame, text="Live Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, relief=tk.SUNKEN, borderwidth=1, font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} > {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def _select_file(self):
        filepath = filedialog.askopenfilename(
            initialdir=self.demo_data_dir,
            title="Select a collected PPG CSV file",
            filetypes=(("CSV files", "*_ppg.csv"), ("All files", "*.*"))
        )
        if filepath:
            self.selected_ppg_file.set(filepath)
            self.file_label.config(text=os.path.basename(filepath))
            self._log(f"Selected file: {os.path.basename(filepath)}")
            self.run_button.config(state=tk.NORMAL)
        else:
            self.run_button.config(state=tk.DISABLED)
            
    def _start_processing(self):
        if self.processing_in_progress: return
        if not self.selected_ppg_file.get():
            messagebox.showerror("Input Error", "Please select a PPG file.")
            return
        if not self.actual_glucose.get():
            messagebox.showerror("Input Error", "Please enter the actual glucose value.")
            return
        
        self.processing_in_progress = True
        self.run_button.config(state=tk.DISABLED)
        self._log("\nStarting prediction thread...")
        
        # Clear previous results
        for label in self.result_labels.values():
            label.config(text="---")

        thread = threading.Thread(target=self._process_and_predict, daemon=True)
        thread.start()

    def _process_and_predict(self):
        try:
            ppg_file = self.selected_ppg_file.get()
            actual_glucose = float(self.actual_glucose.get())
            self.root.after(0, lambda: self.result_labels["Actual Glucose (mg/dL):"].config(text=f"{actual_glucose:.2f}"))

            # 1. Load Data
            self._log("Loading PPG signals from file...")
            df = pd.read_csv(ppg_file)
            raw_signals = [df[f'ppg_finger{i+1}'].to_numpy() for i in range(3)]
            self._log(f"-> Loaded {len(raw_signals[0])} samples per finger.")

            # 2. Preprocess & Feature Extraction
            all_features_unscaled = []
            all_segments_for_fusion = []
            target_fs = config_mendeley.TARGET_FS
            win_dur = config_mendeley.WINDOW_DURATION_SEC
            
            for i, sig in enumerate(raw_signals):
                self._log(f"Processing Finger {i+1}...")
                
                # Resample and filter
                num_samples_target = int(len(sig) * (target_fs / 100.0))
                resampled_sig = resample(sig, num_samples_target)
                filtered_sig = preprocessing_mendeley.apply_bandpass_filter(resampled_sig, config_mendeley.FILTER_LOWCUT, config_mendeley.FILTER_HIGHCUT, target_fs, config_mendeley.FILTER_ORDER)
                smoothed_sig = preprocessing_mendeley.apply_savgol_smoothing(filtered_sig, config_mendeley.SAVGOL_WINDOW, config_mendeley.SAVGOL_POLYORDER)
                
                # Segment
                segments = preprocessing_mendeley.segment_signal(smoothed_sig, int(win_dur * target_fs))
                all_segments_for_fusion.append(segments)
                self._log(f"-> Created {len(segments)} segments.")
                
                # Extract Features
                features = [feature_extraction_mendeley.extract_all_features_from_segment(seg, target_fs) for seg in segments]
                ordered_features = [[f.get(feat_name, np.nan) for feat_name in self.feature_order] for f in features]
                all_features_unscaled.append(ordered_features)
                self._log(f"-> Extracted {len(ordered_features)} feature sets.")

            # 3. Scale Features
            self._log("Scaling features...")
            all_features_scaled = [self.scaler.transform(np.array(f)) for f in all_features_unscaled]
            self._log("-> Scaling complete.")

            # 4. Fuse Features (using SQI-based selection for demo)
            self._log("Fusing features using SQI-selection...")
            fused_features = fuse_features_sqi_selected(all_features_scaled, all_segments_for_fusion, target_fs)
            self._log(f"-> Fusion resulted in {len(fused_features)} final feature sets.")

            # 5. Predict
            self._log("Making predictions with the model...")
            predictions = self.model.predict(np.array(fused_features))
            final_prediction = np.mean(predictions)
            self._log(f"-> Final Average Prediction: {final_prediction:.2f} mg/dL")

            # 6. Calculate Metrics and Display
            mard = abs(final_prediction - actual_glucose) / actual_glucose * 100
            accuracy = max(0, 100 - mard)
            
            self.root.after(0, self._update_results, final_prediction, mard, accuracy)
            self._log("\nDemo complete!")

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Processing Error", f"An error occurred: {e}"))
            self._log(f"ERROR: {e}")
        finally:
            self.processing_in_progress = False
            self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL))

    def _update_results(self, prediction, mard, accuracy):
        self.result_labels["Predicted Glucose (mg/dL):"].config(text=f"{prediction:.2f}", foreground="green")
        self.result_labels["MARD (%):"].config(text=f"{mard:.2f}")
        self.result_labels["Accuracy (%):"].config(text=f"{accuracy:.2f}")

    def _center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        self.root.withdraw() # Hide window until it's centered
        self.root.deiconify()


if __name__ == '__main__':
    if not SCRIPTS_LOADED:
        print("\nCould not start the app because the required processing scripts from your project were not found.")
        print("Please ensure this demo script is in the 'app_evaluator' directory.")
    else:
        themed_root = ThemedTk(theme="arc")
        app = LiveDemoApp(themed_root)
        themed_root.mainloop()