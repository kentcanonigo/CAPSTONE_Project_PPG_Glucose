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
import lightgbm as lgb
import json

# --- For Standalone Execution: Dynamically find the Mendeley scripts ---
try:
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_path, '..', '..'))
    mendeley_src_path = os.path.join(project_root, "02_Machine_Learning_Mendeley", "src")
    if mendeley_src_path not in sys.path:
        sys.path.insert(0, mendeley_src_path)
    
    import config as config_mendeley
    import preprocessing as preprocessing_mendeley
    import feature_extraction_new as feature_extraction_mendeley
    SCRIPTS_LOADED = True
    print("Successfully loaded processing modules.")
except ImportError as e:
    SCRIPTS_LOADED = False
    print(f"Could not load processing modules: {e}")


# --- Helper Functions (Unchanged) ---
def calculate_sqi_for_segment(segment_array, fs):
    if segment_array is None or len(segment_array) < fs: return 0.0
    try:
        from scipy.signal import find_peaks
        std_dev = np.std(segment_array)
        if std_dev < 1e-9: return 0.0
        peak_to_peak_amplitude = np.ptp(segment_array)
        peaks, _ = find_peaks(segment_array, distance=int(fs * 0.3))
        if len(peaks) < 2: return peak_to_peak_amplitude / (std_dev * 2.0)
        else:
            ppi_values, ppi_mean = np.diff(peaks) / fs, np.mean(np.diff(peaks) / fs)
            ppi_std_norm = np.std(ppi_values) / ppi_mean if ppi_mean > 1e-9 else 1.0
            return peak_to_peak_amplitude / (std_dev * (1 + ppi_std_norm))
    except Exception: return 0.0

def calculate_snr_for_segment(segment_array, fs):
    if segment_array is None or len(segment_array) < 2: return 0.01
    try:
        signal_power = np.var(segment_array)
        snr = signal_power * (1.0 / (1.0 + np.std(np.diff(np.diff(segment_array))))) if len(segment_array) > 5 else signal_power
        return max(0.01, snr)
    except Exception: return 0.01

def fuse_features_sqi_selected(features_scaled, segments, fs):
    selected_features_list = []
    if not segments or not segments[0]: return []
    num_segments = len(segments[0])
    for i in range(num_segments):
        sqis = [calculate_sqi_for_segment(seg[i], fs) for seg in segments]
        best_finger_idx = np.argmax(sqis)
        selected_features_list.append(features_scaled[best_finger_idx][i])
    return selected_features_list

def fuse_features_snr_weighted(features_scaled, segments, fs):
    fused_list = []
    if not segments or not segments[0]: return []
    num_segments = len(segments[0])
    for i in range(num_segments):
        snrs = [calculate_snr_for_segment(seg[i], fs) for seg in segments]
        total_snr = sum(snrs)
        weights = [snr / total_snr if total_snr > 1e-6 else 1/3 for snr in snrs]
        fused_features = np.zeros_like(features_scaled[0][i])
        for finger_idx in range(len(features_scaled)):
            fused_features += weights[finger_idx] * np.array(features_scaled[finger_idx][i])
        fused_list.append(fused_features.tolist())
    return fused_list


class LiveDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Glucose Prediction Demo")
        self.root.geometry("750x650")

        # --- Project Paths ---
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.abspath(os.path.join(self.app_dir, '..', '..'))
        self.models_dir = os.path.join(project_root_dir, "04_Collected_Data_Analysis", "evaluation_results")
        self.mendeley_models_dir = os.path.join(project_root_dir, "02_Machine_Learning_Mendeley", "src", "models")
        self.demo_data_dir = os.path.join(project_root_dir, "04_Collected_Data_Analysis", "Collected_Data", "RawData")
        
        # --- Model Asset Paths ---
        self.model_path = os.path.join(self.models_dir, "lgbm_model_finetuned_on_20subjects_ALL_FINGERS.txt")
        self.scaler_path = os.path.join(self.mendeley_models_dir, "mendeley_feature_scaler_retrained_v1.pkl")
        self.features_path = os.path.join(self.mendeley_models_dir, "model_features_retrained_v1.json")

        # --- State Variables ---
        self.selected_ppg_file = tk.StringVar()
        self.actual_glucose = tk.StringVar()
        self.processing_in_progress = False
        self.model, self.scaler, self.feature_order = None, None, []
        
        # --- Prediction Result Storage ---
        self.predicted_sqi = None
        self.predicted_snr = None
        self.predicted_fingers = []
        self.tree_item_ids = {}

        self._setup_gui()
        self._load_model_assets()
        self._center_window()

    def _load_model_assets(self):
        try:
            self._log("Loading model, scaler, and features...")
            self.model = lgb.Booster(model_file=self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            with open(self.features_path, 'r') as f:
                self.feature_order = json.load(f)
            self._log(f"-> Model: {os.path.basename(self.model_path)}")
            self._log(f"-> Scaler & {len(self.feature_order)} features loaded.")
            self._log("\nAll assets loaded successfully. Ready for demo.")
        except Exception as e:
            self._log(f"ERROR loading assets: {e}")
            messagebox.showerror("Error", f"Could not load model assets: {e}")

    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Step 1: File Selection ---
        input_frame = ttk.LabelFrame(main_frame, text="Step 1: Select PPG Data File", padding="10")
        input_frame.pack(fill=tk.X, expand=False, pady=5)
        ttk.Button(input_frame, text="Browse for PPG File...", command=self._select_file).pack(side=tk.LEFT, padx=5, pady=5)
        self.file_label = ttk.Label(input_frame, text="No file selected.", relief=tk.SUNKEN, padding=5)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        # --- Step 2: Run Prediction ---
        self.run_button = ttk.Button(main_frame, text="Step 2: Run Prediction", command=self._start_processing, state=tk.DISABLED)
        self.run_button.pack(fill=tk.X, expand=False, pady=10, ipady=5)
        
        # --- Step 3: Optional Accuracy Calculation ---
        accuracy_frame = ttk.LabelFrame(main_frame, text="Step 3: Calculate Accuracy (Optional)", padding="10")
        accuracy_frame.pack(fill=tk.X, expand=False, pady=5)
        ttk.Label(accuracy_frame, text="Enter Actual Glucose (mg/dL):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.glucose_entry = ttk.Entry(accuracy_frame, textvariable=self.actual_glucose, width=20)
        self.glucose_entry.grid(row=0, column=1, sticky="ew", padx=5)
        self.accuracy_button = ttk.Button(accuracy_frame, text="Calculate Accuracy", command=self._calculate_and_display_accuracy, state=tk.DISABLED)
        self.accuracy_button.grid(row=0, column=2, sticky="ew", padx=10)
        accuracy_frame.columnconfigure(2, weight=1)

        # --- Step 4: Results ---
        results_frame = ttk.LabelFrame(main_frame, text="Step 4: View Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results_cols = ["Approach", "Predicted", "Actual", "MARD (%)", "Accuracy (%)"]
        self.results_tree = ttk.Treeview(results_frame, columns=self.results_cols, show="headings", height=3)
        col_widths = {"Approach": 180, "Predicted": 100, "Actual": 80, "MARD (%)": 100, "Accuracy (%)": 100}
        for col in self.results_cols:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, anchor='center', width=col_widths.get(col, 120))
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        self.results_tree.tag_configure('good_accuracy', foreground='green')

        # --- Live Log ---
        log_frame = ttk.LabelFrame(main_frame, text="Live Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, relief=tk.SUNKEN, borderwidth=1, font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _log(self, message):
        self.root.after(0, lambda: self._update_log_text(message))

    def _update_log_text(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} > {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _select_file(self):
        filepath = filedialog.askopenfilename(initialdir=self.demo_data_dir, title="Select a collected PPG CSV file", filetypes=(("CSV files", "*_ppg.csv"),))
        if filepath:
            self.selected_ppg_file.set(filepath)
            self.file_label.config(text=os.path.basename(filepath))
            self._log(f"Selected file: {os.path.basename(filepath)}")
            self.run_button.config(state=tk.NORMAL)
            self.accuracy_button.config(state=tk.DISABLED)
            # Clear previous results
            self.predicted_sqi, self.predicted_snr = None, None
            for i in self.results_tree.get_children(): self.results_tree.delete(i)


    def _start_processing(self):
        if self.processing_in_progress: return
        if not self.selected_ppg_file.get():
            messagebox.showerror("Input Error", "Please select a PPG file first.")
            return

        self.processing_in_progress = True
        self.run_button.config(state=tk.DISABLED)
        self.accuracy_button.config(state=tk.DISABLED)
        for i in self.results_tree.get_children(): self.results_tree.delete(i)
        
        self._log("\nStarting prediction thread...")
        threading.Thread(target=self._process_and_predict, daemon=True).start()

    def _process_and_predict(self):
        try:
            ppg_file = self.selected_ppg_file.get()
            self._log("Loading PPG signals...")
            df = pd.read_csv(ppg_file)
            raw_signals = [df[f'ppg_finger{i+1}'].to_numpy() for i in range(3)]
            
            all_features_unscaled, all_segments_for_fusion = [], []
            for i, sig in enumerate(raw_signals):
                segments = preprocessing_mendeley.full_preprocess_pipeline(sig, use_mendeley_fs=False, custom_fs=100)
                all_segments_for_fusion.append(segments)
                features = [feature_extraction_mendeley.extract_all_features_from_segment(s, config_mendeley.TARGET_FS) for s in segments]
                ordered_features = [[f.get(feat, np.nan) for feat in self.feature_order] for f in features]
                all_features_unscaled.append(ordered_features)
            self._log("Preprocessing and feature extraction complete.")

            self._log("Scaling features...")
            all_features_scaled = []
            for finger_features in all_features_unscaled:
                features_df = pd.DataFrame(finger_features, columns=self.feature_order)
                scaled_array = self.scaler.transform(features_df.fillna(0))
                all_features_scaled.append(scaled_array)

            self._log("Making predictions for each individual finger...")
            self.predicted_fingers = []
            for i, finger_features_scaled in enumerate(all_features_scaled):
                if finger_features_scaled.shape[0] > 0:
                    pred = np.mean(self.model.predict(finger_features_scaled))
                    self.predicted_fingers.append(pred)
                    self._log(f"-> Finger {i+1} Predicted: {pred:.2f}")
                else:
                    self.predicted_fingers.append(np.nan)
            
            self._log("Evaluating fusion methods and making predictions...")
            fused_features_sqi = fuse_features_sqi_selected(all_features_scaled, all_segments_for_fusion, config_mendeley.TARGET_FS)
            fused_features_snr = fuse_features_snr_weighted(all_features_scaled, all_segments_for_fusion, config_mendeley.TARGET_FS)

            # Store predictions to be used later for accuracy calculation
            self.predicted_sqi = np.mean(self.model.predict(np.array(fused_features_sqi))) if fused_features_sqi else np.nan
            self.predicted_snr = np.mean(self.model.predict(np.array(fused_features_snr))) if fused_features_snr else np.nan
            
            self._log("\nPrediction Complete!")
            self.root.after(0, self._update_prediction_results)

        except Exception as e:
            import traceback
            self._log(f"ERROR: {e}\n{traceback.format_exc()}")
            self.root.after(0, lambda: messagebox.showerror("Processing Error", f"An error occurred: {e}"))
        finally:
            self.processing_in_progress = False
            self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.accuracy_button.config(state=tk.NORMAL))

    def _update_prediction_results(self):
        """Displays the predicted values for each approach in the results table."""
        for i in self.results_tree.get_children(): self.results_tree.delete(i)
        
        self.results_tree.config(height=5)

        def format_val(val, precision=2):
            return f"{val:.{precision}f}" if pd.notna(val) else "N/A"
        
        # A single, clear dictionary for all display names
        method_display_names = {
            'F1': "Index Finger",
            'F2': "Middle Finger",
            'F3': "Ring Finger",
            'SQI': "SQI-Selected Fusion",
            'SNR': "SNR-Weighted Fusion"
        }

        # --- This single block now correctly adds all rows ---

        # Add the finger results
        if self.predicted_fingers:
            for i, key in enumerate(['F1', 'F2', 'F3']):
                pred = self.predicted_fingers[i]
                values = [method_display_names[key], format_val(pred), "N/A", "N/A", "N/A"]
                self.tree_item_ids[key] = self.results_tree.insert("", "end", values=values)
        
        # Add the fusion results
        sqi_values = [method_display_names['SQI'], format_val(self.predicted_sqi), "N/A", "N/A", "N/A"]
        self.tree_item_ids['SQI'] = self.results_tree.insert("", "end", values=sqi_values)
        
        snr_values = [method_display_names['SNR'], format_val(self.predicted_snr), "N/A", "N/A", "N/A"]
        self.tree_item_ids['SNR'] = self.results_tree.insert("", "end", values=snr_values)

        self._log("-> Prediction results displayed.")
        self._log("-> You can now enter the actual glucose value and click 'Calculate Accuracy'.")
        
    def _calculate_and_display_accuracy(self):
        """
        Calculates accuracy for all methods, finds the best performer,
        and updates the table, highlighting only the row with the highest accuracy.
        """
        if not self.predicted_fingers:
            messagebox.showwarning("Warning", "Please run a prediction first before calculating accuracy.")
            return
            
        try:
            actual_glucose = float(self.actual_glucose.get())
        except (ValueError, TypeError):
            messagebox.showerror("Input Error", "Please enter a valid number for actual glucose.")
            return

        self._log(f"\nCalculating accuracy against actual glucose: {actual_glucose} mg/dL")

        def format_val(val, precision=2):
            return f"{val:.{precision}f}" if pd.notna(val) else "N/A"
        
        def calculate_metrics(pred_val, actual_val):
            if pd.isna(pred_val) or pd.isna(actual_val) or actual_val == 0:
                return np.nan, np.nan
            mard = abs(pred_val - actual_val) / actual_val * 100
            acc = max(0, 100 - mard)
            return mard, acc

        # 1. Store all predictions and their calculated metrics in one place
        all_results = {}
        predictions = {
            'F1': self.predicted_fingers[0], 'F2': self.predicted_fingers[1], 'F3': self.predicted_fingers[2],
            'SQI': self.predicted_sqi, 'SNR': self.predicted_snr
        }

        for method, pred_val in predictions.items():
            mard, acc = calculate_metrics(pred_val, actual_glucose)
            all_results[method] = {'pred': pred_val, 'mard': mard, 'acc': acc}

        # 2. Find the best accuracy among all valid results
        best_accuracy = -1.0
        best_methods = [] # Use a list to handle potential ties for the best accuracy
        
        for method, metrics in all_results.items():
            acc = metrics.get('acc')
            if pd.notna(acc):
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_methods = [method]  # New best, so start a new list
                elif acc == best_accuracy:
                    best_methods.append(method) # It's a tie, add to the list of best methods

        # 3. Update the Treeview, applying the tag ONLY to the best method(s)
        method_display_names = {
            'F1': "Index Finger", 'F2': "Middle Finger", 'F3': "Ring Finger",
            'SQI': "SQI-Selected Fusion", 'SNR': "SNR-Weighted Fusion"
        }

        for method_key, display_name in method_display_names.items():
            metrics = all_results.get(method_key, {})
            
            # Determine the tag: 'good_accuracy' if it's a winner, otherwise empty
            tag_to_apply = ('good_accuracy',) if method_key in best_methods else ()

            values = [
                display_name,
                format_val(metrics.get('pred')),
                format_val(actual_glucose, 1),
                format_val(metrics.get('mard')),
                format_val(metrics.get('acc'))
            ]
            
            # Update the item in the tree, setting both values and the determined tag
            self.results_tree.item(self.tree_item_ids[method_key], values=values, tags=tag_to_apply)

        self._log("-> Accuracy metrics updated. Best performing method highlighted.")

    def _center_window(self):
        self.root.update_idletasks()
        self.root.deiconify()
        width, height = self.root.winfo_width(), self.root.winfo_height()
        x, y = (self.root.winfo_screenwidth() - width) // 2, (self.root.winfo_screenheight() - height) // 2
        self.root.geometry(f'{width}x{height}+{x}+{y}')

if __name__ == '__main__':
    if not SCRIPTS_LOADED:
        print("\nCould not start the app because the required processing scripts were not found.")
        # As a fallback for the GUI to still open and show the error
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Startup Error", "Could not load processing modules. Please check the script paths. See console for details.")
    else:
        themed_root = ThemedTk(theme="arc")
        app = LiveDemoApp(themed_root)
        themed_root.mainloop()