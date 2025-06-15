import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ttkthemes import ThemedTk
import pandas as pd
import numpy as np
import os
import sys
import json
import threading
from datetime import datetime
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Dynamic Path and Module Setup ---
# Ensures the script can find and import your custom modules from the Mendeley project.
SCRIPTS_LOADED_SUCCESSFULLY = False
config_mendeley, preprocessing_mendeley, feature_extraction_mendeley, model_trainer_mendeley = None, None, None, None

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    mendeley_src_path = os.path.join(project_root, "02_Machine_Learning_Mendeley", "src")
    if mendeley_src_path not in sys.path:
        sys.path.insert(0, mendeley_src_path)

    import config as config_mendeley
    import preprocessing as preprocessing_mendeley
    import feature_extraction_new as feature_extraction_mendeley
    import model_trainer as model_trainer_mendeley
    
    print(f"Successfully imported processing modules from: {mendeley_src_path}")
    SCRIPTS_LOADED_SUCCESSFULLY = True
except ImportError as e:
    print(f"ERROR: Could not import required modules. Details: {e}")
except Exception as e_path:
    print(f"Error setting up path for Mendeley scripts: {e_path}")


# --- Integrated Fusion Functions ---
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
    except Exception: return 0.01

def calculate_sqi_for_segment(segment_array, fs):
    if segment_array is None or len(segment_array) < fs: return 0.0
    try:
        from scipy.signal import find_peaks
        std_dev = np.std(segment_array)
        if std_dev < 1e-9: return 0.0
        peak_to_peak_amplitude = np.ptp(segment_array)
        if peak_to_peak_amplitude < 1e-6: return 0.0
        peaks, _ = find_peaks(segment_array, distance=int(fs * 0.3))
        if len(peaks) < 2: ppi_std_norm = 1.0
        else:
            ppi_values = np.diff(peaks) / fs
            ppi_mean = np.mean(ppi_values)
            ppi_std = np.std(ppi_values)
            ppi_std_norm = ppi_std / ppi_mean if ppi_mean > 1e-9 else 1.0
        sqi = peak_to_peak_amplitude / (std_dev * (1 + ppi_std_norm))
        return max(0.0, sqi)
    except Exception: return 0.0

def fuse_features_snr_weighted(features, segments, fs):
    fused_list = []
    # Check if there are valid features to process
    if not features or not segments or not segments[0]: return []
    num_segments = len(segments[0])
    
    for i in range(num_segments):
        snrs = [calculate_snr_for_segment(segs[i], fs) if i < len(segs) else 0 for segs in segments]
        total_snr = sum(snrs)
        weights = [snr / total_snr if total_snr > 1e-6 else 1/3 for snr in snrs]
        
        fused_features = np.zeros_like(features[0][i])
        for finger_idx in range(len(features)):
            if len(features[finger_idx]) > i:
                fused_features += weights[finger_idx] * np.array(features[finger_idx][i])
        fused_list.append(fused_features.tolist())
    return fused_list

def fuse_features_sqi_selected(features, segments, fs):
    fused_list = []
    # Check if there are valid features to process
    if not features or not segments or not segments[0]: return []
    num_segments = len(segments[0])

    for i in range(num_segments):
        sqis = [calculate_sqi_for_segment(segs[i], fs) if i < len(segs) else 0 for segs in segments]
        best_finger_idx = np.argmax(sqis)
        if len(features[best_finger_idx]) > i:
            fused_list.append(features[best_finger_idx][i])
    return fused_list


class BatchEvaluatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch Data Evaluator (for Pre-trained Model)")
        self.root.withdraw() # Hide window to prevent flash
        self.root.geometry("650x600")

        self.project_root = project_root
        self.mendeley_model_dir = os.path.join(self.project_root, "02_Machine_Learning_Mendeley", "src", "models")
        self.output_dir = os.path.join(self.project_root, "04_Collected_Data_Analysis", "evaluation_results")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.data_folder_path = tk.StringVar()
        self.processing_in_progress = False
        self.model, self.scaler, self.expected_feature_order = None, None, []

        self._setup_gui()
        self._load_model_and_scaler()
        self._center_window() # Center the window
        self.root.deiconify() # Show the centered window

    def _center_window(self):
        """Centers the main window on the user's screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)

        file_frame = ttk.LabelFrame(main_frame, text="Select Root Folder of Custom Data", padding="10")
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        ttk.Button(file_frame, text="Browse...", command=self._select_folder).grid(row=0, column=0, padx=5)
        self.selected_folder_label = ttk.Label(file_frame, text="No folder selected. (Select the 'Collected_Data' folder)", relief=tk.SUNKEN)
        self.selected_folder_label.grid(row=0, column=1, sticky="ew", padx=5)

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky="ew", pady=10)
        control_frame.columnconfigure(0, weight=1)
        
        self.process_button = ttk.Button(control_frame, text="Process All Samples & Generate Report", command=self._start_processing_thread, state=tk.DISABLED)
        self.process_button.pack(pady=5, ipady=5)

        results_frame = ttk.LabelFrame(main_frame, text="Averaged Evaluation Results (Across All Samples)", padding="10")
        results_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results_cols = ["Approach", "Avg. mARD (%)", "Avg. RMSE (mg/dL)", "Avg. MAE (mg/dL)"]
        self.results_tree = ttk.Treeview(results_frame, columns=self.results_cols, show="headings", height=5)
        
        col_widths = {"Approach": 200, "Avg. mARD (%)": 120, "Avg. RMSE (mg/dL)": 120, "Avg. MAE (mg/dL)": 120}
        
        for col in self.results_cols:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, anchor='center', width=col_widths.get(col, 120), minwidth=100)
            
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        tree_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        tree_scrollbar.grid(row=0, column=1, sticky='ns')
        self.results_tree.configure(yscrollcommand=tree_scrollbar.set)

        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=3, column=0, sticky="ew", pady=(10,0))
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1)
        self.log_text.grid(row=0, column=0, sticky="ew")
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        main_frame.rowconfigure(2, weight=1)

    def _log_message(self, message):
        self.root.after(0, lambda: self._update_log_text(message))

    def _update_log_text(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _load_model_and_scaler(self):
        if not SCRIPTS_LOADED_SUCCESSFULLY:
            self._log_message("ERROR: Cannot load model. Core processing scripts failed to import.")
            return
        try:
            self._log_message("Loading model and scaler trained from scratch...")

            # Directory where the new model/scaler are saved
            artifacts_dir = os.path.join(self.project_root, "04_Collected_Data_Analysis", "evaluation_results")

            # Names of the new model and scaler files
            model_name = "lgbm_model_from_scratch.txt"
            scaler_name = "scaler_from_scratch.pkl"
            
            # The feature order file is still in the original Mendeley directory
            features_name = "model_features_retrained_v1.json"

            # Load the new model from the correct directory
            model_path = os.path.join(artifacts_dir, model_name)
            self.model = lgb.Booster(model_file=model_path)
            
            # Load the new scaler from the correct directory
            scaler_path = os.path.join(artifacts_dir, scaler_name)
            with open(scaler_path, 'rb') as f:
                self.scaler = joblib.load(f)

            # Load the feature order (this path is unchanged)
            features_path = os.path.join(self.mendeley_model_dir, features_name)
            with open(features_path, 'r') as f:
                self.expected_feature_order = json.load(f)

            if self.model and self.scaler:
                self._log_message("Successfully loaded model and scaler trained from scratch.")
            else:
                raise FileNotFoundError("Could not find the model or scaler from scratch.")
        except Exception as e:
            import traceback
            error_msg = f"CRITICAL ERROR loading artifacts: {e}\n{traceback.format_exc()}"
            self._log_message(error_msg)
            messagebox.showerror("Model Load Error", f"Failed to load artifacts.\n\nError: {e}")

    def _select_folder(self):
        initial_dir = os.path.join(self.project_root, "04_Collected_Data_Analysis")
        folder_path = filedialog.askdirectory(title="Select the 'Collected_Data' Folder", initialdir=initial_dir)
        if folder_path and os.path.basename(folder_path) == "Collected_Data":
            self.data_folder_path.set(folder_path)
            self.selected_folder_label.config(text=folder_path)
            self._log_message(f"Selected data folder: {folder_path}")
            self.process_button.config(state=tk.NORMAL)
        elif folder_path:
             messagebox.showwarning("Incorrect Folder", "Please select the folder named 'Collected_Data'.")

    def _start_processing_thread(self):
        if self.processing_in_progress or not self.model: return
        self.processing_in_progress = True
        self.process_button.config(state=tk.DISABLED)
        for i in self.results_tree.get_children(): self.results_tree.delete(i)
        self._log_message("--- Starting Batch Evaluation Pipeline ---")
        threading.Thread(target=self._run_batch_evaluation, daemon=True).start()

    def _run_batch_evaluation(self):
        try:
            folder_path = self.data_folder_path.get()
            labels_path = os.path.join(folder_path, "Labels", "collected_labels.csv")
            raw_data_path = os.path.join(folder_path, "RawData")

            labels_df = pd.read_csv(labels_path)
            self._log_message(f"Found {len(labels_df)} participant entries to process.")

            detailed_results = []

            for _, row in labels_df.iterrows():
                sample_id = f"{row['ID']}_{row['Sample_Num']}"
                ppg_filepath = os.path.join(raw_data_path, f"{sample_id}_ppg.csv")
                if not os.path.exists(ppg_filepath):
                    self._log_message(f"Skipping {sample_id}: Raw PPG file not found.")
                    continue

                self._log_message(f"Processing sample: {sample_id}...")
                actual_glucose = row['Glucose_mgdL']

                ppg_df = pd.read_csv(ppg_filepath)
                signals = [pd.to_numeric(ppg_df[f'ppg_finger{i+1}'], errors='coerce').dropna().to_numpy() for i in range(3)]

                all_features_scaled = []
                all_segments = []
                for sig in signals:
                    segments = preprocessing_mendeley.full_preprocess_pipeline(sig, use_mendeley_fs=False, custom_fs=100)
                    all_segments.append(segments)
                    if not segments:
                        all_features_scaled.append([])
                        continue
                    features = [feature_extraction_mendeley.extract_all_features_from_segment(s, config_mendeley.TARGET_FS) for s in segments]
                    ordered_features = pd.DataFrame(features)[self.expected_feature_order]
                    all_features_scaled.append(self.scaler.transform(ordered_features))

                fused_snr = fuse_features_snr_weighted(all_features_scaled, all_segments, config_mendeley.TARGET_FS)
                fused_sqi = fuse_features_sqi_selected(all_features_scaled, all_segments, config_mendeley.TARGET_FS)

                def get_metrics(feature_vector, actual_glucose_val):
                    if feature_vector is None or feature_vector.size == 0:
                        return [np.nan] * 4
                    
                    self._log_message(f"DEBUG: Input features to model: {feature_vector[:5]}") # Log first 5 features

                    # Reshape for a single prediction and predict
                    prediction = self.model.predict(feature_vector.reshape(1, -1))[0]
                    
                    # Calculate metrics based on the single prediction
                    mard = (np.abs(prediction - actual_glucose_val) / actual_glucose_val) * 100
                    rmse = np.sqrt(mean_squared_error([actual_glucose_val], [prediction]))
                    mae = mean_absolute_error([actual_glucose_val], [prediction])
                    
                    return [prediction, mard, rmse, mae]

                # Define approaches and corresponding feature sets
                approaches = {
                    "Index Finger": all_features_scaled[0],
                    "Middle Finger": all_features_scaled[1],
                    "Ring Finger": all_features_scaled[2],
                    "SNR-Weighted Fusion": fused_snr,
                    "SQI-Selected Fusion": fused_sqi
                }

                for approach, features in approaches.items():
                    # Check if features were successfully generated
                    if features is None or len(features) == 0:
                        pred_glucose, mard, rmse, mae = np.nan, np.nan, np.nan, np.nan
                    else:
                        # Average the features across all segments into a single vector
                        avg_feature_vector = np.mean(np.array(features), axis=0)
                        
                        # Get metrics using the single averaged feature vector
                        pred_glucose, mard, rmse, mae = get_metrics(avg_feature_vector, actual_glucose)

                    detailed_results.append({
                        "SampleID": sample_id, "Approach": approach, "ActualGlucose": actual_glucose,
                        "PredictedGlucose": pred_glucose, "mARD(%)": mard, "RMSE(mg/dL)": rmse, "MAE(mg/dL)": mae
                    })

            self._log_message("\nBatch processing complete. Aggregating results...")
            results_df = pd.DataFrame(detailed_results).dropna()

            output_csv_path = os.path.join(self.output_dir, "detailed_evaluation_log.csv")
            results_df.to_csv(output_csv_path, index=False)
            self._log_message(f"Detailed log with all sample results saved to:\n{output_csv_path}")

            avg_results = results_df.groupby('Approach').agg({
                'mARD(%)': 'mean', 'RMSE(mg/dL)': 'mean', 'MAE(mg/dL)': 'mean'
            }).reset_index()

            display_data = []
            for _, row in avg_results.iterrows():
                display_data.append([row['Approach'], f"{row['mARD(%)']:.2f}", f"{row['RMSE(mg/dL)']:.2f}", f"{row['MAE(mg/dL)']:.2f}"])

            self.root.after(0, self._update_results_display, display_data)
            self._log_message("\n--- Pipeline Finished Successfully ---")

        except Exception as e:
            import traceback
            error_msg = f"ERROR in pipeline: {e}\n{traceback.format_exc()}"
            self._log_message(error_msg)
            messagebox.showerror("Pipeline Error", error_msg)
        finally:
            self.processing_in_progress = False
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))

    def _update_results_display(self, results_list):
        for i in self.results_tree.get_children():
            self.results_tree.delete(i)
        for row in results_list:
            self.results_tree.insert("", tk.END, values=row)

if __name__ == '__main__':
    if not SCRIPTS_LOADED_SUCCESSFULLY:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Initialization Error", "Failed to load critical modules from '02_Machine_Learning_Mendeley'.")
    else:
        root = ThemedTk(theme="arc")
        app = BatchEvaluatorApp(root)
        root.mainloop()
