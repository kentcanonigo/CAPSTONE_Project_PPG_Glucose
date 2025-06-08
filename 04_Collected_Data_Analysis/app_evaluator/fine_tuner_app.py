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
# FIX: Import both GroupShuffleSplit and train_test_split
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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
        if len(peaks) < 2: return peak_to_peak_amplitude / (std_dev * 2.0)
        else:
            ppi_values = np.diff(peaks) / fs
            ppi_mean = np.mean(ppi_values)
            ppi_std = np.std(ppi_values)
            ppi_std_norm = ppi_std / ppi_mean if ppi_mean > 1e-9 else 1.0
            return peak_to_peak_amplitude / (std_dev * (1 + ppi_std_norm))
    except Exception: return 0.0

def fuse_features_snr_weighted(features, segments, fs):
    fused_list = []
    if not all(len(f) == len(segments[0]) for f in features): return []
    num_segments = len(segments[0])
    for i in range(num_segments):
        snrs = [calculate_snr_for_segment(segs[i], fs) for segs in segments]
        total_snr = sum(snrs)
        weights = [snr / total_snr if total_snr > 1e-6 else 1/3 for snr in snrs]
        fused_features = np.zeros_like(features[0][i])
        for finger_idx in range(len(features)):
            fused_features += weights[finger_idx] * np.array(features[finger_idx][i])
        fused_list.append(fused_features.tolist())
    return fused_list

def fuse_features_sqi_selected(features, segments, fs):
    fused_list = []
    if not all(len(f) == len(segments[0]) for f in features): return []
    num_segments = len(segments[0])
    for i in range(num_segments):
        sqis = [calculate_sqi_for_segment(segs[i], fs) for segs in segments]
        best_finger_idx = np.argmax(sqis)
        fused_list.append(features[best_finger_idx][i])
    return fused_list


class FineTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Fine-Tuning & Final Evaluation")
        self.root.geometry("650x600")

        self.project_root = project_root
        self.mendeley_model_dir = os.path.join(self.project_root, "02_Machine_Learning_Mendeley", "src", "models")
        self.custom_data_dir = os.path.join(self.project_root, "04_Collected_Data_Analysis", "Collected_Data")
        self.output_dir = os.path.join(self.project_root, "04_Collected_Data_Analysis", "evaluation_results")
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

        self.processing_in_progress = False
        self.pretrained_model, self.scaler, self.expected_feature_order = None, None, []

        self._setup_gui()
        self._load_pretrained_artifacts()

    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)

        info_frame = ttk.LabelFrame(main_frame, text="Pipeline Configuration", padding="10")
        info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        ttk.Label(info_frame, text="Pre-trained Model:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(info_frame, text="lgbm_glucose_model_retrained_v1.txt", font="TkDefaultFont 9 italic").grid(row=0, column=1, sticky="w")
        ttk.Label(info_frame, text="Custom Data Source:").grid(row=1, column=0, sticky="w", padx=5)
        ttk.Label(info_frame, text=self.custom_data_dir, font="TkDefaultFont 9 italic", wraplength=450).grid(row=1, column=1, sticky="w")
        
        self.process_button = ttk.Button(main_frame, text="Start Fine-Tuning & Evaluation", command=self._start_processing_thread)
        self.process_button.grid(row=1, column=0, pady=10, ipady=5)

        results_frame = ttk.LabelFrame(main_frame, text="Final Fine-Tuned Model Performance (on Hold-Out Test Set)", padding="10")
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

        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=3, column=0, sticky="ew", pady=(10,0))
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1)
        self.log_text.grid(row=0, column=0, sticky="ew")

        main_frame.rowconfigure(2, weight=1)

    def _log_message(self, message):
        self.root.after(0, lambda: self._update_log_text(message))

    def _update_log_text(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _load_pretrained_artifacts(self):
        if not SCRIPTS_LOADED_SUCCESSFULLY:
            self._log_message("ERROR: Core processing scripts failed to import.")
            return
        try:
            model_name = "lgbm_glucose_model_retrained_v1.txt"
            scaler_name = "mendeley_feature_scaler_retrained_v1.pkl"
            features_name = "model_features_retrained_v1.json"
            self.pretrained_model, self.scaler, self.expected_feature_order = model_trainer_mendeley.load_model_scaler_and_features(
                self.mendeley_model_dir, model_name, scaler_name, features_name
            )
            if self.pretrained_model and self.scaler:
                self._log_message("Successfully loaded pre-trained model and scaler.")
            else:
                raise FileNotFoundError("Pre-trained model or scaler not found.")
        except Exception as e:
            self._log_message(f"CRITICAL ERROR loading artifacts: {e}")
            messagebox.showerror("Model Load Error", f"Failed to load artifacts from '{self.mendeley_model_dir}'.\n\nError: {e}")

    def _start_processing_thread(self):
        if self.processing_in_progress or not self.pretrained_model: return
        self.processing_in_progress = True
        self.process_button.config(state=tk.DISABLED)
        for i in self.results_tree.get_children(): self.results_tree.delete(i)
        self._log_message("--- Starting Fine-Tuning Pipeline ---")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        try:
            # --- Phase 1: Load and Process ALL Custom Data ---
            self._log_message("Phase 1: Loading and processing all custom data...")
            labels_df = pd.read_csv(os.path.join(self.custom_data_dir, "Labels", "collected_labels.csv"))
            raw_data_path = os.path.join(self.custom_data_dir, "RawData")
            
            all_participant_data = []
            for _, row in labels_df.iterrows():
                sample_id = f"{row['ID']}_{row['Sample_Num']}"
                ppg_filepath = os.path.join(raw_data_path, f"{sample_id}_ppg.csv")
                if not os.path.exists(ppg_filepath): continue

                ppg_df = pd.read_csv(ppg_filepath)
                signals = [pd.to_numeric(ppg_df[f'ppg_finger{i+1}'], errors='coerce').dropna().to_numpy() for i in range(3)]
                
                segments_per_finger = [preprocessing_mendeley.full_preprocess_pipeline(s, use_mendeley_fs=False, custom_fs=100) for s in signals]
                
                if not all(len(s) == len(segments_per_finger[0]) for s in segments_per_finger) or not segments_per_finger[0]: continue
                
                features_per_finger = []
                for segs in segments_per_finger:
                    features = [feature_extraction_mendeley.extract_all_features_from_segment(s, config_mendeley.TARGET_FS) for s in segs]
                    ordered_df = pd.DataFrame(features)[self.expected_feature_order]
                    features_per_finger.append(self.scaler.transform(ordered_df))

                all_participant_data.append({
                    "id": row['ID'],
                    "actual_glucose": row['Glucose_mgdL'],
                    "features_scaled": features_per_finger,
                    "segments": segments_per_finger
                })
            self._log_message(f"Successfully processed data for {len(all_participant_data)} participants.")

            # --- Phase 2: Split data by participant ---
            self._log_message("Phase 2: Splitting participants into fine-tuning and hold-out test sets...")
            participant_ids = np.unique([p['id'] for p in all_participant_data])
            if len(participant_ids) < 2:
                self._log_message("Not enough participants for a train/test split. Aborting."); return

            train_ids, test_ids = train_test_split(participant_ids, test_size=0.2, random_state=42)
            train_data = [p for p in all_participant_data if p['id'] in train_ids]
            test_data = [p for p in all_participant_data if p['id'] in test_ids]
            self._log_message(f"Data split: {len(train_ids)} participants for tuning, {len(test_ids)} for final testing.")

            # --- Phase 3: Prepare training data (using best fusion method) and fine-tune ---
            self._log_message("Phase 3: Preparing fused data and fine-tuning model...")
            X_train_list, y_train_list = [], []
            for p_data in train_data:
                fused_features = fuse_features_sqi_selected(p_data['features_scaled'], p_data['segments'], config_mendeley.TARGET_FS)
                X_train_list.extend(fused_features)
                y_train_list.extend([p_data['actual_glucose']] * len(fused_features))

            ft_params = self.pretrained_model.params.copy()
            ft_params['learning_rate'] = 0.01
            
            fine_tuned_model = lgb.train(
                ft_params, lgb.Dataset(pd.DataFrame(X_train_list), label=pd.Series(y_train_list)),
                num_boost_round=150, init_model=self.pretrained_model
            )
            self._log_message("Fine-tuning complete.")
            
            # --- Phase 4: Evaluate fine-tuned model on ALL approaches on the test set ---
            self._log_message("Phase 4: Evaluating fine-tuned model on hold-out test set...")
            detailed_log = []
            
            for p_data in test_data:
                sample_id = f"{p_data['id']}_{row['Sample_Num']}" # Note: Sample_num might not be unique here if multiple samples per ID
                actual_glucose = p_data['actual_glucose']
                
                features_sets = {
                    "Index Finger": p_data['features_scaled'][0],
                    "Middle Finger": p_data['features_scaled'][1],
                    "Ring Finger": p_data['features_scaled'][2],
                    "SNR-Weighted Fusion": fuse_features_snr_weighted(p_data['features_scaled'], p_data['segments'], config_mendeley.TARGET_FS),
                    "SQI-Selected Fusion": fuse_features_sqi_selected(p_data['features_scaled'], p_data['segments'], config_mendeley.TARGET_FS)
                }

                for approach, features in features_sets.items():
                    if len(features) == 0: continue
                    preds = fine_tuned_model.predict(features)
                    detailed_log.append({
                        "ParticipantID": p_data['id'],
                        "Approach": approach,
                        "ActualGlucose": actual_glucose,
                        "PredictedGlucose": np.mean(preds),
                        "mARD(%)": np.mean(np.abs(preds - actual_glucose) / actual_glucose) * 100,
                        "RMSE(mg/dL)": np.sqrt(mean_squared_error([actual_glucose] * len(preds), preds)),
                        "MAE(mg/dL)": mean_absolute_error([actual_glucose] * len(preds), preds)
                    })
            
            # --- Phase 5: Aggregate results and update GUI ---
            self._log_message("Phase 5: Aggregating final results...")
            log_df = pd.DataFrame(detailed_log)
            
            output_csv_path = os.path.join(self.output_dir, "fine_tuned_model_evaluation_log.csv")
            log_df.to_csv(output_csv_path, index=False)
            self._log_message(f"Detailed log saved to:\n{output_csv_path}")

            avg_results = log_df.groupby('Approach').agg({
                'mARD(%)': 'mean',
                'RMSE(mg/dL)': 'mean',
                'MAE(mg/dL)': 'mean'
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
        for i in self.results_tree.get_children(): self.results_tree.delete(i)
        for row in sorted(results_list, key=lambda x: x[0]): # Sort alphabetically for consistent order
            self.results_tree.insert("", tk.END, values=row)

if __name__ == '__main__':
    if not SCRIPTS_LOADED_SUCCESSFULLY:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Initialization Error", "Failed to load critical modules.")
    else:
        root = ThemedTk(theme="arc")
        app = FineTunerApp(root)
        root.mainloop()
