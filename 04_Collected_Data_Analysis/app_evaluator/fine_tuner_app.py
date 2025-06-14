import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import pandas as pd
import numpy as np
import os
import sys
import threading
from datetime import datetime
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Dynamic Path and Module Setup ---
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


# --- Configuration ---
class AppConfig:
    HOLDOUT_TEST_SIZE = 0.2
    FT_VALIDATION_SIZE = 0.25
    RANDOM_STATE = 43
    FT_PARAMS = {'learning_rate': 0.01, 'n_jobs': -1, 'seed': 42, 'verbose': -1}
    FT_NUM_BOOST_ROUND = 200
    FT_EARLY_STOPPING_ROUNDS = 25
    INPUT_FS = 100
    TARGET_FS_FEATURES = config_mendeley.TARGET_FS if SCRIPTS_LOADED_SUCCESSFULLY else 50

# --- Helper Functions ---
def calculate_mard(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask): return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100

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
            ppi_consistency_metric = 1 + (ppi_std / ppi_mean if ppi_mean > 1e-9 else 1.0)
            return peak_to_peak_amplitude / (std_dev * ppi_consistency_metric)
    except Exception as e:
        print(f"Warning: Could not calculate SQI. Returning 0. Error: {e}")
        return 0.0

def calculate_snr_for_segment(segment_array, fs):
    if segment_array is None or len(segment_array) < 2: return 0.01
    try:
        signal_power = np.var(segment_array)
        snr = signal_power * (1.0 / (1.0 + np.std(np.diff(np.diff(segment_array))))) if len(segment_array) > 5 else signal_power
        return max(0.01, snr)
    except Exception: return 0.01

def fuse_features_sqi_selected(features_per_finger, segments_per_finger, fs):
    num_segments = len(segments_per_finger[0])
    if num_segments == 0: return []
    fused_list = []
    for i in range(num_segments):
        sqis = [calculate_sqi_for_segment(segs[i], fs) for segs in segments_per_finger]
        best_finger_idx = np.argmax(sqis)
        fused_list.append(features_per_finger[best_finger_idx][i])
    return fused_list

def fuse_features_snr_weighted(features_per_finger, segments_per_finger, fs):
    num_segments = len(segments_per_finger[0])
    if num_segments == 0: return []
    fused_list = []
    for i in range(num_segments):
        snrs = [calculate_snr_for_segment(segs[i], fs) for segs in segments_per_finger]
        total_snr = sum(snrs)
        weights = [snr / total_snr if total_snr > 1e-6 else 1/3 for snr in snrs]
        fused_features = np.zeros_like(features_per_finger[0][i])
        for finger_idx in range(len(features_per_finger)):
            fused_features += weights[finger_idx] * np.array(features_per_finger[finger_idx][i])
        fused_list.append(fused_features.tolist())
    return fused_list


class FineTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Fine-Tuning & Final Evaluation (Group-Split)")
        self.root.withdraw()
        self.root.geometry("750x650")

        self.project_root = project_root
        self.mendeley_model_dir = os.path.join(self.project_root, "02_Machine_Learning_Mendeley", "src", "models")
        self.custom_data_dir = os.path.join(self.project_root, "04_Collected_Data_Analysis", "Collected_Data")
        self.output_dir = os.path.join(self.project_root, "04_Collected_Data_Analysis", "evaluation_results")
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

        self.processing_in_progress = False
        self.pretrained_model, self.scaler, self.expected_feature_order = None, None, []

        self._setup_gui()
        self._load_pretrained_artifacts()
        self._center_window()
        self.root.deiconify()

    def _center_window(self):
        self.root.update_idletasks()
        width, height = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

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
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky="ew", pady=10)
        control_frame.columnconfigure(0, weight=1)
        self.process_button = ttk.Button(control_frame, text="Start Fine-Tuning & Evaluation", command=self._start_processing_thread)
        self.process_button.pack(pady=5, ipady=5)
        results_frame = ttk.LabelFrame(main_frame, text="Final Model Performance (on Hold-Out Test Set)", padding="10")
        results_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        results_frame.columnconfigure(0, weight=1); results_frame.rowconfigure(0, weight=1)
        
        self.results_cols = ["Approach", "Avg. mARD (%)", "Avg. RMSE (mg/dL)", "Avg. MAE (mg/dL)"]
        self.results_tree = ttk.Treeview(results_frame, columns=self.results_cols, show="headings", height=6)
        
        col_widths = {"Approach": 200, "Avg. mARD (%)": 130, "Avg. RMSE (mg/dL)": 140, "Avg. MAE (mg/dL)": 130}
        for col in self.results_cols:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, anchor='center', width=col_widths.get(col, 120), minwidth=100)
            
        self.results_tree.grid(row=0, column=0, sticky="nsew")

        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=3, column=0, sticky="ew", pady=(10,0))
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1)
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
        if not SCRIPTS_LOADED_SUCCESSFULLY: self._log_message("ERROR: Core processing scripts failed to import."); return
        try:
            model_name, scaler_name, features_name = "lgbm_glucose_model_retrained_v1.txt", "mendeley_feature_scaler_retrained_v1.pkl", "model_features_retrained_v1.json"
            self.pretrained_model, self.scaler, self.expected_feature_order = model_trainer_mendeley.load_model_scaler_and_features(
                self.mendeley_model_dir, model_name, scaler_name, features_name)
            if self.pretrained_model and self.scaler: self._log_message("Successfully loaded pre-trained model and scaler.")
            else: raise FileNotFoundError("Pre-trained model or scaler not found.")
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
            self._log_message("Phase 1: Loading and processing all custom data...")
            labels_df = pd.read_csv(os.path.join(self.custom_data_dir, "Labels", "collected_labels.csv"))
            raw_data_path = os.path.join(self.custom_data_dir, "RawData")
            all_feature_rows = []
            for _, row in labels_df.iterrows():
                ppg_filepath = os.path.join(raw_data_path, f"{row['ID']}_{row['Sample_Num']}_ppg.csv")
                if not os.path.exists(ppg_filepath): continue
                ppg_df = pd.read_csv(ppg_filepath)
                signals = [pd.to_numeric(ppg_df[f'ppg_finger{i+1}'], errors='coerce').dropna().to_numpy() for i in range(3)]
                segments_per_finger = [preprocessing_mendeley.full_preprocess_pipeline(s, use_mendeley_fs=False, custom_fs=AppConfig.INPUT_FS) for s in signals]
                if not any(segments_per_finger) or not all(len(s) == len(segments_per_finger[0]) for s in segments_per_finger if s): continue
                num_segments = len(segments_per_finger[0])
                if num_segments == 0: continue
                features_f1 = [feature_extraction_mendeley.extract_all_features_from_segment(s, AppConfig.TARGET_FS_FEATURES) for s in segments_per_finger[0]]
                features_f2 = [feature_extraction_mendeley.extract_all_features_from_segment(s, AppConfig.TARGET_FS_FEATURES) for s in segments_per_finger[1]]
                features_f3 = [feature_extraction_mendeley.extract_all_features_from_segment(s, AppConfig.TARGET_FS_FEATURES) for s in segments_per_finger[2]]
                for i in range(num_segments):
                    all_feature_rows.append({
                        "subject_id": row['ID'], "glucose": row['Glucose_mgdL'],
                        "features_f1": [features_f1[i].get(feat, np.nan) for feat in self.expected_feature_order],
                        "features_f2": [features_f2[i].get(feat, np.nan) for feat in self.expected_feature_order],
                        "features_f3": [features_f3[i].get(feat, np.nan) for feat in self.expected_feature_order],
                        "segment_f1": segments_per_finger[0][i], "segment_f2": segments_per_finger[1][i], "segment_f3": segments_per_finger[2][i]
                    })
            if not all_feature_rows: self._log_message("No data processed, aborting."); return
            master_df = pd.DataFrame(all_feature_rows)
            self._log_message(f"Successfully processed {len(master_df)} total segments from {master_df['subject_id'].nunique()} participants.")

            self._log_message("\nPhase 2: Splitting participants into Fine-Tuning and Hold-Out sets...")
            holdout_splitter = GroupShuffleSplit(n_splits=1, test_size=AppConfig.HOLDOUT_TEST_SIZE, random_state=AppConfig.RANDOM_STATE)
            ft_pool_indices, holdout_indices = next(holdout_splitter.split(master_df, groups=master_df['subject_id']))
            ft_pool_df, holdout_df = master_df.iloc[ft_pool_indices], master_df.iloc[holdout_indices]
            self._log_message(f"  - Hold-Out Subjects ({holdout_df['subject_id'].nunique()}): {sorted(holdout_df['subject_id'].unique())}")
            ft_splitter = GroupShuffleSplit(n_splits=1, test_size=AppConfig.FT_VALIDATION_SIZE, random_state=AppConfig.RANDOM_STATE)
            train_indices, valid_indices = next(ft_splitter.split(ft_pool_df, groups=ft_pool_df['subject_id']))
            train_df, valid_df = ft_pool_df.iloc[train_indices], ft_pool_df.iloc[valid_indices]
            self._log_message(f"  - Fine-Tuning Train Subjects ({train_df['subject_id'].nunique()}): {sorted(train_df['subject_id'].unique())}")
            self._log_message(f"  - Fine-Tuning Valid Subjects ({valid_df['subject_id'].nunique()}): {sorted(valid_df['subject_id'].unique())}")

            self._log_message("\nPhase 3: Preparing combined data from all fingers for robust fine-tuning...")
            X_train_f1 = pd.DataFrame(train_df['features_f1'].tolist(), columns=self.expected_feature_order)
            X_train_f2 = pd.DataFrame(train_df['features_f2'].tolist(), columns=self.expected_feature_order)
            X_train_f3 = pd.DataFrame(train_df['features_f3'].tolist(), columns=self.expected_feature_order)
            X_train = pd.concat([X_train_f1, X_train_f2, X_train_f3], ignore_index=True).fillna(0)
            y_train = pd.concat([train_df['glucose']] * 3, ignore_index=True)
            X_valid_f1 = pd.DataFrame(valid_df['features_f1'].tolist(), columns=self.expected_feature_order)
            X_valid_f2 = pd.DataFrame(valid_df['features_f2'].tolist(), columns=self.expected_feature_order)
            X_valid_f3 = pd.DataFrame(valid_df['features_f3'].tolist(), columns=self.expected_feature_order)
            X_valid = pd.concat([X_valid_f1, X_valid_f2, X_valid_f3], ignore_index=True).fillna(0)
            y_valid = pd.concat([valid_df['glucose']] * 3, ignore_index=True)
            self._log_message(f"  - Total training feature sets (3 fingers combined): {len(X_train)}")
            
            lgb_train = lgb.Dataset(self.scaler.transform(X_train), label=y_train)
            lgb_valid = lgb.Dataset(self.scaler.transform(X_valid), label=y_valid, reference=lgb_train)
            fine_tuned_model = lgb.train(AppConfig.FT_PARAMS, lgb_train, valid_sets=[lgb_valid], valid_names=['validation'],
                                         num_boost_round=AppConfig.FT_NUM_BOOST_ROUND,
                                         callbacks=[lgb.early_stopping(AppConfig.FT_EARLY_STOPPING_ROUNDS, verbose=True), lgb.log_evaluation(50)],
                                         init_model=self.pretrained_model)
            self._log_message("Fine-tuning complete.")
            
            self._log_message("\nPhase 4: Evaluating fine-tuned model on hold-out test set...")
            final_results = []
            approaches_to_test = ["Index Finger", "Middle Finger", "Ring Finger", "SNR-Weighted Fusion", "SQI-Selected Fusion"]
            for approach in approaches_to_test:
                y_true_agg, y_pred_agg = [], []
                for subject_id in holdout_df['subject_id'].unique():
                    subject_data = holdout_df[holdout_df['subject_id'] == subject_id]
                    y_true, features = subject_data['glucose'].iloc[0], None
                    features_all_fingers = [subject_data[f'features_f{i+1}'].tolist() for i in range(3)]
                    segments_all_fingers = [subject_data[f'segment_f{i+1}'].tolist() for i in range(3)]
                    if approach == "SQI-Selected Fusion":
                        features = fuse_features_sqi_selected(features_all_fingers, segments_all_fingers, AppConfig.TARGET_FS_FEATURES)
                    elif approach == "SNR-Weighted Fusion":
                        features = fuse_features_snr_weighted(features_all_fingers, segments_all_fingers, AppConfig.TARGET_FS_FEATURES)
                    else:
                        finger_map = {"Index": "1", "Middle": "2", "Ring": "3"}
                        finger_name = approach.split(' ')[0]
                        f_key = f"features_f{finger_map[finger_name]}"
                        features = subject_data[f_key].tolist()
                    if not features: continue
                    X_test = self.scaler.transform(pd.DataFrame(features, columns=self.expected_feature_order).fillna(0))
                    preds = fine_tuned_model.predict(X_test, num_iteration=fine_tuned_model.best_iteration)
                    y_true_agg.append(y_true); y_pred_agg.append(np.mean(preds))
                if not y_true_agg: continue
                mard, rmse, mae = calculate_mard(y_true_agg, y_pred_agg), np.sqrt(mean_squared_error(y_true_agg, y_pred_agg)), mean_absolute_error(y_true_agg, y_pred_agg)
                final_results.append([approach, f"{mard:.2f}", f"{rmse:.2f}", f"{mae:.2f}"])
            
            self._log_message("\n--- Final Results on Unseen Data ---")
            display_order = {name: i for i, name in enumerate(approaches_to_test)}
            sorted_results = sorted(final_results, key=lambda r: display_order.get(r[0], 99))
            
            # --- FIX: SAVE THE FINAL AGGREGATED RESULTS TO A CSV ---
            results_df = pd.DataFrame(sorted_results, columns=self.results_cols)
            results_log_path = os.path.join(self.output_dir, "fine_tuned_aggregated_results.csv")
            results_df.to_csv(results_log_path, index=False)
            self._log_message(f"Final aggregated results saved to: {results_log_path}")
            # --- END OF FIX ---

            self.root.after(0, self._update_results_display, sorted_results)
            
            self._log_message("Saving the fine-tuned model...")
            model_filename = f"lgbm_model_finetuned_on_{master_df['subject_id'].nunique()}subjects_ALL_FINGERS.txt"
            model_save_path = os.path.join(self.output_dir, model_filename)
            fine_tuned_model.save_model(model_save_path)
            self._log_message(f"Model saved to: {model_save_path}")

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
        for row in results_list:
            self.results_tree.insert("", tk.END, values=row)

if __name__ == '__main__':
    if not SCRIPTS_LOADED_SUCCESSFULLY:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Initialization Error", "Failed to load critical modules from the Mendeley project directory.")
    else:
        root = ThemedTk(theme="arc")
        app = FineTunerApp(root)
        root.mainloop()