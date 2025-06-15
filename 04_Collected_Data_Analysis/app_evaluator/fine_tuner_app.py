import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import pandas as pd
import numpy as np
import os
import sys
import threading
from datetime import datetime
import json
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler

# --- Dynamic Path and Module Setup ---
SCRIPTS_LOADED_SUCCESSFULLY = False
config_mendeley, preprocessing_mendeley, feature_extraction_mendeley = None, None, None

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    mendeley_src_path = os.path.join(project_root, "02_Machine_Learning_Mendeley", "src")
    if mendeley_src_path not in sys.path:
        sys.path.insert(0, mendeley_src_path)

    import config as config_mendeley
    import preprocessing as preprocessing_mendeley
    import feature_extraction_new as feature_extraction_mendeley
    
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
    TRAIN_PARAMS = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
        'learning_rate': 0.02, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
        'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    EARLY_STOPPING_ROUNDS = 50
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
    except Exception as e: return 0.0

def calculate_snr_for_segment(segment_array, fs):
    if segment_array is None or len(segment_array) < 2: return 0.01
    try:
        signal_power = np.var(segment_array)
        snr = signal_power * (1.0 / (1.0 + np.std(np.diff(np.diff(segment_array))))) if len(segment_array) > 5 else signal_power
        return max(0.01, snr)
    except Exception: return 0.01

def fuse_features_sqi_selected(features_per_finger, segments_per_finger, fs):
    if not segments_per_finger or not segments_per_finger[0]: return []
    num_segments = len(segments_per_finger[0])
    if num_segments == 0: return []
    fused_list = []
    for i in range(num_segments):
        sqis = [calculate_sqi_for_segment(segs[i], fs) if i < len(segs) and segs[i] is not None else 0.0 for segs in segments_per_finger]
        best_finger_idx = np.argmax(sqis)
        if len(features_per_finger[best_finger_idx]) > i:
            fused_list.append(features_per_finger[best_finger_idx][i])
    return fused_list

def fuse_features_snr_weighted(features_per_finger, segments_per_finger, fs):
    if not segments_per_finger or not segments_per_finger[0]: return []
    num_segments = len(segments_per_finger[0])
    if num_segments == 0: return []
    fused_list = []
    for i in range(num_segments):
        snrs = [calculate_snr_for_segment(segs[i], fs) if i < len(segs) and segs[i] is not None else 0.01 for segs in segments_per_finger]
        total_snr = sum(snrs)
        weights = [snr / total_snr if total_snr > 1e-6 else 1/3 for snr in snrs]
        fused_features = np.zeros_like(features_per_finger[0][i])
        for finger_idx in range(len(features_per_finger)):
            if len(features_per_finger[finger_idx]) > i:
                fused_features += weights[finger_idx] * np.array(features_per_finger[finger_idx][i])
        fused_list.append(fused_features.tolist())
    return fused_list


class ModelTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Trainer (From Scratch)")
        self.root.withdraw()
        self.root.geometry("750x650")

        self.project_root = project_root
        self.mendeley_model_dir = os.path.join(self.project_root, "02_Machine_Learning_Mendeley", "src", "models")
        self.custom_data_dir = os.path.join(self.project_root, "04_Collected_Data_Analysis", "Collected_Data")
        self.output_dir = os.path.join(self.project_root, "04_Collected_Data_Analysis", "evaluation_results")
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

        self.processing_in_progress = False
        self.expected_feature_order = []

        self._setup_gui()
        self._load_feature_config()
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
        ttk.Label(info_frame, text="Training Data Source:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(info_frame, text=self.custom_data_dir, font="TkDefaultFont 9 italic", wraplength=450).grid(row=0, column=1, sticky="w")
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky="ew", pady=10)
        control_frame.columnconfigure(0, weight=1)
        self.process_button = ttk.Button(control_frame, text="Start Training & Evaluation", command=self._start_processing_thread)
        self.process_button.pack(pady=5, ipady=5)
        results_frame = ttk.LabelFrame(main_frame, text="Model Performance (on Hold-Out Test Set)", padding="10")
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

    def _load_feature_config(self):
        if not SCRIPTS_LOADED_SUCCESSFULLY: self._log_message("ERROR: Core processing scripts failed to import."); return
        try:
            features_name = "model_features_retrained_v1.json"
            features_path = os.path.join(self.mendeley_model_dir, features_name)
            with open(features_path, 'r') as f:
                self.expected_feature_order = json.load(f)
            self._log_message("Successfully loaded feature configuration.")
        except Exception as e:
            self._log_message(f"CRITICAL ERROR loading feature config: {e}")
            messagebox.showerror("Config Load Error", f"Failed to load feature config from '{self.mendeley_model_dir}'.\n\nError: {e}")

    def _start_processing_thread(self):
        if self.processing_in_progress: return
        self.processing_in_progress = True
        self.process_button.config(state=tk.DISABLED)
        for i in self.results_tree.get_children(): self.results_tree.delete(i)
        self._log_message("--- Starting Training Pipeline From Scratch ---")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        try:
            self._log_message("Phase 1: Loading and processing all custom data...")
            labels_df = pd.read_csv(os.path.join(self.custom_data_dir, "Labels", "collected_labels.csv"))
            raw_data_path = os.path.join(self.custom_data_dir, "RawData")
            all_data_rows = []
            for _, row in labels_df.iterrows():
                ppg_filepath = os.path.join(raw_data_path, f"{row['ID']}_{row['Sample_Num']}_ppg.csv")
                if not os.path.exists(ppg_filepath): continue
                ppg_df = pd.read_csv(ppg_filepath)
                signals = [pd.to_numeric(ppg_df[f'ppg_finger{i+1}'], errors='coerce').dropna().to_numpy() for i in range(3)]
                segments_per_finger = [preprocessing_mendeley.full_preprocess_pipeline(s, use_mendeley_fs=False, custom_fs=AppConfig.INPUT_FS) for s in signals]
                if not any(s for s in segments_per_finger): continue
                all_data_rows.append({ "subject_id": row['ID'], "glucose": row['Glucose_mgdL'], "segments": segments_per_finger })
            
            master_df = pd.DataFrame(all_data_rows)
            self._log_message(f"Successfully processed data from {master_df['subject_id'].nunique()} participants.")

            self._log_message("\nPhase 2: Splitting participants into Training and Hold-Out sets...")
            holdout_splitter = GroupShuffleSplit(n_splits=1, test_size=AppConfig.HOLDOUT_TEST_SIZE, random_state=AppConfig.RANDOM_STATE)
            pool_indices, holdout_indices = next(holdout_splitter.split(master_df, groups=master_df['subject_id']))
            pool_df, holdout_df = master_df.iloc[pool_indices], master_df.iloc[holdout_indices]
            
            self._log_message(f"  - Hold-Out Subjects ({holdout_df['subject_id'].nunique()}): {sorted(holdout_df['subject_id'].unique())}")
            
            ft_splitter = GroupShuffleSplit(n_splits=1, test_size=AppConfig.FT_VALIDATION_SIZE, random_state=AppConfig.RANDOM_STATE)
            train_indices, valid_indices = next(ft_splitter.split(pool_df, groups=pool_df['subject_id']))
            train_df, valid_df = pool_df.iloc[train_indices], pool_df.iloc[valid_indices]
            self._log_message(f"  - Training Subjects ({train_df['subject_id'].nunique()}): {sorted(train_df['subject_id'].unique())}")
            self._log_message(f"  - Validation Subjects ({valid_df['subject_id'].nunique()}): {sorted(valid_df['subject_id'].unique())}")

            self._log_message("\nPhase 3: Preparing SQI-Fused data for training...")
            def extract_and_fuse_data(df):
                feature_list = []
                for _, row in df.iterrows():
                    segments_per_finger = row['segments']
                    features_per_finger = [
                        [feature_extraction_mendeley.extract_all_features_from_segment(seg, AppConfig.TARGET_FS_FEATURES) for seg in finger_segs]
                        if finger_segs else [] for finger_segs in segments_per_finger
                    ]
                    ordered_features = [
                        [[f.get(feat, np.nan) for feat in self.expected_feature_order] for f in finger_features]
                        for finger_features in features_per_finger
                    ]
                    sqi_fused = fuse_features_sqi_selected(ordered_features, segments_per_finger, AppConfig.TARGET_FS_FEATURES)
                    for feature_vector in sqi_fused:
                        feature_list.append(feature_vector + [row['glucose']])
                cols = self.expected_feature_order + ['glucose']
                return pd.DataFrame(feature_list, columns=cols).dropna()

            train_data = extract_and_fuse_data(train_df)
            valid_data = extract_and_fuse_data(valid_df)
            X_train, y_train = train_data[self.expected_feature_order], train_data['glucose']
            X_valid, y_valid = valid_data[self.expected_feature_order], valid_data['glucose']
            self._log_message(f"  - Total training feature sets (SQI-Fused): {len(X_train)}")

            self._log_message("Fitting new feature scaler on custom training data...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_valid_scaled = scaler.transform(X_valid)
            
            self._log_message("Calculating sample weights for imbalance...")
            y_train_bins = pd.cut(y_train, bins=4, labels=False) 
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_bins)
            
            lgb_train = lgb.Dataset(X_train_scaled, label=y_train, weight=sample_weights)
            lgb_valid = lgb.Dataset(X_valid_scaled, label=y_valid, reference=lgb_train)

            self._log_message("Starting model training from scratch...")
            model_from_scratch = lgb.train(AppConfig.TRAIN_PARAMS, lgb_train, valid_sets=[lgb_valid],
                                         callbacks=[lgb.early_stopping(AppConfig.EARLY_STOPPING_ROUNDS, verbose=True)])
            self._log_message("Training complete.")
            
            self._log_message("\nPhase 4: Evaluating new model on hold-out test set...")
            
            # --- MODIFICATION START: Create detailed log for hold-out set ---
            detailed_holdout_results = []
            y_true_aggregated = []
            y_pred_aggregated_by_approach = {approach: [] for approach in ["Index Finger", "Middle Finger", "Ring Finger", "SNR-Weighted Fusion", "SQI-Selected Fusion"]}

            for _, row in holdout_df.iterrows():
                y_true = row['glucose']
                y_true_aggregated.append(y_true)
                
                segments_per_finger = row['segments']
                features_per_finger = [
                    [feature_extraction_mendeley.extract_all_features_from_segment(seg, AppConfig.TARGET_FS_FEATURES) for seg in finger_segs]
                    if finger_segs else [] for finger_segs in segments_per_finger
                ]
                ordered_features = [
                    [[f.get(feat, np.nan) for feat in self.expected_feature_order] for f in finger_features]
                    for finger_features in features_per_finger
                ]

                for approach in y_pred_aggregated_by_approach.keys():
                    features_for_approach = []
                    if approach == "SQI-Selected Fusion":
                        features_for_approach = fuse_features_sqi_selected(ordered_features, segments_per_finger, AppConfig.TARGET_FS_FEATURES)
                    elif approach == "SNR-Weighted Fusion":
                        features_for_approach = fuse_features_snr_weighted(ordered_features, segments_per_finger, AppConfig.TARGET_FS_FEATURES)
                    else:
                        finger_map = {"Index": 0, "Middle": 1, "Ring": 2}
                        finger_idx = finger_map[approach.split(' ')[0]]
                        features_for_approach = ordered_features[finger_idx]

                    if features_for_approach:
                        X_test = pd.DataFrame(features_for_approach, columns=self.expected_feature_order).fillna(0)
                        X_test_scaled = scaler.transform(X_test)
                        preds = model_from_scratch.predict(X_test_scaled, num_iteration=model_from_scratch.best_iteration)
                        mean_pred = np.mean(preds)
                        y_pred_aggregated_by_approach[approach].append(mean_pred)
                        
                        detailed_holdout_results.append({
                            "SampleID": row['subject_id'],
                            "Approach": approach,
                            "ActualGlucose": y_true,
                            "PredictedGlucose": mean_pred,
                        })
                    else:
                        y_pred_aggregated_by_approach[approach].append(np.nan)

            detailed_df = pd.DataFrame(detailed_holdout_results)
            detailed_df['mARD(%)'] = np.abs(detailed_df['PredictedGlucose'] - detailed_df['ActualGlucose']) / detailed_df['ActualGlucose'] * 100
            detailed_df['RMSE(mg/dL)'] = (detailed_df['PredictedGlucose'] - detailed_df['ActualGlucose'])**2
            detailed_df['MAE(mg/dL)'] = np.abs(detailed_df['PredictedGlucose'] - detailed_df['ActualGlucose'])

            detailed_log_path = os.path.join(self.output_dir, "from_scratch_detailed_holdout_log.csv")
            detailed_df.to_csv(detailed_log_path, index=False)
            self._log_message(f"Detailed hold-out log saved to: {detailed_log_path}")

            final_results = []
            for approach, y_preds in y_pred_aggregated_by_approach.items():
                valid_indices = ~np.isnan(y_preds)
                y_true_valid = np.array(y_true_aggregated)[valid_indices]
                y_preds_valid = np.array(y_preds)[valid_indices]
                if len(y_true_valid) == 0: continue
                
                mard = calculate_mard(y_true_valid, y_preds_valid)
                rmse = np.sqrt(mean_squared_error(y_true_valid, y_preds_valid))
                mae = mean_absolute_error(y_true_valid, y_preds_valid)
                final_results.append([approach, f"{mard:.2f}", f"{rmse:.2f}", f"{mae:.2f}"])
            # --- MODIFICATION END ---
            
            self._log_message("\n--- Final Results on Unseen Data ---")
            display_order = {name: i for i, name in enumerate(y_pred_aggregated_by_approach.keys())}
            sorted_results = sorted(final_results, key=lambda r: display_order.get(r[0], 99))
            
            results_df = pd.DataFrame(sorted_results, columns=self.results_cols)
            results_log_path = os.path.join(self.output_dir, "from_scratch_aggregated_results.csv")
            results_df.to_csv(results_log_path, index=False)
            self._log_message(f"Final aggregated results saved to: {results_log_path}")

            self.root.after(0, self._update_results_display, sorted_results)
            
            self._log_message("Saving the new model and scaler...")
            model_filename = "lgbm_model_from_scratch.txt"
            scaler_filename = "scaler_from_scratch.pkl"
            model_save_path = os.path.join(self.output_dir, model_filename)
            scaler_save_path = os.path.join(self.output_dir, scaler_filename)
            model_from_scratch.save_model(model_save_path)
            joblib.dump(scaler, scaler_save_path)
            self._log_message(f"Model saved to: {model_save_path}")
            self._log_message(f"Scaler saved to: {scaler_save_path}")

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
        app = ModelTrainerApp(root)
        root.mainloop()