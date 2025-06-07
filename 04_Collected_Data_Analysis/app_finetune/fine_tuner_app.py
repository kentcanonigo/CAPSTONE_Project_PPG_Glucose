import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ttkthemes import ThemedTk
import pandas as pd
import numpy as np
import os
import sys 
import threading
import time 
from datetime import datetime 
import joblib 
from scipy.signal import resample
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Path and Module Setup ---
# This block handles importing your existing scripts from the Mendeley project.
SCRIPTS_LOADED_SUCCESSFULLY = False
config_mendeley, preprocessing_mendeley, feature_extraction_mendeley, model_trainer_mendeley = None, None, None, None

try:
    # Build robust, absolute paths from this script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) # Navigates up to 'capstone-machine-learning'
    
    # Path to the Mendeley 'src' directory
    mendeley_src_path = os.path.join(project_root, "02_Machine_Learning_Mendeley", "src")
    if not os.path.isdir(mendeley_src_path):
        raise ImportError(f"Mendeley src path does not exist: {mendeley_src_path}")
    if mendeley_src_path not in sys.path:
        sys.path.insert(0, mendeley_src_path)

    # Now, import the modules from the Mendeley project using their filenames
    # and assign them the aliases we use in the code
    import config as config_mendeley 
    import preprocessing as preprocessing_mendeley
    import feature_extraction_new as feature_extraction_mendeley 
    import model_trainer as model_trainer_mendeley
    
    print(f"Successfully imported processing modules from: {mendeley_src_path}")
    SCRIPTS_LOADED_SUCCESSFULLY = True
except ImportError as e:
    print(f"ERROR: Could not import required modules. Please ensure all scripts are in place. Details: {e}")
except Exception as e_path:
    print(f"Error setting up path for Mendeley scripts: {e_path}")


# --- Configuration (Internal to this App) ---
class FineTuningConfig:
    PROJECT_ROOT = project_root # Use the dynamically found project root
    CUSTOM_DATA_DIR = os.path.join(PROJECT_ROOT, "04_Collected_Data_Analysis", "Collected_Data")
    CUSTOM_LABELS_FILE = os.path.join(CUSTOM_DATA_DIR, "Labels", "collected_labels.csv")
    CUSTOM_RAW_DATA_DIR = os.path.join(CUSTOM_DATA_DIR, "RawData")
    MENDELEY_MODEL_DIR = os.path.join(PROJECT_ROOT, "02_Machine_Learning_Mendeley", "src", "models")
    
    PRE_TRAINED_MODEL_FILENAME = "lgbm_glucose_model_retrained_v1.txt" 
    PRE_TRAINED_SCALER_FILENAME = "mendeley_feature_scaler_retrained_v1.pkl"
    PRE_TRAINED_FEATURES_FILENAME = "model_features_retrained_v1.json"

    CUSTOM_MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "04_Collected_Data_Analysis", "models_finetuned")
    if not os.path.exists(CUSTOM_MODEL_OUTPUT_DIR):
        os.makedirs(CUSTOM_MODEL_OUTPUT_DIR)

    FT_PARAMS = {'learning_rate': 0.005, 'n_jobs': -1, 'seed': 42}
    FT_NUM_BOOST_ROUND = 300
    FT_EARLY_STOPPING_ROUNDS = 30
    INPUT_SAMPLING_RATE = 100 

config = FineTuningConfig()

def calculate_mard(y_true, y_pred):
    """
    Calculates Mean Absolute Relative Difference (mARD).
    
    Args:
        y_true (array-like): The ground truth (actual) glucose values.
        y_pred (array-like): The predicted glucose values from the model.
        
    Returns:
        float: The mARD value as a percentage, or NaN if calculation is not possible.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Ensure there are values to process
    if y_true_np.size == 0 or y_pred_np.size == 0:
        return np.nan

    # Create a mask to exclude entries where the true value is zero to avoid division by zero
    mask = y_true_np != 0
    
    # If all true values were zero, mARD is undefined
    if not np.any(mask):
        return np.nan

    # Apply the mask to both arrays
    y_true_masked = y_true_np[mask]
    y_pred_masked = y_pred_np[mask]
    
    # Calculate the absolute relative difference for each valid point
    ard = np.abs(y_true_masked - y_pred_masked) / y_true_masked
    
    # Return the mean of the relative differences, as a percentage
    return np.mean(ard) * 100


# --- Data Loading Utilities (Internal to this App) ---
def load_all_custom_data():
    """Loads all custom data samples and their labels from the collection directory."""
    try:
        labels_df = pd.read_csv(config.CUSTOM_LABELS_FILE)
    except FileNotFoundError:
        print(f"ERROR: Custom labels file not found at {config.CUSTOM_LABELS_FILE}")
        return None
    
    all_data = []
    print(f"Found {len(labels_df)} entries in labels file. Loading raw PPG data files...")
    for _, row in labels_df.iterrows():
        subject_id, sample_num = row['ID'], row['Sample_Num']
        ppg_filepath = os.path.join(config.CUSTOM_RAW_DATA_DIR, f"{subject_id}_{sample_num}_ppg.csv")

        if os.path.exists(ppg_filepath):
            try:
                # Use pd.to_numeric to handle potential header issues robustly
                ppg_df = pd.read_csv(ppg_filepath)
                sig1 = pd.to_numeric(ppg_df['ppg_finger1'], errors='coerce').dropna().to_numpy()
                sig2 = pd.to_numeric(ppg_df['ppg_finger2'], errors='coerce').dropna().to_numpy()
                sig3 = pd.to_numeric(ppg_df['ppg_finger3'], errors='coerce').dropna().to_numpy()
                
                if all(s.size > 0 for s in [sig1, sig2, sig3]):
                    all_data.append({
                        "subject_id": subject_id, "sample_num": sample_num, "glucose": row['Glucose_mgdL'],
                        "ppg_finger1": sig1, "ppg_finger2": sig2, "ppg_finger3": sig3
                    })
                else:
                    print(f"Warning: Empty PPG columns after cleaning non-numeric values in {ppg_filepath}.")
            except Exception as e:
                print(f"Warning: Could not read or process {ppg_filepath}: {e}")
        else:
            print(f"Warning: PPG file not found for subject {subject_id}, sample {sample_num}")
            
    print(f"Successfully loaded data for {len(all_data)} samples.")
    return all_data


# --- Main Application Class ---
class FineTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LightGBM Model Fine-Tuner")
        self.root.geometry("800x800")
        
        # ... (GUI setup from previous response) ...
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.pack(fill=tk.X, expand=False, pady=5)
        config_frame.columnconfigure(1, weight=1)
        ttk.Label(config_frame, text="Pre-trained Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(config_frame, text=config.PRE_TRAINED_MODEL_FILENAME, wraplength=500).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(config_frame, text="Scaler File:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(config_frame, text=config.PRE_TRAINED_SCALER_FILENAME, wraplength=500).grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Label(config_frame, text="Custom Data Path:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(config_frame, text=config.CUSTOM_DATA_DIR, wraplength=500).grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Label(config_frame, text="Fine-tuned Model Output:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Label(config_frame, text=config.CUSTOM_MODEL_OUTPUT_DIR, wraplength=500).grid(row=3, column=1, sticky=tk.W, pady=2)
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.pack(fill=tk.X, expand=False, pady=10)
        self.start_button = ttk.Button(control_frame, text="Start Fine-Tuning Pipeline", command=self.start_pipeline_thread)
        self.start_button.pack(pady=5, ipady=5)
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        log_frame.columnconfigure(0, weight=1); log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=20, width=100, state=tk.DISABLED, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text['yscrollcommand'] = log_scrollbar.set
        self.log_message("Fine-Tuning Application Initialized.")
        if not SCRIPTS_LOADED_SUCCESSFULLY:
            self.log_message("ERROR: Failed to load Mendeley processing modules. Please check console.")
            self.start_button.config(state=tk.DISABLED)

    def log_message(self, message):
        self.root.after(0, self._update_log_text, message)

    def _update_log_text(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def start_pipeline_thread(self):
        self.start_button.config(state=tk.DISABLED)
        self.log_message("--- Pipeline Started ---")
        pipeline_thread = threading.Thread(target=self.run_fine_tuning_pipeline, daemon=True)
        pipeline_thread.start()

    def run_fine_tuning_pipeline(self):
        try:
            self.log_message("Phase 1: Loading Custom 30-Respondent Dataset...")
            custom_data = load_all_custom_data()
            if not custom_data:
                self.log_message("ERROR: No custom data loaded. Exiting pipeline.")
                self.root.after(0, self.start_button.config, {'state': tk.NORMAL}); return

            self.log_message("\nPhase 2: Loading Pre-trained Mendeley Model and Scaler...")
            # Use the imported function directly
            pretrained_model, mendeley_scaler, expected_feature_order = model_trainer_mendeley.load_model_scaler_and_features(
                config.MENDELEY_MODEL_DIR, config.PRE_TRAINED_MODEL_FILENAME,
                config.PRE_TRAINED_SCALER_FILENAME, config.PRE_TRAINED_FEATURES_FILENAME
            )
            if pretrained_model is None or mendeley_scaler is None or not expected_feature_order:
                self.log_message("ERROR: Failed to load pre-trained artifacts. Exiting."); self.root.after(0, self.start_button.config, {'state': tk.NORMAL}); return
            
            self.log_message("\nPhase 3: Preparing Custom Data for Fine-Tuning...")
            all_feature_vectors = []
            for sample in custom_data:
                signals = [sample['ppg_finger1'], sample['ppg_finger2'], sample['ppg_finger3']]
                for finger_idx, raw_signal in enumerate(signals):
                    segments = preprocessing_mendeley.full_preprocess_pipeline(raw_signal, use_mendeley_fs=False, custom_fs=config.INPUT_SAMPLING_RATE)
                    if not segments: continue
                    for segment in segments:
                        features_dict = feature_extraction_mendeley.extract_all_features_from_segment(segment, config_mendeley.TARGET_FS)
                        ordered_features = [features_dict.get(feat, np.nan) for feat in expected_feature_order]
                        all_feature_vectors.append({"subject_id": sample['subject_id'], "features": ordered_features, "glucose": sample['glucose']})
            
            if not all_feature_vectors: self.log_message("No feature vectors extracted. Exiting."); self.root.after(0, self.start_button.config, {'state': tk.NORMAL}); return
                
            features_df = pd.DataFrame([item['features'] for item in all_feature_vectors], columns=expected_feature_order)
            labels = pd.Series([item['glucose'] for item in all_feature_vectors]); groups = pd.Series([item['subject_id'] for item in all_feature_vectors])
            if features_df.isnull().values.any():
                self.log_message(f"NaNs found. Filling with median."); features_df = features_df.fillna(features_df.median())
            features_scaled_df = pd.DataFrame(mendeley_scaler.transform(features_df), columns=expected_feature_order)
            self.log_message(f"Custom data preprocessed and scaled. Total feature sets: {len(features_scaled_df)}")

            self.log_message("\nPhase 4: Splitting Custom Data by Participant...")
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            ft_indices, test_indices = next(splitter.split(features_scaled_df, labels, groups=groups))
            X_ft, y_ft, groups_ft = features_scaled_df.iloc[ft_indices], labels.iloc[ft_indices], groups.iloc[ft_indices]
            X_test_custom, y_test_custom = features_scaled_df.iloc[test_indices], labels.iloc[test_indices]
            
            ft_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            ft_train_indices, ft_valid_indices = next(ft_splitter.split(X_ft, y_ft, groups=groups_ft))
            X_ft_train, y_ft_train = X_ft.iloc[ft_train_indices], y_ft.iloc[ft_train_indices]
            X_ft_valid, y_ft_valid = X_ft.iloc[ft_valid_indices], y_ft.iloc[ft_valid_indices]
            self.log_message(f"Data split: {len(X_ft_train)} for FT-Train, {len(X_ft_valid)} for FT-Validation, {len(X_test_custom)} for Final Test.")

            self.log_message("\nPhase 5: Fine-tuning the Model...")
            lgb_ft_train = lgb.Dataset(X_ft_train, label=y_ft_train); lgb_ft_valid = lgb.Dataset(X_ft_valid, label=y_ft_valid)
            ft_params = pretrained_model.params.copy(); ft_params.update(config.FT_PARAMS)
            if 'n_estimators' in ft_params: del ft_params['n_estimators']
            if 'num_iterations' in ft_params: del ft_params['num_iterations']
            
            fine_tuned_model = lgb.train(
                ft_params, lgb_ft_train, num_boost_round=config.FT_NUM_BOOST_ROUND, valid_sets=[lgb_ft_train, lgb_ft_valid],
                callbacks=[lgb.early_stopping(stopping_rounds=config.FT_EARLY_STOPPING_ROUNDS, verbose=True), lgb.log_evaluation(period=50)],
                init_model=pretrained_model
            )
            self.log_message("Fine-tuning finished.")

            self.log_message("\nPhase 6: Saving the Fine-Tuned Model...")
            fine_tuned_model_name = f"lgbm_model_finetuned_on_{len(np.unique(groups))}subjects.txt"
            fine_tuned_model_path = os.path.join(config.CUSTOM_MODEL_OUTPUT_DIR, fine_tuned_model_name)
            fine_tuned_model.save_model(fine_tuned_model_path)
            self.log_message(f"Fine-tuned model saved to: {fine_tuned_model_path}")
            
            self.log_message("\n--- Phase 7: Evaluating on Hold-out Custom Test Set ---")
            y_pred_finetuned = fine_tuned_model.predict(X_test_custom, num_iteration=fine_tuned_model.best_iteration)
            mard_finetuned = calculate_mard(y_test_custom, y_pred_finetuned)
            rmse_finetuned = np.sqrt(mean_squared_error(y_test_custom, y_pred_finetuned))
            self.log_message(f"  **Fine-Tuned Model** on Custom Test Set -> mARD: {mard_finetuned:.2f}%, RMSE: {rmse_finetuned:.2f}")
            y_pred_original = pretrained_model.predict(X_test_custom)
            mard_original = calculate_mard(y_test_custom, y_pred_original)
            rmse_original = np.sqrt(mean_squared_error(y_test_custom, y_pred_original))
            self.log_message(f"  **Original Mendeley Model** on Custom Test Set -> mARD: {mard_original:.2f}%, RMSE: {rmse_original:.2f}")
            improvement = mard_original - mard_finetuned
            self.log_message(f"\nIMPROVEMENT IN MARD DUE TO FINE-TUNING: {improvement:.2f}%")
            
            self.log_message("\n--- Pipeline Finished Successfully ---")

        except Exception as e:
            import traceback
            self.log_message(f"AN ERROR OCCURRED: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Pipeline Error", f"An error occurred: {e}")
        finally:
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))

if __name__ == '__main__':
    root = ThemedTk(theme="arc")
    app = FineTunerApp(root)
    root.mainloop()
