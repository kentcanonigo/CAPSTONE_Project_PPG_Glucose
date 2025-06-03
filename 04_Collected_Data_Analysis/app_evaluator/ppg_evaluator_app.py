import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ttkthemes import ThemedTk # For applying themes
import pandas as pd
import numpy as np
import os
import threading
import time # For simulating processing time
import random # For dummy data
import math   # For dummy data
from datetime import datetime # <--- IMPORT ADDED

# --- Placeholder for your actual processing functions ---
# You will replace these dummy functions with imports and calls
# to your actual scripts in 04_Collected_Data_Analysis/src/

def dummy_load_ppg_signals(ppg_filepath):
    """Simulates loading 3 raw PPG signals from a file."""
    print(f"DUMMY: Loading raw signals from {os.path.basename(ppg_filepath)}")
    time.sleep(0.3)
    # Simulate 30 seconds of data at 100Hz (3000 samples per signal)
    # This would be downsampled to 1500 samples at 50Hz for feature extraction
    return [np.random.randint(1000, 3000, size=30000).astype(float) for _ in range(3)]

def dummy_preprocess_and_segment(raw_signal_list, input_fs=100, target_fs_features=50, window_duration_sec=5):
    """Simulates preprocessing and segmenting for a list of raw signals."""
    print(f"DUMMY: Preprocessing {len(raw_signal_list)} signals (Input FS: {input_fs}, Target FS for Feat: {target_fs_features})...")
    time.sleep(0.7)
    all_finger_segments = []
    samples_per_window_for_features = int(window_duration_sec * target_fs_features) # e.g., 250
    
    for i, raw_signal in enumerate(raw_signal_list):
        # Simulate downsampling: if input_fs=100, target_fs=50, signal length halves
        # For dummy, just assume it's been downsampled to a length that gives some segments
        num_original_samples = len(raw_signal)
        num_downsampled_samples = int(num_original_samples * (target_fs_features / input_fs))

        num_segments = num_downsampled_samples // samples_per_window_for_features
        segments_this_finger = [
            np.random.rand(samples_per_window_for_features) + (i*0.1) # Add offset for visual difference
            for _ in range(max(1, num_segments)) # Ensure at least one segment
        ]
        all_finger_segments.append(segments_this_finger)
        print(f"  Finger {i+1}: Generated {len(segments_this_finger)} segments of {samples_per_window_for_features} samples.")
    return all_finger_segments # List of lists of segments

def dummy_extract_features_from_segments(list_of_segments):
    """Simulates extracting 13 features from a list of segments for one finger."""
    print(f"DUMMY: Extracting features for {len(list_of_segments)} segments...")
    time.sleep(0.3)
    # Each feature vector should be a list or 1D numpy array of 13 features
    return [np.random.rand(13).tolist() for _ in list_of_segments]

def dummy_apply_feature_scaler(feature_vectors_list_of_lists, scaler_path=None):
    """Simulates applying a pre-fitted scaler."""
    print(f"DUMMY: Applying feature scaler (loaded from '{scaler_path}')...")
    time.sleep(0.1)
    # In reality, you would load your scaler object (e.g., from a .pkl file)
    # and call scaler.transform(features). For dummy, just return as is.
    return feature_vectors_list_of_lists

def dummy_fuse_snr_weighted_features(features_f1, features_f2, features_f3):
    """Simulates SNR-weighted feature fusion."""
    print("DUMMY: Performing SNR-Weighted Feature Fusion...")
    time.sleep(0.2)
    fused_features = []
    # Assuming features_f1, f2, f3 are lists of feature vectors (lists themselves)
    # and all have the same number of segments (feature vectors)
    for i in range(len(features_f1)):
        # Simple average for dummy fusion
        avg_feature_vector = np.mean([features_f1[i], features_f2[i], features_f3[i]], axis=0).tolist()
        fused_features.append(avg_feature_vector)
    return fused_features

def dummy_fuse_sqi_selected_features(features_f1, features_f2, features_f3, segments_f1, segments_f2, segments_f3):
    """Simulates SQI-based feature selection."""
    print("DUMMY: Performing SQI-Based Feature Selection...")
    time.sleep(0.2)
    selected_features = []
    # For dummy, let's say finger 1 always has the best SQI
    for i in range(len(features_f1)):
        selected_features.append(features_f1[i]) # Just taking features from finger 1
    return selected_features

class DummyModel:
    """A mock model for testing."""
    def __init__(self, model_path=None):
        print(f"DUMMY MODEL: Initialized (would load from '{model_path}')")
    
    def predict(self, list_of_feature_vectors):
        print(f"DUMMY MODEL: Predicting for {len(list_of_feature_vectors)} feature sets...")
        predictions = []
        for features in list_of_feature_vectors:
            # Simulate a prediction based on the first feature, with some randomness
            base_pred = 100 + features[0] * 10 
            pred = round(random.uniform(base_pred - 10, base_pred + 10), 1)
            predictions.append(max(40, min(500, pred))) # Keep predictions in a broad glucose range
        return np.array(predictions)

def dummy_load_lgbm_model(model_path=None):
    """Simulates loading a pre-trained LightGBM model."""
    return DummyModel(model_path)
# --- End of Placeholder Functions ---


class PPGDataEvaluatorApp:
    def __init__(self, root_window, initial_geometry="750x700"): # Adjusted default size
        self.root = root_window
        self.root.title("PPG Data Evaluator V1.0")
        
        self.root.withdraw()
        self.root.geometry(initial_geometry)

        # --- Define Paths ---
        try:
            self.app_dir = os.path.dirname(os.path.abspath(__file__))
            # Assuming ppg_evaluator_app.py is in Your_Project_Root/04_Collected_Data_Analysis/app_evaluator/
            collected_data_analysis_dir = os.path.dirname(self.app_dir) 
            project_root_dir = os.path.dirname(collected_data_analysis_dir)
        except NameError: # Fallback if __file__ is not defined (e.g. running in an interactive session)
            self.app_dir = os.getcwd()
            # Adjust fallback path if necessary
            project_root_dir = os.path.abspath(os.path.join(self.app_dir, "..", ".."))


        self.collected_labels_path = os.path.join(project_root_dir, "04_Collected_Data_Analysis", "Collected_Data", "Labels", "collected_labels.csv")
        self.collected_raw_data_dir = os.path.join(project_root_dir, "04_Collected_Data_Analysis", "Collected_Data", "RawData")
        self.model_dir = os.path.join(project_root_dir, "02_Machine_Learning_Mendeley", "models")
        self.model_path = os.path.join(self.model_dir, "lgbm_glucose_mendeley_model.txt") # Adjust if name is different
        self.scaler_path = os.path.join(self.model_dir, "mendeley_feature_scaler.pkl") # Example scaler path

        self.selected_ppg_file = tk.StringVar()
        self.reference_glucose = tk.StringVar(value="N/A")
        self.processing_in_progress = False

        self._setup_gui()
        self._center_window()
        self.root.deiconify()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))


    def _center_window(self):
        self.root.update_idletasks() 
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        if width <= 1 or height <= 1: # Window might not have its final size yet from geometry string
            try:
                geom_parts = self.root.geometry().split('+')[0].split('x')
                width = int(geom_parts[0])
                height = int(geom_parts[1])
            except: # Fallback if geometry string is unusual or not yet fully processed
                width = 750 # Default width from initial_geometry
                height = 700 # Default height from initial_geometry
                
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_coordinate = int((screen_width / 2) - (width / 2))
        y_coordinate = int((screen_height / 2) - (height / 2))
        self.root.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}")
        if hasattr(self, 'log_text') and self.log_text: # Check if log_text is ready
            self._log_message(f"Window centered. Size: {width}x{height}")

    def _setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True) # Use pack for main_frame

        # Define row indices for gridding frames within main_frame
        row_file_select = 0
        row_info_action = 1
        row_results = 2
        row_log = 3

        # --- File Selection Frame ---
        file_frame = ttk.LabelFrame(main_frame, text="Select Data Sample", padding="10")
        file_frame.grid(row=row_file_select, column=0, sticky=(tk.W, tk.E), padx=5, pady=(5,10))
        main_frame.columnconfigure(0, weight=1) # Make file_frame expand horizontally

        ttk.Button(file_frame, text="Browse Raw PPG File (*_ppg.csv)...", command=self._select_ppg_file).pack(side=tk.LEFT, padx=5)
        self.selected_file_label = ttk.Label(file_frame, text="No file selected.", width=70, anchor="w", relief=tk.SUNKEN, padding=2)
        self.selected_file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # --- Info and Action Frame ---
        action_frame = ttk.Frame(main_frame, padding="5")
        action_frame.grid(row=row_info_action, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        ttk.Label(action_frame, text="Reference Glucose (mg/dL):").pack(side=tk.LEFT, padx=(0,2))
        self.ref_glucose_label = ttk.Label(action_frame, textvariable=self.reference_glucose, font=('Helvetica', 10, 'bold'), width=10)
        self.ref_glucose_label.pack(side=tk.LEFT, padx=(0,10))

        self.process_button = ttk.Button(action_frame, text="Process & Evaluate Sample", command=self._start_processing_thread, state=tk.DISABLED)
        self.process_button.pack(side=tk.RIGHT, padx=5) # Align to right

        # --- Results Display Frame ---
        results_frame = ttk.LabelFrame(main_frame, text="Evaluation Results", padding="10")
        results_frame.grid(row=row_results, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        results_frame.columnconfigure(0, weight=1) # Allow Treeview to expand horizontally
        results_frame.rowconfigure(0, weight=1) # Allow Treeview to expand vertically

        self.results_cols = ["Approach", "Predicted Glucose (mg/dL)", "ARD (%)", "Accuracy (%)"]
        self.results_tree = ttk.Treeview(results_frame, columns=self.results_cols, show="headings", height=7) # Initial height in rows
        for col in self.results_cols:
            self.results_tree.heading(col, text=col)
            anchor = 'e' if "Glucose" in col or "ARD" in col or "Accuracy" in col else 'w'
            width = 200 if col == "Approach" else 150 # Wider for approach name
            self.results_tree.column(col, width=width, anchor=anchor, minwidth=100)
        
        results_scrollbar_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar_y.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) # Use grid for Treeview
        results_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S)) # Grid scrollbar next to Treeview

        # --- Log Area ---
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=row_log, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=(10,5))
        log_frame.columnconfigure(0, weight=1) # Allow Text widget to expand horizontally
        log_frame.rowconfigure(0, weight=1) # Allow Text widget to expand vertically
        self.log_text = tk.Text(log_frame, height=10, width=80, state=tk.DISABLED, relief=tk.SUNKEN, borderwidth=1)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) # Use grid for Text
        log_scrollbar_y_log = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar_y_log.grid(row=0, column=1, sticky=(tk.N, tk.S)) # Grid scrollbar next to Text
        self.log_text['yscrollcommand'] = log_scrollbar_y_log.set
        
        # Configure row weights for main_frame to allow results and log to expand
        main_frame.rowconfigure(row_results, weight=1) # Results Treeview expands
        main_frame.rowconfigure(row_log, weight=1)   # Log area also expands
        
        self._log_message("Evaluator App Initialized. Select a Raw PPG file.")


    def _log_message(self, message):
        if hasattr(self, 'log_text') and self.log_text: # Check if log_text is initialized
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
            if self.root and self.root.winfo_exists(): # Check if root window still exists
                 self.root.update_idletasks()
        else: # Fallback if log_text isn't ready (e.g., during very early __init__)
            print(f"LOG_FALLBACK: {message}")

    def _select_ppg_file(self):
        filepath = filedialog.askopenfilename(
            initialdir=self.collected_raw_data_dir, # Start browsing in your collected raw data
            title="Select Raw PPG Data File (*_ppg.csv)",
            filetypes=(("CSV files", "*_ppg.csv"), ("All files", "*.*"))
        )
        if filepath:
            self.selected_ppg_file.set(filepath)
            self.selected_file_label.config(text=os.path.basename(filepath))
            self._log_message(f"Selected PPG file: {os.path.basename(filepath)}")
            self._load_label_for_file(filepath) # Attempt to load corresponding label
            self.process_button.config(state=tk.NORMAL)
            # Clear previous results from treeview
            for i in self.results_tree.get_children():
                self.results_tree.delete(i)
        else:
            self.selected_file_label.config(text="No file selected.")
            self.reference_glucose.set("N/A")
            self.process_button.config(state=tk.DISABLED)

    def _load_label_for_file(self, ppg_filepath):
        try:
            filename = os.path.basename(ppg_filepath)
            # Assuming filename is SubjectID_SampleNum_ppg.csv
            base = filename.replace("_ppg.csv", "")
            parts = base.split("_")
            if len(parts) < 2: # Expecting at least SubjectID_SampleNum
                self._log_message(f"Could not parse Subject ID and Sample # from filename: {filename}")
                self.reference_glucose.set("Error: Parse Filename")
                return

            subject_id_from_file = parts[0]
            sample_num_from_file = parts[1] # Assuming second part is sample number

            if not os.path.exists(self.collected_labels_path):
                self._log_message(f"Labels file not found: {self.collected_labels_path}")
                self.reference_glucose.set("Error: No Labels CSV")
                return

            labels_df = pd.read_csv(self.collected_labels_path)
            # Ensure comparison is robust (e.g., string to string for Sample_Num)
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
        if self.processing_in_progress:
            messagebox.showwarning("Busy", "Processing already in progress.")
            return
        if not self.selected_ppg_file.get():
            messagebox.showwarning("No File", "Please select a PPG data file first.")
            return
        ref_glucose_str = self.reference_glucose.get()
        if ref_glucose_str == "N/A" or "Error" in ref_glucose_str or not ref_glucose_str:
            messagebox.showwarning("No Reference Glucose", "Reference glucose not loaded or found. Cannot evaluate.")
            return

        self.processing_in_progress = True
        self.process_button.config(state=tk.DISABLED)
        # Clear previous results from treeview
        for i in self.results_tree.get_children():
            self.results_tree.delete(i)
        
        self._log_message("Starting processing & evaluation thread...")
        thread = threading.Thread(target=self._process_and_evaluate, daemon=True)
        thread.start()

    def _process_and_evaluate(self):
        """
        This method orchestrates the data loading, preprocessing, feature extraction,
        fusion, model prediction, and evaluation.
        REPLACE DUMMY FUNCTIONS WITH YOUR ACTUAL IMPLEMENTATIONS.
        """
        try:
            ppg_file = self.selected_ppg_file.get()
            actual_glucose = float(self.reference_glucose.get())

            self.root.after(0, lambda: self._log_message(f"Processing: {os.path.basename(ppg_file)}"))

            # 1. Load Raw Data (3 finger signals)
            # Replace with: raw_signals_list = data_loader_custom.load_ppg_signals(ppg_file)
            raw_signals_list = dummy_load_ppg_signals(ppg_file) 
            self.root.after(0, lambda: self._log_message("Raw PPG signals loaded."))

            # 2. Preprocess Signals
            # Replace with: all_finger_segments = preprocessing_custom.process_and_segment_all_fingers(raw_signals_list, ...)
            all_finger_segments = dummy_preprocess_and_segment(raw_signals_list, input_fs=100, target_fs_features=50) 
            self.root.after(0, lambda: self._log_message("Signals preprocessed and segmented."))

            # 3. Extract Features
            # Replace with: features_f1 = feature_extraction_custom.extract_all_features_for_finger(all_finger_segments[0], fs=50)
            features_f1 = dummy_extract_features_from_segments(all_finger_segments[0])
            features_f2 = dummy_extract_features_from_segments(all_finger_segments[1])
            features_f3 = dummy_extract_features_from_segments(all_finger_segments[2])
            self.root.after(0, lambda: self._log_message("Features extracted for individual fingers."))

            # 4. Feature Scaling (Load scaler fitted on Mendeley training data)
            # Replace with: 
            # scaler = model_utilities.load_scaler(self.scaler_path) 
            # features_f1_scaled = model_utilities.apply_scaling(features_f1, scaler)
            features_f1_scaled = dummy_apply_feature_scaler(features_f1, self.scaler_path)
            features_f2_scaled = dummy_apply_feature_scaler(features_f2, self.scaler_path)
            features_f3_scaled = dummy_apply_feature_scaler(features_f3, self.scaler_path)
            self.root.after(0, lambda: self._log_message("Features scaled."))
            
            # 5. Fusion
            # Replace with: fused_features_snr = signal_fusion_custom.fuse_snr(features_f1_scaled, features_f2_scaled, features_f3_scaled, snr_signals_f1, ...)
            fused_features_snr = dummy_fuse_snr_weighted_features(features_f1_scaled, features_f2_scaled, features_f3_scaled)
            self.root.after(0, lambda: self._log_message("SNR fusion applied."))
            # Replace with: fused_features_sqi = signal_fusion_custom.fuse_sqi(features_f1_scaled, ..., all_finger_segments[0], ...)
            fused_features_sqi = dummy_fuse_sqi_selected_features(features_f1_scaled, features_f2_scaled, features_f3_scaled,
                                                         all_finger_segments[0], all_finger_segments[1], all_finger_segments[2])
            self.root.after(0, lambda: self._log_message("SQI fusion applied."))

            # 6. Load Pre-trained Model
            # Replace with: model = model_utilities.load_lgbm_model(self.model_path)
            model = dummy_load_lgbm_model(self.model_path) 
            self.root.after(0, lambda: self._log_message("Pre-trained model loaded."))

            # 7. Make Predictions (Average predictions if multiple segments per sample)
            pred_f1_segments = model.predict(features_f1_scaled)
            pred_f2_segments = model.predict(features_f2_scaled)
            pred_f3_segments = model.predict(features_f3_scaled)
            pred_snr_segments = model.predict(fused_features_snr)
            pred_sqi_segments = model.predict(fused_features_sqi)

            # Ensure predictions are lists/arrays before np.mean and np.all(np.isnan())
            pred_f1 = np.mean(pred_f1_segments) if isinstance(pred_f1_segments, (list, np.ndarray)) and len(pred_f1_segments) > 0 and not np.all(np.isnan(pred_f1_segments)) else np.nan
            pred_f2 = np.mean(pred_f2_segments) if isinstance(pred_f2_segments, (list, np.ndarray)) and len(pred_f2_segments) > 0 and not np.all(np.isnan(pred_f2_segments)) else np.nan
            pred_f3 = np.mean(pred_f3_segments) if isinstance(pred_f3_segments, (list, np.ndarray)) and len(pred_f3_segments) > 0 and not np.all(np.isnan(pred_f3_segments)) else np.nan
            pred_snr = np.mean(pred_snr_segments) if isinstance(pred_snr_segments, (list, np.ndarray)) and len(pred_snr_segments) > 0 and not np.all(np.isnan(pred_snr_segments)) else np.nan
            pred_sqi = np.mean(pred_sqi_segments) if isinstance(pred_sqi_segments, (list, np.ndarray)) and len(pred_sqi_segments) > 0 and not np.all(np.isnan(pred_sqi_segments)) else np.nan
            self.root.after(0, lambda: self._log_message("Predictions generated."))

            # 8. Calculate Metrics and Update GUI
            results_to_display = []
            approaches = {
                "Finger 1 (Individual)": pred_f1, 
                "Finger 2 (Individual)": pred_f2, 
                "Finger 3 (Individual)": pred_f3,
                "SNR-Weighted Fusion": pred_snr, 
                "SQI-Selected Fusion": pred_sqi
            }
            for name, pred_glucose_val in approaches.items():
                if pd.isna(pred_glucose_val) or actual_glucose == 0 : 
                    ard_percent_str, acc_percent_str = "N/A", "N/A"
                    pred_glucose_str = "N/A" if pd.isna(pred_glucose_val) else f"{pred_glucose_val:.2f}"
                else:
                    ard = abs(actual_glucose - pred_glucose_val) / actual_glucose
                    ard_percent_str = f"{ard * 100:.2f}"
                    # Accuracy as 100 - ARD%, capped at 0 if ARD > 100%
                    acc_val = (1 - ard) * 100
                    acc_percent_str = f"{max(0, acc_val):.2f}" 
                    pred_glucose_str = f"{pred_glucose_val:.2f}"
                results_to_display.append([name, pred_glucose_str, ard_percent_str, acc_percent_str])
            
            self.root.after(0, lambda r=results_to_display: self._update_results_display(r))

        except Exception as e:
            self.root.after(0, lambda err=e: self._log_message(f"ERROR during processing: {err}"))
            self.root.after(0, lambda err=e: messagebox.showerror("Processing Error", f"An error occurred: {err}"))
        finally:
            self.processing_in_progress = False
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))

    def _update_results_display(self, results_list):
        for i in self.results_tree.get_children():
            self.results_tree.delete(i)
        for row_data in results_list:
            self.results_tree.insert("", tk.END, values=row_data)
        self._log_message("Results display updated.")

# --- Main Application Execution ---
if __name__ == '__main__':
    main_window = ThemedTk(theme="arc") # Or "plastik", "clearlooks", etc.
    app = PPGDataEvaluatorApp(main_window, initial_geometry="750x700") # Pass initial geometry
    
    def on_closing():
        # Add any cleanup needed before closing
        main_window.destroy()

    main_window.protocol("WM_DELETE_WINDOW", on_closing)
    main_window.mainloop()
