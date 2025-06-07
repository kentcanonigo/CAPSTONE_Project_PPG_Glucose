import os

# --- Paths ---
# This assumes this file is in 04_Collected_Data_Analysis/src/
ANALYSIS_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ANALYSIS_ROOT, "..", ".."))

# Path to your collected custom data
CUSTOM_DATA_DIR = os.path.join(PROJECT_ROOT, "04_Collected_Data_Analysis", "Collected_Data")
CUSTOM_LABELS_FILE = os.path.join(CUSTOM_DATA_DIR, "Labels", "collected_labels.csv")
CUSTOM_RAW_DATA_DIR = os.path.join(CUSTOM_DATA_DIR, "RawData")

# --- Path to the PRE-TRAINED Model Artifacts to be Fine-Tuned ---
MENDELEY_MODEL_DIR = os.path.join(PROJECT_ROOT, "02_Machine_Learning_Mendeley", "src", "models")
# *** IMPORTANT: Update these to the filenames of the best model you trained on Mendeley ***
PRE_TRAINED_MODEL_FILENAME = "lgbm_glucose_model_retrained_v1.txt" 
PRE_TRAINED_SCALER_FILENAME = "mendeley_feature_scaler_retrained_v1.pkl"
PRE_TRAINED_FEATURES_FILENAME = "model_features_retrained_v1.json"

# --- Output Paths for the FINE-TUNED Model ---
CUSTOM_MODEL_OUTPUT_DIR = os.path.join(ANALYSIS_ROOT, "..", "models_finetuned")
if not os.path.exists(CUSTOM_MODEL_OUTPUT_DIR):
    os.makedirs(CUSTOM_MODEL_OUTPUT_DIR)

# --- Fine-Tuning Hyperparameters ---
# Use a VERY SMALL learning rate to gently adjust the pre-trained model
FT_PARAMS = {
    'learning_rate': 0.005, 
    'n_jobs': -1,
    'seed': 42
    # Other parameters will be inherited from the pre-trained model
}
FT_NUM_BOOST_ROUND = 300 # Set a max number of training rounds for fine-tuning
FT_EARLY_STOPPING_ROUNDS = 30 # Stop if validation score doesn't improve for 30 rounds
