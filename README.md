# PPG Glucose Estimation - Model Training from Dataset

This project focuses on training and evaluating a machine learning model to estimate blood glucose levels from an existing dataset of PPG (photoplethysmography) signals.

## Repository Overview

This repository contains the Python scripts necessary to:

1.  Load and preprocess PPG signals and their corresponding glucose labels.
2.  Extract a comprehensive set of physiological features from the PPG data.
3.  Train a LightGBM machine learning model for glucose estimation.
4.  Evaluate the model's performance.

## Files

- **`PPG_Dataset/`**: Contains the input dataset.
  - `Labels/Total.csv`: Subject IDs and their glucose levels.
  - `RawData/signal_*.csv`: Raw PPG signal files.
- **`models/`**: Target directory for saved trained models.
- **`config.py`**: Central configuration for file paths, signal processing parameters (sampling rates, filter settings), and model hyperparameters. **Adjust paths here first.**
- **`data_loader.py`**: Scripts for loading the raw signal data and the glucose labels.
- **`preprocessing.py`**: Contains functions for signal preprocessing, including downsampling, filtering (Butterworth bandpass, Savitzky-Golay), and segmentation into windows.
- **`feature_extraction.py`**: Implements the extraction of various morphological, interval, statistical, and frequency-domain features from the processed PPG segments.
- **`model_trainer.py`**: Manages the machine learning workflow: splits data, trains the LightGBM model, evaluates it using metrics like mARD, RMSE, MAE, and saves the trained model.
- **`main.py`**: The main executable script that orchestrates the entire pipeline from data loading to model training and saving.

## How to Run

1.  **Setup**:
    - Ensure Python and required libraries are installed: `pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`.
    - Verify that the `BASE_DATA_PATH` in `config.py` correctly points to your `PPG_Dataset` folder.
2.  **Execute**:
    - Run the main script from the project's root directory:
      ```bash
      python main.py
      ```
    - The script will process the data, train the model, print evaluation metrics, and save the trained model to the `models/` directory.
