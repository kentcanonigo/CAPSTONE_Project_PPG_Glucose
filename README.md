# Non-Invasive Glucose Estimation using Multi-Finger PPG

## Capstone Project - Department of Computer Engineering, Cebu Technological University - Main Campus

This repository contains all components for the capstone project titled: "Comparative Analysis of Single-Site and Multi-Finger PPG Systems for Non-Invasive Glucose Estimation." The project involves developing a custom multi-finger PPG data acquisition system, training machine learning models on existing and newly collected data, and evaluating the efficacy of multi-finger approaches compared to single-site PPG for glucose estimation.

## Project Components

This project is organized into several key functional areas:

- **`01_Data_Collection_Tool/`**: Contains the Python Tkinter application for data entry and device interfacing, along with the microcontroller firmware (ESP32/Arduino) for acquiring PPG signals from participants.

- **`02_Machine_Learning_Mendeley/`**: Dedicated to the machine learning pipeline (data loading, preprocessing, feature extraction, LightGBM model training, and evaluation) developed using the existing public Mendeley PPG dataset to establish a single-site baseline model.

- **`03_Hardware_Interface_Development/`**: Houses all hardware design aspects, including schematics, component lists, and assembly notes for the custom multi-finger PPG acquisition device.

- **`04_Collected_Data_Analysis/`**: Intended for storing the data collected by the custom system (`Collected_Data/` subfolder) and the scripts/notebooks for its processing, multi-finger fusion analysis, and performance evaluation against the baseline model or newly trained models.

- **`docs/`**: Stores general project documentation such as the thesis manuscript, presentation slides, and research notes.

## General Workflow

1.  **Baseline Model Training**: Use components in `02_Machine_Learning_Mendeley/` to train a glucose estimation model on the existing dataset.
2.  **Hardware & Firmware Setup**: Assemble the device based on `03_Hardware_Interface_Development/` and flash firmware from `01_Data_Collection_Tool/firmware/`.
3.  **Custom Data Collection**: Utilize the Tkinter app in `01_Data_Collection_Tool/app/` to collect new multi-finger PPG data.
4.  **Analysis & Comparison**: Process and analyze the newly collected data using tools in `04_Collected_Data_Analysis/` to compare single-finger and multi-finger performance.

Refer to specific `README.md` files within each main directory for more detailed instructions.
