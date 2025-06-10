import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_scatter_predicted_vs_actual(df, actual_col, predicted_col, filename="scatter_plot.png"):
    """
    Generates and saves a scatter plot of predicted vs. actual glucose values.

    Args:
        df (pd.DataFrame): DataFrame containing the results.
        actual_col (str): Name of the column with actual glucose values.
        predicted_col (str): Name of the column with predicted glucose values from the best model.
        filename (str): Name of the file to save the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create the scatter plot
    sns.scatterplot(data=df, x=actual_col, y=predicted_col, ax=ax, alpha=0.7, s=80, label="Predictions")

    # Add a 45-degree line for perfect correlation
    lims = [
        np.min([df[actual_col].min(), df[predicted_col].min()]) - 10,
        np.max([df[actual_col].max(), df[predicted_col].max()]) + 10,
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Prediction (y=x)")

    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_title("Predicted vs. Actual Glucose Levels\n(Fine-Tuned SQI-Selected Fusion Model)", fontsize=16, pad=20)
    ax.set_xlabel("Actual Glucose (mg/dL) - Invasive Glucometer", fontsize=12)
    ax.set_ylabel("Predicted Glucose (mg/dL) - Non-Invasive Model", fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Scatter plot saved as '{filename}'")
    plt.show()

def plot_error_distribution_boxplots(df, ard_columns, filename="boxplot.png"):
    """
    Generates and saves box plots comparing the ARD distributions of different methods.

    Args:
        df (pd.DataFrame): DataFrame containing the ARD results for each method.
        ard_columns (dict): A dictionary mapping column names to display names.
        filename (str): Name of the file to save the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Rename columns for plotting
    df_renamed = df.rename(columns=ard_columns)
    
    # Melt the DataFrame to transform it from wide to long format
    df_melted = pd.melt(df_renamed, value_vars=list(ard_columns.values()),
                        var_name='Method', value_name='Absolute Relative Difference (ARD %)')

    # Create the box plots
    sns.boxplot(data=df_melted, x='Method', y='Absolute Relative Difference (ARD %)', ax=ax, 
                palette="viridis", showfliers=True)
    
    ax.set_title("Comparison of Error Distributions Across Methods", fontsize=16, pad=20)
    ax.set_xlabel("Estimation Method", fontsize=12)
    ax.set_ylabel("Absolute Relative Difference (ARD %)", fontsize=12)
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Box plot saved as '{filename}'")
    plt.show()


if __name__ == '__main__':
    # --- Load the actual results from the batch evaluator ---
    try:
        # Define the CORRECT path to the log file relative to the script's location
        log_filepath = '../evaluation_results/detailed_evaluation_log.csv'
        df_long = pd.read_csv(log_filepath)
        print(f"✅ Successfully loaded results from '{log_filepath}'")
    except FileNotFoundError:
        print(f"❌ ERROR: The results file was not found at '{log_filepath}'.")
        print("Please ensure you have run the 'batch_evaluator_app.py' first to generate the results.")
        exit()

    # --- Prepare the data for plotting ---
    print("\nProcessing data for report generation...")
    # 1. For the Box Plot: Pivot the mARD data from long to wide format
    df_ard_wide = df_long.pivot_table(index='SampleID', columns='Approach', values='mARD(%)').reset_index()

    # Rename columns to a consistent format
    df_ard_wide = df_ard_wide.rename(columns={
        'Index Finger': 'ARD_Finger1',
        'Middle Finger': 'ARD_Finger2',
        'Ring Finger': 'ARD_Finger3',
        'SNR-Weighted Fusion': 'ARD_SNR_Fusion',
        'SQI-Selected Fusion': 'ARD_SQI_Fusion'
    })

    # 2. For the Scatter Plot: Get the actual and predicted values for the best model (SQI Fusion)
    df_sqi = df_long[df_long['Approach'] == 'SQI-Selected Fusion'].copy()
    df_sqi = df_sqi.rename(columns={
        'ActualGlucose': 'Actual_Glucose',
        'PredictedGlucose': 'Predicted_SQI_Fusion'
    })

    # 3. Merge the datasets to create one final DataFrame for plotting
    df_results = pd.merge(df_ard_wide, df_sqi[['SampleID', 'Actual_Glucose', 'Predicted_SQI_Fusion']], on='SampleID')
    print("✅ Data has been successfully prepared for plotting.")

    # --- Generate Report Graph 1: Scatter Plot ---
    print("\nGenerating Scatter Plot...")
    plot_scatter_predicted_vs_actual(
        df=df_results,
        actual_col='Actual_Glucose',
        predicted_col='Predicted_SQI_Fusion',
        filename="Figure_4_1_Predicted_vs_Actual_CustomData.png"
    )

    # --- Generate Report Graph 2: Box Plot ---
    print("\nGenerating Box Plot...")
    ard_cols_to_plot = {
        'ARD_Finger1': 'Index Finger',
        'ARD_Finger2': 'Middle Finger',
        'ARD_Finger3': 'Ring Finger',
        'ARD_SNR_Fusion': 'SNR Fusion',
        'ARD_SQI_Fusion': 'SQI Fusion'
    }

    # Verify that all necessary columns exist in the DataFrame
    if all(col in df_results.columns for col in ard_cols_to_plot.keys()):
        plot_error_distribution_boxplots(
            df=df_results,
            ard_columns=ard_cols_to_plot,
            filename="Figure_4_2_ARD_Distribution_CustomData.png"
        )
    else:
        print("\n❌ Warning: Could not generate box plot because one or more required ARD columns were not found.")