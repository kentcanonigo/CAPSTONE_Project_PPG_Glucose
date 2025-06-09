import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    sns.scatterplot(data=df, x=actual_col, y=predicted_col, ax=ax, alpha=0.7, s=50, label="Predictions")

    # Add a 45-degree line for perfect correlation
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # Get the minimum of the axis limits
        np.max([ax.get_xlim(), ax.get_ylim()]),  # Get the maximum of the axis limits
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
    
    # We need to "melt" the DataFrame to make it suitable for seaborn's boxplot
    # This transforms the data from wide format to long format
    df_melted = pd.melt(df_renamed, value_vars=list(ard_columns.values()),
                        var_name='Method', value_name='Absolute Relative Difference (ARD %)')

    # Create the box plots
    sns.boxplot(data=df_melted, x='Method', y='Absolute Relative Difference (ARD %)', ax=ax, 
                palette="viridis", showfliers=True) # showfliers=True to show outliers
    
    ax.set_title("Comparison of Error Distributions Across Methods", fontsize=16, pad=20)
    ax.set_xlabel("Estimation Method", fontsize=12)
    ax.set_ylabel("Absolute Relative Difference (ARD %)", fontsize=12)
    plt.xticks(rotation=15, ha='right') # Rotate labels slightly for better readability

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Box plot saved as '{filename}'")
    plt.show()


if __name__ == '__main__':
    # --- This is where you load your ACTUAL results file ---
    # For demonstration, we will create a sample DataFrame.
    # Replace this with: df_results = pd.read_csv("path/to/your/final_evaluation_results.csv")
    
    print("Creating sample data for demonstration...")
    data = {
        'Actual_Glucose': [90, 105, 120, 85, 99, 115, 130, 92, 108, 125, 95, 112],
        'ARD_Finger1':    [15.1, 12.5, 18.2, 10.1, 14.5, 20.1, 13.5, 16.2, 11.8, 19.5, 14.8, 15.5],
        'ARD_Finger2':    [14.5, 13.1, 17.5, 11.5, 13.9, 19.5, 14.1, 15.5, 12.2, 18.9, 15.2, 16.1],
        'ARD_Finger3':    [16.2, 14.2, 19.1, 12.2, 15.1, 21.0, 15.2, 17.1, 13.1, 20.1, 16.5, 17.2],
        'ARD_SNR_Fusion': [11.2, 10.1, 14.5, 9.5, 11.9, 15.2, 11.1, 12.5, 9.8, 14.9, 11.5, 12.1],
        'ARD_SQI_Fusion': [10.5, 9.8, 13.2, 8.9, 10.5, 14.8, 10.2, 11.8, 9.1, 13.5, 10.8, 11.5],
        'Predicted_SQI_Fusion': [85, 115, 105, 92, 108, 98, 142, 82, 115, 110, 104, 100] # Predicted values from best model
    }
    df_results = pd.DataFrame(data)
    print("Sample data created.")

    # --- Generate Recommendation 1: Scatter Plot ---
    plot_scatter_predicted_vs_actual(
        df=df_results,
        actual_col='Actual_Glucose',
        predicted_col='Predicted_SQI_Fusion',
        filename="Figure_4_1_Predicted_vs_Actual.png" # Example filename
    )

    # --- Generate Recommendation 2: Box Plot ---
    # Define the columns and their nice display names for the plot
    ard_cols_to_plot = {
        'ARD_Finger1': 'Index Finger',
        'ARD_Finger2': 'Middle Finger',
        'ARD_Finger3': 'Ring Finger',
        'ARD_SNR_Fusion': 'SNR Fusion',
        'ARD_SQI_Fusion': 'SQI Fusion'
    }
    plot_error_distribution_boxplots(
        df=df_results,
        ard_columns=ard_cols_to_plot,
        filename="Figure_4_2_ARD_Distribution.png" # Example filename
    )