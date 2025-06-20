import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from pathlib import Path
# To import the config file from the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def get_filename(path):
    """ Extract the filename from a given path.
    
    Parameters:
        path (str): The file path from which to extract the filename.
    
    Returns:
        str: The filename extracted from the path."""
    
    return os.path.basename(path)



def from_json_to_df(filepaths, obs='intra'):
    """
    Load two JSON files and convert them into a DataFrame with filenames as indices.

    Parameters:
        filepaths (list): A list containing two file paths to JSON files.
    
    Returns:
        combined_df (pd.DataFrame): A DataFrame with filenames as indices and the contents of the JSON files as columns. """

    # Load the JSON files
    with open(filepaths[0], 'r') as f1:
        data1 = json.load(f1)
    with open(filepaths[1], 'r') as f2:
        data2 = json.load(f2)
    if obs == 'inter':
        with open(filepaths[2], 'r') as f3:
            data3 = json.load(f3)

    if obs == 'intra':
        names = ['Initial', 'Repeat']
    if obs == 'inter':
        names = ['Obs. 1', 'Obs. 2', 'Obs. 3']

    # Convert the dictionaries to DataFrames
    df1 = pd.DataFrame.from_dict({get_filename(k): v for k, v in data1.items()}, orient='index', columns=[names[0]])
    df2 = pd.DataFrame.from_dict({get_filename(k): v for k, v in data2.items()}, orient='index', columns=[names[1]])
    if obs == 'inter':
        df3 = pd.DataFrame.from_dict({get_filename(k): v for k, v in data3.items()}, orient='index', columns=[names[2]])

    # Sort both DataFrames by index
    df1 = df1.sort_index()
    df2 = df2.sort_index()
    if obs == 'inter':
        df3 = df3.sort_index()

    # Combine the two DataFrames
    combined_df = pd.concat([df1, df2], axis=1)
    if obs == 'inter':
        combined_df = pd.concat([combined_df, df3], axis=1)

    # print('number of NaN values found:', combined_df.isna().sum().sum())

    # Sort index or reset index
    combined_df.index.name = 'filename'

    return combined_df



def angular_diff(a, b):
    """ Calculate the angular difference between two series of angles.  
    
    Parameters:
        a (np.ndarray): First series of angles.
        b (np.ndarray): Second series of angles.

    Returns:
        angular_diff (np.ndarray): Angular differences wrapped to the range [-180, 180]. """

    diff = a - b  # Compute raw difference
    angular_diff = np.remainder(diff + 180, 360) - 180  # Wrap difference to [-180, 180]
    return angular_diff



def calculate_differences(combined_df):
    """ Calculate angular differences between two columns in a DataFrame and add them as new columns.
    
    Parameters:
        combined_df (pd.DataFrame): DataFrame with two columns containing angles.
    
    Returns:
        combined_df (pd.DataFrame): DataFrame with additional columns for angular differences and absolute differences. """

    # Store all pairwise differences
    for (i, j) in combinations(combined_df.columns, 2):
        diff_col_name = f"diff_{i}_{j}"
        abs_diff_col_name = f"abs_diff_{i}_{j}"
        combined_df[diff_col_name] = angular_diff(combined_df[i], combined_df[j])
        combined_df[abs_diff_col_name] = combined_df[diff_col_name].abs()

    return combined_df




def describe_angular_series(df):
    """ Print descriptive statistics for angular differences in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing angular differences with columns prefixed by 'diff_' and 'abs_diff_'.
    
    Returns:
        None: This function prints descriptive statistics to the console. """
    
    abs_diff_cols = [col for col in df.columns if col.startswith("abs_diff")]

    for col in abs_diff_cols:
        diff = df[col]
        print(f"\n=== {col} ===")
        print(f"Median: {diff.median():.2f}°")
        iqr = np.percentile(diff, 75) - np.percentile(diff, 25)
        print(f"IQR: {iqr:.2f}°, 75th percentile: {np.percentile(diff, 75):.2f}°, 25th percentile: {np.percentile(diff, 25):.2f}°")
        print(f"Agreement ≤ 5°: {(diff <= 5).mean()*100:.1f}%")
        print(f"Agreement ≤ 10°: {(diff <= 10).mean()*100:.1f}%")



def bland_altman_plot(df, output_path, obs=''):
    """ Create a Bland-Altman plot to visualize the agreement between two sets of angular measurements.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing angular differences and annotations.
        output_path (str): Path to save the Bland-Altman plot.
        obs (str): Observer label for the plot title.
    
    Returns:
        None: This function saves the plot to the specified output path. """

    # Calculate means and differences
    # Calculate means and differences
    means = df.mean(axis=1)
    diffs = angular_diff(df.iloc[:, 0], df.iloc[:, 1])


    # Mean and limits of agreement
    mean_error = diffs.mean()
    std_error = diffs.std()
    upper_limit = mean_error + 1.96 * std_error
    lower_limit = mean_error - 1.96 * std_error

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(means, diffs, alpha=0.4, s=10)
    plt.axhline(mean_error, color='blue', linestyle='--', label=f"Mean: {mean_error:.2f}°")
    plt.axhline(upper_limit, color='red', linestyle='--', label=f"+1.96 SD: {upper_limit:.2f}°")
    plt.axhline(lower_limit, color='red', linestyle='--', label=f"-1.96 SD: {lower_limit:.2f}°")
    plt.xlabel('Mean of paired rotations (°)', fontsize=13)
    axis_title = f'Angular difference ({obs}) (°)'
    plt.ylabel(axis_title, fontsize=13)
    plt.legend()
    # plt.title('Bland-Altman Plot', fontsize=15)
    plt.grid(True)
    plt.ylim(-180, 180)
    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()
    plt.close()



def run_bland_altman_all_pairs(df, observer_cols, stain, base_output_path=''):
    """ Run Bland-Altman analysis for all pairs of observers in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing angular measurements from observers.
        observer_cols (list): List of column names corresponding to observers.
        stain (str): Stain type (e.g., 'HE', 'IHC').
        base_output_path (str): Base path to save the Bland-Altman plots.
    
    Returns:
        None: This function saves Bland-Altman plots for each pair of observers. """
    
    for i, j in combinations(observer_cols, 2):
        pair_df = df[[i, j]].copy()
        obs_label = f'{i} - {j}'
        output_file = f"{base_output_path}bland_altman_{stain}_{i}_{j}.png" if base_output_path else None
        bland_altman_plot(pair_df, output_path=output_file, obs=obs_label)


if __name__ == "__main__":

    ###### INTER-OBSERVER VARIABILITY ######
    # Get confuguration
    observer_1_files = [config.observer_1_1_HE, config.observer_1_1_IHC]
    observer_2_files = [config.observer_2_HE, config.observer_2_IHC]
    observer_3_files = [config.observer_3_HE, config.observer_3_IHC]
    observer_variability_path = config.observer_variability_path

    for stain in ['HE', 'IHC']:
        if stain == 'HE':
            filepaths = [observer_1_files[0], observer_2_files[0], observer_3_files[0]]
        else:
            filepaths = [observer_1_files[1], observer_2_files[1], observer_3_files[1]]
        
        eval_intra_observer_path = observer_variability_path + f'inter_observer_{stain}/'
        # Create the directory if it does not exist
        if not os.path.exists(eval_intra_observer_path):
            os.makedirs(eval_intra_observer_path)
        
        print(f"\nProcessing inter-observer variability... {stain}")
        df_obs123 = from_json_to_df(filepaths, obs='inter')
        df_obs123 = calculate_differences(df_obs123)
        describe_angular_series(df_obs123)
        observer_cols = df_obs123.columns[:3] 
        run_bland_altman_all_pairs(df_obs123, observer_cols, stain, base_output_path=eval_intra_observer_path)

    ###### INTRA-OBSERVER VARIABILITY ######
    # Get confuguration
    observer_1_HE_files = [config.observer_1_1_HE, config.observer_1_2_HE]
    observer_1_IHC_files = [config.observer_1_1_IHC, config.observer_1_2_IHC]
    observer_variability_path = config.observer_variability_path

    for stain in ['HE', 'IHC']:
        if stain == 'HE':
            filepaths = observer_1_HE_files
        else:
            filepaths = observer_1_IHC_files
        
        eval_intra_observer_path = observer_variability_path + f'intra_observer_{stain}/'
        # Create the directory if it does not exist
        if not os.path.exists(eval_intra_observer_path):
            os.makedirs(eval_intra_observer_path)
        
        print(f"\nProcessing intra-observer variability... {stain}")
        df_obs1 = from_json_to_df(filepaths, obs='intra')
        df_obs1 = calculate_differences(df_obs1)
        describe_angular_series(df_obs1)
        observer_cols = df_obs1.columns[:2] 
        run_bland_altman_all_pairs(df_obs1, observer_cols, stain, base_output_path=eval_intra_observer_path)
        
