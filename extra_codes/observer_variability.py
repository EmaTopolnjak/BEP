import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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



def from_json_to_df(filepaths):
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

    # Convert the dictionaries to DataFrames
    df1 = pd.DataFrame.from_dict({get_filename(k): v for k, v in data1.items()}, orient='index', columns=[os.path.basename(filepaths[0])])
    df2 = pd.DataFrame.from_dict({get_filename(k): v for k, v in data2.items()}, orient='index', columns=[os.path.basename(filepaths[1])])

    # Sort both DataFrames by index
    df1 = df1.sort_index()
    df2 = df2.sort_index()

    # Combine the two DataFrames
    combined_df = pd.concat([df1, df2], axis=1)

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

    # Angular differences
    combined_df['diffs'] = angular_diff(combined_df.iloc[:, 0], combined_df.iloc[:, 1])

    # Absolute angular differences
    combined_df['abs_diffs'] = angular_diff(combined_df.iloc[:, 0], combined_df.iloc[:, 1]).abs()

    return combined_df





def describe_angular_series(df):
    """ Print descriptive statistics of the angular differences in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing angular differences in a column named 'abs_diffs'.

    Returns:
        None: This function prints the statistics directly. """
    
    diff = df['abs_diffs']
    print(f"Median: {diff.median():.2f}°")
    print(f"IQR: {(np.percentile(diff, 75) - np.percentile(diff, 25)):.2f}°, 75th percentile: {np.percentile(diff, 75):.2f}°, 25th percentile: {np.percentile(df, 25):.2f}°")
    print(f"Agreement ≤ 5°: {(diff <= 5).mean()*100:.1f}%")
    print(f"Agreement ≤ 10°: {(diff <= 10).mean()*100:.1f}%")



def bland_altman_plot(df, output_path):
    """ Create a Bland-Altman plot to visualize the agreement between two sets of angular measurements.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing angular differences and annotations.
        output_path (str): Path to save the Bland-Altman plot.
    
    Returns:
        None: This function saves the plot to the specified output path. """

    # Calculate means and differences
    means = df.mean(axis=1)
    diffs = df['diffs']

    # Mean and limits of agreement
    mean_error = diffs.mean()
    std_error = diffs.std()
    upper_limit = mean_error + 1.96 * std_error
    lower_limit = mean_error - 1.96 * std_error

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(means, diffs, alpha=0.6)
    plt.axhline(mean_error, color='blue', linestyle='--', label=f"Mean: {mean_error:.2f}°")
    plt.axhline(upper_limit, color='red', linestyle='--', label=f"+1.96 SD: {upper_limit:.2f}°")
    plt.axhline(lower_limit, color='red', linestyle='--', label=f"-1.96 SD: {lower_limit:.2f}°")
    plt.xlabel('Mean of Annotations (degrees)')
    plt.ylabel('Augular difference (Annot1 - Annot2)')
    plt.title('Bland-Altman Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'bland_altman_plot_intra_observer.png'))
    # plt.show()
    plt.close()




if __name__ == "__main__":

    ###### INTRA-OBSERVER VARIABILITY ######
    # Get confuguration
    observer_1_files = [config.observer_1_1_HE, config.observer_1_1_IHC]
    observer_2_files = [config.observer_2_HE, config.observer_2_IHC]
    observer_variability_path = config.observer_variability_path

    for stain in ['HE', 'IHC']:
        if stain == 'HE':
            filepaths = [observer_1_files[0], observer_2_files[0]]
        else:
            filepaths = [observer_1_files[1], observer_2_files[1]]
        
        eval_intra_observer_path = observer_variability_path + f'inter_observer_{stain}/'
        # Create the directory if it does not exist
        if not os.path.exists(eval_intra_observer_path):
            os.makedirs(eval_intra_observer_path)
        
        print(f"\nProcessing inter-observer variability... {stain}")
        df_obs1_2 = from_json_to_df(filepaths)
        df_obs1_2 = calculate_differences(df_obs1_2)
        describe_angular_series(df_obs1_2)
        bland_altman_plot(df_obs1_2, eval_intra_observer_path)



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
        df_obs1 = from_json_to_df(filepaths)
        df_obs1 = calculate_differences(df_obs1)
        describe_angular_series(df_obs1)
        bland_altman_plot(df_obs1, eval_intra_observer_path)
        
