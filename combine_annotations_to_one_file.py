import json
import config

def combine_information_from_json_files(file_paths, output_path):
    """
    Combine information from multiple JSON files into a single JSON file.
    Each file contains a dictionary where the keys are image paths and the values are rotation angles.
    The combined file will have the structure: {filename: rotation}.

    Parameters:
    file_paths (list): List of paths to the JSON files to be combined.
    output_path (str): Path to save the combined JSON file. 
    
    Returns:
    None. The combined data is saved to 'data/image_rotations.json'.
    """      

    # Initialize an empty dictionary to hold the combined data (structure: filename -> rotation)
    combined_data = {}

    # Loop through and load each file
    for path in file_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            
            # Transform the keys to just filenames
            for full_path, value in data.items():
                filename = full_path.split('/')[-1]  # Get only the last part of the path (i.e. image filename)
                
                # Check for duplicates
                if filename in combined_data:
                    print(f"Duplicate found: '{filename}'")
                
                combined_data[filename] = value

    combined_data = dict(sorted(combined_data.items())) # Sort the dictionary by filename

    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)



if __name__ == "__main__":
    # Load the configuration
    filepaths_HE = config.HE_ground_truth_rotations_seperate
    output_path_HE = config.HE_ground_truth_rotations
    filepaths_IHC = config.IHC_ground_truth_rotations_seperate
    output_path_IHC = config.IHC_ground_truth_rotations

    filepaths = [filepaths_HE, filepaths_IHC]
    output_path = [output_path_HE, output_path_IHC]

    for filepaths, output_path in zip(filepaths, output_path):
        # Combine the JSON files
        combine_information_from_json_files(filepaths, output_path)
        print(f"Combined JSON file created successfully at {output_path}.")
