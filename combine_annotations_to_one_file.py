import json

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
    # List of file paths to the JSON files to be combined
    file_paths = ['data/rotations_IHC/image_rotations_IHC_pt1.json', 'data/rotations_IHC/image_rotations_IHC_pt2.json', 'data/rotations_IHC/image_rotations_IHC_pt3.json']
    output_path = 'data/rotations_IHC/image_rotations_IHC.json'

    combine_information_from_json_files(file_paths, output_path)
    print("Combined JSON file created successfully.")