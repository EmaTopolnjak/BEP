import json
from collections import Counter, defaultdict
import statistics

import sys
from pathlib import Path
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



def analyze_skipped_entries(data):
    """ Analyzes the skipped entries in the given data.

    Parameters:
    data (dict): The data to analyze. The data should be a dictionary where each key is an image filename and the value is a dictionary
                 containing rotation information and possibly a "skipped" key.

    Returns:
    None. The function prints the number of images before rotating, the number of skipped images, and a breakdown of the reasons for skipping. """

    # Total number of keys
    total_keys = len(data)

    # Count skipped entries and reasons
    skipped_count = 0
    skipped_reasons = Counter()

    for key, value in data.items():
        if isinstance(value, dict) and "skipped" in value:
            skipped_count += 1
            reason = value["skipped"]
            skipped_reasons[reason] += 1

    # Output results
    print(f"Number images before rotating: {total_keys}")
    print(f"Number of skipped images: {skipped_count}")
    print("Skipped reasons breakdown:")
    for reason, count in skipped_reasons.items():
        print(f"  {reason}: {count}")

    

def analyze_number_slices_left(data, stain, details=False):
    """ Analyzes the number of slices left in the data. 

    Parameters:
    data (dict): The data to analyze. The data should be a dictionary where each key is an image filename and the value is a dictionary
                 containing rotation information and possibly a "skipped" key.
    stain (str): The type of stain to analyze. It can be "HE" or "IHC".
    details (bool): If True, prints detailed information about the number of slices left per stain. Default is False.

    Returns:
    None. The function prints the number of slices left. """

    if stain == "HE":
        counts = 0
        for key, value in data.items():
            if not (isinstance(value, dict) and "skipped" in value):
                counts += 1
        print("Number of images after rotating:", counts)

    if stain == "IHC":
        stain_counts = defaultdict(int)
        for key, value in data.items():
            if not (isinstance(value, dict) and "skipped" in value):
                parts = key.split("_")
                if len(parts) > 1:
                    stain = parts[2]
                    stain_counts[stain] += 1

        ihc_values = list(stain_counts.values())

        print("Number of images after rotating:", sum(ihc_values))
        print("  median per stain:", statistics.median(ihc_values))
        print("  min per stain:", min(ihc_values))
        print("  max per stain:", max(ihc_values))
        print("  number of stain:", len(ihc_values))

        if details:
            print("  details per stain:")
            for stain, count in stain_counts.items():
                print(f"    {stain}: {count}")



def get_removed_cases(data):
    """ Analyzes the removed cases in the given data. 

    Parameters:
    data (dict): The data to analyze. The data should be a dictionary where each key is an image filename and the value is a dictionary
                 containing rotation information and possibly a "skipped" key.

    Returns:
    None. The function prints the number of completely removed cases. """

    grouped = defaultdict(list)

    # Group all keys by the number prefix before the first underscore
    for key, value in data.items():
        prefix = key.split("_")[0]
        grouped[prefix].append(value)

    fully_removed = []

    # Check if all entries in each group are skipped
    for prefix, items in grouped.items():
        if all(isinstance(item, dict) and 'skipped' in item for item in items):
            fully_removed.append(prefix)

    total_cases = len(grouped)
    remaining_cases = total_cases - len(fully_removed)

    print(f"Number of removed cases: {len(fully_removed)}")
    print(f"Number of remaining cases: {remaining_cases}")



if __name__ == "__main__":
    # Load the configuration
    stains = ['HE', 'IHC']
    HE_ground_truth_rotations = config.HE_ground_truth_rotations
    IHC_ground_truth_rotations = config.IHC_ground_truth_rotations

    rotation_info_paths = [HE_ground_truth_rotations, IHC_ground_truth_rotations]

    for stain, rotations_path in zip(stains, rotation_info_paths):
        # Load JSON file
        with open(rotations_path, "r") as f:
            data = json.load(f)
        
        print(f"Analyzing {stain} data...")	

        # Analyze skipped entries
        analyze_skipped_entries(data)
                
        # Analyze number of slices left (per stain)
        analyze_number_slices_left(data, stain, details=True)

        # Analyze removed cases
        get_removed_cases(data)

        print("\n")	