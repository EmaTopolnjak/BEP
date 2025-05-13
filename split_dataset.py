import random
from collections import defaultdict, Counter
import json
from pathlib import Path

# set seed for reproducibility
random.seed(42)


def flatten_dict(id_to_patient):
    """ Flattens a nested dictionary where keys are strings with multiple idnrs separated by '+'
    and values are patient names.
        
    Parameters:
        id_to_patient (dict): A dictionary where keys are strings of idnrs and values are patient names.
    
    Returns:
        dict: A flattened dictionary where each idnr is a key and the corresponding value is the patient name. """ 

    flat_id_to_patient = {}
    for key, patient in id_to_patient.items():
        idnrs = key.split("+")  # split multi-idnr keys
        for idnr in idnrs:
            flat_id_to_patient[idnr] = patient
    return flat_id_to_patient



def build_stain_patient_counts(filepaths_IHC, flat_id_to_patient):
    """ Groups images by IHC stain and patient via idnr → patient.
    
    Parameters:
        filepaths_IHC (list): List of file paths for IHC images.
        flat_id_to_patient (dict): Dictionary mapping idnr to patient.
    
    Returns:
        stain_patient_counts (dict): Dictionary with stains as keys and dictionaries of patients and their image counts as values.
        patient_image_counts (dict): Dictionary with patients as keys and their total image counts as values. """
    
    # Group images by stain and patient via idnr → patient
    stain_patient_counts = defaultdict(lambda: defaultdict(int)) # dict. containing stain: {patient: nr_images, ...}
    patient_image_counts = defaultdict(int) # dict. containing patient: nr_images

    for file in filepaths_IHC:
        parts = file.split("_")
        stain = parts[2]
        idnr = parts[3] + '_' + parts[4]
        patient = flat_id_to_patient.get(idnr)
        if patient:
            stain_patient_counts[stain][patient] += 1
            patient_image_counts[patient] += 1
        else:
            print(f"Warning: for ID {idnr}, no patient is found in patient_mapping. Skipping this file.")

    return stain_patient_counts, patient_image_counts  



def patient_overlap_across_stains(stain_patient_counts, print_results=False):
    """ Check for patients that are in multiple stains. 
    
    Parameters:
        stain_patient_counts (dict): Dictionary with stains as keys and dictionaries of patients and their image counts as values.
        print_results (bool): If True, prints which patients are in multiple stains and which stains they are associated with.
    
    Returns:
        patient_to_stains (dict): Dictionary with patients as keys and sets of stains they are associated with as values. """
    
    # Check for patients that are in multiple stains
    patient_to_stains = defaultdict(set) # dict. containing patient: {stain1, stain2, ...}
    for stain, patient_counts in stain_patient_counts.items():
        for patient in patient_counts:
            patient_to_stains[patient].add(stain)

    # Filter for patients that are in multiple stains
    overlapping_patients = {patient: stains for patient, stains in patient_to_stains.items() if len(stains) > 1} # dict. containing patient: {stain1, stain2, ...} for patients in multiple stains

    # Print the results
    if print_results:
        if overlapping_patients:
            print(f"Found {len(overlapping_patients)} patients in multiple stains.")
            for patient, stains in overlapping_patients.items():
                print(f"  {patient}: {sorted(stains)}")
        else:
            print("No patients shared across stains.")

    return patient_to_stains 



def split_dataset(flat_id_to_patient, stain_patient_counts, patient_to_stains, val_ratio=0.2, target_per_stain_test=47):
    """ Splits the dataset into train, validation, and test sets based on the specified ratios and target number of images per stain in the test set.

    Parameters:
        flat_id_to_patient (dict): Dictionary mapping idnr to patient.
        stain_patient_counts (dict): Dictionary with stains as keys and dictionaries of patients and their image counts as values.
        patient_to_stains (dict): Dictionary with patients as keys and sets of stains they are associated with as values.
        val_ratio (float): Ratio of patients to be assigned to the validation set.
        target_per_stain_test (int): Target number of images per stain in the test set.

    Returns:
        assigned_patients (dict): Dictionary with patients as keys and their assigned set (train/val/test) as values. """

    # Step 1: Shuffle patients 
    all_patients = list(patient_to_stains.keys())
    random.shuffle(all_patients)

    # Keep track of splitting
    assigned_patients = {} # dict. containing patient: train/val/test
    remaining_patients = set(all_patients) # patients that are not assigned yet

    # Step 2: test set filling
    stain_test_counts = Counter() # dict. containing stain: nr_image_in_test_set
    for stain, patient_counts in stain_patient_counts.items():
        while stain_test_counts[stain] < target_per_stain_test:
            candidates = [(patient, patient_counts[patient]) for patient in patient_counts if patient in remaining_patients]
            if not candidates:
                break  # No unassigned candidates left for this stain
            
            images_needed = target_per_stain_test - stain_test_counts[stain]

            # if more than 2/3 full, use closest-fit strategy
            if stain_test_counts[stain] >= (2/3) * target_per_stain_test:
                # Closest-fit: minimize overshoot
                best_patient, _ = min(candidates, key=lambda x: abs(images_needed - x[1])) 
            else:
                # Encourage more patients: choose randomly from the candidates
                best_patient, _ = random.choice(candidates) # or minimal to encourage more patients? --> min(candidates, key=lambda x: x[1])
                        
            # Assign to test set
            assigned_patients[best_patient] = "test"
            remaining_patients.remove(best_patient)

            # Update test counts for all stains this patient touches because they are all in the test set
            for s in patient_to_stains[best_patient]:
                stain_test_counts[s] += stain_patient_counts[s].get(best_patient, 0)

    # Step 3: Assign remaining patients to train and validation sets
    val_cutoff = int(len(remaining_patients) * val_ratio)
    val_patients = list(remaining_patients)[:val_cutoff]
    train_patients = list(remaining_patients)[val_cutoff:]

    for p in val_patients:
        assigned_patients[p] = "val"
    for p in train_patients:
        assigned_patients[p] = "train"

    # Step 4: Assign remainging patients to train set (patients that are not in IHC images but in HE images)
    for patient in set(flat_id_to_patient.values()):
        if patient not in assigned_patients:
            assigned_patients[patient] = "train"

    return assigned_patients



def analyze_split(assigned_patients, patient_image_counts_IHC, patient_image_counts_HE, stain_patient_counts_IHC):
    """ Analyzes the split of the dataset into train, validation, and test sets.
    
    Parameters:
        assigned_patients (dict): Dictionary with patients as keys and their assigned set (train/val/test) as values.
        patient_image_counts_IHC (dict): Dictionary with patients as keys and their total image counts for IHC images as values.
        patient_image_counts_HE (dict): Dictionary with patients as keys and their total image counts for HE images as values.
        stain_patient_counts_IHC (dict): Dictionary with stains as keys and dictionaries of patients and their image counts as values.
        
    Returns:
        None. The function prints the analysis results. """
    
    # Look at how many images there are in each set and how many patients are in the test set per stain
    set_image_counts_IHC = {"train": 0, "val": 0, "test": 0}
    set_image_counts_HE = {"train": 0, "val": 0, "test": 0}
    test_stain_distribution = Counter()
    stain_to_test_patients = defaultdict(set)

    for patient, subset in assigned_patients.items():
        set_image_counts_IHC[subset] += patient_image_counts_IHC[patient]
        set_image_counts_HE[subset] += patient_image_counts_HE[patient]
        if subset == "test":
            for stain in patient_to_stains_IHC[patient]:
                test_stain_distribution[stain] += stain_patient_counts_IHC[stain].get(patient, 0)
                stain_to_test_patients[stain].add(patient)

    print("Patient-to-set assignment:")
    total_patients = len(assigned_patients)
    for subset in ["train", "val", "test"]:
        patients_in_set = [p for p, s in assigned_patients.items() if s == subset]
        print(f"  {subset}: {len(patients_in_set)} patients, {len(patients_in_set)/total_patients*100:.2f}% of total patients") 

    print("\nImage-to-set assignment for IHC images:")
    total_images = sum(set_image_counts_IHC.values())
    for subset in ["train", "val", "test"]:
        print(f"  {subset}: {set_image_counts_IHC[subset]} images, {set_image_counts_IHC[subset]/total_images*100:.2f}% of total patients")
    
    print("\nImage-to-set assignment for HE images:")
    total_images = sum(set_image_counts_HE.values())
    for subset in ["train", "val", "test"]:
        print(f"  {subset}: {set_image_counts_HE[subset]} images, {set_image_counts_HE[subset]/total_images*100:.2f}% of total patients")

    print("\nImage-to-stain assignment in test set:")
    for stain, count in test_stain_distribution.items():
        print(f"  {stain}: {count} images")

    print("\nPatient-to-stain assignment in test set:")
    for stain, patient_set in stain_to_test_patients.items():
        print(f"  {stain}: {len(patient_set)} patients")



if __name__ == "__main__":

    # VARIABLES     
    patient_mapping_path = '../../Data/splitting_data/patient_mapping.json'
    files_HE = '../../Data/images/HE_crops_masked_rotated'
    files_IHC = '../../Data/images/IHC_crops_masked_rotated'
    assigned_split_path = '../../Data/splitting_data/assigned_split.json'
    
    # Load the patient to ID mapping
    with open(patient_mapping_path, "r") as f:
        patient_mapping = json.load(f)

    # Get all images
    files_HE = Path(files_HE)
    files_IHC = Path(files_IHC)
    filepaths_HE = [f.name for f in files_HE.iterdir() if f.is_file()]
    filepaths_IHC = [f.name for f in files_IHC.iterdir() if f.is_file()]
    
    # Get mapping of idnr to patient
    flat_id_to_patient = flatten_dict(patient_mapping)

    # Prepare mappings so so we can split per patient 
    stain_patient_counts_IHC, patient_image_counts_IHC  = build_stain_patient_counts(filepaths_IHC, flat_id_to_patient)
    _, patient_image_counts_HE  = build_stain_patient_counts(filepaths_HE, flat_id_to_patient)
    patient_to_stains_IHC = patient_overlap_across_stains(stain_patient_counts_IHC)
    
    # Split the dataset
    assigned_split = split_dataset(flat_id_to_patient, stain_patient_counts_IHC, patient_to_stains_IHC, val_ratio=0.2, target_per_stain_test=47)

    # Analyze the split
    analyze_split(assigned_split, patient_image_counts_IHC, patient_image_counts_HE, stain_patient_counts_IHC)

    # Save the assigned split to a JSON file
    with open(assigned_split_path, "w") as f:
        json.dump(assigned_split, f, indent=4)
