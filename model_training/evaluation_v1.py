from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

import sys
from model_training_v4 import ImageDataset, initialize_model

from pathlib import Path
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



def apply_model_on_test_set(model, test_loader):
    """ Apply the trained model on the test set and return the predictions and labels.

    Parameters:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test set.
    Returns:
        all_labels (np.ndarray): Array of labels.
        all_preds (np.ndarray): Array of predictions.
         
    Note: This function is currently set to only process 5 batches for testing. """
    
    model.eval()
    all_labels = []
    all_preds = []

    counter = 0

    with torch.no_grad():
        for image, label, pos in test_loader:
            # TEMPORARY ONLY 5 BATCHES
            if counter >= 5: 
                break
            counter += 1

            image = image.to(device)
            label = label.to(device).float().view(-1)

            output = model(image, pos).view(-1)
            all_labels.append(label.cpu().numpy())
            all_preds.append(output.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    return all_labels, all_preds



def model_evaluation(true_labels, all_preds):
    """ Evaluate the model performance using Mean Absolute Error (MAE), Median Absolute Error (MedAE), Mean Squared Error (MSE) and accuracy within 5 
    degrees, all for angluar error. The predictions and labels are expected to be in degrees. The function also plots the predictions vs labels and the 
    histogram of angular differences.

    Parameters:
        true_labels (np.ndarray): Array of true labels.
        all_preds (np.ndarray): Array of predictions.
    Returns:
        None. Prints the evaluation metrics.

    Note: The labels and predictions are expected to be in degrees.
    Note: Choose correct metrics to evaluate the model on later. For now, random metrics were chosen to see if the model is working. """

    diff = true_labels - all_preds  # Compute raw difference 
    wrapped_diff = np.remainder(diff + 180, 360) - 180 # wrap difference to [-180, 180]

    # Metrics
    mae = np.mean(np.abs(wrapped_diff))
    medae = np.median(np.abs(wrapped_diff))
    mse = np.mean(wrapped_diff ** 2)
    acc_5 = np.mean(np.abs(wrapped_diff) <= 5)  # Accuracy within 5 degrees

    # Print the metrics
    print(f"MAE: {mae:.4f} degrees")
    print(f"Median MAE: {medae:.4f} degrees")
    print(f"MSE: {mse:.4f} degrees")
    print(f"Accuracy within 5 degrees: {acc_5:.4f}")

    # Plot the predictions vs labels
    plt.figure()
    plt.scatter(true_labels, all_preds, alpha=0.6)
    plt.plot([0, 360], [0, 360], '--', color='gray')  # y=x line
    plt.xlabel("Ground Truth (°)")
    plt.ylabel("Prediction (°)")
    plt.title("Prediction vs Ground Truth")
    plt.grid(True)
    plt.show()

    # Plote the histogram of angluar differencesd
    plt.figure()
    plt.hist(wrapped_diff, bins=50, edgecolor='black')
    plt.xlabel("Angular Error (degrees)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Angular Errors")
    plt.show()
    



if __name__ == "__main__":
    
    # Load the configurations
    HE_ground_truth_rotations = config.HE_ground_truth_rotations
    HE_images_for_model = config.HE_crops_masked_rotated
    HE_masks_for_model = config.HE_masks_rotated

    PRETRAINED_MODEL = config.pretrained_model
    TRAINED_MODEL_PATH = config.trained_model_path
    RANDOM_SEED = config.random_seed

    # Set the random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load the images
    # Load the images
    HE_images_train = HE_images_for_model + '/train'
    image_paths_train = [os.path.join(HE_images_train, fname) for fname in os.listdir(HE_images_train) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    HE_masks_train = HE_masks_for_model + '/train'
    mask_paths_train = [os.path.join(HE_masks_train, fname) for fname in os.listdir(HE_masks_train) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

    HE_images_val = HE_images_for_model + '/val'
    image_paths_val = [os.path.join(HE_images_val, fname) for fname in os.listdir(HE_images_val) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    HE_masks_val = HE_masks_for_model + '/val'
    mask_paths_val = [os.path.join(HE_masks_val, fname) for fname in os.listdir(HE_masks_val) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Load the labels
    with open(HE_ground_truth_rotations, 'r') as f:
        label_dict = json.load(f)
    labels_train = [label_dict[os.path.basename(path)] for path in image_paths_train]
    labels_val = [label_dict[os.path.basename(path)] for path in image_paths_val]

    # Create the dataset and dataloader
    train_data = ImageDataset(image_paths=image_paths_train, mask_paths=mask_paths_train, labels=labels_train)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False) 
    val_data = ImageDataset(image_paths=image_paths_val, mask_paths=mask_paths_val, labels=labels_val)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = initialize_model(pretrained_weights_path=PRETRAINED_MODEL)

    # Load saved weights
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH))

    # Model evaluation
    test_labels, test_pred = apply_model_on_test_set(model, val_loader)
    model_evaluation(test_labels, test_pred)