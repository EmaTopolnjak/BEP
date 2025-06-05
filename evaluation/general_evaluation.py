# NOTE: ...

import os
import sys
import math
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from PIL import Image

# Codes from other folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_training.model_training_loop import ImageDataset, initialize_model, filter_by_rotated_size_threshold
from preprocessing.rotate_images_correctly import rotate_image_with_correct_padding

# To import the config file from the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



def extract_datasets(img_path, filter_large_images):
    """ Extract the datasets for training, validation and test sets. The images and masks are filtered by size to ensure that they are small enough
    to not let the model run out of memory. The maximum number of pixels is set to 2.700.000, which is the maximum size of the image after rotation.

    Parameters:
        img_path_ (str): Path to the folder containing the images. The folder should contain subfolders 'train', 'val' and 'test'.
    filter_large_images (function): Function to filter the images based on size. The function should take a list of image paths and return a list of

    Returns:
        image_paths_train (list): List of filenames for the training set.
        image_paths_val (list): List of filenames for the validation set. """

    results = {}

    for subset in ['val']:
        img_dir  = os.path.join(img_path, subset)
        filenames = sorted([fname for fname in os.listdir(img_dir) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))])

        # Build full paths to apply filtering
        image_paths = [os.path.join(img_dir, fname) for fname in filenames]

        # Filter images based on size
        image_paths_filtered, suppressed = filter_large_images(image_paths)
        filtered_filenames = [os.path.basename(p) for p in image_paths_filtered]

        results[subset] = filtered_filenames
        print(f"Number of {subset} images suppressed: {suppressed}\n")

    return results['val']



def get_filenames_and_labels(images_path, ground_truth_rotations):
    # Load the images and masks for training and validation sets and filter them by size 
    filenames_test = extract_datasets(
        images_path,
        lambda img_p: filter_by_rotated_size_threshold(img_p)
    )

    # TEMPORARY: Limit the number images
    filenames_test = filenames_test[:30]

    # Load the labels
    with open(ground_truth_rotations, 'r') as f:
        label_dict = json.load(f)
    labels_test = [label_dict[fname] for fname in filenames_test]
    
    return filenames_test, labels_test




def apply_model_on_test_set(model, test_loader, device):
    """ Apply the trained model on the test set and return the predictions and labels.

    Parameters:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test set.
    Returns:
        all_labels (np.ndarray): Array of labels.
        all_preds (np.ndarray): Array of predictions.
         
    Note: This function is currently set to only process 5 batches for testing. """
    
    model.eval()
    true_labels = []
    preds = []

    with torch.no_grad():
        for image, label, pos in test_loader:           
            image = image.to(device)
            label = label.to(device).float().view(-1)
            
            # Predict the angle of rotation
            output = model(image, pos).view(-1)

            # Convert the label and output to degrees (0-360)
            cos_label, sin_label = label.squeeze(0) 
            angle_rad_label = torch.atan2(sin_label, cos_label)
            angle_deg_label = angle_rad_label * 180.0 / math.pi % 360
            true_labels.append(angle_deg_label.cpu().numpy())

            cos_pred, sin_pred = output
            angle_rad_pred = torch.atan2(sin_pred, cos_pred)
            angle_deg_pred = angle_rad_pred * 180.0 / math.pi % 360
            preds.append(angle_deg_pred.cpu().numpy())

    true_labels = np.stack(true_labels)
    preds = np.stack(preds)

    return true_labels, preds



def calculate_error(true_labels, preds):
    """ Calculate the angular difference between true labels and predictions. The difference is wrapped to the range [-180, 180] degrees.
    
    Parameters:
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.
    
    Returns:
        angluar_diff (np.ndarray): Array of angular differences wrapped to [-180, 180] degrees.
        abs_angluar_diff (np.ndarray): Array of absolute angular differences. """
    
    diff = true_labels - preds  # Compute raw difference
    angluar_diff = np.remainder(diff + 180, 360) - 180  # wrap difference to [-180, 180]
    abs_angluar_diff = np.abs(angluar_diff) # Convert to absolute angular difference

    return angluar_diff, abs_angluar_diff


def get_error_metrics(true_labels, preds):
    """ Calculate and print the error metrics for the predictions.
   
    Parameters:
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.
    
    Returns:
        None. Prints the error metrics (include mean absolute error, median absolute error, mean squared error, 
        and accuracy within 5 and 10 degrees) to the console. """
    
    angluar_diff, abs_angular_diff = calculate_error(true_labels, preds)

    # Metrics
    mae = np.mean(abs_angular_diff)
    medae = np.median(abs_angular_diff)
    mse = np.mean(angluar_diff ** 2)
    acc_5 = np.mean(abs_angular_diff <= 5) # Accuracy within 5 degrees
    acc_10 = np.mean(abs_angular_diff <= 10) # Accuracy within 10 degrees

    # Print the metrics
    print(f"Mean abs error: {mae:.4f} deg") 
    print(f"Median abs error: {medae:.4f} deg") 
    print(f"Mean squared error: {mse:.4f} deg") 
    print(f"Accuracy within 5 deg: {acc_5:.4f}") 
    print(f"Accuracy within 10 deg: {acc_10:.4f}")



def visualize_predictions_vs_labels(true_labels, preds, evaluation_plots_path):
    """ Visualize the predictions vs labels and the distribution of angular differences.
    Parameters:
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.
        evaluation_plots_path (str): Path to the folder where the evaluation plots will be saved.
    Returns:
        None. Plots are saved to the evaluation_plots_path. """
       
    angluar_diff, _ = calculate_error(true_labels, preds)

    # Plot the predictions vs labels
    plt.figure()
    plt.scatter(true_labels, preds, s=10, alpha=0.6)
    plt.plot([0, 360], [0, 360], '--', color='gray')  # y=x line
    plt.xlabel("Ground Truth (°)")
    plt.ylabel("Prediction (°)")
    plt.title("Prediction vs Ground Truth")
    plt.grid(True)
    path = evaluation_plots_path + 'predictions_vs_labels.pdf'
    plt.savefig(path)
    # plt.show()

    # Plot the histogram of angular differences
    plt.figure()
    plt.hist(angluar_diff, bins=30, edgecolor='black')
    plt.xlabel("Absolute Angular Error (°)")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.grid(True)
    path = evaluation_plots_path + 'histogram_angluar_differences.pdf'
    plt.savefig(path)
    # plt.show()

    # Error bias plot
    mean_error = np.mean(angluar_diff)
    std_error = np.std(angluar_diff)
    upper_limit = mean_error + 1.96 * std_error
    lower_limit = mean_error - 1.96 * std_error

    plt.figure()
    plt.scatter(true_labels, angluar_diff, alpha=0.5, s=10)
    plt.axhline(mean_error, color='blue', linestyle='--', label=f"Mean: {mean_error:.2f}°")
    plt.axhline(upper_limit, color='red', linestyle='--', label=f"+1.96 SD: {upper_limit:.2f}°")
    plt.axhline(lower_limit, color='red', linestyle='--', label=f"-1.96 SD: {lower_limit:.2f}°")
    plt.xlabel("Ground Truth")
    plt.ylabel("Angular Difference (°)")
    plt.title('Errors vs Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = evaluation_plots_path + 'error_bias_plot.pdf'
    plt.savefig(path)
    # plt.show()



def get_filenames_based_on_error_percentiles(true_labels, preds, filenames):
    """ Get the filenames of the images that are at the 1st, 50th and 99th percentile of absolute angular differences.

    Parameters:
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.
        filenames (list): List of image filenames.
    Returns:
        filenames_selected (list): List of filenames of the images that are at the 1st, 50th and 99th percentile of absolute angular differences.
        true_labels_selected (list): List of true labels of the images that are at the 1st, 50th and 99th percentile of absolute angular differences.
        preds_selected (list): List of predictions of the images that are at the 1st, 50th and 99th percentile of absolute angular differences. """

    _, abs_angluar_diff = calculate_error(true_labels, preds)

    percentiles = [1, 50, 99]

    thresholds = np.percentile(abs_angluar_diff, percentiles)
    selected_indices = [np.argmin(np.abs(abs_angluar_diff - t)) for t in thresholds]

    # Get the filenames based on the indices
    filenames_selected = [filenames[i] for i in selected_indices]
    true_labels_selected = [true_labels[i] for i in selected_indices]
    preds_selected = [preds[i] for i in selected_indices]

    return filenames_selected, true_labels_selected, preds_selected



def plot_predictions_based_on_percentile(image_path, mask_path, filenames, true_labels, preds, evaluation_plots_path):
    """ Visualize the errors with the images and masks. The images are displayed with the predicted angle of rotation and the true angle of rotation.
    
    Parameters:
        image_path (str): Path to the folder containing the images.
        mask_path (str): Path to the folder containing the masks.
        filenames (list): List of filenames of the images.
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.
        evaluation_plots_path (str): Path to the folder where the evaluation plots will be saved.
    Returns:
        None. Plots are saved to the evaluation_plots_path. """

    # Get the filenames, true labels and predictions for the 1st, 50th and 99th percentile of absolute angular differences
    filenames_selected, true_labels_selected, preds_selected = get_filenames_based_on_error_percentiles(true_labels, preds, filenames)

    fig, ax = plt.subplots(3, 3, figsize=(10, 9))

    for i, filename in enumerate(filenames_selected):
        img = Image.open(os.path.join(image_path, 'val', filename)).convert("RGB")
        mask = Image.open(os.path.join(mask_path, 'val', filename)).convert("L")

        img, _ = rotate_image_with_correct_padding(img, mask, 0, bg_color=(255, 255, 255, 255))
        predicted_rotated_img, _ = rotate_image_with_correct_padding(img, mask, preds_selected[i], bg_color=(255, 255, 255, 255))
        correctly_rotated_img, _ = rotate_image_with_correct_padding(img, mask, true_labels_selected[i], bg_color=(255, 255, 255, 255))

        ax[0, i].imshow(img)
        ax[1, i].imshow(predicted_rotated_img)
        ax[2, i].imshow(correctly_rotated_img)

    # Turn off the axes for all subplots
    for row in ax:
        for col in row:
            col.axis('off')

    # Set row labels on the left-most column only
    row_labels = ['Original image', 'Predicted orientation', 'Manual orientation']
    for row_idx, label in enumerate(row_labels):
        fig.text(0.03, # x-position (adjust if needed)
            0.97 - (row_idx + 0.5) / 3, # y-position centered in each row
            label, ha='left', va='center', rotation='vertical', fontsize=12)
    
    col_labels = ['1st percentile', '50th percentile', '99th percentile']
    for col_idx, label in enumerate(col_labels):
        fig.text(0.22 + 0.31 * col_idx, # x-position (tune these based on 3 columns)
            0.96, # y-position (same for all columns)
            label, ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, top=0.95) # Adjust margin for text labels
    path = evaluation_plots_path + 'qualitative_analysis_errors.pdf'
    fig.savefig(path)
    # plt.show()



def det_top10_worst_predictions(true_labels, preds, filenames):

    angular_diff, abs_angluar_diff = calculate_error(true_labels, preds)

    # Get indices of top 10 worst errors
    worst_indices = np.argsort(abs_angluar_diff)[-10:][::-1]  # descending order

    # Get the worst samples
    filenames_worst = [filenames[i] for i in worst_indices]
    true_labels_worst = [true_labels[i] for i in worst_indices]
    preds_worst = [preds[i] for i in worst_indices]
    errors_worst = [angular_diff[i] for i in worst_indices] 

    return filenames_worst, true_labels_worst, preds_worst, errors_worst



def plot_top10_worst_predictions(image_path, mask_path, filenames, true_labels, preds, evaluation_plots_path):
    
    filenames_worst, true_labels_worst, preds_worst, errors_worst = det_top10_worst_predictions(true_labels, preds, filenames)

    fig, ax = plt.subplots(3, 10, figsize=(20, 6))  # 3 rows × 10 columns

    # Turn off the axes for all subplots
    for row in ax:
        for col in row:
            col.axis('off')
            col.set_aspect('equal')      # same width and height

    for i, filename in enumerate(filenames_worst):
        img = Image.open(os.path.join(image_path, 'val', filename)).convert("RGB")
        mask = Image.open(os.path.join(mask_path, 'val', filename)).convert("L")

        img, _ = rotate_image_with_correct_padding(img, mask, 0, bg_color=(255, 255, 255, 255))
        predicted_rotated_img, _ = rotate_image_with_correct_padding(img, mask, preds_worst[i], bg_color=(255, 255, 255, 255))
        correctly_rotated_img, _ = rotate_image_with_correct_padding(img, mask, true_labels_worst[i], bg_color=(255, 255, 255, 255))

        ax[0, i].imshow(img)
        ax[1, i].imshow(predicted_rotated_img)
        ax[2, i].imshow(correctly_rotated_img)
    
    # Set row labels on the left-most column only
    row_labels = ['Original image', 'Predicted orientation', 'Manual orientation']
    y_positions = [0.80, 0.52, 0.23]
    for row_idx, label in enumerate(row_labels):
        fig.text(0.03, # x-position (adjust if needed)
           y_positions[row_idx], # y-position centered in each row
            label, ha='left', va='center', rotation='vertical', fontsize=12)
        
    col_labels = ['Worst pred. ' + str(i+1) + f'\nError: {errors_worst[i]:.1f}°' for i in range(10)]
    for col_idx, label in enumerate(col_labels):
        fig.text(0.1 + 0.084 * col_idx, 
            0.93, # y-position (same for all columns)
            label, ha='center', va='bottom', fontsize=12)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.04, top=0.91)
    path = evaluation_plots_path + 'worst_predictions.pdf'
    fig.savefig(path)
    # plt.show()





if __name__ == "__main__":
    
    # Load configuration
    STAIN = config.stain1
    PRETRAINED_MODEL = config.pretrained_model
    TRAINED_MODEL_PATH = config.trained_model_path1
    EVALUATION_PLOTS_PATH = config.evaluation_plots_path1
    RANDOM_SEED = config.random_seed
    
    # Set the random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load the images, masks and corresponding rotations
    if STAIN == 'HE':
        ground_truth_rotations = config.HE_ground_truth_rotations
        images_path = config.HE_crops_masked_padded 
        masks_path = config.HE_masks_padded

        filenames_test, labels_test = get_filenames_and_labels(images_path, ground_truth_rotations)
        test_data = ImageDataset(image_path=images_path, mask_path=masks_path, subset='val', filenames=filenames_test, labels=labels_test, perform_transforms=False)

    elif STAIN == 'IHC':
        ground_truth_rotations = config.IHC_ground_truth_rotations
        images_path = config.IHC_crops_masked_padded 
        masks_path = config.IHC_masks_padded

        filenames_test, labels_test = get_filenames_and_labels(images_path, ground_truth_rotations)
        test_data = ImageDataset(image_path=images_path, mask_path=masks_path, subset='val', filenames=filenames_test, labels=labels_test, perform_transforms=False)

    elif STAIN == 'HE+IHC':
        ground_truth_rotations_HE = config.HE_ground_truth_rotations
        images_path_HE = config.HE_crops_masked_padded 
        masks_path_HE = config.HE_masks_padded
        filenames_test_HE, labels_test_HE = get_filenames_and_labels(images_path_HE, ground_truth_rotations_HE)
        test_data_HE = ImageDataset(image_path=images_path_HE, mask_path=masks_path_HE, subset='val', filenames=filenames_test_HE, labels=labels_test_HE, perform_transforms=False)

        ground_truth_rotations_IHC = config.IHC_ground_truth_rotations
        images_path_IHC = config.IHC_crops_masked_padded 
        masks_path_IHC = config.IHC_masks_padded
        filenames_test_IHC, labels_test_IHC = get_filenames_and_labels(images_path_IHC, ground_truth_rotations_IHC)
        test_data_IHC = ImageDataset(image_path=images_path_IHC, mask_path=masks_path_IHC, subset='val', filenames=filenames_test_IHC, labels=labels_test_IHC, perform_transforms=False)

        # Combine
        test_data = ConcatDataset([test_data_HE, test_data_IHC])
        
    # Create dataloader
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model wjth correct weights
    model = initialize_model(pretrained_weights_path=PRETRAINED_MODEL)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    model.to(device)

    # Apply the model on the test set
    test_labels, test_pred = apply_model_on_test_set(model, test_loader, device)

    # Calculate error metrics
    get_error_metrics(test_labels, test_pred)

    # Visualize predictions vs labels
    visualize_predictions_vs_labels(test_labels, test_pred, EVALUATION_PLOTS_PATH)

    # Visualize errors with images: TODO: DOES NOT WORK FOR HE+IHC CASE
    plot_predictions_based_on_percentile(images_path, masks_path, filenames_test, test_labels, test_pred, EVALUATION_PLOTS_PATH)
    plot_top10_worst_predictions(images_path, masks_path, filenames_test, test_labels, test_pred, EVALUATION_PLOTS_PATH)

