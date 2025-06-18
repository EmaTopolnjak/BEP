from torch.utils.data import DataLoader, ConcatDataset
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import torchvision
import random

# Codes from other files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_training.model_training_loop import initialize_model
from preprocessing.rotate_images_correctly import rotate_image_with_correct_padding, get_centroid_of_mask
import evaluation.evaluation_utils as eval_utils

from pathlib import Path
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



def seed_worker(worker_id):
    """ Set the random seed for each worker to ensure reproducibility."""
    worker_seed = RANDOM_SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)



def plot_errors(true_labels, preds, evaluation_plots_path):
    """ Visualize THE predictions by making plots of the predictions vs labels, the histogram of angular 
    differences, histogram of absolute of absolute angular differences and the error bias plot.
    
    Parameters:
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.
        evaluation_plots_path (str): Path to the folder where the evaluation plots will be saved.

    Returns:
        None. Plots are saved to the evaluation_plots_path. """
       
    angluar_diff, abs_angular_diff = eval_utils.calculate_error(true_labels, preds)

    # Plot the predictions vs labels
    plt.figure()
    plt.scatter(true_labels, preds, s=10, alpha=0.6)
    plt.plot([0, 360], [0, 360], '--', color='gray')  # y=x line
    plt.xlabel("Ground Truth (°)")
    plt.ylabel("Prediction (°)")
    plt.title("Prediction vs Ground Truth")
    plt.grid(True)
    path = evaluation_plots_path + '/predictions_vs_labels.pdf'
    plt.savefig(path)
    # plt.show()
    plt.close()

    # Plot the histogram of angular differences
    plt.figure()
    plt.hist(angluar_diff, bins=180, edgecolor='black')
    plt.xlabel("Angular Error (°)")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.grid(True)
    path = evaluation_plots_path + '/histogram_angluar_differences.pdf'
    plt.savefig(path)
    # plt.show()
    plt.close()  

    # Plot the histogram of absolute angular differences
    plt.figure()
    plt.hist(abs_angular_diff, bins=180, edgecolor='black')
    plt.xlabel("Absolute Angular Error (°)")
    plt.ylabel("Count")
    plt.title("Absolute Error Distribution")
    plt.grid(True)
    path = evaluation_plots_path + '/histogram_abs_angluar_differences.pdf'
    plt.savefig(path)
    # plt.show()
    plt.close()

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
    path = evaluation_plots_path + '/error_bias_plot.pdf'
    plt.savefig(path)
    # plt.show()
    plt.close()



def get_filenames_based_on_error_percentiles(test_data, true_labels, preds):
    """ Get the filenames of the images that are at the 1st, 50th and 99th percentile of absolute angular differences.

    Parameters:
        test_data (Dataset): Dataset containing the images and masks.
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.

    Returns:
        dataset_selected (list): List of selected dataset items (images, masks, true labels, pos).
        preds_selected (list): List of selected predictions. """

    _, abs_angluar_diff = eval_utils.calculate_error(true_labels, preds)

    percentiles = [1, 50, 99]
    thresholds = np.percentile(abs_angluar_diff, percentiles)
    selected_indices = [np.argmin(np.abs(abs_angluar_diff - t)) for t in thresholds]

    # Get the filenames based on the indices
    dataset_selected = [test_data[i] for i in selected_indices]
    preds_selected = [preds[i] for i in selected_indices]

    return dataset_selected, preds_selected



def plot_predictions_based_on_percentile(test_data, true_labels, preds, evaluation_plots_path):
    """ Visualize the errors with the images and masks. The images are displayed with the predicted angle of rotation and the true angle of rotation.
    
    Parameters:
        test_data (Dataset): Dataset containing the images and masks.
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.
        evaluation_plots_path (str): Path to the folder where the evaluation plots will be saved.

    Returns:
        None. Plots are saved to the evaluation_plots_path. """

    # Get the filenames, true labels and predictions for the 1st, 50th and 99th percentile of absolute angular differences
    dataset_selected, preds_selected = get_filenames_based_on_error_percentiles(test_data, true_labels, preds)

    fig, ax = plt.subplots(3, 3, figsize=(10, 9))

    for i, item in enumerate(dataset_selected):
        img, mask, true_label, _ = item
        true_angle = float(eval_utils.vector_to_angle_deg(true_label))
        
        # Original image and mask
        img, mask = rotate_image_with_correct_padding(torchvision.transforms.functional.to_pil_image(img.squeeze(0).cpu()), torchvision.transforms.functional.to_pil_image(mask.squeeze(0).cpu()), 0, bg_color=(255, 255, 255, 255))

        # Plot line for orientation
        centroid_original = get_centroid_of_mask(mask)
        dx = np.cos(np.deg2rad(90 - true_angle)) * max(img.size[0], img.size[1])
        dy = np.sin(np.deg2rad(90 - true_angle)) * max(img.size[0], img.size[1])
        x1, y1 = centroid_original[0] - dx, centroid_original[1] - dy
        x2, y2 = centroid_original[0] + dx, centroid_original[1] + dy
        ax[0, i].plot([x1, x2], [y1, y2], color='black', linewidth=0.8)

        ax[0, i].imshow(img)
        
        # Predicted rotated image and mask
        predicted_rotated_img, predicted_rotated_mask = rotate_image_with_correct_padding(img, mask, preds_selected[i], bg_color=(255, 255, 255, 255))
        
        # Plot line for orientation
        centroid_predicted = get_centroid_of_mask(predicted_rotated_mask)
        dx = np.cos(np.deg2rad(90 - (true_angle - preds_selected[i]))) * max(predicted_rotated_img.size[0], predicted_rotated_img.size[1])
        dy = np.sin(np.deg2rad(90 - (true_angle - preds_selected[i]))) * max(predicted_rotated_img.size[0], predicted_rotated_img.size[1])
        x1, y1 = centroid_predicted[0] - dx, centroid_predicted[1] - dy
        x2, y2 = centroid_predicted[0] + dx, centroid_predicted[1] + dy
        ax[1, i].plot([x1, x2], [y1, y2], color='black', linewidth=0.8)

        ax[1, i].imshow(predicted_rotated_img)      

        # Correctly rotated image and mask
        correctly_rotated_img, correctly_rotated_mask = rotate_image_with_correct_padding(img, mask, true_angle, bg_color=(255, 255, 255, 255))
        
        # Plot line for orientation
        centroid_correct = get_centroid_of_mask(correctly_rotated_mask)
        x1, y1 = centroid_correct[0], centroid_correct[1] - max(correctly_rotated_img.size[0], correctly_rotated_img.size[1])
        x2, y2 = centroid_correct[0], centroid_correct[1] + max(correctly_rotated_img.size[0], correctly_rotated_img.size[1])
        ax[2, i].plot([x1, x2], [y1, y2], color='black', linewidth=0.8)

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
    path = evaluation_plots_path + '/qualitative_analysis_errors.pdf'
    fig.savefig(path)
    # plt.show()



def det_top10_worst_predictions(test_data, true_labels, preds):
    """ Determine the top 10 worst predictions based on absolute angular differences.
    
    Parameters:
        test_data (Dataset): Dataset containing the images and masks.
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.
    
    Returns:
        dataset_worst (list): List of the worst dataset items (images, masks, true labels, pos).
        preds_worst (list): List of predictions of the worst predictions.
        errors_worst (list): List of angular differences of the worst predictions. """

    angular_diff, abs_angluar_diff = eval_utils.calculate_error(true_labels, preds)

    # Get indices of top 10 worst errors
    worst_indices = np.argsort(abs_angluar_diff)[-10:][::-1]  # descending order

    # Get the worst samples
    dataset_worst = [test_data[i] for i in worst_indices]
    preds_worst = [preds[i] for i in worst_indices]
    errors_worst = [angular_diff[i] for i in worst_indices] 

    return dataset_worst, preds_worst, errors_worst



def plot_top10_worst_predictions(test_data, true_labels, preds, evaluation_plots_path):
    """ Plot the top 10 worst predictions based on absolute angular differences.

    Parameters:
        test_data (Dataset): Dataset containing the images and masks.
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.
        evaluation_plots_path (str): Path to the folder where the evaluation plots will be saved.

    Returns:
        None. Plots are saved to the evaluation_plots_path. """
    
    dataset_worst, preds_worst, errors_worst = det_top10_worst_predictions(test_data, true_labels, preds)

    fig, ax = plt.subplots(3, 10, figsize=(20, 6))  # 3 rows × 10 columns

    # Turn off the axes for all subplots
    for row in ax:
        for col in row:
            col.axis('off')
            col.set_aspect('equal')    

    for i, item in enumerate(dataset_worst):
        img, mask, true_label, _ = item
        true_angle = float(eval_utils.vector_to_angle_deg(true_label))

        # Original image and mask
        img, mask = rotate_image_with_correct_padding(torchvision.transforms.functional.to_pil_image(img.squeeze(0).cpu()), torchvision.transforms.functional.to_pil_image(mask.squeeze(0).cpu()), 0, bg_color=(255, 255, 255, 255))

        # Plot line for orientation
        centroid_original = get_centroid_of_mask(mask)
        dx = np.cos(np.deg2rad(90 - true_angle)) * max(img.size[0], img.size[1])
        dy = np.sin(np.deg2rad(90 - true_angle)) * max(img.size[0], img.size[1])
        x1, y1 = centroid_original[0] - dx, centroid_original[1] - dy
        x2, y2 = centroid_original[0] + dx, centroid_original[1] + dy
        ax[0, i].plot([x1, x2], [y1, y2], color='black', linewidth=0.8)

        ax[0, i].imshow(img)
        
        # Predicted rotated image and mask
        predicted_rotated_img, predicted_rotated_mask = rotate_image_with_correct_padding(img, mask, preds_worst[i], bg_color=(255, 255, 255, 255))
        
        # Plot line for orientation
        centroid_predicted = get_centroid_of_mask(predicted_rotated_mask)
        dx = np.cos(np.deg2rad(90 - (true_angle - preds_worst[i]))) * max(predicted_rotated_img.size[0], predicted_rotated_img.size[1])
        dy = np.sin(np.deg2rad(90 - (true_angle - preds_worst[i]))) * max(predicted_rotated_img.size[0], predicted_rotated_img.size[1])
        x1, y1 = centroid_predicted[0] - dx, centroid_predicted[1] - dy
        x2, y2 = centroid_predicted[0] + dx, centroid_predicted[1] + dy
        ax[1, i].plot([x1, x2], [y1, y2], color='black', linewidth=0.8)

        ax[1, i].imshow(predicted_rotated_img)     

        # Correctly rotated image and mask
        correctly_rotated_img, correctly_rotated_mask = rotate_image_with_correct_padding(img, mask, true_angle, bg_color=(255, 255, 255, 255))
        
        # Plot line for orientation
        centroid_correct = get_centroid_of_mask(correctly_rotated_mask)
        x1, y1 = centroid_correct[0], centroid_correct[1] - max(correctly_rotated_img.size[0], correctly_rotated_img.size[1])
        x2, y2 = centroid_correct[0], centroid_correct[1] + max(correctly_rotated_img.size[0], correctly_rotated_img.size[1])
        ax[2, i].plot([x1, x2], [y1, y2], color='black', linewidth=0.8)

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

    plt.subplots_adjust(left=0.04, top=0.91)
    path = evaluation_plots_path + '/worst_predictions.pdf'
    fig.savefig(path)
    # plt.show()



if __name__ == "__main__":
    
    # Load configuration
    RANDOM_SEED = config.random_seed
    PRETRAINED_MODEL = config.pretrained_model
    STAIN = config.stain  # 'HE', 'IHC' or 'HE+IHC'
    UNIFORM_DISTRIBUTION = config.uniform_distribution
    TRAINED_MODEL_PATH = config.trained_model_path
    EVALUATION_PLOTS_PATH = config.evaluation_plots_path
    
    # Set the random seed for reproducibility
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    # Load the images, masks and corresponding rotations
    if STAIN == 'HE':
        ground_truth_rotations = config.HE_ground_truth_rotations
        images_path = config.HE_crops_masked_padded 
        masks_path = config.HE_masks_padded

        filenames_test, labels_test = eval_utils.get_filenames_and_labels(images_path, ground_truth_rotations)
        test_data = eval_utils.ImageDataset(image_path=images_path, mask_path=masks_path, subset='val', filenames=filenames_test, labels=labels_test, uniform_distribution=UNIFORM_DISTRIBUTION)

    elif STAIN == 'IHC':
        ground_truth_rotations = config.IHC_ground_truth_rotations
        images_path = config.IHC_crops_masked_padded 
        masks_path = config.IHC_masks_padded

        filenames_test, labels_test = eval_utils.get_filenames_and_labels(images_path, ground_truth_rotations)
        test_data = eval_utils.ImageDataset(image_path=images_path, mask_path=masks_path, subset='val', filenames=filenames_test, labels=labels_test,  uniform_distribution=UNIFORM_DISTRIBUTION)

    elif STAIN == 'HE+IHC':
        ground_truth_rotations_HE = config.HE_ground_truth_rotations
        images_path_HE = config.HE_crops_masked_padded 
        masks_path_HE = config.HE_masks_padded
        filenames_test_HE, labels_test_HE = eval_utils.get_filenames_and_labels(images_path_HE, ground_truth_rotations_HE)
        test_data_HE = eval_utils.ImageDataset(image_path=images_path_HE, mask_path=masks_path_HE, subset='val', filenames=filenames_test_HE, labels=labels_test_HE,  uniform_distribution=UNIFORM_DISTRIBUTION)

        ground_truth_rotations_IHC = config.IHC_ground_truth_rotations
        images_path_IHC = config.IHC_crops_masked_padded 
        masks_path_IHC = config.IHC_masks_padded
        filenames_test_IHC, labels_test_IHC = eval_utils.get_filenames_and_labels(images_path_IHC, ground_truth_rotations_IHC)
        test_data_IHC = eval_utils.ImageDataset(image_path=images_path_IHC, mask_path=masks_path_IHC, subset='val', filenames=filenames_test_IHC, labels=labels_test_IHC,  uniform_distribution=UNIFORM_DISTRIBUTION)

        # Combine
        test_data = ConcatDataset([test_data_HE, test_data_IHC])
        
    # Create dataloader
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model wjth correct weights
    model = initialize_model(pretrained_weights_path=PRETRAINED_MODEL)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    # Apply the model on the test set
    test_labels, test_pred = eval_utils.apply_model_on_test_set(model, test_loader, device)
    print(f"Test labels: {test_labels}")

    # Calculate error metrics
    eval_utils.get_error_metrics(test_labels, test_pred)

    # Visualize predictions vs labels
    plot_errors(test_labels, test_pred, EVALUATION_PLOTS_PATH)

    # Visualize errors with images - does not work for HE+IHC case
    if STAIN != 'HE+IHC':
        plot_predictions_based_on_percentile(test_data, test_labels, test_pred, EVALUATION_PLOTS_PATH)
        plot_top10_worst_predictions(test_data, test_labels, test_pred, EVALUATION_PLOTS_PATH)
