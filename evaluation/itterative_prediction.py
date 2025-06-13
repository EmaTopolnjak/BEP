import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms.functional as F

# Codes from other folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.rotate_images_correctly import rotate_image_with_correct_padding
from model_training.model_training_loop import initialize_model
import evaluation.evaluation_utils as eval_utils

# To import the config file from the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



def pass_image_through_model_multiple_times(model, test_data, img, true_angle_vec, mask, device, max_iters=5):
    """ Pass an image through the model multiple times, rotating it based on predicted angles.
    
    Parameters:
        model (torch.nn.Module): The trained model to use for predictions.
        test_data (ImageDataset): The dataset with images, masks, labels.
        img (torch.Tensor): The input image tensor.
        true_angle_vec (torch.Tensor): The true angle vector.
        mask (PIL.Image): The mask for the image.
        device (torch.device): The device to run the model on.
        max_iters (int): The number of iterations to run the model. Default is 5.
    
    Returns:
        images (list): List of images after each iteration.
        masks (list): List of masks after each iteration.
        angular_errors (list): List of angular errors after each iteration.
        true_angle_deg (float): The true angle in degrees. 
        cumulative_angle_deg (float): The cumulative angle after all iterations. """

    # Initalize image and true angle
    true_angle_deg = eval_utils.vector_to_angle_deg(true_angle_vec)
    image = img
    cumulative_angle_deg = 0.0

    # Initialize lists to store results
    images = []
    masks = []
    angular_errors = []

    for i in range(max_iters):
        # Compute position matrix for current image
        pos = test_data.get_pos(image, mask)
        pos = pos.unsqueeze(0).to(device)
        image = image.unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            pred_angle_vec = model(image, pos).squeeze()
            pred_angle_deg = eval_utils.vector_to_angle_deg(pred_angle_vec)

        # Rotate image based on predicted angle
        image, mask = rotate_image_with_correct_padding(F.to_pil_image(image.squeeze(0).cpu()), mask, pred_angle_deg)
        
        # Update cumulative angle
        cumulative_angle_deg += pred_angle_deg
        cumulative_angle_deg = cumulative_angle_deg % 360

        # Log results 
        total_angluar_diff = eval_utils.angluar_error_deg(true_angle_deg, cumulative_angle_deg)
        angular_errors.append(total_angluar_diff)
        images.append(image)
        masks.append(mask)

        image = F.to_tensor(image)
        pos = pos.to(device)

    return images, masks, angular_errors, true_angle_deg, cumulative_angle_deg



def plot_results_through_iterations(img, mask, images, masks, angular_errors, true_angle_deg, max_iters=5):
    """ Plot the original image and the images after each iteration with their corresponding angular errors.
    
    Parameters:
        img (torch.Tensor): The original image tensor.
        mask (PIL.Image): The mask for the image.
        images (list): List of images after each iteration.
        masks (list): List of masks after each iteration.
        angular_errors (list): List of angular errors after each iteration.
        true_angle_deg (float): The true angle in degrees.
        max_iters (int): The number of iterations to plot.
        
    Returns:
        None. Plots the original image and the images after each iteration. """

    fig, ax = plt.subplots(1, max_iters + 1, figsize=(20, 5))

    # Turn off the axes for all subplots
    for col in ax:
        col.axis('off')
        col.set_aspect('equal')

    # Plot the original image
    image, _ = rotate_image_with_correct_padding(F.to_pil_image(img.squeeze(0).cpu()), mask, 0, bg_color=(255, 255, 255, 255))  # Remove background
    ax[0].imshow(image)

    # Plot the images for each iteration
    for i in range(max_iters):
        image, _ = rotate_image_with_correct_padding(images[i], masks[i], 0, bg_color=(255, 255, 255, 255))  # Remove background
        ax[i + 1].imshow(image)
    
    # Set titles for each subplot
    col_labels = [f"Org. img - True angle: {true_angle_deg:.4f}°"]
    for i in range(max_iters):
        col_labels.append(f"Iter. {i+1} - Error: {angular_errors[i]:.4f}°")
    for col_idx, label in enumerate(col_labels):
        fig.text(0.08 + 0.16 * col_idx, 
            0.96, # y-position (same for all columns)
            label, ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    # plt.show()



def pass_dataset_through_model_multiple_times(model, test_data, test_loader, device, max_iters=5):
    """ Pass the entire dataset through the model multiple times and return the results.
    
    Parameters:
        model (torch.nn.Module): The trained model to use for predictions.
        test_data (ImageDataset): The dataset with images, masks, labels.
        test_loader (DataLoader): The DataLoader for the test dataset.
        device (torch.device): The device to run the model on.
        max_iters (int): The number of iterations to run the model.
        
    Returns:
        results (list): List of dictionaries containing images, masks, angular errors, and true angles for each image in the dataset. """
    
    final_preds = []
    true_labels = []


    for idx, (img, true_angle_vec, pos) in enumerate(test_loader):
        img, true_angle_vec, pos = img.squeeze(0).to(device), true_angle_vec.squeeze(0).to(device), pos.squeeze(0).to(device)
        mask_path = os.path.join(test_data.mask_path, test_data.filenames[idx])
        mask = Image.open(mask_path).convert("L")

        # Pass the image through the model multiple times
        images, masks, angular_errors, true_angle_deg, cummulative_angle_deg = pass_image_through_model_multiple_times(model, test_data, img, true_angle_vec, mask, device, max_iters=max_iters)
        
        # Log results
        final_preds.append(cummulative_angle_deg)
        true_labels.append(true_angle_deg)

        # if idx == 0: # Plot results for the first image
        #     plot_results_through_iterations(img, mask, images, masks, angular_errors, true_angle_deg, max_iters=5)

    final_preds = np.array(final_preds)
    true_labels = np.array(true_labels)

    return final_preds, true_labels



if __name__ == "__main__":
    
    # Load configuration
    RANDOM_SEED = config.random_seed
    PRETRAINED_MODEL = config.pretrained_model
    STAIN = config.stain1  # 'HE', 'IHC' or 'HE+IHC'
    UNNIFORM_DISTRIBUTION = config.uniform_distribution
    TRAINED_MODEL_PATH = config.trained_model_path
    EVALUATION_PLOTS_PATH = config.evaluation_plots_path
    
    # Set the random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load the images, masks and corresponding rotations
    if STAIN == 'HE':
        ground_truth_rotations = config.HE_ground_truth_rotations
        images_path = config.HE_crops_masked_padded 
        masks_path = config.HE_masks_padded

        filenames_test, labels_test = eval_utils.get_filenames_and_labels(images_path, ground_truth_rotations)
        test_data = eval_utils.ImageDataset(image_path=images_path, mask_path=masks_path, subset='val', filenames=filenames_test, labels=labels_test, uniform_distribution=UNNIFORM_DISTRIBUTION)

    elif STAIN == 'IHC':
        ground_truth_rotations = config.IHC_ground_truth_rotations
        images_path = config.IHC_crops_masked_padded 
        masks_path = config.IHC_masks_padded

        filenames_test, labels_test = eval_utils.get_filenames_and_labels(images_path, ground_truth_rotations)
        test_data = eval_utils.ImageDataset(image_path=images_path, mask_path=masks_path, subset='val', filenames=filenames_test, labels=labels_test,  uniform_distribution=UNNIFORM_DISTRIBUTION)

    elif STAIN == 'HE+IHC':
        ground_truth_rotations_HE = config.HE_ground_truth_rotations
        images_path_HE = config.HE_crops_masked_padded 
        masks_path_HE = config.HE_masks_padded
        filenames_test_HE, labels_test_HE = eval_utils.get_filenames_and_labels(images_path_HE, ground_truth_rotations_HE)
        test_data_HE = eval_utils.ImageDataset(image_path=images_path_HE, mask_path=masks_path_HE, subset='val', filenames=filenames_test_HE, labels=labels_test_HE,  uniform_distribution=UNNIFORM_DISTRIBUTION)

        ground_truth_rotations_IHC = config.IHC_ground_truth_rotations
        images_path_IHC = config.IHC_crops_masked_padded 
        masks_path_IHC = config.IHC_masks_padded
        filenames_test_IHC, labels_test_IHC = eval_utils.get_filenames_and_labels(images_path_IHC, ground_truth_rotations_IHC)
        test_data_IHC = eval_utils.ImageDataset(image_path=images_path_IHC, mask_path=masks_path_IHC, subset='val', filenames=filenames_test_IHC, labels=labels_test_IHC,  uniform_distribution=UNNIFORM_DISTRIBUTION)

        # Combine
        test_data = ConcatDataset([test_data_HE, test_data_IHC])
        
    # Create dataloader
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model wjth correct weights
    model = initialize_model(pretrained_weights_path=PRETRAINED_MODEL)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    # Let each image pass through the model multiple times
    results, test_pred, test_labels = pass_dataset_through_model_multiple_times(model, test_data, test_loader, device, max_iters=5)

    # Evaluate the results
    eval_utils.get_error_metrics(test_labels, test_pred)

    