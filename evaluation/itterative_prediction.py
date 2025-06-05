# NOTE: ...

import os
import sys
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms.functional as F

# Codes from other folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_training.model_training_loop import ImageDataset, initialize_model
from preprocessing.rotate_images_correctly import rotate_image_with_correct_padding
from evaluation.general_evaluation import get_filenames_and_labels

# To import the config file from the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



def vector_to_angle(vec):
    """ Convert a 2D vector to an angle in degrees. """

    angle_rad = torch.atan2(vec[1], vec[0]).item()
    angle_deg = angle_rad * 180.0 / math.pi % 360
    return angle_deg



def angle_to_vector(angle_deg):
    """ Convert an angle in degrees to a 2D vector. """

    angle_rad = angle_deg * math.pi / 180.0
    angle_vec = torch.tensor([math.cos(angle_rad), math.sin(angle_rad)], dtype=torch.float32)
    return angle_vec



def angluar_error(true_angle_deg, pred_angle_deg):
    """ Calculate the angular error between true and predicted angles. """

    diff = true_angle_deg - pred_angle_deg  # Compute raw difference
    angular_diff = np.remainder(diff + 180, 360) - 180  # Wrap difference to [-180, 180]
    return angular_diff



def pass_image_through_model_multiple_times(img, true_angle_vec, mask, device, max_iters=5):
    """ Pass an image through the model multiple times, rotating it based on predicted angles.
    
    Parameters:
        img (torch.Tensor): The input image tensor.
        true_angle_vec (torch.Tensor): The true angle vector.
        mask (PIL.Image): The mask for the image.
        device (torch.device): The device to run the model on.
        max_iters (int): The number of iterations to run the model.
    
    Returns:
        images (list): List of images after each iteration.
        masks (list): List of masks after each iteration.
        angular_errors (list): List of angular errors after each iteration.
        true_angle_deg (float): The true angle in degrees. """

    # Initalize image and true angle
    true_angle_deg = vector_to_angle(true_angle_vec)
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
            pred_angle_deg = vector_to_angle(pred_angle_vec)

        # Calculate angle error
        angluar_diff = angluar_error(true_angle_deg, pred_angle_deg)
        # print(f"Step {i}: predicted angle = {pred_angle_deg:.2f}°, angular_error = {angluar_diff:.2f}°")

        # Rotate image ibased on predicted angle
        image, mask = rotate_image_with_correct_padding(F.to_pil_image(image.squeeze(0).cpu()), mask, pred_angle_deg)
        
        # Update cumulative angle
        cumulative_angle_deg += pred_angle_deg
        cumulative_angle_deg = cumulative_angle_deg % 360

        # Log results 
        total_angluar_diff = angluar_error(true_angle_deg, cumulative_angle_deg)
        angular_errors.append(total_angluar_diff)
        images.append(image)
        masks.append(mask)

        image = F.to_tensor(image)
        pos = pos.to(device)

    return images, masks, angular_errors, true_angle_deg



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
    plt.show()



def pass_dataset_through_model_multiple_times(test_data, test_loader, device, max_iters=5):
    """ Pass the entire dataset through the model multiple times and return the results.
    
    Parameters:
        test_data (ImageDataset): The dataset with images, masks, labels, ....
        test_loader (DataLoader): The DataLoader for the test dataset.
        device (torch.device): The device to run the model on.
        max_iters (int): The number of iterations to run the model.
        
    Returns:
        results (list): List of dictionaries containing images, masks, angular errors, and true angles for each image in the dataset. """
    
    results = []

    for idx, (img, true_angle_vec, pos) in enumerate(test_loader):
        print(f"Processing image {idx + 1}/{len(test_loader)}", end='\r')

        img, true_angle_vec, pos = img.squeeze(0).to(device), true_angle_vec.squeeze(0).to(device), pos.squeeze(0).to(device)

        mask_path = os.path.join(test_data.mask_path, test_data.filenames[idx])
        mask = Image.open(mask_path).convert("L")
        images, masks, angular_errors, true_angle_deg = pass_image_through_model_multiple_times(img, true_angle_vec, mask, device, max_iters=max_iters)
        results.append({'images': images, 'masks': masks, 'angular_errors': angular_errors, 'true_angle_deg': true_angle_deg})

        if idx == 0: # Plot results for the first image
            plot_results_through_iterations(img, mask, images, masks, angular_errors, true_angle_deg, max_iters=5)

    return results



if __name__ == "__main__":
    
    # Load configuration
    STAIN = config.stain2
    PRETRAINED_MODEL = config.pretrained_model
    TRAINED_MODEL_PATH = config.trained_model_path2
    EVALUATION_PLOTS_PATH = config.evaluation_plots_path2
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

    # Let each image pass through the model multiple times
    pass_dataset_through_model_multiple_times(test_data, test_loader, device, max_iters=5)


    