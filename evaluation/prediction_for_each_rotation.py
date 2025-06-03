# NOTE: Not yet checked how code should be adjusted for the HE+IHC case.

import os
import sys
import math
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

# Codes from other folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_training.model_training_loop import ImageDataset, initialize_model
from preprocessing.rotate_images_correctly import rotate_image_with_correct_padding, get_centroid_of_mask
from evaluation.general_evaluation import get_filenames_and_labels, apply_model_on_test_set

# To import the config file from the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



class ImageDataset_all_rotations(Dataset):
    """ A class to create a custom dataset for loading images and labels.
    
    Attributes:
        image_path (str): Path to the folder containing the images.
        mask_path (str): Path to the folder containing the masks.
        filenames (list): List of filenames for the images in the subset.
        labels (list): List of labels corresponding to the images in the subset.
        
    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(idx): Returns the image, label and position matrix of the patches for a given index (image in the dataset).
        get_pos(img, mask): Returns the position of each patch in the image. 
        augment_image(img, mask): Augments the image and mask by applying random transformations. """

    def __init__(self, image_path, mask_path, filenames, labels, angle_step=40):
        """ Custom dataset for loading images and labels.
        
        Parameters:
            image_path (str): Path to the folder containing the images.
            mask_path (str): Path to the folder containing the masks.
            filenames (list): List of filenames for the images in the subset.
            labels (list): List of labels corresponding to the images in the subset. """

        self.image_path = os.path.join(image_path, 'val')
        self.mask_path = os.path.join(mask_path, 'val')
        self.filenames = filenames 
        self.labels = labels 
        self.angle_step = angle_step

        # Create list of (image_idx, rotation_angle)
        self.index_map = []
        for i in range(len(self.filenames)):
            for angle in range(0, 360, angle_step):
                self.index_map.append((i, angle))

  

    def __len__(self):
        """ Returns the number of images in the dataset. """
        return len(self.index_map) 


    def __getitem__(self, idx):
        """ Returns the images, labels and position matrices for each possible rotation of the image. 

        Parameters:
            idx (int): Index of the image and label to retrieve.
        Returns:
            img_rotated (torch.Tensor): Rotated image tensor.
            angle_vector (torch.Tensor): Angle vector in degrees. 
            pos (torch.Tensor): Position matrix of the patches in the image. """

        image_idx, angle = self.index_map[idx]

        # Load image and mask
        fname = self.filenames[image_idx]
        img_path = os.path.join(self.image_path, fname)
        mask_path = os.path.join(self.mask_path, fname)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        true_angle = self.labels[image_idx]

        # Rotate the image and mask to the true rotation
        correctly_rotated_img, correctly_rotated_mask = rotate_image_with_correct_padding(img, mask, float(true_angle))

        # rotate the image and mask to the specified angle
        img_rotated, mask_rotated = rotate_image_with_correct_padding(correctly_rotated_img, correctly_rotated_mask, -float(angle))
        
        # Convert angle in degrees to vector
        angle_rad = angle * math.pi / 180.0
        angle_vector = torch.tensor([math.cos(angle_rad), math.sin(angle_rad)], dtype=torch.float32)
        
        # Apply transformations normalization to image
        img_rotated = torchvision.transforms.functional.to_tensor(img_rotated)  
        
        pos = self.get_pos(img_rotated, mask_rotated)     

        return img_rotated, angle_vector, pos
    
    
    def get_pos(self, img, mask, patch_size=16):
        """ Returns the position of each patch in the image. The position is calculated based on the centroid of the mask.

        Parameters:
            img (torch.Tensor): Image for position matrix needs to be calculated.
            mask (PIL.Image): Mask for the image.
            patch_size (int): Size of the patches. Default is 16.
        
        Returns:
            pos (torch.Tensor): Position matrix of the patches. """
        
        # Calculate centroid and convert to pixel value
        centroid = get_centroid_of_mask(mask) # Get the centroid of the mask
        cx = math.floor(centroid[0])
        cy = math.floor(centroid[1])

        # Get the size of the image and number of patches
        C, H, W = img.shape 
        h_patches = H // patch_size
        w_patches = W // patch_size

        # Convert pixel centroid to patch coords
        centroid_patch_x = int(cx // patch_size)
        centroid_patch_y = int(cy // patch_size)

        x_range = torch.arange(w_patches) - centroid_patch_x
        y_range = -(torch.arange(h_patches) - centroid_patch_y)

        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        pos = torch.stack([y_grid, x_grid], dim=2).reshape(-1,2)  # Stack into (H, W, 2) adn then flatten to (H*W, 2)

        return pos



def plot_prediction_distribution_heatmap(angles, predictions, evaluation_plots_path, bins=72):
    """ Plot a heatmap of the distribution of predictions based on rotation angles.
    
    Parameters:
        angles (np.ndarray): Array of rotation angles in degrees.
        predictions (np.ndarray): Array of predicted values.
        evaluation_plots_path (str): Path to the folder where the evaluation plots will be saved.
        bins (int): Number of bins for the histogram. Default is 72.
    
    Returns:
        None. Plots are saved to the evaluation_plots_path. """

    plt.figure(figsize=(12, 5))
    plt.hist2d(
        x=angles,
        y=predictions,
        bins=[bins, bins],
        cmap="viridis"
    )
    plt.xlabel("Rotation angle (°)")
    plt.ylabel("Predicted value")
    plt.title("Prediction Distribution by Rotation Angle")
    plt.colorbar(label="Frequency")
    plt.tight_layout()
    title = evaluation_plots_path + 'prediction_distribution_heatmap.pdf'
    plt.savefig(title)
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
        test_data = ImageDataset_all_rotations(image_path=images_path, mask_path=masks_path, filenames=filenames_test, labels=labels_test)

    elif STAIN == 'IHC':
        ground_truth_rotations = config.IHC_ground_truth_rotations
        images_path = config.IHC_crops_masked_padded 
        masks_path = config.IHC_masks_padded

        filenames_test, labels_test = get_filenames_and_labels(images_path, ground_truth_rotations)
        test_data = ImageDataset_all_rotations(image_path=images_path, mask_path=masks_path, filenames=filenames_test, labels=labels_test)

    elif STAIN == 'HE+IHC':
        ground_truth_rotations_HE = config.HE_ground_truth_rotations
        images_path_HE = config.HE_crops_masked_padded 
        masks_path_HE = config.HE_masks_padded
        filenames_test_HE, labels_test_HE = get_filenames_and_labels(images_path_HE, ground_truth_rotations_HE)
        test_data_HE = ImageDataset_all_rotations(image_path=images_path_HE, mask_path=masks_path_HE, filenames=filenames_test_HE, labels=labels_test_HE)

        ground_truth_rotations_IHC = config.IHC_ground_truth_rotations
        images_path_IHC = config.IHC_crops_masked_padded 
        masks_path_IHC = config.IHC_masks_padded
        filenames_test_IHC, labels_test_IHC = get_filenames_and_labels(images_path_IHC, ground_truth_rotations_IHC)
        test_data_IHC = ImageDataset_all_rotations(image_path=images_path_IHC, mask_path=masks_path_IHC, filenames=filenames_test_IHC, labels=labels_test_IHC)

        # Combine
        test_data = ConcatDataset([test_data_HE, test_data_IHC])
        
    # Create dataloader
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model wjth correct weights
    model = initialize_model(pretrained_weights_path=PRETRAINED_MODEL)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    model.to(device)

    # Evaluate rotation sensitivity
    true_labels, preds = apply_model_on_test_set(model, test_loader, device)
    plot_prediction_distribution_heatmap(true_labels, preds, EVALUATION_PLOTS_PATH, bins=72)