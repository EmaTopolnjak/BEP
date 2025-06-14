import os
import math
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import random
import sys

# Codes from other files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.rotate_images_correctly import rotate_image_with_correct_padding, get_centroid_of_mask
from model_training.model_training_loop import filter_by_rotated_size_threshold



class ImageDataset(Dataset):
    """ A class to create a custom dataset for loading images and labels.
    
    Attributes:
        image_paths (list): List of image file paths.
        mask_paths (list): List of mask file paths.
        filenames (list): List of filenames for the images in the dataset.
        labels (list): List of labels corresponding to the images in the dataset.
        perform_transform (bool): Whether to apply augmentations to the images.
        threshold_fine_angle (int): Epoch after which fine-angle augmentations are applied.
        epoch (int): Current epoch number.

        
    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(idx): Returns the image, label and position matrix of the patches for a given index (image in the dataset).
        set_epoch(epoch): Sets the current epoch number.
        update_transform(): Updates the augmentations based on the current epoch. It is used to switch to fine-angle augmentations after a certain epoch.
        augment_image(img, mask, idx): Augments the image and mask by applying random transformations and returns the augmented image, mask and angle.
        get_pos(img, mask): Returns the position of each patch in the image. The position is calculated based on the centroid of the mask. """

    def __init__(self, image_path, mask_path, subset, filenames, labels, uniform_distribution=True):
        """ Custom dataset for loading images and labels.
        
        Parameters:
            image_path (str): Path to the folder containing the images.
            mask_path (str): Path to the folder containing the masks.
            subset (str): Subset of the dataset ('train', 'val', 'test').
            filenames (list): List of filenames for the images in the subset.
            labels (list): List of labels corresponding to the images in the subset.
            uniform_distribution (bool): If True, the angle is uniformly distributed between 0 and 360 degrees. If False, the original angle is used."""
            
        self.image_path = os.path.join(image_path, subset)
        self.mask_path = os.path.join(mask_path, subset)
        self.filenames = filenames 
        self.labels = labels 
        self.uniform_distribution = uniform_distribution  # If True, the angle is uniformly distributed between 0 and 360 degrees


    def __len__(self):
        """ Returns the number of images in the dataset."""
        return len(self.filenames)


    def __getitem__(self, idx):
        """ Returns the image, label and position matrix of the patches for a given index (image in the dataset).
        The image is augmented and transformed to a tensor. The position matrix is calculated based on the centroid of the mask.
        The image is normalized and the mask is converted to a tensor. The angle of rotation is also returned as vector.

        Parameters:
            idx (int): Index of the image and label to retrieve.
        Returns:
            img (torch.Tensor): Image tensor.
            mask (torch.Tensor): Mask tensor.
            angle_vector (torch.Tensor): Angle vector representing the rotation of the image in radians.
            pos (torch.Tensor): Position matrix of the patches. """

        # Load image and mask
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_path, fname)
        mask_path = os.path.join(self.mask_path, fname)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.uniform_distribution: # Apply augmentations
            img, mask, angle_deg = self.rotate_image(img, mask, idx)
        else:
            angle_deg = self.labels[idx]

        # Convert angle in degrees to vector
        angle_rad = angle_deg * math.pi / 180.0
        angle_vector = torch.tensor([math.cos(angle_rad), math.sin(angle_rad)], dtype=torch.float32)

        # Apply transformations normalization to image
        img = torchvision.transforms.functional.to_tensor(img)  
        mask = torchvision.transforms.functional.to_tensor(mask)  

        # Get the position of each patch in the image
        pos = self.get_pos(img, mask)      

        return img, mask, angle_vector, pos
    

    def rotate_image(self, img, mask, idx):
        """ Rotates the image and mask by a random angle, ensuring that the angle is correctly adjusted based on the original label.
        This function is used to augment the images in the dataset by rotating them such overall the angles are uniformly distributed between 0 and 360 degrees.
        
        Parameters:
            img (PIL.Image): Image to be augmented.
            mask (PIL.Image): Mask to be augmented.
        
        Returns:
            img (PIL.Image): Augmented image.
            mask (PIL.Image): Augmented mask.
            idx (int): Index of the image in the dataset. The angle is adjusted based on the original label."""
        
        angle = self.labels[idx] # Get the angle from the labels 

        # Rotate the image and mask by a random angle
        final_angle = random.uniform(0, 360)
        added_angle = (final_angle - angle) % 360
        img, mask = rotate_image_with_correct_padding(img, mask, -float(added_angle))
        angle = (angle + added_angle) % 360 # Update angle to final (clipped to [0, 360])

        return img, mask, angle
    
    
    def get_pos(self, img, mask, patch_size=16):
        """ Returns the position of each patch in the image. The position is calculated based on the centroid of the mask.

        Parameters:
            img (torch.Tensor): Image for position matrix needs to be calculated.
            mask (PIL.Image): Mask for the image.
            patch_size (int): Size of the patches. Default is 16.
        
        Returns:
            pos (torch.Tensor): Position matrix of the patches. """
        
        # Calculate centroid and convert to pixel value
        mask = mask.squeeze(0)  # Remove the channel dimension if present
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
        pos = torch.stack([y_grid, x_grid], dim=2).reshape(-1,2)  # Stack into (H, W, 2) and then flatten to (H*W, 2)

        return pos
    


def extract_datasets(img_path, filter_large_images):
    """ Extract the dataset (test) from the given image path. The images are filtered based on size using the provided filter function.

    Parameters:
        img_path (str): Path to the folder containing the images. The folder should contain subfolders 'train', 'val' and 'test'.
        filter_large_images (function): Function to filter images based on size. 

    Returns:
        image_paths_test (list): List of filenames for the test set. """

    results = {}

    for subset in ['val']:
        img_dir  = os.path.join(img_path, subset)
        filenames = sorted([fname for fname in os.listdir(img_dir) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))])

        # Build full paths to apply filtering
        image_paths = [os.path.join(img_dir, fname) for fname in filenames]

        # Filter images based on size
        image_paths_filtered, suppressed = filter_large_images(image_paths, subset)
        filtered_filenames = [os.path.basename(p) for p in image_paths_filtered]

        results[subset] = filtered_filenames
        print(f"Number of {subset} images suppressed: {suppressed}")

    return results['val']



def get_filenames_and_labels(images_path, ground_truth_rotations):
    """ Get the filenames and labels for the training and validation sets. The images are filtered by size using the provided filter function.
    
    Parameters:
        images_path (str): Path to the folder containing the images. The folder should contain subfolder 'test'.
        ground_truth_rotations (str): Path to the JSON file containing the ground truth rotations for the images.

    Returns:
        filenames_test (list): List of filenames for the test set.
        labels_test (list): List of labels corresponding to the images in the test set. """

    # Load the images and masks for training and validation sets and filter them by size 
    filenames_test = extract_datasets(images_path, filter_by_rotated_size_threshold)

    # Load the labels
    with open(ground_truth_rotations, 'r') as f:
        label_dict = json.load(f)
    labels_test = [label_dict[fname] for fname in filenames_test]
    
    return filenames_test, labels_test



def vector_to_angle_deg(v):
    """ Convert a 2D vector to an angle in degrees.

    Parameters:
        v (torch.Tensor): A 1D or 2D tensor representing the vector(s).
    
    Returns:
        torch.Tensor: The angle in degrees, normalized to the range [0, 360). """
    
    if v.dim() == 1:
        v = v / (v.norm() + 1e-8)
        return torch.atan2(v[1], v[0]) * 180 / torch.pi % 360
    else:
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        return torch.atan2(v[:, 1], v[:, 0]) * 180 / torch.pi % 360



def angluar_error_deg(true_angle_deg, pred_angle_deg):
    """ Calculate the angular error between true and predicted angle. 
    
    Parameters:
        true_angle_deg (float): True angle in degrees.
        pred_angle_deg (float): Predicted angle in degrees.
    
    Returns:
        angular_diff (float): Angular difference wrapped to the range [-180, 180] degrees. """

    # Ensure tensors are on CPU and converted to NumPy
    if torch.is_tensor(true_angle_deg):
        true_angle_deg = true_angle_deg.cpu().numpy()
    if torch.is_tensor(pred_angle_deg):
        pred_angle_deg = pred_angle_deg.cpu().numpy()
        
    diff = true_angle_deg - pred_angle_deg  # Compute raw difference
    angular_diff = np.remainder(diff + 180, 360) - 180  # Wrap difference to [-180, 180]
    return angular_diff



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



def apply_model_on_test_set(model, test_loader, device):
    """ Apply the trained model on the test set and return the predictions and labels.

    Parameters:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        true_labels (np.ndarray): Array of labels.
        preds (np.ndarray): Array of predictions. """
    
    model.eval()
    true_labels = []
    preds = []

    with torch.no_grad():
        for image, _, label, pos in test_loader:        
            image = image.to(device)
            label = label.to(device).float().view(-1)
            
            # Predict the angle of rotation
            output = model(image, pos).view(-1)

            # Convert the label and output to degrees (0-360)
            angle_deg_label = vector_to_angle_deg(label)
            true_labels.append(angle_deg_label.cpu().numpy())

            angle_deg_pred = vector_to_angle_deg(output)
            preds.append(angle_deg_pred.cpu().numpy())

    true_labels = np.stack(true_labels)
    preds = np.stack(preds)

    return true_labels, preds



def get_error_metrics(true_labels, preds):
    """ Calculate and print the error metrics for the predictions.
   
    Parameters:
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.
    
    Returns:
        None. Prints the error metrics (include median absolute error, IQR, and accuracy 
        within 5 and 10 degrees) to the console. """
    
    _, abs_angular_diff = calculate_error(true_labels, preds)

    # Metrics
    medae = np.median(abs_angular_diff)
    q1 = np.percentile(abs_angular_diff, 25)
    q3 = np.percentile(abs_angular_diff, 75)
    iqr = q3 - q1  # Interquartile range
    acc_5 = np.mean(abs_angular_diff <= 5) # Accuracy within 5 degrees
    acc_10 = np.mean(abs_angular_diff <= 10) # Accuracy within 10 degrees

    # Print the metrics
    print(f"Median abs error: {medae:.4f} deg") 
    print(f"IQR: {iqr:.4f} deg [Q1: {q1:.4f}, Q3: {q3:.4f}]")
    print(f"Accuracy within 5 deg: {acc_5:.4f}") 
    print(f"Accuracy within 10 deg: {acc_10:.4f}")