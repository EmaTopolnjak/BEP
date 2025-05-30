# NOTE: code incomplete

from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import math
import torchvision

import sys
from model_training import ImageDataset, initialize_model, filter_by_rotated_size_threshold
from rotate_images_correctly import rotate_image_with_correct_padding, get_centroid_of_mask


from pathlib import Path
# Add parent directory to sys.path
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

    for subset in ['test']:
        img_dir  = os.path.join(img_path, subset)
        filenames = sorted([fname for fname in os.listdir(img_dir) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))])

        # Build full paths to apply filtering
        image_paths = [os.path.join(img_dir, fname) for fname in filenames]

        # Filter images based on size
        image_paths_filtered, suppressed = filter_large_images(image_paths)
        filtered_filenames = [os.path.basename(p) for p in image_paths_filtered]

        results[subset] = filtered_filenames
        print(f"Number of {subset} images suppressed: {suppressed}")

    return results['test']



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
    """ ... """
    
    diff = true_labels - preds  # Compute raw difference
    angluar_diff = np.remainder(diff + 180, 360) - 180  # wrap difference to [-180, 180]
    abs_angluar_diff = np.abs(angluar_diff) # Convert to absolute angular difference

    return angluar_diff, abs_angluar_diff


def get_error_metrics(true_labels, preds):
    
    angluar_diff, abs_angular_diff = calculate_error(true_labels, preds)

    # Metrics
    mae = np.mean(abs_angular_diff)
    medae = np.median(abs_angular_diff)
    mse = np.mean(angluar_diff ** 2)
    acc_5 = np.mean(abs_angular_diff <= 5) # Accuracy within 5 degrees

    # Print the metrics
    print(f"Mean abs error: {mae:.4f} deg") 
    print(f"Median abs error: {medae:.4f} deg") 
    print(f"Mean squared error: {mse:.4f} deg") 
    print(f"Accuracy within 5 deg: {acc_5:.4f}") 



def visualize_predictions_vs_labels(true_labels, preds, evaluation_plots_path):
    
    angluar_diff, _ = calculate_error(true_labels, preds)

    # Plot the predictions vs labels
    plt.figure()
    plt.scatter(true_labels, preds, s=1, alpha=0.6)
    plt.plot([0, 360], [0, 360], '--', color='gray')  # y=x line
    plt.xlabel("Ground Truth (°)")
    plt.ylabel("Prediction (°)")
    plt.title("Prediction vs Ground Truth")
    plt.grid(True)
    path = evaluation_plots_path + 'predictions_vs_labels.png'
    plt.savefig(path)
    plt.show()

    # Plot the histogram of angular differences
    plt.figure()
    plt.hist(angluar_diff, bins=30, edgecolor='black')
    plt.xlabel("Absolute Angular Error (°)")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.grid(True)
    path = evaluation_plots_path + 'histogram_angluar_differences.png'
    plt.savefig(path)
    plt.show()



def get_filenames_based_on_error(true_labels, preds, image_paths):
    """ Get the filenames of the images that are at the 1st, 50th and 99th percentile of absolute angular differences.

    Parameters:
        true_labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predictions.
        image_paths (list): List of image paths.
    Returns:
        filenames_selected (list): List of filenames of the images that are at the 1st, 50th and 99th percentile of absolute angular differences.
        true_labels_selected (list): List of true labels of the images that are at the 1st, 50th and 99th percentile of absolute angular differences.
        preds_selected (list): List of predictions of the images that are at the 1st, 50th and 99th percentile of absolute angular differences. """

    angluar_diff, abs_angluar_diff = calculate_error(true_labels, preds)

    percentiles = [1, 50, 99]

    thresholds = np.percentile(abs_angluar_diff, percentiles)
    selected_indices = [np.argmin(np.abs(abs_angluar_diff - t)) for t in thresholds]
    
    # Get the filenames based on the indices
    filenames_selected = [image_paths[i] for i in selected_indices]
    true_labels_selected = [true_labels[i] for i in selected_indices]
    preds_selected = [preds[i] for i in selected_indices]

    return filenames_selected, true_labels_selected, preds_selected



def visualize_errors_with_images(image_path, mask_path, filenames, true_labels, preds, evaluation_plots_path):
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
    filenames_selected, true_labels_selected, preds_selected = get_filenames_based_on_error(true_labels, preds, filenames)

    fig, ax = plt.subplots(3, 3, figsize=(10, 9))

    row_labels = ['Original image', 'Predicted orientation', 'Manual orientation']
    col_labels = ['1st percentile', '50th percentile', '99th percentile']

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
    for row_idx, label in enumerate(row_labels):
        fig.text(0.03, # x-position (adjust if needed)
            0.97 - (row_idx + 0.5) / 3, # y-position centered in each row
            label, ha='left', va='center', rotation='vertical', fontsize=12)
    
    for col_idx, label in enumerate(col_labels):
        fig.text(0.22 + 0.31 * col_idx, # x-position (tune these based on 3 columns)
            0.98, # y-position (same for all columns)
            label, ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, top=0.95) # Adjust margin for text labels
    path = evaluation_plots_path + 'qualitative_analysis_errors.png'
    fig.savefig(path)
    fig.show()



class ImageDataset_all_rotations(Dataset):
    """ A class to create a custom dataset for loading images and labels.
    
    Attributes:
        image_paths (list): List of image file paths.
        mask_paths (list): List of mask file paths.
        labels (list): List of labels corresponding to the images.
        perform_transform (bool): Whether to apply augmentations to the images.
        
    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(idx): Returns the image, label and position matrix of the patches for a given index (image in the dataset).
        get_pos(img, mask): Returns the position of each patch in the image. 
        augment_image(img, mask): Augments the image and mask by applying random transformations. """

    def __init__(self, image_path, mask_path, filenames, labels):
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
  

    def __len__(self):
        """ Returns the number of images in the dataset."""
        return len(self.filenames) 


    def __getitem__(self, idx):
        """ Returns the image, label and position matrix of the patches for a given index (image in the dataset).
        The image is augmented and transformed to a tensor. The position matrix is calculated based on the centroid of the mask.
        The image is normalized and the mask is converted to a tensor. The angle of rotation is also returned.

        Parameters:
            idx (int): Index of the image and label to retrieve.
        Returns:
            img (torch.Tensor): Image tensor.
            label (torch.Tensor): Label tensor.
            pos (torch.Tensor): Position matrix of the patches. """

        # Load image and mask
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_path, fname)
        mask_path = os.path.join(self.mask_path, fname)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        angle_deg = self.labels[idx]

        correctly_rotated_img, correctly_rotated_mask = rotate_image_with_correct_padding(img, mask, float(angle_deg))
        
        images, labels, positions = [], [], []

        for angle in range(0, 360, 5):
            # Rotate the image and mask
            img_rotated, mask_rotated = rotate_image_with_correct_padding(correctly_rotated_img, correctly_rotated_mask, -float(angle))
            
            # Convert angle in degrees to vector
            angle_rad = angle_deg * math.pi / 180.0
            angle_vector = torch.tensor([math.cos(angle_rad), math.sin(angle_rad)], dtype=torch.float32)
            
            # Apply transformations normalization to image
            img_rotated = torchvision.transforms.functional.to_tensor(img_rotated)  
            
            pos = self.get_pos(img_rotated, mask_rotated)     
            images.append(img_rotated)
            labels.append(angle_vector)
            positions.append(pos)         

        return images, labels, positions
    
    
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



def evaluate_rotation_sensitivity(model, test_loader):

    model.eval()
    true_labels = []
    preds = []

    with torch.no_grad():
        for step, (images, labels, positions) in enumerate(test_loader):
            # Iterate over each rotation
            for i in range(len(images)):
                print('step:', step, 'substep:', i, end='\r')
                image = images[i].to(device)
                label = labels[i].to(device).float().view(-1)
                pos = positions[i].to(device)

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


def plot_prediction_distribution_heatmap(angles, predictions, evaluation_plots_path, bins=72):
    """ angles: list or array of angles (e.g., [0, 1, ..., 359, 0, 1, ..., 359])
    predictions: list of predicted values """

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
    title = evaluation_plots_path + 'prediction_distribution_heatmap.png'
    plt.savefig(title)
    plt.show()


if __name__ == "__main__":
    
    # Load configuration
    PRETRAINED_MODEL = config.pretrained_model
    TRAINED_MODEL_PATH = config.trained_model_path
    RANDOM_SEED = config.random_seed
    evaluation_plots_path = config.evaluation_plots_path

    # Load the images, masks and corresponding rotations
    HE_ground_truth_rotations = config.HE_ground_truth_rotations
    HE_images_path = config.HE_crops_masked_padded 
    HE_masks_path = config.HE_masks_padded

    # Set the random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load the images and masks for test set and filter them by size 
    filenames_test = extract_datasets(
        HE_images_path,
        lambda img_p: filter_by_rotated_size_threshold(img_p)
    )

    # TEMPORARY: Limit the number of test images
    filenames_test = filenames_test[:3]

    # Load the labels
    with open(HE_ground_truth_rotations, 'r') as f:
        label_dict = json.load(f)
    labels_test = [label_dict[fname] for fname in filenames_test]

    # Create the dataset and dataloader
    test_data = ImageDataset(image_path=HE_images_path, mask_path=HE_masks_path, subset='test', filenames=filenames_test, labels=labels_test, perform_transforms=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model wjth correct weights
    model = initialize_model(pretrained_weights_path=PRETRAINED_MODEL)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    model.to(device)

    # Apply the model on the test set
    test_labels, test_pred = apply_model_on_test_set(model, test_loader)

    # Calculate error metrics
    get_error_metrics(test_labels, test_pred)

    # Visualize predictions vs labels
    visualize_predictions_vs_labels(test_labels, test_pred, evaluation_plots_path)

    # Get filenames based on error percentiles
    filenames = get_filenames_based_on_error(test_labels, test_pred, filenames_test)

    # Visualize errors with images
    visualize_errors_with_images(HE_images_path, HE_masks_path, filenames, test_labels, test_pred, evaluation_plots_path)

    # Evaluate rotation sensitivity
    test_data = ImageDataset_all_rotations(image_path=HE_images_path, mask_path=HE_masks_path, filenames=filenames_test, labels=labels_test)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False) 
    true_labels, preds = evaluate_rotation_sensitivity(model, test_loader)
    plot_prediction_distribution_heatmap(true_labels, preds, evaluation_plots_path, bins=72)


