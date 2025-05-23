# NOTE: This does not include augmentations yet, only the basic training loop.

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import json
import matplotlib.pyplot as plt
import math
from ViT import ViT, convert_state_dict


import sys
from pathlib import Path
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



def get_centroid_of_mask(mask):
    """ Get the centroid of the mask.

    Parameters:
    mask (PIL.Image): The mask to find the centroid of.

    Returns:
    centroid (tuple): The (x, y) coordinates of the centroid. """

    # Convert to numpy array
    mask_array = np.array(mask)

    # Get coordinates of foreground pixels (non-zero)
    y_indices, x_indices = np.where(mask_array > 0)

    # Compute centroid
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Mask is empty — no foreground pixels found.")

    # Compute centroid
    x_centroid = x_indices.mean()
    y_centroid = y_indices.mean()
    centroid = (x_centroid, y_centroid)
    return centroid


class ImageDataset(Dataset):
    """ A class to create a custom dataset for loading images and labels.
    
    Attributes:
        image_paths (list): List of image file paths.
        mask_paths (list): List of mask file paths.
        labels (list): List of labels corresponding to the images.
        
    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(idx): Returns the image, label and position matrix of the patches for a given index (image in the dataset).
        get_pos(img, mask): Returns the position of each patch in the image. """

    def __init__(self, image_paths, mask_paths, labels):
        """ Custom dataset for loading images and labels.
        
        Parameters:
            image_paths (list): List of image file paths.
            mask_paths (list): List of mask file paths.
            labels (list): List of labels corresponding to the images. """

        self.image_paths = image_paths # List of image file paths
        self.mask_paths = mask_paths # List of mask file paths
        self.labels = labels # List of labels corresponding to the images


    def __len__(self):
        """ Returns the number of images in the dataset."""
        return len(self.image_paths) # Needed for PyTorch Dataset


    def __getitem__(self, idx):
        """ Returns the image, label and position matrix of the patches for a given index (image in the dataset).
        1. Load the image and label.
        2. Apply transformations (normalization, patchify).
        3. Get the position of each patch in the image.
        4. Return the image, label and position matrix.

        Parameters:
            idx (int): Index of the image and label to retrieve.
        Returns:
            img (torch.Tensor): Image tensor.
            label (torch.Tensor): Label tensor.
            pos (torch.Tensor): Position matrix of the patches. """

        # Load image and label, and apply transformations (normalization, patchify)
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = TF.to_tensor(img)  # Convert to tensor and normalize
        label = self.labels[idx]

        # Load mask to calculate position matrix
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # get the position of each patch in the image
        pos = self.get_pos(img, mask)      

        return img, label, pos
    

    def get_pos(self, img, mask, patch_size=16):
        """ Returns the position of each patch in the image cen

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
    


def initialize_model(pretrained_weights_path):
    """ Initialize the model with pretrained weights.
    
    Parameters:
        pretrained_weights_path (str): Path to the pretrained weights.
    
    Returns:
        model (nn.Module): Initialized model with pretrained weights. """
    
    # Initialize the model
    model = ViT(
        patch_shape=16,
        input_dim=3,
        embed_dim=256,
        n_classes=1,
        depth=14,
        n_heads=4,
        mlp_ratio=5,
        pytorch_attn_imp=False,
        init_values=1e-5,
    )

    # Load the pretrained weights
    state_dict = torch.load(pretrained_weights_path)

    # Convert the state dict to match the model
    converted_state_dict = convert_state_dict(state_dict=state_dict)
    converted_state_dict['pos_embedder.pos_embed'] = model.state_dict()['pos_embedder.pos_embed']
    converted_state_dict['classifier.weight'] = model.state_dict()['classifier.weight']
    converted_state_dict['classifier.bias'] = model.state_dict()['classifier.bias']

    model.load_state_dict(converted_state_dict)

    return model


def circular_mse_loss(pred_deg, target_deg):
    """ Circular Mean Squared Error Loss for angles in degrees.
    
    Parameters:
        pred_deg (torch.Tensor): Predicted angles in degrees.
        target_deg (torch.Tensor): Target angles in degrees.
    
    Returns:
        torch.Tensor: Circular Mean Squared Error Loss. """
    
    diff = pred_deg - target_deg  # Compute raw difference 
    wrapped_diff = torch.remainder(diff + 180, 360) - 180 # Wrap difference to [-180, 180]
    circular_mse = torch.mean(wrapped_diff ** 2) # Calculate mean squared error
    
    return circular_mse



def training_plot(train_losses, val_losses, save_path):
    """ Plot the training and validation loss over epochs.
    
    Parameters:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        save_path (str): Path to save the plot.
    
    Returns:
        None. Plots the training and validation loss. """
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    # plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    # plt.ylim(0, max(max(train_losses), max(val_losses))*1.01)
    plt.savefig(save_path)
    plt.show()



def train_model(model, train_loader, val_loader, device, learning_rate, epochs, accumulation_steps, save_training_plot_path, trained_model_path):
    """ Train the model on the training set and validate on the validation set. The loss is calculated using circular mean squared error.
    The model is trained using gradient accumulation to reduce memory usage. The training and validation loss are plotted and saved. The 
    trained model is saved to a file.

    Parameters:
        model (nn.Module): Model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to train the model on (CPU or GPU).
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train the model.
        accumulation_steps (int): Number of steps to accumulate gradients before updating weights.
        save_training_plot_path (str): Path to save the training plot.
        trained_model_path (str): Path to save the trained model.

    Returns:
        model (nn.Module): Trained model.
    
    Note: For now, the model is trained on 2 accumlation steps per epoch and validated on 5 batches. """

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    print("\nStarting training...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        optimizer.zero_grad()
        
        running_loss = 0.0
        accumulation_count = 0  # Track number of accumulation steps
        
        # ---- Training ----
        for step, (image, label, pos) in enumerate(train_loader):

            # TEMPORARY ONLY 2 UPDATING WEIGHT STEPS FOR ONE EPOCH
            if accumulation_count >= 1:
                accumulation_count = 0 # reset counter
                break

            image = image.to(device)
            label = label.to(device).float().view(-1)  # ensure shape (B,)
            pos = pos.to(device)

            # Forward pass
            output = model(image, pos).view(-1)  # shape output: (B, 1) -> (B,)

            # Compute loss, scale it and backpropagate
            loss = circular_mse_loss(output, label) / accumulation_steps
            loss.backward()

            # Log the loss for the current batch
            running_loss += loss.item() * accumulation_steps  # undo scaled loss for logging
            print(f"Image {step+1}/{accumulation_steps*2} | Loss: {running_loss:.4f}", end="\r")

            # Update weights every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                print('Updating weights...')
                optimizer.step()
                optimizer.zero_grad()
                accumulation_count += 1

        avg_train_loss = running_loss / step # Scale the loss for the number of images that were used 

        # ---- Validation ----
        print("Validation step")
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for step, (image, label, pos) in enumerate(val_loader):
                # TEMPORARY ONLY 5 BATCHES FOR VALIDATION
                if step >= 2:
                    break
                
                # Get the image, label and position
                image = image.to(device)
                label = label.to(device).float().view(-1)
                pos = pos.to(device)

                # Forward pass
                output = model(image, pos).view(-1)
                loss = circular_mse_loss(output, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / step # Scale the loss for the number of images that were used

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss (5 batches): {avg_val_loss:.4f}")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    training_plot(train_losses, val_losses, save_training_plot_path)
    print("Training complete.")

    # Save the model
    torch.save(model.state_dict(), trained_model_path)

    return model



if __name__ == "__main__":
    
    # Load the configurations
    HE_ground_truth_rotations = config.HE_ground_truth_rotations
    HE_images_for_model = config.HE_crops_masked_rotated
    HE_masks_for_model = config.HE_masks_rotated
    
    PRETRAINED_MODEL = config.pretrained_model
    TRAINED_MODEL_PATH = config.trained_model_path
    TRAINING_PLOT_PATH = config.training_plot_path
    BATCH_SIZE = config.batch_size 
    NUM_EPOCHS = config.num_epochs
    LEARNING_RATE = config.learning_rate
    ACCUMULATIONS_STEPS = config.accumulation_steps
    RANDOM_SEED = config.random_seed
    DROPOUT_PROB = config.dropout_prob

    # Set the random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

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
    
    # Model training
    optimized_model = train_model(model, train_loader, val_loader, device, LEARNING_RATE, epochs=NUM_EPOCHS, accumulation_steps=ACCUMULATIONS_STEPS, save_training_plot_path=TRAINING_PLOT_PATH, trained_model_path=TRAINED_MODEL_PATH)

