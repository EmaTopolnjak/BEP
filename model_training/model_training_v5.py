# NOTE: Images that are too large for the model to handle are filtered out before training. The labels are scaled to [0,1] to make it easier for the model to learn. Code does not work because it 
# works on the old position embedding implementation.

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import json

from ViT_old import ViT, convert_state_dict
from rotate_images_correctly import rotate_image_with_correct_padding, get_centroid_of_mask

import sys
from pathlib import Path
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



def filter_by_rotated_size_threshold(image_paths, mask_paths, max_pixels=2700000):
    """ Filter images and masks based on the estimated size after rotation. The maximum number of pixels is set to 
    2.700.000. The estimated size is calculated based on the maximum size of the image after rotation, which is when the image is 
    rotated by 45 degrees. If the estimated size is less than or equal to the maximum number of pixels, the image and mask are kept. 
    
    Parameters:
        image_paths (list): List of image file paths.
        mask_paths (list): List of mask file paths.
        max_pixels (int): Maximum number of pixels allowed after rotation. Default is 2700000.
        
    Returns:
        filtered_images (list): List of filtered image file paths.
        filtered_masks (list): List of filtered mask file paths.
        suppressed_count (int): Number of images and masks that were suppressed due to exceeding the maximum pixel threshold. """
    
    filtered_images = []
    filtered_masks = []
    suppressed_count = 0

    for img_path, mask_path in zip(image_paths, mask_paths):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                new_size = int(math.ceil( (w  + h)*0.5*math.sqrt(2) )) # Max size of the image after rotation is when image is rotated by 45 degrees
                est_pixels = new_size**2
                if est_pixels <= max_pixels:
                    filtered_images.append(img_path)
                    filtered_masks.append(mask_path)
                else:
                    suppressed_count += 1

        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
            suppressed_count += 1

    return filtered_images, filtered_masks, suppressed_count



def get_image_mask_paths_from_filenames(image_folder, mask_folder):
    """ Get the image and mask paths from the filenames in the given directories. The images and masks are sorted by filename.
    
    Parameters:
        image_folder (str): Folder containing the images.
        mask_folder (str): Folder containing the masks.
        
    Returns:
        image_paths (list): List of image file paths.
        mask_paths (list): List of mask file paths. """
    
    filenames = sorted([
        fname for fname in os.listdir(image_folder) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))])
    filenames.sort()

    image_paths = [os.path.join(image_folder, fname) for fname in filenames]
    mask_paths  = [os.path.join(mask_folder, fname)  for fname in filenames]
    return image_paths, mask_paths



def extract_datasets(img_path_train, masks_path_train, img_path_val_test, masks_path_val_test, filter_large_images):
    """ Extract the datasets for training, validation and test sets. The images and masks are filtered by size to ensure that they are small enough
    to not let the model run out of memory. The maximum number of pixels is set to 2.700.000, which is the maximum size of the image after rotation.

    Parameters:
        img_path_train (str): Path to the training images.
        masks_path_train (str): Path to the training masks.
        img_path_val_test (str): Path to the validation and test images.
        masks_path_val_test (str): Path to the validation and test masks.
        filter_large_images (function): Function to filter images and masks by size.

    Returns:
        image_paths_train (list): List of image paths for the training set.
        mask_paths_train (list): List of mask paths for the training set.
        image_paths_val (list): List of image paths for the validation set.
        mask_paths_val (list): List of mask paths for the validation set.
        image_paths_test (list): List of image paths for the test set.
        mask_paths_test (list): List of mask paths for the test set. """

    subsets = {
        "train": (os.path.join(img_path_train, 'train'), os.path.join(masks_path_train, 'train')),
        "val":   (os.path.join(img_path_val_test, 'val'), os.path.join(masks_path_val_test, 'val')),
        "test":  (os.path.join(img_path_val_test, 'test'), os.path.join(masks_path_val_test, 'test'))
    }

    results = {}
    for subset, (img_dir, mask_dir) in subsets.items():
        image_paths, mask_paths = get_image_mask_paths_from_filenames(img_dir, mask_dir)
        image_paths, mask_paths, suppressed = filter_large_images(image_paths, mask_paths)
        results[subset] = (image_paths, mask_paths, suppressed)
        print(f"Number of {subset} images suppressed: {suppressed}")

    return results['train'][0], results['train'][1], \
           results['val'][0], results['val'][1], \
           results['test'][0], results['test'][1]



class ImageDataset(Dataset):
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

    def __init__(self, image_paths, mask_paths, labels, perform_transforms=False):
        """ Custom dataset for loading images and labels.
        
        Parameters:
            image_paths (list): List of image file paths.
            mask_paths (list): List of mask file paths.
            labels (list): List of labels corresponding to the images.
            perform_transforms (bool): Whether to apply augmentations to the images. Default is False. """

        self.image_paths = image_paths # List of image file paths
        self.mask_paths = mask_paths # List of mask file paths
        self.labels = labels # List of labels (angles of rotation) corresponding to the image - this is only known for the validation set
        self.perform_transform = perform_transforms # Transformations to be applied to the images


    def __len__(self):
        """ Returns the number of images in the dataset."""
        return len(self.image_paths) # Needed for PyTorch Dataset


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
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        if self.perform_transform: # Apply augmentations
            img, mask, angle = self.augment_image(img, mask)
        else:
            angle = self.labels[idx]

        angle = angle / 360  # Normalize angle to [0, 1] 

        # Apply transformations (normalization, patchify) to image
        img = TF.to_tensor(img)  

        # Get the position of each patch in the image
        pos = self.get_pos(img, mask)      

        return img, angle, pos
    

    def augment_image(self, img, mask):
        """ Augment the image and mask by addjusting brightness, contrast, saturation and hue. Also the image is flipped vertically with
        a probability of 30%. The image and mask are rotated by a random angle between 0 and 360 degrees. 
        
        Parameters:
            img (PIL.Image): Image to be augmented.
            mask (PIL.Image): Mask to be augmented.
        
        Returns:
            img (PIL.Image): Augmented image.
            mask (PIL.Image): Augmented mask.
            angle (float): Angle of rotation applied to the image and mask. """
        
        # Adjust brightness, contrast, saturation and hue
        transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        img = transform(img) # Apply color jitter to image

        # Randomly flip the image and mask vertically
        flip_image_vertically = random.choices([True, False], weights=[0.3, 0.7])[0] # Probability of setting Boolean to True is 30%
        
        if flip_image_vertically: # Flip vertically
            img = ImageOps.mirror(img) 
            mask = ImageOps.mirror(mask) 

        # Apply random rotation 
        angle = np.random.random() * 360 # Random rotation angle between 0 and 360 degrees
        img, mask = rotate_image_with_correct_padding(img, mask, -float(angle)) # Rotate image and mask with correct padding
        
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
    wrapped_diff = torch.remainder(diff + 0.5, 1) - 0.5 # Wrap difference to [-0.5, 0.5] (because angles are normalized to [0, 1])
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
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.ylim(0, max(max(train_losses), max(val_losses))*1.01)
    plt.savefig(save_path)
    plt.show()



def append_losses_to_file(filename, epoch, train_loss, val_loss):
    """ Append the training and validation loss to a file. If the file does not exist, create it and write the header.
    
    Parameters:
        filename (str): Name of the file to append the losses to.
        epoch (int): Current epoch number.
        train_loss (float): Training loss for the current epoch.
        val_loss (float): Validation loss for the current epoch.
    
    Returns:
        None. Appends the losses to the file. If the file does not exist, creates it and writes the header. """
    
    header_needed = not os.path.exists(filename)
    with open(filename, "a") as f:
        if header_needed:
            f.write(f"Epoch\tTrain Loss\tValidation Loss\n")
        f.write(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\n")



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
            # if accumulation_count >= 20:
            #     accumulation_count = 0 # reset counter
            #     break
            
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
                # # TEMPORARY ONLY 5 BATCHES FOR VALIDATION
                # if step >= 1:
                #     break

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
    
        append_losses_to_file("training_log.txt", epoch+1, avg_train_loss, avg_val_loss)

    training_plot(train_losses, val_losses, save_training_plot_path)
    print("Training complete.")

    # Save the model
    torch.save(model.state_dict(), trained_model_path)

    return model



if __name__ == "__main__":
    
    # Load the configurations
    HE_ground_truth_rotations = config.HE_ground_truth_rotations

    # For training, use the correctly rotated images so it is easier to rotate them for training
    HE_images_for_training = config.HE_crops_masked_rotated 
    HE_masks_for_training = config.HE_masks_rotated

    # For validation, use the images that are not rotated but have original rotation
    HE_images_for_val_test = config.HE_crops_masked_padded
    HE_masks_for_val_test = config.HE_masks_padded
    
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
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load the images and masks for training, validation and test sets and filter them by size 
    image_paths_train, mask_paths_train, image_paths_val, mask_paths_val, image_paths_test, mask_paths_test = extract_datasets(
        HE_images_for_training,
        HE_masks_for_training,
        HE_images_for_val_test,
        HE_masks_for_val_test,
        lambda img_p, mask_p: filter_by_rotated_size_threshold(img_p, mask_p)
    )

    # Load the labels
    with open(HE_ground_truth_rotations, 'r') as f:
        label_dict = json.load(f)
    labels_val = [label_dict[os.path.basename(path)] for path in image_paths_val]

    # Create the dataset and dataloader
    train_data = ImageDataset(image_paths=image_paths_train, mask_paths=mask_paths_train, labels=None, perform_transforms=True)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True) 
    val_data = ImageDataset(image_paths=image_paths_val, mask_paths=mask_paths_val, labels=labels_val, perform_transforms=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = initialize_model(pretrained_weights_path=PRETRAINED_MODEL)
    model = model.to(device)
    
    # Model training
    optimized_model = train_model(model, train_loader, val_loader, device, learning_rate=LEARNING_RATE, epochs=NUM_EPOCHS, accumulation_steps=ACCUMULATIONS_STEPS, save_training_plot_path=TRAINING_PLOT_PATH, trained_model_path=TRAINED_MODEL_PATH)

