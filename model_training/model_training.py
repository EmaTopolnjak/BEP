# NOTE: ...

import json
import math
import os
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

# Local imports
from ViT import ViT, convert_state_dict
from rotate_images_correctly import rotate_image_with_correct_padding, get_centroid_of_mask

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config




def filter_by_rotated_size_threshold(image_paths, max_pixels=500000): # On GPU, max. is 2,700,000
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
    suppressed_count = 0

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                new_size = int(math.ceil( (w  + h)*0.5*math.sqrt(2) )) # Max size of the image after rotation is when image is rotated by 45 degrees
                est_pixels = new_size**2
                if est_pixels <= max_pixels:
                    filtered_images.append(img_path)
                else:
                    suppressed_count += 1

        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
            suppressed_count += 1

    return filtered_images, suppressed_count



def extract_datasets(img_path, filter_large_images):
    """ Extract the datasets for training, validation and test sets. The images and masks are filtered by size to ensure that they are small enough
    to not let the model run out of memory. The maximum number of pixels is set to 2.700.000, which is the maximum size of the image after rotation.

    Parameters:
        img_path (str): Path to the folder containing the images. The folder should contain subfolders 'train', 'val' and 'test'.
        filter_large_images (function): Function to filter images based on size. It should take a list of image paths and return a list of filtered image paths and the number of suppressed images.

    Returns:
        image_paths_train (list): List of filenames for the training set.
        image_paths_val (list): List of filenames for the validation set. """

    results = {}

    for subset in ['train', 'val']:
        img_dir  = os.path.join(img_path, subset)
        filenames = sorted([fname for fname in os.listdir(img_dir) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))])

        # Build full paths to apply filtering
        image_paths = [os.path.join(img_dir, fname) for fname in filenames]

        # Filter images based on size
        image_paths_filtered, suppressed = filter_large_images(image_paths)
        filtered_filenames = [os.path.basename(p) for p in image_paths_filtered]

        results[subset] = filtered_filenames
        print(f"Number of {subset} images suppressed: {suppressed}")

    return results['train'], results['val']



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

    def __init__(self, image_path, mask_path, subset, filenames, labels, perform_transforms=False):
        """ Custom dataset for loading images and labels.
        
        Parameters:
            image_path (str): Path to the folder containing the images.
            mask_path (str): Path to the folder containing the masks.
            subset (str): Subset of the dataset ('train', 'val', 'test').
            filenames (list): List of filenames for the images in the subset.
            labels (list): List of labels corresponding to the images in the subset.
            perform_transforms (bool): Whether to apply augmentations to the images. Default is False."""
            
        self.image_path = os.path.join(image_path, subset)
        self.mask_path = os.path.join(mask_path, subset)
        self.filenames = filenames 
        self.labels = labels 
        self.perform_transform = perform_transforms 


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
            angle_vector (torch.Tensor): Angle vector representing the rotation of the image in radians.
            pos (torch.Tensor): Position matrix of the patches. """

        # Load image and mask
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_path, fname)
        mask_path = os.path.join(self.mask_path, fname)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.perform_transform: # Apply augmentations
            img, mask, angle_deg = self.augment_image(img, mask, idx)
        else:
            angle_deg = self.labels[idx]

        # Convert angle in degrees to vector
        angle_rad = angle_deg * math.pi / 180.0
        angle_vector = torch.tensor([math.cos(angle_rad), math.sin(angle_rad)], dtype=torch.float32)

        # Apply transformations normalization to image
        img = torchvision.transforms.functional.to_tensor(img)  

        # Get the position of each patch in the image
        pos = self.get_pos(img, mask)      

        return img, angle_vector, pos
    

    def augment_image(self, img, mask, idx):
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
        transform = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        img = transform(img) # Apply color jitter to image

        angle = self.labels[idx] # Get the angle from the labels 

        # Randomly flip the image and mask vertically
        flip_image_vertically = random.choice([True, False]) # Probability of setting Boolean to True is 50%
        if flip_image_vertically: # Flip vertically
            img = ImageOps.mirror(img) 
            mask = ImageOps.mirror(mask)
            angle = 360 - angle # Adjust angle accordingly 

        # Apply random rotation 
        added_angle = random.random()*360-180 
        img, mask = rotate_image_with_correct_padding(img, mask, -float(added_angle))
        angle = (angle + added_angle) % 360

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
    converted_state_dict.pop('classifier.weight', None)
    converted_state_dict.pop('classifier.bias', None)

    model.load_state_dict(converted_state_dict, strict=False)

    return model



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



def train_model(model, train_loader, val_loader, device, learning_rate, epochs, accumulation_steps, save_training_plot_path, trained_model_path, training_log_path="training_log.txt"):
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
        training_log_path (str): Path to save the training log file. Default is "training_log.txt".

    Returns:
        model (nn.Module): Trained model.
    
    Note: For now, the model is trained on 2 accumlation steps per epoch and validated on 5 batches. """

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    loss_fn = torch.nn.MSELoss() 
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
            # Get the image, label and position matrix
            image = image.to(device)
            label = label.to(device).float().view(-1)  # ensure shape (B,)
            pos = pos.to(device)

            # Forward pass, compute loss and backpropagate
            output = model(image, pos).view(-1)  # shape output: (B, 1) -> (B,)
            loss = loss_fn(output, label) / accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                # print('Updating weights...')
                optimizer.step()
                optimizer.zero_grad()
                accumulation_count += 1

        # Log the loss for the current batch
        running_loss += loss.item() * accumulation_steps  # Undo scaled loss for logging
        avg_train_loss = running_loss / step # Scale the loss for the number of images that were used 
        train_losses.append(avg_train_loss)
        
        # ---- Validation ----
        print("Validation step")
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for step, (image, label, pos) in enumerate(val_loader):
                # Get the image, label and position matrix
                image = image.to(device)
                label = label.to(device).float().view(-1)
                pos = pos.to(device)

                # Forward pass and compute loss
                output = model(image, pos).view(-1)
                loss = loss_fn(output, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / step # Scale the loss for the number of images that were used
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)  # Update learning rate

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        append_losses_to_file(training_log_path, epoch+1, avg_train_loss, avg_val_loss)

    # Save the model and plot the losses
    torch.save(model.state_dict(), trained_model_path)
    training_plot(train_losses, val_losses, save_training_plot_path)
    print("Training complete.")

    return model



if __name__ == "__main__":
    
    # Load configuration
    PRETRAINED_MODEL = config.pretrained_model
    TRAINED_MODEL_PATH = config.trained_model_path
    TRAINING_PLOT_PATH = config.training_plot_path
    TRAINING_LOG_PATH = config.training_log_path
    BATCH_SIZE = config.batch_size 
    NUM_EPOCHS = config.num_epochs
    LEARNING_RATE = config.learning_rate
    ACCUMULATIONS_STEPS = config.accumulation_steps
    RANDOM_SEED = config.random_seed
    DROPOUT_PROB = config.dropout_prob

    # Load the images, masks and corresponding rotations
    HE_ground_truth_rotations = config.HE_ground_truth_rotations
    HE_images_path = config.HE_crops_masked_padded 
    HE_masks_path = config.HE_masks_padded

    # Set the random seed for reproducibility
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load the images and masks for training and validation sets and filter them by size 
    filenames_train, filenames_val = extract_datasets(
        HE_images_path,
        lambda img_p: filter_by_rotated_size_threshold(img_p)
    )

    # TEMPORARY: Limit the number of training and validation images
    filenames_train = filenames_train[:10]
    filenames_val = filenames_val[:10]

    # Load the labels
    with open(HE_ground_truth_rotations, 'r') as f:
        label_dict = json.load(f)
    labels_train = [label_dict[fname] for fname in filenames_train]
    labels_val = [label_dict[fname] for fname in filenames_val]
    
    # Create the dataset and dataloader
    train_data = ImageDataset(image_path=HE_images_path, mask_path=HE_masks_path, subset='train', filenames=filenames_train, labels=labels_train, perform_transforms=True)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False) # TEMPORARY: dont shuffle
    val_data = ImageDataset(image_path=HE_images_path, mask_path=HE_masks_path, subset='val', filenames=filenames_val, labels=labels_val, perform_transforms=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = initialize_model(pretrained_weights_path=PRETRAINED_MODEL)
    model = model.to(device)
    
    # Model training
    optimized_model = train_model(model, train_loader, val_loader, device, learning_rate=LEARNING_RATE, epochs=NUM_EPOCHS, accumulation_steps=ACCUMULATIONS_STEPS, save_training_plot_path=TRAINING_PLOT_PATH, trained_model_path=TRAINED_MODEL_PATH, training_log_path=TRAINING_LOG_PATH)
