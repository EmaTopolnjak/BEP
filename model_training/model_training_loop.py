import random
import os
import sys
import math
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Codes from other files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_training.ViT_model import ViT, convert_state_dict
from preprocessing.rotate_images_correctly import rotate_image_with_correct_padding, get_centroid_of_mask

# To import the config file from the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Set environment variable to allow expandable CUDA memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch



def filter_by_rotated_size_threshold(image_paths, set, max_pixels=1900000): # On GPU, max. is 2,700,000
    """ Filter images and masks based on the estimated size after rotation. The maximum number of pixels is set to 
    1,900,000. The estimated size is calculated based on the maximum size of the image after rotation, which is when the image is 
    rotated by 45 degrees. If the estimated size is less than or equal to the maximum number of pixels, the image and mask are kept. 
    
    Parameters:
        image_paths (list): List of image file paths.
        set (int): Set of the dataset ('train', 'val' or 'test').
        max_pixels (int): Maximum number of pixels allowed after rotation. Default is 2700000.
        
    Returns:
        filtered_images (list): List of filtered image filenames.
        suppressed_count (int): Number of images and masks that were suppressed due to exceeding the maximum pixel threshold. """
    
    filtered_images = []
    suppressed_count = 0

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if set == 'train':
                    new_size = int(math.ceil( (w  + h)*0.5*math.sqrt(2) )) # Max size of the image after rotation is when image is rotated by 45 degrees
                    est_pixels = new_size**2
                else:
                    est_pixels = w * h  # For validation and test sets, use original size because they are not rotated
                if est_pixels <= max_pixels:
                    filtered_images.append(img_path)
                else:
                    suppressed_count += 1

        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
            suppressed_count += 1

    return filtered_images, suppressed_count



def extract_datasets(img_path, filter_large_images):
    """ Extract the datasets (training and validation) from the given image path. The images are filtered based on size using the provided filter function.

    Parameters:
        img_path (str): Path to the folder containing the images. The folder should contain subfolders 'train', 'val' and 'test'.
        filter_large_images (function): Function to filter images based on size. 

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
        image_paths_filtered, suppressed = filter_large_images(image_paths, subset)
        filtered_filenames = [os.path.basename(p) for p in image_paths_filtered]

        results[subset] = filtered_filenames
        print(f"Number of {subset} images suppressed: {suppressed}")

    return results['train'], results['val']



def get_filenames_and_labels(images_path, ground_truth_rotations):
    """ Get the filenames and labels for the training and validation sets. The images are filtered by size using the provided filter function.
    
    Parameters:
        images_path (str): Path to the folder containing the images. The folder should contain subfolders 'train' and 'val'.
        ground_truth_rotations (str): Path to the JSON file containing the ground truth rotations for the images.

    Returns:
        filenames_train (list): List of filenames for the training set.
        filenames_val (list): List of filenames for the validation set.
        labels_train (list): List of labels corresponding to the images in the training set.
        labels_val (list): List of labels corresponding to the images in the validation set. """
    
    # Load the images and masks for training and validation sets and filter them by size 
    filenames_train, filenames_val = extract_datasets(images_path, filter_by_rotated_size_threshold)

    filenames_train = filenames_train[:2]
    filenames_val = filenames_val[:2]

    # Load the labels
    with open(ground_truth_rotations, 'r') as f:
        label_dict = json.load(f)
    labels_train = [label_dict[fname] for fname in filenames_train]
    labels_val = [label_dict[fname] for fname in filenames_val]
    
    return filenames_train, filenames_val, labels_train, labels_val



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

    def __init__(self, image_path, mask_path, subset, filenames, labels, perform_transforms=False, threshold_fine_angle=50):
        """ Custom dataset for loading images and labels.
        
        Parameters:
            image_path (str): Path to the folder containing the images.
            mask_path (str): Path to the folder containing the masks.
            subset (str): Subset of the dataset ('train', 'val', 'test').
            filenames (list): List of filenames for the images in the subset.
            labels (list): List of labels corresponding to the images in the subset.
            perform_transforms (bool): Whether to apply augmentations to the images. Default is False.
            threshold_fine_angle (int): Epoch after which fine-angle augmentations are applied. Default is 50. """
            
        self.image_path = os.path.join(image_path, subset)
        self.mask_path = os.path.join(mask_path, subset)
        self.filenames = filenames 
        self.labels = labels 
        self.epoch = 0 # initial epoch
        self.perform_transform = perform_transforms # Whether to apply augmentations to the images
        self.threshold_fine_angle = threshold_fine_angle # After this epoch, fine-angle augmentations are applied


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
    

    def set_epoch(self, epoch):
        """ Sets the current epoch number. This is used to update the augmentations based on the current epoch.
        Parameters:
            epoch (int): Current epoch number. """
        
        self.epoch = epoch
        

    def update_transform(self):
        """ Updates the augmentations based on the current epoch. It is used to switch to extra fine-angle augmentations after a certain epoch.
        If the current epoch is greater than or equal to the threshold for fine-angle augmentations, the extra small angles are set to True.
        If the current epoch is less than the threshold, the extra small angles are set to False. """

        if self.epoch >= self.threshold_fine_angle:
            self.extra_small_angles = True
        else:
            self.extra_small_angles = False


    def augment_image(self, img, mask, idx):
        """ Augment the image and mask by addjusting brightness, contrast, saturation and hue. Also the image is flipped vertically with
        a probability of 50%. The image and mask are rotated by.. If the extra small angles are set to True, the angle is biased towards 0 
        degrees with a probability of 25%. Otherwise, the angle is uniformly distributed between 10 and 350 degrees.
        
        Parameters:
            img (PIL.Image): Image to be augmented.
            mask (PIL.Image): Mask to be augmented.
        
        Returns:
            img (PIL.Image): Augmented image.
            mask (PIL.Image): Augmented mask.
            angle (float): Rotation of image and mask. """
        
        # Adjust brightness, contrast, saturation and hue
        transform = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        img = transform(img) 

        angle = self.labels[idx] # Get the angle from the labels 

        # Randomly flip the image and mask vertically
        flip_image_vertically = random.choice([True, False]) # Probability of setting Boolean to True is 50%
        if flip_image_vertically: 
            img = ImageOps.mirror(img) 
            mask = ImageOps.mirror(mask)
            angle = 360 - angle # Adjust angle accordingly 

        # Rotate the image and mask based on the calculated angle
        if self.extra_small_angles:
            if random.random() < 0.25:
                # Bias to near 0°
                final_angle = random.uniform(0, 10) if random.random() < 0.5 else random.uniform(350, 360)
            else:
                # General angle range
                final_angle = random.uniform(10, 350)
        else:
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
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.ylim(0, max(max(train_losses), max(val_losses))*1.01)
    plt.savefig(save_path)
    plt.show()



def append_losses_to_file(filename, epoch, train_loss, val_loss, learning_rate):
    """ Append the training and validation loss to a file. If the file does not exist, create it and write the header.
    
    Parameters:
        filename (str): Name of the file to append the losses to.
        epoch (int): Current epoch number.
        train_loss (float): Training loss for the current epoch.
        val_loss (float): Validation loss for the current epoch.
        learning_rate (float): Learning rate used for the current epoch.
    
    Returns:
        None. Appends the losses to the file. If the file does not exist, creates it and writes the header. """
    
    header_needed = not os.path.exists(filename)
    with open(filename, "a") as f:
        if header_needed:
            f.write(f"epoch\ttrain_loss\tvalidation_loss\tlearning_rate\n")
        f.write(f"{epoch}\t{train_loss:.5f}\t{val_loss:.5f}\t{learning_rate:.2e}\n")



def train_model(model, stain, train_dataset, train_loader, val_loader, device, learning_rate, epochs, accumulation_steps, save_training_plot_path, trained_model_path, training_log_path):
    """ Train the model on the training set and validate on the validation set. The loss is calculated using circular mean squared error.
    The model is trained using gradient accumulation to reduce memory usage. The training and validation loss are plotted and saved. The 
    trained model is saved to a file.

    Parameters:
        model (nn.Module): Model to be trained.
        train_dataset (ImageDataset): Dataset for training.
        train_loader (DataLoader): DataLoader for training dataset.
        val_loader (DataLoader): DataLoader for validation dataset.
        device (torch.device): Device to train the model on (CPU or GPU).
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train the model.
        accumulation_steps (int): Number of gradient accumulation steps.
        save_training_plot_path (str): Path to save the training plot.
        trained_model_path (str): Path to save the trained model.
        training_log_path (str): Path to save the training log.

    Returns:
        model (nn.Module): Trained model. """

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    train_losses = []
    val_losses = []

    print("\nStarting training...")

    for epoch in range(epochs):
        # print(f"\nEpoch {epoch + 1}/{epochs}")
        if stain == 'HE+IHC':
            # If HE+IHC, set the epoch for each dataset separately
            for dataset in train_dataset:
                dataset.set_epoch(epoch)
                dataset.update_transform()
        else:
            # For HE or IHC, set the epoch for the single dataset
            train_dataset.set_epoch(epoch)
            train_dataset.update_transform()  

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
            loss = torch.nn.functional.mse_loss(output, label, reduction='sum') / accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accumulation_count += 1

            # Log the loss for the current batch
            running_loss += loss.item() * accumulation_steps  # Undo scaled loss for logging
        
        avg_train_loss = running_loss / (step+1) # Scale the loss for the number of images that were used 
        train_losses.append(avg_train_loss)
        
        # ---- Validation ----
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
                loss = torch.nn.functional.mse_loss(output, label, reduction='sum')
                val_loss += loss.item()

        torch.cuda.empty_cache() # Clear the cache to free up memory

        avg_val_loss = val_loss / (step+1) # Scale the loss for the number of images that were used
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step(avg_val_loss)  # Update learning rate
        append_losses_to_file(training_log_path, epoch+1, avg_train_loss, avg_val_loss, scheduler.optimizer.param_groups[0]['lr'])

        # every 10 epochs, save the model 
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{os.path.splitext(trained_model_path)[0]}_epoch{epoch+1}{os.path.splitext(trained_model_path)[1]}")

    # Save the model and plot the losses
    torch.save(model.state_dict(), trained_model_path)
    training_plot(train_losses, val_losses, save_training_plot_path)
    print("Training complete.")

    return model



def seed_worker(worker_id):
    """ Set the random seed for each worker to ensure reproducibility. The seed is set based on the global RANDOM_SEED and the worker_id.
    
    Parameters:
        worker_id (int): ID of the worker.
    
    Returns:
        None. Sets the random seed for the worker. """
    
    worker_seed = RANDOM_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)



if __name__ == "__main__":
    
    # Load configuration
    RANDOM_SEED = config.random_seed
    PRETRAINED_MODEL = config.pretrained_model
    LEARNING_RATE = config.learning_rate
    ACCUMULATIONS_STEPS = config.accumulation_steps
    STAIN = config.stain  # 'HE', 'IHC' or 'HE+IHC'
    NUM_EPOCHS = config.num_epochs
    FINE_ANGLE_THRESHOLD = config.fine_angle_epoch_threshold  # Threshold for switching to fine-angle augmentations
    TRAINED_MODEL_PATH = config.trained_model_path
    TRAINING_PLOT_PATH = config.training_plot_path
    TRAINING_LOG_PATH = config.training_log_path
   
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

        filenames_train, filenames_val, labels_train, labels_val = get_filenames_and_labels(images_path, ground_truth_rotations)
        train_data = ImageDataset(image_path=images_path, mask_path=masks_path, subset='train', filenames=filenames_train, labels=labels_train, perform_transforms=True, threshold_fine_angle=FINE_ANGLE_THRESHOLD)
        val_data = ImageDataset(image_path=images_path, mask_path=masks_path, subset='val', filenames=filenames_val, labels=labels_val, perform_transforms=False)

    elif STAIN == 'IHC':
        ground_truth_rotations = config.IHC_ground_truth_rotations
        images_path = config.IHC_crops_masked_padded 
        masks_path = config.IHC_masks_padded

        filenames_train, filenames_val, labels_train, labels_val = get_filenames_and_labels(images_path, ground_truth_rotations)
        train_data = ImageDataset(image_path=images_path, mask_path=masks_path, subset='train', filenames=filenames_train, labels=labels_train, perform_transforms=True, threshold_fine_angle=FINE_ANGLE_THRESHOLD)
        val_data = ImageDataset(image_path=images_path, mask_path=masks_path, subset='val', filenames=filenames_val, labels=labels_val, perform_transforms=False)

    elif STAIN == 'HE+IHC':
        ground_truth_rotations_HE = config.HE_ground_truth_rotations
        images_path_HE = config.HE_crops_masked_padded 
        masks_path_HE = config.HE_masks_padded
        filenames_train_HE, filenames_val_HE, labels_train_HE, labels_val_HE = get_filenames_and_labels(images_path_HE, ground_truth_rotations_HE)
        train_data_HE = ImageDataset(image_path=images_path_HE, mask_path=masks_path_HE, subset='train', filenames=filenames_train_HE, labels=labels_train_HE, perform_transforms=True, threshold_fine_angle=FINE_ANGLE_THRESHOLD)
        val_data_HE = ImageDataset(image_path=images_path_HE, mask_path=masks_path_HE, subset='val', filenames=filenames_val_HE, labels=labels_val_HE, perform_transforms=False)

        ground_truth_rotations_IHC = config.IHC_ground_truth_rotations
        images_path_IHC = config.IHC_crops_masked_padded 
        masks_path_IHC = config.IHC_masks_padded
        filenames_train_IHC, filenames_val_IHC, labels_train_IHC, labels_val_IHC = get_filenames_and_labels(images_path_IHC, ground_truth_rotations_IHC)
        train_data_IHC = ImageDataset(image_path=images_path_IHC, mask_path=masks_path_IHC, subset='train', filenames=filenames_train_IHC, labels=labels_train_IHC, perform_transforms=True, threshold_fine_angle=FINE_ANGLE_THRESHOLD)
        val_data_IHC = ImageDataset(image_path=images_path_IHC, mask_path=masks_path_IHC, subset='val', filenames=filenames_val_IHC, labels=labels_val_IHC, perform_transforms=False)

        # Combine
        train_data = [train_data_HE, train_data_IHC]
        val_data = ConcatDataset([val_data_HE, val_data_IHC])

    else:
        raise ValueError("Invalid stain type. Choose 'HE', 'IHC' or 'HE+IHC'.")  

    # Create dataloader for training and validation set
    if STAIN == 'HE+IHC':
        train_loader = DataLoader(ConcatDataset(train_data), batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=g)
    else:
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = initialize_model(pretrained_weights_path=PRETRAINED_MODEL)
    model = model.to(device)
    
    # Model training
    optimized_model = train_model(model, STAIN, train_data, train_loader, val_loader, device, learning_rate=LEARNING_RATE, epochs=NUM_EPOCHS, accumulation_steps=ACCUMULATIONS_STEPS, save_training_plot_path=TRAINING_PLOT_PATH, trained_model_path=TRAINED_MODEL_PATH, training_log_path=TRAINING_LOG_PATH)

    # Also output all settings
    with open(TRAINING_LOG_PATH, "a") as f:
        f.write(f"\n\nSettings:\n")
        f.write(f"STAIN: {STAIN}\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"FINE ANGLE THRESHOLD: {FINE_ANGLE_THRESHOLD}\n")
        f.write(f"STARTING LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"ACCUMULATIONS_STEPS: {ACCUMULATIONS_STEPS}\n")