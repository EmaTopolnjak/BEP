import config
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
import timm


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # Disable the warning for symlinks



class ImageDataset(Dataset):
    """ A class to create a custom dataset for loading images and labels.
    
    Attributes:
        image_paths (list): List of image file paths.
        labels (list): List of labels corresponding to the images.
        
    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(idx): Returns the image, label and position matrix of the patches for a given index (image in the dataset).
        get_pos(img): Returns the position of each patch in the image. """

    def __init__(self, image_paths, labels):
        """ Custom dataset for loading images and labels.
        
        Parameters:
            image_paths (list): List of image file paths.
            labels (list): List of labels corresponding to the images. """

        self.image_paths = image_paths # List of image file paths
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

        # get the position of each patch in the image
        pos = self.get_pos(img)

        return img, label, pos
    

    def get_pos(self, img, patch_size=16):
        """ Returns the position of each patch in the image cen

        Parameters:
            img (torch.Tensor): Image for position matrix needs to be calculated.
            patch_size (int): Size of the patches. Default is 16.
        
        Returns:
            pos (torch.Tensor): Position matrix of the patches. """
        
        C, H, W = img.shape 
        h_patches = H // patch_size
        w_patches = W // patch_size

        y_range = -(torch.arange(h_patches) - (h_patches // 2))
        x_range = torch.arange(w_patches) - (w_patches // 2)

        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        pos = torch.stack([y_grid, x_grid], dim=2).reshape(-1,2)  # Stack into (H, W, 2) adn then flatten to (H*W, 2)

        return pos
    


class PatchEmbedder(nn.Module):
    """ A class to create a patch embedding layer for the Vision Transformer (ViT) model.

    Attributes:
        proj (nn.Conv2d): Convolutional layer for patch embedding.
    
    Methods:
        forward(x): Applies the patch embedding to the input tensor x. """
    
    def __init__(self, patch_size, in_channels, embed_dim):
        """ Initializes the PatchEmbedder class.
        
        Parameters:
            patch_size (int): Size of the patches.
            in_channels (int): Number of input channels.
            embed_dim (int): Embedding dimension. """
        
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):
        """ Applies the patch embedding to the input tensor x.

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            x (torch.Tensor): Output tensor of shape (B, S, D), where S is the number of patches and D is the embedding dimension. """
        
        x = self.proj(x) # x: (B, C, H, W) -> (B, C', H', W')
        x = x.flatten(2, 3).transpose(1, 2) # [B, C, H, W] -> [B, H*W, C] = [B, S, D]
        return x



class PositionalEmbedder(nn.Module):
    """ A class to create a positional embedding layer for the Vision Transformer (ViT) model.
    
    Attributes:
        embed_dim (int): Embedding dimension.
        pos_embed (nn.Parameter): Positional embedding parameter.
        pos_drop (nn.Dropout): Dropout layer for positional embedding.
    
    Methods:
        forward(x, pos): Applies the positional embedding to the input tensor x and position matrix pos. """

    max_position_index = 100 # maximum number of positions

    def __init__(self, embed_dim, dropout_prob):
        """ Initializes the PositionalEmbedder class.

        Parameters:
            embed_dim (int): Embedding dimension.
            dropout_prob (float): Dropout probability. """
        
        super().__init__()
        self.embed_dim = embed_dim 
        self.pos_embed = nn.Parameter(data=torch.zeros((self.max_position_index+1, self.embed_dim//2)), requires_grad=False) # (max_position_index+1, embed_dim//2)
        X = torch.arange(self.max_position_index+1, dtype=torch.float32).reshape(-1, 1) # (max_position_index+1, 1)
        X = X / torch.pow(10000, torch.arange(0, self.embed_dim//2, 2, dtype=torch.float32) / (self.embed_dim//2)) 
        self.pos_embed[:, 0::2] = torch.sin(X)
        self.pos_embed[:, 1::2] = torch.cos(X)

        self.pos_drop = nn.Dropout(p=dropout_prob) 


    def forward(self, x, pos):
        """ Applies the positional embedding to the input tensor x and position matrix pos.

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, S, D).
            pos (torch.Tensor): Position matrix of shape (B, S, 2).

        Returns:
            x (torch.Tensor): Output tensor of shape (B, S, D). """
        
        pos = torch.round(pos).to(int)
        if torch.max(pos[:, :, 1:]) > self.max_position_index:
            raise ValueError('Maximum requested position index exceeds the prepared position indices.')
        
        # Get the number of items in the batch and the number of tokens in the sequence
        B, S, _ = pos.shape 
        device = self.pos_embed.get_device()
        if device == -1:
            device = 'cpu'

        # Define embeddings for x and y dimension
        embeddings = [self.pos_embed[pos[:, :, 0], :],
                      self.pos_embed[pos[:, :, 1], :]]
        
        # Add a row of zeros as padding in case the embedding dimension has an odd length
        if self.embed_dim % 2 == 1:
            embeddings.append(torch.zeros((B, S, 1), device=device))

        # Prepare positional embedding
        pos_embedding = torch.concat(embeddings, dim=-1)

        # Account for [CLS] token
        pos_embedding = torch.concatenate([torch.zeros((B, 1, self.embed_dim), device=device), pos_embedding], dim=1)

        # Check if the shape of the features and positional embeddings match
        if x.shape != pos_embedding.shape:
            raise ValueError('Shape of features and positional embedding tensors do not match.')
        
        # Add the combined embedding to each element in the sequence
        x = self.pos_drop(x+pos_embedding)
        
        return x



class ViT_for_rotation_prediction(nn.Module):
    """ A class to create a Vision Transformer (ViT) model for rotation prediction.
    
    Attributes:
        patch_size (int): Size of the patches.
        dropout_prob (float): Dropout probability.
        embed_dim (int): Embedding dimension.
        patch_embed (PatchEmbedder): Patch embedding layer.
        positional_embedder (PositionalEmbedder): Positional embedding layer.
        blocks (nn.ModuleList): Transformer blocks.
        norm (nn.LayerNorm): Layer normalization.
        cls_token (nn.Parameter): Class token parameter.
        head (nn.Linear): Linear layer for regression.
    
    Methods:
        forward(x, pos): Forward pass through the model. """

    def __init__(self, model_name, patch_size, dropout_prob):
        """ Initializes the ViT_for_rotation_prediction class.
        
        Parameters:
            model_name (str): Name of the pre-trained ViT model.
            patch_size (int): Size of the patches.
            dropout_prob (float): Dropout probability. """
        
        super().__init__()
        self.patch_size = patch_size
        self.dropout_prob = dropout_prob
        
        # Load the pre-trained ViT model
        vit = timm.create_model(model_name, pretrained=True)
        self.embed_dim = vit.embed_dim

        # Patch embedding layer
        self.patch_embed = PatchEmbedder(patch_size=patch_size, in_channels=3, embed_dim=self.embed_dim)

        # Replace the positional embedding layer with a custom one
        self.positional_embedder = PositionalEmbedder(embed_dim=self.embed_dim, dropout_prob=dropout_prob)

        # Reuse blocks and norm from the pretrained model
        self.blocks = vit.blocks
        self.norm = vit.norm

        # Add cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Replace the head with a regression layer
        # self.head = nn.Linear(self.embed_dim, 1) # linear head as extra option, but I want to use MLP head
        self.head = nn.Sequential( # MLP head
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, 1)
        )


    def forward(self, x, pos):
        """ Forward pass through the model.
        Parameters:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            pos (torch.Tensor): Position matrix of shape (B, S, 2). 
        Returns:
            out (torch.Tensor): Output tensor of shape (B, 1). """
        
        # Patchify the image
        x = self.patch_embed(x)

        # CLS token added
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, S, D] -> [B, 1+S, D]

        # Add positional embeddings
        x = self.positional_embedder(x, pos) # [B, 1+S, D] -> [B, 1+S, D]   

        # Pass through the transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x) # Normalize the output
        out = self.head(x[:, 0])  # Regression value
        return out



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
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.ylim(0, max(max(train_losses), max(val_losses))*1.01)
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
            if accumulation_count >= 2:
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
                if step >= 5:
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
    all_labels = []
    all_preds = []

    counter = 0

    with torch.no_grad():
        for image, label, pos in test_loader:
            # TEMPORARY ONLY 5 BATCHES
            if counter >= 5: 
                break
            counter += 1

            image = image.to(device)
            label = label.to(device).float().view(-1)

            output = model(image, pos).view(-1)
            all_labels.append(label.cpu().numpy())
            all_preds.append(output.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    return all_labels, all_preds



def model_evaluation(true_labels, all_preds):
    """ Evaluate the model performance using Mean Absolute Error (MAE), Median Absolute Error (MedAE), Mean Squared Error (MSE) and accuracy within 5 
    degrees, all for angluar error. The predictions and labels are expected to be in degrees. The function also plots the predictions vs labels and the 
    histogram of angular differences.

    Parameters:
        true_labels (np.ndarray): Array of true labels.
        all_preds (np.ndarray): Array of predictions.
    Returns:
        None. Prints the evaluation metrics.

    Note: The labels and predictions are expected to be in degrees.
    Note: Choose correct metrics to evaluate the model on later. For now, random metrics were chosen to see if the model is working. """

    diff = true_labels - all_preds  # Compute raw difference 
    wrapped_diff = np.remainder(diff + 180, 360) - 180 # wrap difference to [-180, 180]

    # Metrics
    mae = np.mean(np.abs(wrapped_diff))
    medae = np.median(np.abs(wrapped_diff))
    mse = np.mean(wrapped_diff ** 2)
    acc_5 = np.mean(np.abs(wrapped_diff) <= 5)  # Accuracy within 5 degrees

    # Print the metrics
    print(f"MAE: {mae:.4f} degrees")
    print(f"Median MAE: {medae:.4f} degrees")
    print(f"MSE: {mse:.4f} degrees")
    print(f"Accuracy within 5 degrees: {acc_5:.4f}")

    # Plot the predictions vs labels
    plt.figure()
    plt.scatter(true_labels, all_preds, alpha=0.6)
    plt.plot([0, 360], [0, 360], '--', color='gray')  # y=x line
    plt.xlabel("Ground Truth (°)")
    plt.ylabel("Prediction (°)")
    plt.title("Prediction vs Ground Truth")
    plt.grid(True)
    plt.show()

    # Plote the histogram of angluar differencesd
    plt.figure()
    plt.hist(wrapped_diff, bins=50, edgecolor='black')
    plt.xlabel("Angular Error (degrees)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Angular Errors")
    plt.show()
    



if __name__ == "__main__":
    
    # Load the configurations
    HE_ground_truth_rotations = config.HE_ground_truth_rotations
    HE_crops_for_model = config.HE_crops_for_model

    MODEL_NAME = config.model_name
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
    HE_train = HE_crops_for_model + '/train'
    image_paths_train = [os.path.join(HE_train, fname) for fname in os.listdir(HE_train) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

    HE_val = HE_crops_for_model + '/val'
    image_paths_val = [os.path.join(HE_val, fname) for fname in os.listdir(HE_val) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Load the labels
    with open(HE_ground_truth_rotations, 'r') as f:
        label_dict = json.load(f)
    labels_train = [label_dict[os.path.basename(path)] for path in image_paths_train]
    labels_val = [label_dict[os.path.basename(path)] for path in image_paths_val]

    # Create the dataset and dataloader
    train_data = ImageDataset(image_paths=image_paths_train, labels=labels_train)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True) 
    val_data = ImageDataset(image_paths=image_paths_val, labels=labels_val)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = ViT_for_rotation_prediction(model_name=MODEL_NAME, patch_size=16, dropout_prob=DROPOUT_PROB).to(device)

    # # Model training
    # optimized_model = train_model(model, train_loader, val_loader, device, LEARNING_RATE, epochs=NUM_EPOCHS, accumulation_steps=ACCUMULATIONS_STEPS, save_training_plot_path=TRAINING_PLOT_PATH, trained_model_path=TRAINED_MODEL_PATH)

    # # Apply the model on the test set: for now on validation set
    # test_labels, test_pred = apply_model_on_test_set(optimized_model, val_loader)
    # print("Test set predictions and labels directly after training model:")
    # print(test_labels)
    # print(test_pred)

    # # Model evaluation
    # model_evaluation(test_labels, test_pred)

    ## TRYING TO LOAD THE SAVED MODEL
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH))

    # Model evaluation
    test_labels, test_pred = apply_model_on_test_set(model, val_loader)
    model_evaluation(test_labels, test_pred)