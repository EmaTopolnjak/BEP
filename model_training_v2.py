import config
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification
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
    def __init__(self, image_paths, labels):
        # Initialize the dataset with image paths and labels
        self.image_paths = image_paths # List of image file paths
        self.labels = labels # List of labels corresponding to the images


    def __len__(self):
        return len(self.image_paths) # Needed for PyTorch Dataset


    def __getitem__(self, idx):
        # Load image and label, and apply transformations (normalization, patchify)
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = TF.to_tensor(img)  # Convert to tensor and normalize
        label = self.labels[idx]

        # get the position of each patch in the image
        pos = self.get_pos(img)

        return img, label, pos
    
    def get_pos(self, img, patch_size=16):
        # Get the position of each patch in the image
        C, H, W = img.shape 
        h_patches = H // patch_size
        w_patches = W // patch_size

        y_range = -(torch.arange(h_patches) - (h_patches // 2))
        x_range = torch.arange(w_patches) - (w_patches // 2)

        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        pos = torch.stack([y_grid, x_grid], dim=2).reshape(-1,2)  # Stack into (H, W, 2) adn then flatten to (H*W, 2)

        return pos
    



class PatchEmbedder(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        # define the patch embedding layer
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) # (x: (B, C, H, W) -> (B, C', H', W')
        x = x.flatten(2, 3).transpose(1, 2) # [B, C, H, W] -> [B, H*W, C] = [B, S, D]

        return x



class PositionalEmbedder(nn.Module):

    max_position_index = 100 # maximum number of positions

    def __init__(self, embed_dim: int, dropout_prob: float):
        super().__init__()
        # initialize instance attributes
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(data=torch.zeros((self.max_position_index+1, self.embed_dim//2)), requires_grad=False) # (max_position_index+1, embed_dim//2)
        X = torch.arange(self.max_position_index+1, dtype=torch.float32).reshape(-1, 1) # (max_position_index+1, 1)
        X = X / torch.pow(10000, torch.arange(0, self.embed_dim//2, 2, dtype=torch.float32) / (self.embed_dim//2)) 
        self.pos_embed[:, 0::2] = torch.sin(X)
        self.pos_embed[:, 1::2] = torch.cos(X)

        # initialize dropout layer
        self.pos_drop = nn.Dropout(p=dropout_prob)



    def forward(
        self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        pos = torch.round(pos).to(int)
        if torch.max(pos[:, :, 1:]) > self.max_position_index:
            raise ValueError(
                'Maximum requested position index exceeds the prepared position indices.'
            )
        # get the number of items in the batch and the number of tokens in the sequence
        B, S, _ = pos.shape 
        device = self.pos_embed.get_device()
        if device == -1:
            device = 'cpu'

        # define embeddings for x and y dimension
        embeddings = [self.pos_embed[pos[:, :, 0], :],
                      self.pos_embed[pos[:, :, 1], :]]
        # add a row of zeros as padding in case the embedding dimension has an odd length
        if self.embed_dim % 2 == 1:
            embeddings.append(torch.zeros((B, S, 1), device=device))

        # prepare positional embedding
        pos_embedding = torch.concat(embeddings, dim=-1)

        # account for [CLS] token
        pos_embedding = torch.concatenate(
            [torch.zeros((B, 1, self.embed_dim), device=device), pos_embedding], dim=1,
        )
        
        # plt.imshow(pos_embedding[0, ...])
        # plt.show()

        # check if the shape of the features and positional embeddings match
        if x.shape != pos_embedding.shape:
            raise ValueError(
                'Shape of features and positional embedding tensors do not match.',
            )
        # add the combined embedding to each element in the sequence
        x = self.pos_drop(x+pos_embedding)
        
        return x




class CustomViT(nn.Module):
    def __init__(self, model_name, patch_size, dropout_prob):

        super().__init__()
        self.patch_size = patch_size
        self.dropout_prob = dropout_prob
        

        # Load the pre-trained ViT model
        vit = timm.create_model(model_name, pretrained=True)
        self.embed_dim = vit.embed_dim

        # Patch embedding layer
        self.patch_embed = PatchEmbedder(patch_size=patch_size, in_channels=3, embed_dim=self.embed_dim)

        # Replace the patch embedding layer with our custom patch embedder
        self.positional_embedder = PositionalEmbedder(embed_dim=self.embed_dim, dropout_prob=dropout_prob)

        # Reuse blocks and norm from the pretrained model
        self.blocks = vit.blocks
        self.norm = vit.norm

        # Add cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Replace the head with a regression layer
        # self.head = nn.Linear(self.embed_dim, 1) # linear head

        self.head = nn.Sequential( # MLP head
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, 1)
        )


    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:

        # Patchify the image
        x = self.patch_embed(x)

        # CLS toen added
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # [B, S, D] -> [B, 1+S, D]
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional embeddings
        # [B, 1+S, D] -> [B, 1+S, D]
        x = self.positional_embedder(x, pos)     

        # Pass through the transformer blocks
        for block in self.blocks:
            x = block(x)

        # Normalize the output
        x = self.norm(x)

        out = self.head(x[:, 0])  # Regression value

        return out




def circular_mse_loss(pred_deg, target_deg):
    """
    Both pred_deg and target_deg: tensors in degrees, shape (B,)
    """
    
    diff = pred_deg - target_deg  # Step 1: compute raw difference 
    wrapped_diff = torch.remainder(diff + 180, 360) - 180 # wrap difference to [-180, 180]
    circular_mse = torch.mean(wrapped_diff ** 2) # calculate mean squared error
    
    return circular_mse






def train_model(model, train_loader, val_loader, device, learning_rate, epochs=10, accumulation_steps=8, scheduler=None):
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("\nStarting training...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0

        optimizer.zero_grad()

        accumulation_count = 0  # Track number of accumulation steps


        for step, (image, label, pos) in enumerate(train_loader):
            if step > 10:
                break
            image = image.to(device)
            label = label.to(device).float().view(-1)  # ensure shape (B,)
            pos = pos.to(device)

            # Forward pass
            output = model(image, pos)  # shape: (B, 1)
            output = output.view(-1)  # shape: (B,)

            loss = circular_mse_loss(output, label) / accumulation_steps
            loss.backward()

            # Accumulate gradients
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accumulation_count += 1

            running_loss += loss.item() * accumulation_steps  # undo scaled loss for logging
            print(f"Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}", end="\r")

            if accumulation_count >= 5:
                break  # Exit epoch after 5 accumulation steps

        # Optional learning rate scheduler step
        if scheduler:
            scheduler.step()

        avg_train_loss = running_loss / len(train_loader)

        # ---- Validation ----
        print("Validation step")
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for image, label, pos in val_loader:
                if val_batches >= 5:
                    break

                image = image.to(device)
                label = label.to(device).float().view(-1)

                output = model(image, pos).view(-1)
                loss = circular_mse_loss(output, label)
                val_loss += loss.item()

                val_batches += 1
                

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss (5 batches): {avg_val_loss:.4f}")





if __name__ == "__main__":
    
    # Load the configurations
    HE_ground_truth_rotations = config.HE_ground_truth_rotations
    HE_crops_for_model = config.HE_crops_for_model

    MODEL_NAME = config.model_name
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
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False) 
    val_data = ImageDataset(image_paths=image_paths_val, labels=labels_val)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False) 

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomViT(model_name=MODEL_NAME, patch_size=16, dropout_prob=DROPOUT_PROB).to(device)


    train_model(model, train_loader, val_loader, device, LEARNING_RATE, epochs=NUM_EPOCHS, accumulation_steps=ACCUMULATIONS_STEPS, scheduler=None)