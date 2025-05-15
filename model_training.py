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

        return img, label
    


class CustomViT(nn.Module):
    def __init__(self, pretrained_vit, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.vit = pretrained_vit # Load the pretrained ViT model
        hidden_size = pretrained_vit.config.hidden_size
        
        # Patch embedding using Conv2d
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Use the same positional embedding and class token as the pretrained model
        self.pos_embed_pretrained = pretrained_vit.vit.embeddings.position_embeddings
        self.cls_token = pretrained_vit.vit.embeddings.cls_token

        self.dropout = nn.Dropout(0.1) # Dropout layer for the input embeddings
        self.encoder = pretrained_vit.vit.encoder
        self.norm = pretrained_vit.vit.layernorm
        self.head = nn.Linear(hidden_size, 1)  # for single output regression


    def interpolate_pos_embed(self, grid_h, grid_w):
        cls_pos = self.pos_embed_pretrained[:, :1, :] # (1, 1, D)
        patch_pos = self.pos_embed_pretrained[:, 1:, :] # (1, N, D)

        orig_num = patch_pos.shape[1]
        orig_grid_size = int(orig_num ** 0.5)
        assert orig_grid_size ** 2 == orig_num, "Pretrained pos_embed must be square"
        
        # Reshape positional embedding to 2D grid
        patch_pos = patch_pos.reshape(1, orig_grid_size, orig_grid_size, -1).permute(0, 3, 1, 2)  # (1, D, H, W)

        # Interpolate to new (H, W)
        new_patch_pos = F.interpolate(
            patch_pos,
            size=(grid_h, grid_w),
            mode='bicubic',
            align_corners=False
        )
        
        # Reshape back to (1, N_new, D)
        new_patch_pos = new_patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
        
        return torch.cat((cls_pos, new_patch_pos), dim=1)  # (1, N+1, D)


    def forward(self, x):
        B, C, H, W = x.shape # (batch_size, num_patches, channels, height, width)
        
        # Patchify using Conv2d
        x = self.patch_embed(x)  # (B, hidden, H/16, W/16)
        grid_h, grid_w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden)
        
        # Interpolate positional embeddings and add class token
        pos_embed = self.interpolate_pos_embed(grid_h, grid_w)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  
        x = x + pos_embed[:, :x.size(1), :]

        x = self.dropout(x)
        x = self.encoder(x).last_hidden_state
        x = self.norm(x)
        x = self.head(x[:, 0]) # regression head by only using the class token output 
        return x



def run_validation(model, val_loader, max_batches=5):
    model.eval()
    total_loss = 0.0
    count = 0

    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break  # Stop after 5 images

            patches = patches
            labels = labels.float().unsqueeze(1)

            outputs = model(image)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            count += 1

    model.train()  # back to training mode
    return total_loss / count if count > 0 else float('nan')



def model_training(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4, accumulation_steps=8):

    model.train()
    print("\nStarting training...")
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    num_batches_to_try = 10  # for testing purposes, only train on a few batches 

    for batch_idx, (image, label) in enumerate(train_loader):
        if batch_idx >= num_batches_to_try:
            break  # stop after a few batches

        label = label.float().unsqueeze(1)

        # Forward pass
        outputs = model(image)

        # Compute loss
        loss = loss_fn(outputs, label)

        # Scale loss so final update is an average
        loss = loss / accumulation_steps

        # Backward pass 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the model every 'accumulation_steps' batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"[Batch {batch_idx}] Update! Loss: {loss.item():.4f}")

            # Run validation
            val_loss = run_validation(model, val_loader, max_batches=5)
            print(f"[Validation] Loss after update: {val_loss:.4f}")

        else:
            print(f"[Batch {batch_idx}] Accumulating. Loss: {loss.item():.4f}")





if __name__ == "__main__":
    
    # Load the configurations
    model_name = config.model_name
    HE_ground_truth_rotations = config.HE_ground_truth_rotations
    HE_crops_for_model = config.HE_crops_for_model
    batch_size = config.batch_size 
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    accumulation_steps = config.accumulation_steps

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
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False) # TODO: for now, only batch size of 1 works - modify this later?
    val_data = ImageDataset(image_paths=image_paths_val, labels=labels_val)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False) 

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = ViTForImageClassification.from_pretrained(model_name).to(device) # Load the pretrained ViT model

    # Create the custom model
    model = CustomViT(pretrained_vit=pretrained_model)

    
    model_training(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, accumulation_steps=accumulation_steps)




    