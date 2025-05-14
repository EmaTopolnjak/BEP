import config
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification
from PIL import Image
import os
import torch
import torch.nn.functional as F
import numpy as np
import json


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # Disable the warning for symlinks



class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, patch_size=16):
        # Initialize the dataset with image paths and labels
        self.image_paths = image_paths # List of image file paths
        self.labels = labels # List of labels corresponding to the images
        self.patch_size = patch_size # Size of the patches to be extracted from the images


    def __len__(self):
        return len(self.image_paths) # Needed for PyTorch Dataset


    def __getitem__(self, idx):
        # Load image and label, and apply transformations (normalization, patchify)
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = torch.from_numpy(np.array(img)).float() / 255.0  # Normalize to [0,1]
        img = img.permute(2, 0, 1)  # From (H, W, C) to (C, H, W)
        label = self.labels[idx]

        # Manually patchify
        patches = self._patchify(img) # (num_patches, C, patch_size, patch_size)

        # Get grid size
        grid_h = img.shape[1] // self.patch_size  # Height in patches
        grid_w = img.shape[2] // self.patch_size  # Width in patches

        return patches, label, (grid_h, grid_w)  
    

    def _patchify(self, img):
        # Convert image to patches
        C, H, W = img.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image dimensions ({H}, {W}) must be divisible by patch size {self.patch_size}"

        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size) # Decompose into patches
        
        # Rearrange to (C, grid_h * grid_w, patch_size, patch_size)
        patches = patches.contiguous().view(C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1, 0, 2, 3)  # (num_patches, C, patch_size, patch_size)
        
        return patches



class CustomViT(nn.Module):
    def __init__(self, pretrained_vit, patch_dim=768):
        super().__init__()
        self.vit = pretrained_vit # Load the pretrained ViT model
        hidden_size = pretrained_vit.config.hidden_size
        self.proj = nn.Linear(3 * 16 * 16, hidden_size)  # flatten patch and project to hidden size

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


    def forward(self, x, grid_h, grid_w):
        B, N, C, H, W = x.shape # (batch_size, num_patches, channels, height, width)
        x = x.view(B, N, -1) # Flatten patches to (B, N, C*H*W)
        x = self.proj(x)  # (B, N, hidden_size)
        
        # Interpolate positional embeddings and add class token
        pos_embed = self.interpolate_pos_embed(grid_h, grid_w)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  
        x = x + pos_embed[:, :x.size(1), :]

        x = self.dropout(x)
        x = self.encoder(x).last_hidden_state
        x = self.norm(x)
        x = torch.sigmoid(self.head(x[:, 0])) # regression head by only using the class token output and applying sigmoid to bound it to [0, 1]
        return x



if __name__ == "__main__":
    
    # Load the configurations
    model_name = config.model_name
    HE_ground_truth_rotations = config.HE_ground_truth_rotations
    HE_crops_for_model = config.HE_crops_for_model

    # Load the images
    HE_train = HE_crops_for_model + '/train'
    image_paths_train = [os.path.join(HE_train, fname) for fname in os.listdir(HE_train) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Load the labels
    with open(HE_ground_truth_rotations, 'r') as f:
        label_dict = json.load(f)
    labels_train = [label_dict[os.path.basename(path)] for path in image_paths_train]
    labels_train = torch.tensor(labels_train) / 360 # Normalize labels to [0, 1] range 

    # Create the dataset and dataloader
    dataset = ImageDataset(image_paths=image_paths_train, labels=labels_train)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # TODO: for now, only batch size of 1 works - modify this later?

    # Load the model
    pretrained_model = ViTForImageClassification.from_pretrained(model_name)

    # # Create the custom model
    model = CustomViT(pretrained_vit=pretrained_model)
    model.eval() 

    # # Get one batch (image) from the dataloader
    # patches, label, (grid_h, grid_w) = next(iter(dataloader))

    # # Forward pass (no gradients needed)
    # for i, (patches, label, (grid_h, grid_w)) in enumerate(dataloader):
    
    #     label = label.float().unsqueeze(1)
        
    #     with torch.no_grad():
    #         output = model(patches, grid_h=grid_h.item(), grid_w=grid_w.item())

    #     print(f"Image {i}: Prediction = {output.item()}, Ground truth = {label.item()}")

    #     if i >= 20:  # test first 5 images
    #         break





    