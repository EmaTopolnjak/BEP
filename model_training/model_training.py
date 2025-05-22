# NOTE: Old code, it imports the whole model from timm and uses the pretrained weights. It is incomplete. 

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
    def __init__(self, model_name, dropout_prob, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.hidden_size = self.vit.embed_dim
        self.dropout_prob = dropout_prob
        
        # Remove internal patch embedding
        self.vit.patch_embed = nn.Identity()
        self.vit.pos_drop = nn.Identity()
        self.vit.pos_embed = None
        self.vit.head = nn.Identity()

        # Keep CLS toke, encoder blocks, and norm layer
        self.cls_token = self.vit.cls_token
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm

        self.dropout = nn.Dropout(self.dropout_prob) # Dropout layer for the input embeddings

        # Patch embedding using Conv2d
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Define positional embedder
        self.pos_embedder = PositionalEmbedder(
            embed_dim=self.hidden_size,
            dropout_prob=self.dropout_prob,
        )

        
        self.head = nn.Linear(self.hidden_size, 1)  # For single output regression


    def forward(self, x): 
        B, C, H, W = x.shape # (batch_size, num_patches, channels, height, width)
        
        # Patchify using Conv2d
        x = self.patch_embed(x)  # (B, hidden, H/16, W/16)
        x = x.flatten(2, 3).transpose(1, 2)  # (B, N, hidden)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  

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

    # Load the model
    model = CustomViT(model_name=MODEL_NAME, dropout_prob=DROPOUT_PROB).to(device) # Set the dropout probability

    
    # model_training(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, accumulation_steps=ACCUMULATIONS_STEPS)


    # look at first batch of images to see if the data is loaded correctly
    for batch_idx, (image, label, pos) in enumerate(train_loader):
        if batch_idx < 2:
            continue  # stop after a few batches
        if batch_idx > 2:
            break

        # print(image.shape)
            
        image = image.squeeze(0).permute(1, 2, 0).numpy()  # Convert to numpy array
        label = label.item()

        # plt.imshow(image)
        # plt.title(f"Label: {label}")
        # plt.axis('off')
        # plt.show()

        print(pos)
        print(pos.shape)
    
    