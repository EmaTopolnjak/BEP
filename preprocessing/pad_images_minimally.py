# NOTE: This is the same code as rotate_images_correctly.py, but it does not rotate the images. It only pads the images correctly.

import os
import json
from PIL import Image
import numpy as np
import math

import sys
from pathlib import Path
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

from rotate_images_correctly import rotate_image_with_correct_padding



if __name__ == "__main__":

    # Load the configuration
    rotation_info_path = config.IHC_ground_truth_rotations
    original_images_path = config.IHC_crops_masked
    original_masks_path = config.IHC_masks
    images_for_model_path = config.IHC_crops_masked_padded
    masks_for_model_path = config.IHC_masks_padded

    # Load the rotation info
    with open(rotation_info_path, 'r') as f:
        rotation_info = json.load(f)

    os.makedirs(images_for_model_path, exist_ok=True) # Create the output directory if it doesn't exist
    os.makedirs(masks_for_model_path, exist_ok=True) # Create the output directory if it doesn't exist

    for filename, angle in rotation_info.items(): # Iterate through the dictionary with rotation info
        # Construct the full path to the input and output images
        input_path_image = os.path.join(original_images_path, filename)
        input_path_mask = os.path.join(original_masks_path, filename)
        output_path_image = os.path.join(images_for_model_path, filename)
        output_path_mask = os.path.join(masks_for_model_path, filename)

        # Skip images if marked as "skipped"
        if isinstance(angle, dict) and "skipped" in angle:
            print(f"Skipping {filename} due to 'skipped' flag.") 
            continue
        
        # Rotate the image and save it to the output path
        try:
            img = Image.open(input_path_image)
            mask = Image.open(input_path_mask)
            output_image, output_mask = rotate_image_with_correct_padding(img, mask, 0)
            output_image.save(output_path_image)
            output_mask.save(output_path_mask)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
