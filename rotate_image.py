import matplotlib.pyplot as plt
import json
from PIL import Image
import os
import random

path = 'data/rotations_HE/image_rotations_HE.json'
folder_images = '../../tissue_alignment/data/images/HE_crops_masked'


with open(path, 'r') as f:
    data = json.load(f)


# Take 5 random images from the dictionary
items = random.sample(list(data.items()), 5)


fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(4, 8))

for i, (image_filename, angle) in enumerate(items):
    full_path = os.path.join(folder_images, image_filename)

    original = Image.open(full_path)
    angle_for_rotation = 360-float(angle)
    rotated = original.rotate(angle_for_rotation, expand=True, fillcolor='white')

    # Left column: original
    axes[i, 0].imshow(original)
    axes[i, 0].set_title(f"Original image")
    axes[i, 0].axis('off')

    # Right column: rotated
    axes[i, 1].imshow(rotated)
    axes[i, 1].set_title(f"Rotated {angle:.0f}°")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()