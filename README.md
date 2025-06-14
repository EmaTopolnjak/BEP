# Bachelor End Project - Standardizing Rotation in WSIs 
A Vision Transformer (ViT) adapted for angle regression to predict object orientation.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#Project-structure)
- [Configuration](#Configuration)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)


## Overview

This project uses a ViT model adapted to perform angle regression, allowing it to predict the rotation angle of objects in histopatholy images of skin tissue. The model was trained and evaluated on H&E- and IHC-stained images using PyTorch.


## Installation

### 1. Clone the repository
To get started, download the repository to your machine: 
```bash
git clone https://github.com/EmaTopolnjak/BEP.git
cd BEP
```

###  2. Create a virtual environment
To create a new environment, run the following: 
```bash
conda create --name <myenv>
conda activate <myenv>
```

### 3. Install dependencies
You can install the required packages directly in your environment by running the following command:
```bash
conda install --file requirements.txt
```

## Project Structure
This project is organized as follows:
```bash
BEP/
├── preprocessing/                          # Contains scripts for preprocessing raw data
│   ├── data_selection_description.py       # Describes the dataset after manual rotation  
│   ├── rotate_images_correctly.py          # Rotates and pads images based on manual angles (used as a helper, not essential for preprocessing)
│   ├── pad_images_minimally.py             # Pads the images for model input
│   └── split_dataset.py                    # Splits the data by patient into training, validation, and test sets
├── model_training/                         # Directory for training-related code
│   ├── model_training_loop.py              # Script to train the ViT model
│   ├── ViT_model.py                        # Contains the ViT architecture used in this project
│   └── ViT_utils.py                        # Helper functions related to the ViT model
├── evaluation/                             # Directory for evaluating the trained model
│   ├── evaluation_utils.py                 # Helper functions for evaluation
│   ├── complete_evaluation.py              # Runs complete evaluation across all test cases
│   ├── general_evaluation.py               # If evaluation of the model based on one pass through the model is preferred
│   └── itterative_prediction.py            # If evaluation of the model based on itterative predictions is preferred 
├── extra_codes                             # Extra codes
│   └── combine_annotations_to_one_file.py  # Merges multiple manual rotation files into a single one (if applicable)
├── config.py                               # Configuration file (not yet created, but required to run the project)
├── requirements.txt                        # Lists Python dependencies for the project
└── README.md                               # Project documentation and usage instructions
```

## Configuration

The `config.py` file contains all the parameters needed to train and evaluate the model. This file should be created or edited before running the codes. 

### Example `config.py`

```python
# config.py

### Manual rotation files ###
# If multiple files need to merged with the 'combine_annotations_to_one_file.py' file, list the files here
HE_ground_truth_rotations_seperate = ['../Data/image_rotations_HE_pt1.json', '../Data/image_rotations_HE_pt2.json']
IHC_ground_truth_rotations_seperate = ['../Data/image_rotations_IHC_pt1.json', '../Data/image_rotations_IHC_pt2.json']

# The final manual rotation filepaths
HE_ground_truth_rotations = '../Data/manual_rotations/image_rotations_HE.json'
IHC_ground_truth_rotations = '../Data/manual_rotations/image_rotations_IHC.json'

### Image and mask paths ###
# Image and mask paths before any preprocessing is done (thus raw crops that are that have background mask)
HE_crops_masked = '../Data/images/HE_crops_masked'
IHC_crops_masked = '../Data/images/IHC_crops_masked'
HE_masks = '../Data/annotations/HE_crops'
IHC_masks = '../Data/annotations/IHC_crops'

# Image and mask paths after they are padded minimally
HE_crops_masked_padded = '../Data/images/HE_crops_masked_padded'
IHC_crops_masked_padded = '../Data/images/IHC_crops_masked_padded'
HE_masks_padded = '../Data/annotations/HE_crops_padded'
IHC_masks_padded = '../Data/annotations/IHC_crops_padded'

# Image and mask paths for saving the images and masks if you want to rotate them based on manual rotation
HE_crops_masked_rotated = '../Data/images/HE_crops_masked_rotated'
IHC_crops_masked_rotated = '../Data/images/IHC_crops_masked_rotated'
HE_masks_rotated = '../Data/annotations/HE_crops_rotated'
IHC_masks_rotated = '../Data/annotations/IHC_crops_rotated'

### Spliting dataset ###
patient_mapping = '../Data/splitting_data/patient_mapping.json'
assigned_split = '../Data/splitting_data/assigned_split.json'

### Model training variables ###
# These can be adjusted if different taining settings are preferred
pretrained_model = '../Data/models/vit_wee_patch16_reg1_gap_256.sbb_in1k.pth'
stain = 'HE'  # 'HE', 'IHC' or 'HE+IHC'
num_epochs = 80
learning_rate = 2e-5 # Learning rate that is started with
accumulation_steps = 20 # After this many batches, gradient accumulation happens
fine_angle_epoch_threshold = 30 # After this many epochs, the model becomes biased for small angles

### Model output ###
training_plot_path = '../Data/models/training_plot.png'
trained_model_path = '../Data/models/model.pth'
training_log_path = '../Data/models/training_log.txt'

### Model evaluation ###
evaluation_plots_path = '../Data/models/eval/'
max_iters = 5 # After how many iterations through the model, you want to give an evaluation of the prediction
uniform_distribution = True # If True, the test images are randomly rotated to create a uniform distribution across all possible rotations

### Other configurations ###
random_seed = 42
```

## Usage
After installing the necessary packages, you can run the project as follows:

### Data preprocessing



## Contributors
Ema Topolnjak <br>
Course: Bachelor End Project - Medical Image Analysis
