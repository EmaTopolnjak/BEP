# Bachelor End Project - Standardizing Rotation in WSIs 
A Vision Transformer (ViT) adapted for angle regression to predict object orientation.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#Project-structure)
- [Configuration](#Configuration)
- [Usage](#usage)
- [Dataset](#dataset)

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
To create a new (pip) environment, run the following: 
```bash
python -m venv venv_name
venv_name\Scripts\activate    
```

### 3. Install dependencies
You can install the required packages directly in your environment by running the following command:
```bash
pip install -r requirements.txt
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
│   └── iterative_prediction.py            # If evaluation of the model based on iterative predictions is preferred 
├── extra_codes/                            # Extra codes
│   ├── observer_variability.py             # Calculates the inter-oberver variability between 3 observers and intra-oberver variability between 2 obervations
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
HE_ground_truth_rotations_seperate = ['../Data/manual_rotations/image_rotations_HE_pt1.json', '../Data/image_rotations_HE_pt2.json']
IHC_ground_truth_rotations_seperate = ['../Data/manual_rotations/image_rotations_IHC_pt1.json', '../Data/image_rotations_IHC_pt2.json']

# The final manual rotation filepaths
HE_ground_truth_rotations = '../Data/manual_rotations/image_rotations_HE.json'
IHC_ground_truth_rotations = '../Data/manual_rotations/image_rotations_IHC.json'

### Image and mask paths ###
# Image and mask paths before any preprocessing is done (thus raw crops that are that have background mask)
HE_crops_masked = '../Data/images/HE_crops_masked'
IHC_crops_masked = '../Data/images/IHC_crops_masked'
HE_masks = '../Data/annotations/HE_crops'
IHC_masks = '../Data/annotations/IHC_crops'

# Image and mask paths after they are padded minimally - these images are used for input in the model
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
 
### Observer variability ###
observer_1_1_HE = '../Data/observer_variability/obs1_1_HE.json'
observer_1_2_HE = '../Data/observer_variability/obs1_2_HE.json'
observer_1_1_IHC = '../Data/observer_variability/obs1_1_IHC.json'
observer_1_2_IHC = '../Data/observer_variability/obs1_2_IHC.json'
observer_2_HE = '../Data/observer_variability/obs2_HE.json'
observer_2_IHC = '../Data/observer_variability/obs2_IHC.json'
observer_3_HE = '../Data/observer_variability/obs3_HE.json'
observer_3_IHC = '../Data/observer_variability/obs3_IHC.json'
observer_variability_path = '../Data/observer_variability/evaluation/'

### Other configurations ###
random_seed = 42
```

## Usage
After installing the necessary packages and making a `config.py` file, you can run the project as follows:

### 1. Data Preparation (preprocessing directory)
#### 1.1. Open and run pad_images_minimally.py 
This script processed the individual tissue sections from the dataset and saves the padded images and masks such that they are suitable for model input. The tissue sections that do not have a suitable rotation or are of poor quality are skipped and thus not used for model training. 

#### 1.2 OPTIONAL: Open and run data_selection_description.py
If you want to see the evaluation of how many tissue sections are left after this preparation steps, this script can be run.

#### 1.3 OPTIONAL: Open and run rotate_images_correctly.py
To see the results of rotating the images manually, this script can be run. The tissue sections and their masks are saved into seperate folders.

#### 1.4 Open and run split_dataset.py
This scipt split the dataset on patient-level into a training, validation and test set after non-usable tissue sections are removed. The directories with the data are restructured into a train, val and test subdirectories. 

### 2. Model Training (model_training directory)
#### 2.1 Open and run model_training_loop.py
The model can now be trained. Every 10 epochs, a model is saved to log intermediate results. To choose specific training settings, the `config.py` file can be adjusted.

### 3. Model Evaluation (evaluation directory)
#### 3.1 Open and run complete_evaluation.py
The model is evaluated across 4 combinations, based on two factors: 

a. Rotation:
- Original rotation: The original rotation of the tissue sections is used to provide the evaluation that was present in the dataset. 
- Uniformly rotated: The tissue sections are randomly rotated to remove possible biased towards certain rotations present in the dataset.

b. Number of passes through model:
- One pass through the model: The tissue sections are passed once through the model to evaluate their direct orientation estimation.
- Multiple passes through the model: The tissue sections are passed multiple times through the model and updated in between before the output is evaluated, to evaluate iterative refinement. 

The models are evaluated based on median squared error, interquarile range of mean squared error, accuracy within 5 deg and accuracy within 10 deg. Moreover, the following plots are created: histogram of angular differences, histogram of absolute angular differences, true labels vs predictions, error bias and plots visualizing the predictions. 


## Dataset
The models are trained on H&E- and IHC-stained histology images of skin tissue. Each image has an associated mask and manual rotation label.

Folder structure if the example `config.py` file is used (an alternative structure can be used, but the `config.py` file needs to be adjusted accordingly):
```bash
Data/
├── images/                          
│   ├── HE_crops_masked/                            # Original images of individual tissue sections
│   ├── HE_crops_masked_padded/                     # Images after preparing them for model input - generated after running 'pad_images_minimally.py'
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── IHC_crops_masked/                           # Original images of individual tissue sections
│   └── IHC_crops_masked_padded/                    # Images after preparing them for model input - generated after running 'pad_images_minimally.py'
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
├── annotations/                         
│   ├── HE_crops/                                   # Original masks of individual tissue sections
│   ├── HE_crops_padded/                            # Masks after preparing them for model input - generated after running 'pad_images_minimally.py'
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── IHC_crops/                                  # Original masks of individual tissue sections 
│   └── IHC_crops_padded/                           # Masks after preparing them for model input - generated after running 'pad_images_minimally.py'
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
├── manual_rotations/                               
│   ├── image_rotations_HE.json                     # Manutal rotations of HE-stained images
│   └── image_rotations_IHC.json                    # Manutal rotations of IHC-stained images
├── models/                                     
│   ├── vit_wee_patch16_reg1_gap_256.sbb_in1k.pth   # File with pretrained model weights
│   ├── model.pth                                   # File with trained model weights - generated after running 'model_training_loop.py'
│   ├── training_plot.png                           # Shows the training process - generated after running 'model_training_loop.py'
│   ├── training_log.txt                            # Logs the training process - generated during running 'model_training_loop.py'
│   └── eval/                                       # Folder where evaluation of the model is saved
├── splitting_data/
│   ├── patient_mapping.json                        # File that maps image IDs to patient IDs
│   └── assigned_split.json                         # File that maps image IDs to subset (train, val or test) - generated after running 'split_dataset.py'
└── observer_variability/                           # Folder containing the files to evaluate inter- and intra-observer variability
```

## Contributors
Ema Topolnjak <br>
Course: Bachelor End Project - Medical Image Analysis
