# Bachelor End Project - Standardizing Rotation in WSIs 
A Vision Transformer (ViT) adapted for angle regression to predict object orientation.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#Project-structure)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)


## Overview

This project uses a ViT model adapted to perform angle regression, allowing it to predict the rotation angle of objects in histopatholy images of skin tissue. The model was trained and evaluated on H&E- and IHC-stained images using PyTorch.


## Installation

### 1. Clone the repository
To get started, download the repository to your machine. 
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
├── preprocessing/
├── model_training/
├── evaluation/
├── extra_codes
├── requirements.txt
└── README.md
```

## Configuration

The `config.py` file contains all the parameters needed to train and evaluate the model. You should create or edit this file before running `train.py` or `evaluate.py`.

### Example `config.py`

```python
# config.py


# Paths
a = 'test'
```

## Usage
After installing the necessary packages, you can run the project as follows:

### Data preprocessing



## Contributors
Ema Topolnjak
Course: Bachelor End Project - Medical Image Analysis
