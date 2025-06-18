from torch.utils.data import DataLoader, ConcatDataset
import os
import torch
import numpy as np
import sys
import random

# Codes from other files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_training.model_training_loop import initialize_model
import evaluation.evaluation_utils as eval_utils
import evaluation.general_evaluation as general_eval
import evaluation.iterative_prediction as iterative_pred

from pathlib import Path
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def seed_worker(worker_id):
    """ Set the random seed for each worker to ensure reproducibility."""
    worker_seed = RANDOM_SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)



if __name__ == "__main__":
    
    # Load configuration
    RANDOM_SEED = config.random_seed
    PRETRAINED_MODEL = config.pretrained_model
    TRAINED_MODEL_PATH = config.trained_model_path
    EVALUATION_PLOTS_PATH = config.evaluation_plots_path
    STAIN = config.stain  # 'HE' or 'IHC'
    MAX_ITERS = config.max_iters  # Number of iterations for iterative prediction
    
    # Set the random seed for reproducibility
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model with correct weights
    model = initialize_model(pretrained_weights_path=PRETRAINED_MODEL)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    # Both for uniform and non-uniform distribution of angles in test set, get predictions
    for UNIFORM_DISTRIBUTION in [True, False]:
        # Load the images, masks and corresponding rotations
        if STAIN == 'HE':
            ground_truth_rotations = config.HE_ground_truth_rotations
            images_path = config.HE_crops_masked_padded 
            masks_path = config.HE_masks_padded

            filenames_test, labels_test = eval_utils.get_filenames_and_labels(images_path, ground_truth_rotations)
            test_data = eval_utils.ImageDataset(image_path=images_path, mask_path=masks_path, subset='val', filenames=filenames_test, labels=labels_test, uniform_distribution=UNIFORM_DISTRIBUTION)

        elif STAIN == 'IHC':
            ground_truth_rotations = config.IHC_ground_truth_rotations
            images_path = config.IHC_crops_masked_padded 
            masks_path = config.IHC_masks_padded

            filenames_test, labels_test = eval_utils.get_filenames_and_labels(images_path, ground_truth_rotations)
            test_data = eval_utils.ImageDataset(image_path=images_path, mask_path=masks_path, subset='val', filenames=filenames_test, labels=labels_test,  uniform_distribution=UNIFORM_DISTRIBUTION)

        elif STAIN == 'HE+IHC':
            ground_truth_rotations_HE = config.HE_ground_truth_rotations
            images_path_HE = config.HE_crops_masked_padded 
            masks_path_HE = config.HE_masks_padded
            filenames_test_HE, labels_test_HE = eval_utils.get_filenames_and_labels(images_path_HE, ground_truth_rotations_HE)
            test_data_HE = eval_utils.ImageDataset(image_path=images_path_HE, mask_path=masks_path_HE, subset='val', filenames=filenames_test_HE, labels=labels_test_HE,  uniform_distribution=UNIFORM_DISTRIBUTION)

            ground_truth_rotations_IHC = config.IHC_ground_truth_rotations
            images_path_IHC = config.IHC_crops_masked_padded 
            masks_path_IHC = config.IHC_masks_padded
            filenames_test_IHC, labels_test_IHC = eval_utils.get_filenames_and_labels(images_path_IHC, ground_truth_rotations_IHC)
            test_data_IHC = eval_utils.ImageDataset(image_path=images_path_IHC, mask_path=masks_path_IHC, subset='val', filenames=filenames_test_IHC, labels=labels_test_IHC,  uniform_distribution=UNIFORM_DISTRIBUTION)

            # Combine
            test_data = ConcatDataset([test_data_HE, test_data_IHC])
        
        # Create dataloader
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g) 


        for ITERATIVE in [False, True]:
            distribution_type = 'uniform' if UNIFORM_DISTRIBUTION else 'non_uniform'
            iterative_type = 'iterative' if ITERATIVE else 'non_iterative'
            EVALUATION_PLOTS_PATH_CURRENT = os.path.join(EVALUATION_PLOTS_PATH, f"{distribution_type}_{iterative_type}")
            os.makedirs(EVALUATION_PLOTS_PATH_CURRENT, exist_ok=True)

            if ITERATIVE:
                print(f"\nEvaluating ITTERATIVE application of model with {'UNIFORM' if UNIFORM_DISTRIBUTION else 'NON-UNIFORM'} distribution of angles in test set...")
                test_pred, test_labels = iterative_pred.pass_dataset_through_model_multiple_times(model, test_data, test_loader, device, EVALUATION_PLOTS_PATH_CURRENT, max_iters=MAX_ITERS)
                np.savez(f'{EVALUATION_PLOTS_PATH_CURRENT}/arrays.npz', array1=test_pred, array2=test_labels)                 
            else:
                print(f"\nEvaluating ONE TIME application of model with {'UNIFORM' if UNIFORM_DISTRIBUTION else 'NON-UNIFORM'} distribution of angles in test set...")
                test_labels, test_pred = eval_utils.apply_model_on_test_set(model, test_loader, device) 
                np.savez(f'{EVALUATION_PLOTS_PATH_CURRENT}/arrays.npz', array1=test_pred, array2=test_labels)


            eval_utils.get_error_metrics(test_labels, test_pred) # Get error metrics
            general_eval.plot_errors(test_labels, test_pred, EVALUATION_PLOTS_PATH_CURRENT) # Visualize errors
            general_eval.plot_predictions_based_on_percentile(test_data, test_labels, test_pred, EVALUATION_PLOTS_PATH_CURRENT) # Visualize errors with images
