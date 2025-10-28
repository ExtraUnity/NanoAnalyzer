from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from src.shared.EvaluationResult import EvaluationResult

class ModelEvaluator():
    @staticmethod
    def __get_single_image_iou(prediction: np.ndarray, ground_truth: np.ndarray):
        assert np.isin(prediction, [0, 1]).all(), "prediction must be binary image"
        assert np.isin(ground_truth, [0, 1]).all(), "ground truth must be binary image"
        epsilon = 1e-6
        intersection = np.logical_and(prediction, ground_truth).sum()
        union = np.logical_or(prediction, ground_truth).sum()
        
        iou = (intersection + epsilon) / (union + epsilon)
        return iou
    
    @staticmethod
    def __get_single_image_dice_score(prediction: np.ndarray, ground_truth: np.ndarray):
        assert np.isin(prediction, [0, 1]).all(), "prediction must be binary image"
        assert np.isin(ground_truth, [0, 1]).all(), "ground truth must be binary image"
        epsilon = 1e-6
        intersection = np.logical_and(prediction, ground_truth).sum()
        dice_score = (2 * intersection + epsilon) / (prediction.sum() + ground_truth.sum() + epsilon)
        return dice_score

    @staticmethod
    def calculate_ious(predictions, ground_truths):
        ious = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            ious.append(ModelEvaluator.__get_single_image_iou(prediction, ground_truth))
        return ious

    @staticmethod
    def calculate_dice_scores(predictions, ground_truths):
        dice_scores = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            dice_scores.append(ModelEvaluator.__get_single_image_dice_score(prediction, ground_truth))
        return dice_scores
    
    @staticmethod
    def _log_individual_results(file_names, ious, dice_scores, log_file_path):
        """
        Log individual results to a tab-separated file.
        
        Args:
            file_names: List of filenames
            ious: List of IoU scores
            dice_scores: List of Dice scores
            log_file_path: Path to the log file
        """
        import os
        
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file_path)
        if log_dir:  # Only create if there's actually a directory path
            os.makedirs(log_dir, exist_ok=True)
        
        with open(log_file_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("Filename\tIOU\tDice\n")
            
            # Write individual results
            for filename, iou, dice in zip(file_names, ious, dice_scores):
                f.write(f"{filename}:\t{iou:.6f}\t{dice:.6f}\n")
            
            # Write summary statistics
            f.write("\n")
            f.write(f"Average:\t{np.mean(ious):.6f}\t{np.mean(dice_scores):.6f}\n")
            f.write(f"Std Dev:\t{np.std(ious):.6f}\t{np.std(dice_scores):.6f}\n")
            f.write(f"Min:\t{np.min(ious):.6f}\t{np.min(dice_scores):.6f}\n")
            f.write(f"Max:\t{np.max(ious):.6f}\t{np.max(dice_scores):.6f}\n")
        
        print(f"Individual results logged to: {log_file_path}")
    
    def get_predictions(unet, dataloader: DataLoader):
        from src.model.DataTools import binarize_segmentation_output, center_crop, construct_image_from_patches, mirror_fill, extract_slices
        inputs = []
        predictions = []
        labels = []
        unet.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                input, label = data           
                input, label = input.to(unet.device), label.to(unet.device)
                label = (label > 0.5).long().squeeze(1)
                stride_length = unet.preferred_input_size[0]*4//5
                tensor_mirror_filled = mirror_fill(input, unet.preferred_input_size, (stride_length,stride_length))
                patches = extract_slices(tensor_mirror_filled, unet.preferred_input_size, (stride_length,stride_length))
                segmentations = np.empty((patches.shape[0], 2, patches.shape[2], patches.shape[3]), dtype=patches.dtype)
                unet.to(input.device)
                patches_tensor = torch.tensor(patches, dtype=input.dtype, device=input.device)

                if unet.device.type == 'cuda':
                    with torch.autocast("cuda"):
                        segmentations = unet(patches_tensor).cpu().detach().numpy()
                else:
                    segmentations = unet(patches_tensor).cpu().detach().numpy()
                segmented_image = construct_image_from_patches(segmentations, tensor_mirror_filled.shape[2:], (stride_length,stride_length))
                segmented_image = center_crop(segmented_image, (input.shape[2], input.shape[3]))
                segmented_image = binarize_segmentation_output(segmented_image)
                predictions.append(torch.tensor(segmented_image, dtype=input.dtype, device=input.device))
                labels.append(label)
                inputs.append(input.cpu())
        return inputs, predictions, labels

    @staticmethod
    def evaluate_model(unet, test_dataloader: DataLoader, test_callback = None, log_file_path = None) -> EvaluationResult:
        inputs, predictions, labels = ModelEvaluator.get_predictions(unet, test_dataloader)
        predictions = [pred.cpu() for pred in predictions]
        labels = [label.cpu() for label in labels]
        ious = ModelEvaluator.calculate_ious(predictions, labels)
        dice_scores = ModelEvaluator.calculate_dice_scores(predictions, labels)
        file_names = test_dataloader.dataset.image_filenames

        # Log individual results to file if path is provided
        if log_file_path:
            ModelEvaluator._log_individual_results(file_names, ious, dice_scores, log_file_path)
        
        print(f"IOUS: {ious}")
        print(f"Dice scores: {dice_scores}")

        number_of_predictions_to_show = np.min([5, len(predictions)]) 
        indicies = random.sample(range(len(predictions)), number_of_predictions_to_show)
        if not test_callback:
            return EvaluationResult(ious, dice_scores)
        try:
            for i in indicies:
                test_callback(inputs[i], predictions[i], labels[i], ious[i], dice_scores[i])
        except Exception as e:
            print(f"Error in test callback: {e}")
        finally:
            return EvaluationResult(ious, dice_scores)

        
        