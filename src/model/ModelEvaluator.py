from torch.utils.data import DataLoader
import numpy as np
import torch
import random
from src.shared.EvaluationResult import EvaluationResult
from src.model.ParticleMetrics import compute_size_stratified_metrics, save_size_stratified_metrics

class ModelEvaluator():
    DEFAULT_OBJECT_IOU_THRESHOLD = 0.3

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
    def __get_single_image_precision(prediction: np.ndarray, ground_truth: np.ndarray):
        assert np.isin(prediction, [0, 1]).all(), "prediction must be binary image"
        assert np.isin(ground_truth, [0, 1]).all(), "ground truth must be binary image"
        true_positives = np.logical_and(prediction, ground_truth).sum()
        predicted_positives = prediction.sum()
        ground_truth_positives = ground_truth.sum()

        if predicted_positives == 0:
            return 1.0 if ground_truth_positives == 0 else 0.0
        return true_positives / predicted_positives

    @staticmethod
    def __get_single_image_recall(prediction: np.ndarray, ground_truth: np.ndarray):
        assert np.isin(prediction, [0, 1]).all(), "prediction must be binary image"
        assert np.isin(ground_truth, [0, 1]).all(), "ground truth must be binary image"
        true_positives = np.logical_and(prediction, ground_truth).sum()
        predicted_positives = prediction.sum()
        ground_truth_positives = ground_truth.sum()

        if ground_truth_positives == 0:
            return 1.0 if predicted_positives == 0 else 0.0
        return true_positives / ground_truth_positives

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
    def calculate_precision_scores(predictions, ground_truths):
        precision_scores = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            precision_scores.append(ModelEvaluator.__get_single_image_precision(prediction, ground_truth))
        return precision_scores

    @staticmethod
    def calculate_recall_scores(predictions, ground_truths):
        recall_scores = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            recall_scores.append(ModelEvaluator.__get_single_image_recall(prediction, ground_truth))
        return recall_scores
    
    @staticmethod
    def _log_individual_results(file_names, ious, dice_scores, precision_scores, recall_scores, log_file_path):
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
            f.write("Filename\tIOU\tDice\tPrecision\tRecall\n")
            
            # Write individual results
            for filename, iou, dice, precision, recall in zip(file_names, ious, dice_scores, precision_scores, recall_scores):
                f.write(f"{filename}:\t{iou:.6f}\t{dice:.6f}\t{precision:.6f}\t{recall:.6f}\n")
            
            # Write summary statistics
            f.write("\n")
            f.write(
                f"Average:\t{np.mean(ious):.6f}\t{np.mean(dice_scores):.6f}"
                f"\t{np.mean(precision_scores):.6f}\t{np.mean(recall_scores):.6f}\n"
            )
            f.write(
                f"Std Dev:\t{np.std(ious):.6f}\t{np.std(dice_scores):.6f}"
                f"\t{np.std(precision_scores):.6f}\t{np.std(recall_scores):.6f}\n"
            )
            f.write(
                f"Min:\t{np.min(ious):.6f}\t{np.min(dice_scores):.6f}"
                f"\t{np.min(precision_scores):.6f}\t{np.min(recall_scores):.6f}\n"
            )
            f.write(
                f"Max:\t{np.max(ious):.6f}\t{np.max(dice_scores):.6f}"
                f"\t{np.max(precision_scores):.6f}\t{np.max(recall_scores):.6f}\n"
            )
        
        print(f"Individual results logged to: {log_file_path}")

    @staticmethod
    def _resolve_file_names(dataset, dataset_length):
        file_names = getattr(dataset, "image_filenames", None)
        if file_names and len(file_names) == dataset_length:
            return list(file_names)
        return [f"image_{index:03d}" for index in range(dataset_length)]

    @staticmethod
    def _resolve_file_infos(dataset, dataset_length):
        file_infos = getattr(dataset, "file_infos", None)
        if file_infos and len(file_infos) == dataset_length:
            return file_infos

        image_dir = getattr(dataset, "image_dir", None)
        image_filenames = getattr(dataset, "image_filenames", None)
        if not image_dir or not image_filenames or len(image_filenames) != dataset_length:
            return None

        from src.shared.ParticleImage import ParticleImage
        import os

        resolved_file_infos = []
        for file_name in image_filenames:
            try:
                image_path = os.path.join(image_dir, file_name)
                resolved_file_infos.append(ParticleImage.load_and_preprocess(image_path).file_info)
            except Exception:
                resolved_file_infos.append(None)
        return resolved_file_infos

    @staticmethod
    def _log_size_stratified_results(size_metrics_result, log_file_path):
        import os

        log_root, _ = os.path.splitext(log_file_path)
        csv_path = f"{log_root}_size_stratified.csv"
        summary_path = f"{log_root}_size_stratified.txt"
        plot_path = f"{log_root}_size_stratified.png"

        output_paths = save_size_stratified_metrics(
            size_metrics_result,
            csv_path=csv_path,
            summary_path=summary_path,
            plot_path=plot_path,
        )
        print(f"Size-stratified metrics logged to: {csv_path}")
        return output_paths
    
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
                patches_tensor = patches.to(input.device, memory_format=torch.channels_last, non_blocking=True)

                if unet.device.type == 'cuda':
                    with torch.autocast("cuda"):
                        segmentations = unet(patches_tensor)
                else:
                    segmentations = unet(patches_tensor)

                segmented_image = construct_image_from_patches(
                    segmentations, tensor_mirror_filled.shape[2:], (stride_length,stride_length)
                )
                segmented_image = center_crop(segmented_image, (input.shape[2], input.shape[3]))
                segmented_image = binarize_segmentation_output(segmented_image)
                predictions.append(torch.tensor(segmented_image, dtype=input.dtype, device=input.device))
                labels.append(label)
                inputs.append(input.cpu())
        return inputs, predictions, labels

    @staticmethod
    def evaluate_model(unet, test_dataloader: DataLoader, test_callback = None, log_file_path = None) -> EvaluationResult:
        inputs, predictions, labels = ModelEvaluator.get_predictions(unet, test_dataloader)
        predictions = [pred.cpu().numpy() for pred in predictions]
        labels = [label.cpu().numpy() for label in labels]
        ious = ModelEvaluator.calculate_ious(predictions, labels)
        dice_scores = ModelEvaluator.calculate_dice_scores(predictions, labels)
        precision_scores = ModelEvaluator.calculate_precision_scores(predictions, labels)
        recall_scores = ModelEvaluator.calculate_recall_scores(predictions, labels)
        dataset = test_dataloader.dataset
        file_names = ModelEvaluator._resolve_file_names(dataset, len(predictions))
        file_infos = ModelEvaluator._resolve_file_infos(dataset, len(predictions))
        size_stratified_metrics = compute_size_stratified_metrics(
            ground_truths=labels,
            predictions=predictions,
            file_names=file_names,
            file_infos=file_infos,
            iou_threshold=ModelEvaluator.DEFAULT_OBJECT_IOU_THRESHOLD,
        )
        size_stratified_paths = {}

        # Log individual results to file if path is provided
        if log_file_path:
            ModelEvaluator._log_individual_results(file_names, ious, dice_scores, precision_scores, recall_scores, log_file_path)
            size_stratified_paths = ModelEvaluator._log_size_stratified_results(size_stratified_metrics, log_file_path)
        
        print(f"IOUS: {ious}")
        print(f"Dice scores: {dice_scores}")
        print(f"Precision scores: {precision_scores}")
        print(f"Recall scores: {recall_scores}")

        number_of_predictions_to_show = np.min([5, len(predictions)]) 
        indicies = random.sample(range(len(predictions)), number_of_predictions_to_show)
        if not test_callback:
            return EvaluationResult(
                ious,
                dice_scores,
                precision_scores,
                recall_scores,
                size_stratified_metrics=size_stratified_metrics,
                size_stratified_paths=size_stratified_paths,
            )
        try:
            for i in indicies:
                test_callback(inputs[i], predictions[i], labels[i], ious[i], dice_scores[i])
        except Exception as e:
            print(f"Error in test callback: {e}")
        finally:
            return EvaluationResult(
                ious,
                dice_scores,
                precision_scores,
                recall_scores,
                size_stratified_metrics=size_stratified_metrics,
                size_stratified_paths=size_stratified_paths,
            )

        
        
