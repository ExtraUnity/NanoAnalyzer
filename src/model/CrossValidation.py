import os
import datetime
from torch.utils.data import DataLoader, Subset
import numpy as np
from torch.utils.data import DataLoader, random_split
from src.model.SegmentationDataset import SegmentationDataset
from src.model.PlottingTools import *
from src.model.DataTools import get_dataloaders, get_dataloaders_kfold_already_split, process_and_slice, process_no_slice, slice_dataset_in_four, get_normalizer
from src.model.ModelEvaluator import ModelEvaluator
from src.shared.ModelConfig import ModelConfig
from src.shared.EvaluationResult import EvaluationResult

def cv_holdout(unet, model_config: ModelConfig, stop_training_event = None, loss_callback = None, testing_callback = None, log_dir = None) -> EvaluationResult:
    print(f"Training model using holdout [train_split_size={model_config.train_subset_size}, epochs={model_config.epochs}, learnRate={model_config.learning_rate}]...")
    print("---------------------------------------------------------------------------------------")
    dataset = SegmentationDataset(model_config.images_path, model_config.masks_path)
    train_dataloader, validation_dataloader, test_dataloader = None, None, None

    log_file_path = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "training_evaluation.txt")

    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(dataset, model_config, unet.preferred_input_size, log_file_path)
    unet.normalizer = get_normalizer(train_dataloader.dataset.dataset)
    unet.train_model(
        training_dataloader=train_dataloader, 
        validation_dataloader=validation_dataloader, 
        epochs=model_config.epochs, 
        learningRate=model_config.learning_rate, 
        model_name="UNet_" + datetime.datetime.now().strftime('%d.%m.%Y_%H-%M-%S')+".pt",
        with_early_stopping=model_config.with_early_stopping,
        loss_function="combined",
        scheduler_type=getattr(model_config, 'scheduler_type', 'none'),  # Default to none if not specified
        stop_training_event=stop_training_event,
        loss_callback=loss_callback
        )
        
    evaluation_result = ModelEvaluator.evaluate_model(unet, test_dataloader, testing_callback, log_file_path)
    return evaluation_result

def cv_kfold(images_path, masks_path):
    fold_results = []   
    
    # Set parameters:
    K = 5
    learning_rates = [0.0001] 
    #schedulers = ["none", "plateau"] 
    loss_functions = ["cross_entropy", "dice2", "focal", "combined", "tversky"]#, "dice", "weighted_cross_entropy", "weighted_dice"] 
    augmentations = [(True, True, False, False, False, False)]
    random_cropping = [False, True]
    S = len(loss_functions)#len(learning_rates)
    #models = [UNet() for _ in range(S)]
    epochs = 500
    print(f"\nTraining model using one-level cross-validation with K={K}")
    results_dir = "cv_loss_functions_logs"
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    dataset = SegmentationDataset(images_path, masks_path)
    dataset_size = len(dataset)
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=K, shuffle=True)

    fold_results = {
        s: {
            "test_sizes": [],
            "test_losses": [],
            "test_ious": [],
            "test_dice_scores": [],
            "test_precision_scores": [],
            "test_recall_scores": [],
            "object_precisions": [],
            "object_recalls": [],
            "mean_relative_ecd_errors": [],
            "mean_absolute_relative_ecd_errors": [],
            "size_stratified_rows": [],
            "size_stratified_paths": [],
        }
        for s in range(1, S+1)
    }
    for fold, (par_idx, test_idx) in enumerate(cv.split(np.arange(dataset_size))): 
        inner_fold(fold, K, dataset, loss_functions, epochs, par_idx, test_idx, fold_results, results_dir)

    
    E_gen_loss_s = []
    E_gen_iou_s = []
    E_gen_dice_s = []
    for s in range(1, S+1):
        test_sizes = fold_results[s]["test_sizes"]
        test_losses = fold_results[s]["test_losses"]
        test_ious = fold_results[s]["test_ious"]
        test_dice_scores = fold_results[s]["test_dice_scores"]
        total_test_size = sum(test_sizes)
        gen_error_estimate_loss = sum(test_size * test_loss for test_size, test_loss in zip(test_sizes, test_losses)) / total_test_size
        gen_error_estimate_iou = np.mean(test_ious)
        gen_error_estimate_dice = np.mean(test_dice_scores)
        E_gen_loss_s.append(gen_error_estimate_loss)
        E_gen_iou_s.append(gen_error_estimate_iou)
        E_gen_dice_s.append(gen_error_estimate_dice)
    best_s = E_gen_iou_s.index(max(E_gen_iou_s))
    best_parameter = loss_functions[best_s]

    print(f"\nSelected best model: UNet{best_s+1} with Mean IOU: {E_gen_iou_s[best_s]:.5f} and loss function: {best_parameter}")

    log_one_layer_cv_results(loss_functions, fold_results, best_parameter)
    log_kfold_size_stratified_rows(loss_functions, fold_results, results_dir)

def inner_fold(idx, K2, par_split, parameters, epochs, train_idx, test_idx, test_results, results_dir="cv_loss_functions_logs"):
    print(f"\n ------------ Inner Fold {idx+1}/{K2} -------------") 
    train_split = Subset(par_split, train_idx.tolist())
    inner_test_data = Subset(par_split, test_idx.tolist())
    train_data, val_data = _split_train_validation(train_split)
    train_data = slice_dataset_in_four(train_data)
    val_data = slice_dataset_in_four(val_data)
    inner_test_data = slice_dataset_in_four(inner_test_data)
    inner_test_loss_data = process_and_slice(inner_test_data)
    inner_test_data = process_no_slice(inner_test_data)
    inner_test_loss_dataloader = DataLoader(inner_test_loss_data, batch_size=1, shuffle=False)
    inner_test_dataloader = DataLoader(inner_test_data, batch_size=1, shuffle=False)
    from src.model.UNet import UNet

    for s in range(1, len(parameters)+1):
        unet = UNet()
        inner_train_dataloader, inner_validation_dataloader = get_dataloaders_kfold_already_split(train_data, val_data, 32, (256, 256))
        print(len(inner_validation_dataloader.dataset))
        unet.normalizer = get_normalizer(inner_train_dataloader.dataset.dataset)
        #inner_train_dataloader.dataset.dataset.transform = data_augmenter.get_transformer(True, *parameters[s-1])
        
        model_name = f"UNet{K2}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pt"
        learning_rate = 0.0001#parameters[s-1]  
        loss_function = parameters[s-1]
        scheduler = "none"#parameters[s-1]
        print(parameters[s-1])
        print(f"\nTraining model {s} with \nName: {model_name}\n Loss function: {loss_function}\n Learning rate: {learning_rate}")
        unet.train_model(
            training_dataloader=inner_train_dataloader,
            validation_dataloader=inner_validation_dataloader,
            epochs=epochs,
            learningRate=learning_rate,
            model_name=model_name,
            with_early_stopping=True,
            loss_function=loss_function,
            scheduler_type=scheduler  # Add scheduler type
        )

        test_loss = unet.get_validation_loss(inner_test_loss_dataloader)
        evaluation_log_path = os.path.join(
            results_dir,
            f"fold_{idx+1:02d}_model_{s}_{loss_function}_evaluation.txt",
        )
        evaluation_result = ModelEvaluator.evaluate_model(
            unet,
            inner_test_dataloader,
            log_file_path=evaluation_log_path,
        )
        size_summary = _extract_overall_size_summary(evaluation_result)

        test_results[s]["test_sizes"].append(len(inner_test_loss_data))
        test_results[s]["test_losses"].append(test_loss)
        test_results[s]["test_ious"].append(evaluation_result.mean_iou)
        test_results[s]["test_dice_scores"].append(evaluation_result.mean_dice)
        test_results[s]["test_precision_scores"].append(evaluation_result.mean_precision)
        test_results[s]["test_recall_scores"].append(evaluation_result.mean_recall)
        test_results[s]["object_precisions"].append(size_summary["precision"])
        test_results[s]["object_recalls"].append(size_summary["recall"])
        test_results[s]["mean_relative_ecd_errors"].append(size_summary["mean_relative_ecd_error"])
        test_results[s]["mean_absolute_relative_ecd_errors"].append(size_summary["mean_absolute_relative_ecd_error"])
        test_results[s]["size_stratified_rows"].append(
            {
                "fold": idx + 1,
                "rows": getattr(evaluation_result.size_stratified_metrics, "rows", []),
            }
        )
        test_results[s]["size_stratified_paths"].append(
            {
                "fold": idx + 1,
                "paths": evaluation_result.size_stratified_paths,
            }
        )
        print(f"Test IOU: {evaluation_result.mean_iou}")
        with open(f"cv_loss_functions_inner{idx}_model{s}.txt", "w") as f:
            f.write(f"Model {s} in fold {idx}\n")
            f.write(f"Mean IOU: {evaluation_result.mean_iou}\n")
            f.write(f"Mean Dice: {evaluation_result.mean_dice}\n")
            f.write(f"Mean Precision: {evaluation_result.mean_precision}\n")
            f.write(f"Mean Recall: {evaluation_result.mean_recall}\n")
            f.write(f"Object Precision: {size_summary['precision']}\n")
            f.write(f"Object Recall: {size_summary['recall']}\n")
            f.write(f"Mean Relative ECD Error: {size_summary['mean_relative_ecd_error']}\n")
            f.write(f"Mean Absolute Relative ECD Error: {size_summary['mean_absolute_relative_ecd_error']}\n")
            f.write(f"Evaluation log: {evaluation_log_path}\n")
            f.write(f"Size-stratified paths: {evaluation_result.size_stratified_paths}")

def _split_train_validation(train_split, validation_fraction=0.2):
    validation_size = max(1, round(len(train_split) * validation_fraction))
    validation_size = min(validation_size, len(train_split) - 1)
    train_size = len(train_split) - validation_size
    return random_split(train_split, [train_size, validation_size])

def _extract_overall_size_summary(evaluation_result):
    default_summary = {
        "precision": np.nan,
        "recall": np.nan,
        "mean_relative_ecd_error": np.nan,
        "mean_absolute_relative_ecd_error": np.nan,
    }
    metrics = getattr(evaluation_result, "size_stratified_metrics", None)
    rows = getattr(metrics, "rows", None)
    if not rows:
        return default_summary

    overall_row = next(
        (
            row
            for row in rows
            if row.get("row_type") == "ground_truth_overall" and row.get("size_bin") == "overall"
        ),
        None,
    )
    if overall_row is None:
        return default_summary

    return {
        "precision": _metric_to_float(overall_row.get("precision")),
        "recall": _metric_to_float(overall_row.get("recall")),
        "mean_relative_ecd_error": _metric_to_float(overall_row.get("mean_relative_ecd_error")),
        "mean_absolute_relative_ecd_error": _metric_to_float(
            overall_row.get("mean_absolute_relative_ecd_error")
        ),
    }

def _metric_to_float(value):
    if value in ("", None):
        return np.nan
    return float(value)

def _mean_metric(values):
    numeric_values = [_metric_to_float(value) for value in values]
    numeric_values = [value for value in numeric_values if not np.isnan(value)]
    if not numeric_values:
        return np.nan
    return float(np.mean(numeric_values))

def _metrics_to_floats(values):
    return [_metric_to_float(value) for value in values]

def _format_metric(value):
    value = _metric_to_float(value)
    if np.isnan(value):
        return "nan"
    return f"{value:.5f}"

def log_kfold_size_stratified_rows(parameters, fold_results, results_dir):
    import csv

    output_path = os.path.join(results_dir, "cross_validation_size_stratified_rows.csv")
    flattened_rows = []
    metric_fieldnames = None

    for s in range(1, len(parameters) + 1):
        for fold_entry in fold_results[s]["size_stratified_rows"]:
            for row in fold_entry["rows"]:
                if metric_fieldnames is None:
                    metric_fieldnames = list(row.keys())
                flattened_rows.append(
                    {
                        "model": s,
                        "loss_function": parameters[s - 1],
                        "fold": fold_entry["fold"],
                        **row,
                    }
                )

    if metric_fieldnames is None:
        return None

    os.makedirs(results_dir, exist_ok=True)
    fieldnames = ["model", "loss_function", "fold"] + metric_fieldnames
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened_rows)

    print(f"Combined size-stratified CV rows logged to: {output_path}")
    return output_path

def log_inner_fold_results(idx, parameters, inner_test_results, S):
    results_dir = "cv_loss_functions_logs"
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, f"inner_fold_results_outer{idx+1}.txt")

    with open(log_file, "w") as f:
        f.write(f"Inner Fold Results for Outer Fold {idx+1}\n")
        f.write("=" * 50 + "\n")
        for s in range(1, S+1):
            f.write(f"\nModel {s} (Loss function = {parameters[s-1]}):\n")
            for i in range(len(inner_test_results[s]["test_ious"])):
                f.write(f"  Inner Fold {i+1}:\n")
                f.write(f"    Test Size: {inner_test_results[s]['test_sizes'][i]}\n")
                f.write(f"    Loss: {inner_test_results[s]['test_losses'][i]:.5f}\n")
                f.write(f"    IOU: {inner_test_results[s]['test_ious'][i]:.5f}\n")
                f.write(f"    Dice: {inner_test_results[s]['test_dice_scores'][i]:.5f}\n")
                if "test_precision_scores" in inner_test_results[s]:
                    f.write(f"    Precision: {_format_metric(inner_test_results[s]['test_precision_scores'][i])}\n")
                    f.write(f"    Recall: {_format_metric(inner_test_results[s]['test_recall_scores'][i])}\n")
                    f.write(f"    Object Precision: {_format_metric(inner_test_results[s]['object_precisions'][i])}\n")
                    f.write(f"    Object Recall: {_format_metric(inner_test_results[s]['object_recalls'][i])}\n")
                    f.write(
                        "    Mean Relative ECD Error: "
                        f"{_format_metric(inner_test_results[s]['mean_relative_ecd_errors'][i])}\n"
                    )
                    f.write(
                        "    Mean Absolute Relative ECD Error: "
                        f"{_format_metric(inner_test_results[s]['mean_absolute_relative_ecd_errors'][i])}\n"
                    )

def log_one_layer_cv_results(parameters, fold_results, best_parameter):
    with open("cross_validation_final_model_results.txt", "w") as f:
        f.write(f"############## K-Fold Cross Validation Summary ##############\n")
        for s in range(1, len(parameters)+1):
            f.write(f"Model with {parameters[s-1]}:\n")
            f.write(f"  Test Sizes: {fold_results[s]['test_sizes']}\n")
            f.write(f"  Test Losses: {[float(x) for x in fold_results[s]['test_losses']]} -> Mean = {np.mean(fold_results[s]['test_losses'])}\n")
            f.write(f"  Test IOUs: {[float(x) for x in fold_results[s]['test_ious']]} -> Mean = {np.mean(fold_results[s]['test_ious'])}\n")
            f.write(f"  Test Dices Scores: {[float(x) for x in fold_results[s]['test_dice_scores']]} -> Mean = {np.mean(fold_results[s]['test_dice_scores'])}\n")
            f.write(
                f"  Test Precision Scores: {_metrics_to_floats(fold_results[s]['test_precision_scores'])} "
                f"-> Mean = {_mean_metric(fold_results[s]['test_precision_scores'])}\n"
            )
            f.write(
                f"  Test Recall Scores: {_metrics_to_floats(fold_results[s]['test_recall_scores'])} "
                f"-> Mean = {_mean_metric(fold_results[s]['test_recall_scores'])}\n"
            )
            f.write(
                f"  Object Precision Scores: {_metrics_to_floats(fold_results[s]['object_precisions'])} "
                f"-> Mean = {_mean_metric(fold_results[s]['object_precisions'])}\n"
            )
            f.write(
                f"  Object Recall Scores: {_metrics_to_floats(fold_results[s]['object_recalls'])} "
                f"-> Mean = {_mean_metric(fold_results[s]['object_recalls'])}\n"
            )
            f.write(
                "  Mean Relative ECD Errors: "
                f"{_metrics_to_floats(fold_results[s]['mean_relative_ecd_errors'])} "
                f"-> Mean = {_mean_metric(fold_results[s]['mean_relative_ecd_errors'])}\n"
            )
            f.write(
                "  Mean Absolute Relative ECD Errors: "
                f"{_metrics_to_floats(fold_results[s]['mean_absolute_relative_ecd_errors'])} "
                f"-> Mean = {_mean_metric(fold_results[s]['mean_absolute_relative_ecd_errors'])}\n"
            )
            f.write(f"  Size-Stratified Outputs: {fold_results[s]['size_stratified_paths']}\n\n")
        f.write(f"Best loss function: {best_parameter}")
