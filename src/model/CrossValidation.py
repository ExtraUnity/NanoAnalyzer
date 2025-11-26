import os
import datetime
from torch.utils.data import DataLoader, Subset
import numpy as np
from torch.utils.data import DataLoader, random_split
from src.model.SegmentationDataset import SegmentationDataset
from src.model.PlottingTools import *
from src.model.DataTools import get_dataloaders, get_dataloaders_kfold_already_split, process_and_slice, slice_dataset_in_four, get_normalizer
from src.model.ModelEvaluator import ModelEvaluator
from src.shared.ModelConfig import ModelConfig
from src.shared.EvaluationResult import EvaluationResult

def cv_holdout(unet, model_config: ModelConfig, stop_training_event = None, loss_callback = None, testing_callback = None, log_dir = None, fine_tune: bool = False) -> EvaluationResult:
    mode = "Fine-tuning" if fine_tune else "Training"
    print(f"{mode} model using holdout [train_split_size={model_config.train_subset_size}, epochs={model_config.epochs}, learnRate={model_config.learning_rate}]...")
    print("---------------------------------------------------------------------------------------")
    dataset = SegmentationDataset(model_config.images_path, model_config.masks_path)
    train_dataloader, validation_dataloader, test_dataloader = None, None, None

    log_file_path = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "training_evaluation.txt")

    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(dataset, model_config, unet.preferred_input_size, log_file_path)
    unet.normalizer = get_normalizer(train_dataloader.dataset.dataset)
    model_name_prefix = "UNet_finetuned_" if fine_tune else "UNet_"
    training_fn = unet.fine_tune_model if fine_tune else unet.train_model
    training_fn(
        training_dataloader=train_dataloader, 
        validation_dataloader=validation_dataloader, 
        epochs=model_config.epochs, 
        learningRate=model_config.learning_rate, 
        model_name=model_name_prefix + datetime.datetime.now().strftime('%d.%m.%Y_%H-%M-%S')+".pt",
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

    # Load data
    dataset = SegmentationDataset(images_path, masks_path)
    dataset = slice_dataset_in_four(dataset)
    dataset_size = len(dataset)
    sliced_dataset_size = len(process_and_slice(dataset))
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=K, shuffle=True)

    fold_results = {s: {"test_sizes": [], "test_losses": [], "test_ious": [], "test_dice_scores": []} for s in range(1, S+1)}
    for fold, (par_idx, test_idx) in enumerate(cv.split(np.arange(dataset_size))): 
        inner_fold(fold, K, dataset, loss_functions, epochs, par_idx, test_idx, fold_results)

    
    E_gen_loss_s = []
    E_gen_iou_s = []
    E_gen_dice_s = []
    for s in range(1, S+1):
        test_sizes = fold_results[s]["test_sizes"]
        test_losses = fold_results[s]["test_losses"]
        test_ious = fold_results[s]["test_ious"]
        test_dice_scores = fold_results[s]["test_dice_scores"]
        gen_error_estimate_loss = sum(test_size * test_loss for test_size, test_loss in zip(test_sizes, test_losses)) / sliced_dataset_size
        gen_error_estimate_iou = np.mean(test_ious)
        gen_error_estimate_dice = np.mean(test_dice_scores)
        E_gen_loss_s.append(gen_error_estimate_loss)
        E_gen_iou_s.append(gen_error_estimate_iou)
        E_gen_dice_s.append(gen_error_estimate_dice)
    best_s = E_gen_iou_s.index(max(E_gen_iou_s))
    best_parameter = loss_functions[best_s]

    print(f"\nSelected best model: UNet{best_s+1} with Mean IOU: {E_gen_iou_s[best_s]:.5f} and loss function: {best_parameter}")

    log_one_layer_cv_results(loss_functions, fold_results, best_parameter)

def inner_fold(idx, K2, par_split, parameters, epochs, train_idx, test_idx, test_results):
    print(f"\n ------------ Inner Fold {idx+1}/{K2} -------------") 
    train_split = Subset(par_split, train_idx.tolist())
    inner_test_data = Subset(par_split, test_idx.tolist())
    train_data, val_data = random_split(train_split, [0.8, 0.2])
    inner_test_data = process_and_slice(inner_test_data)
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

        test_loss = unet.get_validation_loss(inner_test_dataloader)
        from src.model.PlottingTools import plot_difference
        test_iou, test_dice = ModelEvaluator.evaluate_model(unet, inner_test_dataloader)

        test_results[s]["test_sizes"].append(len(inner_test_data))
        test_results[s]["test_losses"].append(test_loss)
        test_results[s]["test_ious"].append(test_iou)
        test_results[s]["test_dice_scores"].append(test_dice)
        print(f"Test IOU: {test_iou}")
        with open(f"cv_loss_functions_inner{idx}_model{s}.txt", "w") as f:
            f.write(f"Model {s} in fold {idx}\n")
            f.write(f"Mean IOU: {test_iou}\n")
            f.write(f"Mean Dice: {test_dice}")

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

def log_one_layer_cv_results(parameters, fold_results, best_parameter):
    with open("cross_validation_final_model_results.txt", "w") as f:
        f.write(f"############## K-Fold Cross Validation Summary ##############\n")
        for s in range(1, len(parameters)+1):
            f.write(f"Model with {parameters[s-1]}:\n")
            f.write(f"  Test Sizes: {fold_results[s]['test_sizes']}\n")
            f.write(f"  Test Losses: {[float(x) for x in fold_results[s]['test_losses']]} -> Mean = {np.mean(fold_results[s]['test_losses'])}\n")
            f.write(f"  Test IOUs: {[float(x) for x in fold_results[s]['test_ious']]} -> Mean = {np.mean(fold_results[s]['test_ious'])}\n")
            f.write(f"  Test Dices Scores: {[float(x) for x in fold_results[s]['test_dice_scores']]} -> Mean = {np.mean(fold_results[s]['test_dice_scores'])}\n\n")
        f.write(f"Best loss function: {best_parameter}")
