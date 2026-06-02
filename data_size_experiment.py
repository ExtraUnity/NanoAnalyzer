"""
Experiment: Training Data Size vs Model Performance

This script trains multiple models using different amounts of training data
to analyze how the size of the training dataset affects model performance.

Dataset: medres_images (20 images total)
Training increments: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%
"""

import os
import datetime
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from src.model.SegmentationDataset import SegmentationDataset
from src.model.UNet import UNet
from src.model.ModelEvaluator import ModelEvaluator
from src.model.DataTools import slice_dataset_in_four, get_normalizer, process_and_slice
from src.model.DataAugmenter import DataAugmenter
from src.shared.ModelConfig import ModelConfig


class DataSizeExperiment:
    def __init__(self, images_path, masks_path, output_dir="data/experiments/data_size", random_seed: int = 42):
        """
        Initialize the data size experiment.
        
        Args:
            images_path: Path to training images
            masks_path: Path to training masks
            output_dir: Directory to save results and logs
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.output_dir = output_dir
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.random_seed = random_seed
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.results = {
            'training_sizes': [],
            'mean_iou': [],
            'mean_dice': [],
            'std_iou': [],
            'std_dice': [],
            'training_times': []
        }

    def _log_split_filenames(self, train_indices, val_indices, test_indices):
        """Write train/validation/test split filenames to a timestamped text file."""
        split_file = os.path.join(self.output_dir, f"split_filenames_{self.timestamp}.txt")

        train_files = [self.dataset.image_filenames[i] for i in train_indices]
        val_files = [self.dataset.image_filenames[i] for i in val_indices]
        test_files = [self.dataset.image_filenames[i] for i in test_indices]

        with open(split_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("DATA SIZE EXPERIMENT - IMAGE SPLITS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Random seed: {self.random_seed}\n")
            f.write(f"Images path: {self.images_path}\n")
            f.write(f"Masks path: {self.masks_path}\n\n")

            f.write(f"Training files ({len(train_files)}):\n")
            for name in train_files:
                f.write(f"  {name}\n")

            f.write(f"\nValidation files ({len(val_files)}):\n")
            for name in val_files:
                f.write(f"  {name}\n")

            f.write(f"\nTest files ({len(test_files)}):\n")
            for name in test_files:
                f.write(f"  {name}\n")

        print(f"Split filenames saved to: {split_file}")
        
    def prepare_data_splits(self, train_percentages, val_split=0.2, test_split=0.2, input_size=(256, 256)):
        """
        Prepare data splits for the experiment.
        
        Args:
            train_percentages: List of training data percentages to test (e.g., [0.1, 0.2, 0.3])
            val_split: Validation set size as fraction of remaining data after test split
            test_split: Test set size as fraction of total data
            random_seed: Random seed for reproducibility
            input_size: Size for slicing the dataset into patches
        """
        # Set random seed for reproducibility (seed is set in constructor)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Load full dataset and split original images before any patching.
        dataset = SegmentationDataset(self.images_path, self.masks_path)
        self.dataset = dataset
        print(f"Total images in dataset: {len(dataset)}")

        total_size = len(dataset)
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        # Split into test and train+val
        test_size = int(total_size * test_split)
        test_indices = indices[:test_size]
        train_val_indices = indices[test_size:]
        
        # Further split train+val into validation and available training pool
        val_size = int(total_size * val_split)
        val_indices = train_val_indices[:val_size]
        available_train_indices = train_val_indices[val_size:]
        
        print(f"Test set size: {len(test_indices)} images")
        print(f"Validation set size: {len(val_indices)} images")
        print(f"Available training pool: {len(available_train_indices)} images")

        # Persist the exact image-level split used for this experiment run.
        self._log_split_filenames(available_train_indices, val_indices, test_indices)

        # Build patch datasets immediately after image-level split.
        # From this point forward, training/validation/testing operate on patches only.
        train_subset = Subset(self.dataset, available_train_indices)
        val_subset = Subset(self.dataset, val_indices)
        test_subset = Subset(self.dataset, test_indices)

        train_quadrants = slice_dataset_in_four(train_subset, input_size)
        val_quadrants = slice_dataset_in_four(val_subset, input_size)
        test_quadrants = slice_dataset_in_four(test_subset, input_size)

        self.train_patch_pool = process_and_slice(Subset(train_quadrants, list(range(len(train_quadrants)))), input_size)
        self.val_patches = process_and_slice(Subset(val_quadrants, list(range(len(val_quadrants)))), input_size)
        self.test_patches = process_and_slice(Subset(test_quadrants, list(range(len(test_quadrants)))), input_size)

        print(f"Available training patch pool: {len(self.train_patch_pool)} patches")
        print(f"Validation patch count: {len(self.val_patches)}")
        print(f"Test patch count: {len(self.test_patches)}")
        
        # Create experiment configs per percentage (sampling happens on patch pool)
        self.data_splits = {}
        for percentage in train_percentages:
            requested_num_train_patches = max(1, int(len(self.train_patch_pool) * percentage))
            requested_num_train_patches = min(requested_num_train_patches, len(self.train_patch_pool))

            self.data_splits[percentage] = {
                'requested_num_train_patches': requested_num_train_patches,
                'num_train_patches_pool': len(self.train_patch_pool),
                'num_val_patches': len(self.val_patches),
                'num_test_patches': len(self.test_patches)
            }
            print(
                f"{int(percentage*100)}% training data requested: "
                f"{requested_num_train_patches} patches "
                f"(sampling from {len(self.train_patch_pool)} available training patches)"
            )

        self.test_indices = test_indices
        self.val_indices = val_indices
        
        return self.data_splits
    
    def create_dataloaders(self, train_percentage: float, with_augmentation=True, batch_size=8):
        """
        Create dataloaders for a specific data split.
        """

        # Sample patches from the precomputed training patch pool according to train_percentage
        from torch.utils.data import Subset as TorchSubset
        import numpy as _np
        percentage_seed = self.random_seed + int(train_percentage * 1000)
        total_patches = len(self.train_patch_pool)
        desired_patches = max(1, int(total_patches * train_percentage))
        desired_patches = min(desired_patches, total_patches)
        if desired_patches >= total_patches:
            sampled_train_subset = self.train_patch_pool
        else:
            rng = _np.random.default_rng(percentage_seed)
            sampled_indices = rng.choice(total_patches, size=desired_patches, replace=False)
            sampled_train_subset = TorchSubset(self.train_patch_pool, sampled_indices.tolist())

        # Apply data augmentation to the sampled training patches
        data_augmenter = DataAugmenter()
        if with_augmentation:
            train_data = data_augmenter.augment_dataset(sampled_train_subset)
        else:
            train_data = data_augmenter.augment_dataset(
                sampled_train_subset,
                [False, False, False, False, False, False, False],
            )

        val_data = self.val_patches
        test_data = self.test_patches

        # Create dataloaders
        dataloader_generator = torch.Generator().manual_seed(percentage_seed)
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=dataloader_generator,
        )
        val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        # Return dataloaders and the number of unique training patches sampled (before augmentation repeats)
        num_train_patches = len(sampled_train_subset)
        return train_dataloader, val_dataloader, test_dataloader, num_train_patches
    
    def train_single_model(self, train_percentage, epochs=150, learning_rate=0.0001, 
                          input_size=(256, 256), with_augmentation=True):
        """
        Train a single model with a specific percentage of training data.
        """
        print(f"\n{'='*80}")
        print(f"Training with {int(train_percentage*100)}% of training data")
        print(f"{'='*80}")

        # Per-percentage deterministic seed for training, augmentation, and sampling order
        run_seed = self.random_seed + int(train_percentage * 1000)
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
        
        # Get data split
        split = self.data_splits[train_percentage]
        print(f"Training patch pool size: {split['num_train_patches_pool']}")
        
        # Create dataloaders (now sampling patches according to train_percentage)
        train_dataloader, val_dataloader, test_dataloader, num_train_patches = self.create_dataloaders(
            train_percentage=train_percentage,
            with_augmentation=with_augmentation
        )

        print(f"Training patch count (used): {num_train_patches}")
        print(f"Validation patch count: {split['num_val_patches']}")
        print(f"Test patch count: {split['num_test_patches']}")
        
        # Initialize model
        unet = UNet()
        unet.preferred_input_size = input_size
        unet = unet.to(device)

        # Get normalizer from training data
        normalizer = get_normalizer(train_dataloader.dataset.dataset)
        unet.normalizer = normalizer
        
        # Train model
        model_name = f"UNet_datasize_{int(train_percentage*100)}pct_{self.timestamp}.pt"
        
        start_time = datetime.datetime.now()
        unet.train_model(
            training_dataloader=train_dataloader,
            validation_dataloader=val_dataloader,
            epochs=epochs,
            learningRate=learning_rate,
            model_name=model_name,
            with_early_stopping=True,
            loss_function="combined",
            scheduler_type="none"
        )
        training_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Evaluate on test set
        print(f"\nEvaluating model with {int(train_percentage*100)}% training data...")
        evaluation_result = ModelEvaluator.evaluate_model(unet, test_dataloader)
        
        # Store results
        result = {
            'train_percentage': train_percentage,
            'num_train_patches': num_train_patches,
            'mean_iou': evaluation_result.mean_iou,
            'mean_dice': evaluation_result.mean_dice,
            'std_iou': np.std(evaluation_result.iou_scores),
            'std_dice': np.std(evaluation_result.dice_scores),
            'min_iou': evaluation_result.min_iou,
            'max_iou': evaluation_result.max_iou,
            'min_dice': evaluation_result.min_dice,
            'max_dice': evaluation_result.max_dice,
            'training_time': training_time,
            'model_name': model_name
        }
        
        print(f"\nResults for {int(train_percentage*100)}% training data:")
        print(f"  Mean IoU: {result['mean_iou']:.4f} ± {result['std_iou']:.4f}")
        print(f"  Mean Dice: {result['mean_dice']:.4f} ± {result['std_dice']:.4f}")
        print(f"  Training time: {training_time:.2f} seconds")
        
        return result
    
    def run_experiment(self, train_percentages=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                      epochs=50, learning_rate=0.0001, input_size=(256, 256), 
                      with_augmentation=True):
        """
        Run the complete experiment with multiple training data sizes.
        """
        print(f"\n{'='*80}")
        print(f"DATA SIZE EXPERIMENT")
        print(f"{'='*80}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Images path: {self.images_path}")
        print(f"Masks path: {self.masks_path}")
        print(f"Training percentages: {[int(p*100) for p in train_percentages]}%")
        print(f"Epochs per model: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Data augmentation: {with_augmentation}")
        print(f"Random seed: {self.random_seed}")

        # Prepare data splits
        self.prepare_data_splits(train_percentages, input_size=input_size)
        
        # Train models for each data size
        all_results = []
        for percentage in train_percentages:
            result = self.train_single_model(
                train_percentage=percentage,
                epochs=epochs,
                learning_rate=learning_rate,
                input_size=input_size,
                with_augmentation=with_augmentation
            )
            all_results.append(result)
        
        # Save results
        self.save_results(all_results)
        
        # Plot results
        self.plot_results(all_results)
        
        return all_results
    
    def save_results(self, results):
        """
        Save experiment results to a text file.
        """
        results_file = os.path.join(self.output_dir, f"results_{self.timestamp}.txt")
        
        with open(results_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA SIZE EXPERIMENT RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Images path: {self.images_path}\n")
            f.write(f"Masks path: {self.masks_path}\n\n")
            
            f.write("RESULTS SUMMARY:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Train %':<10} {'#Patches':<10} {'Mean IoU':<15} {'Mean Dice':<15} {'Time (s)':<12}\n")
            f.write("-"*80 + "\n")
            
            for result in results:
                f.write(f"{int(result['train_percentage']*100):<10} "
                      f"{result['num_train_patches']:<10} "
                       f"{result['mean_iou']:.4f}±{result['std_iou']:.4f}  "
                       f"{result['mean_dice']:.4f}±{result['std_dice']:.4f}  "
                       f"{result['training_time']:.2f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED RESULTS:\n")
            f.write("="*80 + "\n\n")
            
            for result in results:
                f.write(f"Training Data: {int(result['train_percentage']*100)}% ({result['num_train_patches']} patches)\n")
                f.write(f"Model: {result['model_name']}\n")
                f.write(f"  Mean IoU:  {result['mean_iou']:.4f} ± {result['std_iou']:.4f}\n")
                f.write(f"  IoU Range: [{result['min_iou']:.4f}, {result['max_iou']:.4f}]\n")
                f.write(f"  Mean Dice: {result['mean_dice']:.4f} ± {result['std_dice']:.4f}\n")
                f.write(f"  Dice Range: [{result['min_dice']:.4f}, {result['max_dice']:.4f}]\n")
                f.write(f"  Training Time: {result['training_time']:.2f} seconds\n")
                f.write("-"*80 + "\n")
        
        print(f"\nResults saved to: {results_file}")
    
    def plot_results(self, results):
        """
        Create visualization plots of the experiment results.
        """
        train_percentages = [r['train_percentage'] * 100 for r in results]
        num_patches = [r['num_train_patches'] for r in results]
        mean_iou = [r['mean_iou'] for r in results]
        mean_dice = [r['mean_dice'] for r in results]
        std_iou = [r['std_iou'] for r in results]
        std_dice = [r['std_dice'] for r in results]
        training_times = [r['training_time'] for r in results]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Data Size vs Model Performance', fontsize=16, fontweight='bold')
        
        # Plot 1: IoU vs Training Data Percentage
        ax1 = axes[0, 0]
        ax1.errorbar(train_percentages, mean_iou, yerr=std_iou, marker='o', 
                     capsize=5, capthick=2, linewidth=2, markersize=8)
        ax1.set_xlabel('Training Data (%)', fontsize=12)
        ax1.set_ylabel('Mean IoU', fontsize=12)
        ax1.set_title('IoU Score vs Training Data Size', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Dice vs Training Data Percentage
        ax2 = axes[0, 1]
        ax2.errorbar(train_percentages, mean_dice, yerr=std_dice, marker='s', 
                     color='orange', capsize=5, capthick=2, linewidth=2, markersize=8)
        ax2.set_xlabel('Training Data (%)', fontsize=12)
        ax2.set_ylabel('Mean Dice Score', fontsize=12)
        ax2.set_title('Dice Score vs Training Data Size', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: IoU and Dice on same plot vs Number of Patches
        ax3 = axes[1, 0]
        ax3.plot(num_patches, mean_iou, marker='o', label='IoU', linewidth=2, markersize=8)
        ax3.plot(num_patches, mean_dice, marker='s', label='Dice', linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Training Patches', fontsize=12)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Performance Metrics vs Number of Training Patches', fontsize=14)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Training Time vs Data Size
        ax4 = axes[1, 1]
        ax4.plot(train_percentages, training_times, marker='D', color='green', 
                linewidth=2, markersize=8)
        ax4.set_xlabel('Training Data (%)', fontsize=12)
        ax4.set_ylabel('Training Time (seconds)', fontsize=12)
        ax4.set_title('Training Time vs Data Size', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, f"plots_{self.timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_file}")
        
        # Show plot
        plt.show()


def main():
    """
    Main function to run the data size experiment.
    """
    # Configuration
    images_path = "data/medres_images"
    masks_path = "data/medres_masks"
    
    # Training percentages to test
    train_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Hyperparameters
    epochs = 150
    learning_rate = 0.0001
    input_size = (256, 256)
    with_augmentation = True
    random_seed = 42
    
    # Create and run experiment (set seed here)
    experiment = DataSizeExperiment(images_path, masks_path, random_seed=random_seed)
    results = experiment.run_experiment(
        train_percentages=train_percentages,
        epochs=epochs,
        learning_rate=learning_rate,
        input_size=input_size,
        with_augmentation=with_augmentation
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED!")
    print("="*80)
    print(f"Total models trained: {len(results)}")
    print(f"Results saved to: data/experiments/data_size/")


if __name__ == "__main__":
    main()
