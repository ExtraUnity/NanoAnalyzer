"""
Experiment: Training Data Size vs Model Performance

This script trains multiple models using different amounts of training data
to analyze how the size of the training dataset affects model performance.

Dataset: medres_images (20 images total)
Training increments: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%
"""

import os
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from src.model.SegmentationDataset import SegmentationDataset
from src.model.UNet import UNet
from src.model.ModelEvaluator import ModelEvaluator
from src.model.DataTools import slice_dataset_in_four, process_and_slice, get_normalizer
from src.model.DataAugmenter import DataAugmenter
from src.shared.ModelConfig import ModelConfig


class DataSizeExperiment:
    def __init__(self, images_path, masks_path, output_dir="data/experiments/data_size"):
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
        
    def prepare_data_splits(self, train_percentages, val_split=0.2, test_split=0.2, random_seed=42, input_size=(256, 256)):
        """
        Prepare data splits for the experiment.
        
        Args:
            train_percentages: List of training data percentages to test (e.g., [0.1, 0.2, 0.3])
            val_split: Validation set size as fraction of remaining data after test split
            test_split: Test set size as fraction of total data
            random_seed: Random seed for reproducibility
            input_size: Size for slicing the dataset into patches
        """
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Load full dataset
        dataset = SegmentationDataset(self.images_path, self.masks_path)
        print(f"Total images in dataset: {len(dataset)}")
        
        # Slice dataset into patches FIRST
        dataset_sliced = slice_dataset_in_four(dataset, input_size)
        print(f"Total patches after slicing: {len(dataset_sliced)}")
        
        # Get all indices from the SLICED dataset
        total_size = len(dataset_sliced)
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        # Split into test and train+val
        test_size = int(total_size * test_split)
        test_indices = indices[:test_size]
        train_val_indices = indices[test_size:]
        
        # Further split train+val into validation and available training pool
        remaining_size = len(train_val_indices)
        val_size = int(remaining_size * val_split)
        val_indices = train_val_indices[:val_size]
        available_train_indices = train_val_indices[val_size:]
        
        print(f"Test set size: {len(test_indices)} patches")
        print(f"Validation set size: {len(val_indices)} patches")
        print(f"Available training pool: {len(available_train_indices)} patches")
        
        # Create training sets for each percentage
        self.data_splits = {}
        for percentage in train_percentages:
            num_train = max(1, int(len(available_train_indices) * percentage))
            train_indices = available_train_indices[:num_train]
            
            self.data_splits[percentage] = {
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'num_train': num_train,
                'num_val': len(val_indices),
                'num_test': len(test_indices)
            }
            print(f"{int(percentage*100)}% training data: {num_train} patches")
        
        self.dataset_sliced = dataset_sliced
        self.test_indices = test_indices
        self.val_indices = val_indices
        
        return self.data_splits
    
    def create_dataloaders(self, train_indices, val_indices, test_indices, input_size=(256, 256), 
                          with_augmentation=True, batch_size=8):
        """
        Create dataloaders for a specific data split.
        """
        # Use the pre-sliced dataset
        # Create subsets using indices from the sliced dataset
        train_subset = Subset(self.dataset_sliced, train_indices)
        val_subset = Subset(self.dataset_sliced, val_indices)
        test_subset = Subset(self.dataset_sliced, test_indices)
        
        # Apply data augmentation to training set
        data_augmenter = DataAugmenter()
        if with_augmentation:
            train_data = data_augmenter.augment_dataset(train_subset, input_size)
        else:
            train_data = data_augmenter.augment_dataset(train_subset, input_size, 
                                                       [False, False, False, False, False, False, False])
        
        # Process validation and test sets
        val_data = process_and_slice(val_subset, input_size)
        test_data = process_and_slice(test_subset, input_size)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        
        return train_dataloader, val_dataloader, test_dataloader
    
    def train_single_model(self, train_percentage, epochs=150, learning_rate=0.0001, 
                          input_size=(256, 256), with_augmentation=True):
        """
        Train a single model with a specific percentage of training data.
        """
        print(f"\n{'='*80}")
        print(f"Training with {int(train_percentage*100)}% of training data")
        print(f"{'='*80}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
        
        # Get data split
        split = self.data_splits[train_percentage]
        print(f"Training samples: {split['num_train']}")
        print(f"Validation samples: {split['num_val']}")
        print(f"Test samples: {split['num_test']}")
        
        # Create dataloaders
        train_dataloader, val_dataloader, test_dataloader = self.create_dataloaders(
            split['train_indices'], split['val_indices'], split['test_indices'],
            input_size=input_size, with_augmentation=with_augmentation
        )
        
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
            epochs=3,
            learningRate=learning_rate,
            model_name=model_name,
            cross_validation="holdout",
            with_early_stopping=True,
            loss_function="combined",
            scheduler_type="plateau"
        )
        training_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Evaluate on test set
        print(f"\nEvaluating model with {int(train_percentage*100)}% training data...")
        evaluation_result = ModelEvaluator.evaluate_model(unet, test_dataloader)
        
        # Store results
        result = {
            'train_percentage': train_percentage,
            'num_train_images': split['num_train'],
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
    
    def run_experiment(self, train_percentages=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                      epochs=150, learning_rate=0.0001, input_size=(256, 256), 
                      with_augmentation=True, random_seed=42):
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
        print(f"Random seed: {random_seed}")
        
        # Prepare data splits
        self.prepare_data_splits(train_percentages, random_seed=random_seed, input_size=input_size)
        
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
            f.write(f"{'Train %':<10} {'#Images':<10} {'Mean IoU':<15} {'Mean Dice':<15} {'Time (s)':<12}\n")
            f.write("-"*80 + "\n")
            
            for result in results:
                f.write(f"{int(result['train_percentage']*100):<10} "
                       f"{result['num_train_images']:<10} "
                       f"{result['mean_iou']:.4f}±{result['std_iou']:.4f}  "
                       f"{result['mean_dice']:.4f}±{result['std_dice']:.4f}  "
                       f"{result['training_time']:.2f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED RESULTS:\n")
            f.write("="*80 + "\n\n")
            
            for result in results:
                f.write(f"Training Data: {int(result['train_percentage']*100)}% ({result['num_train_images']} images)\n")
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
        num_images = [r['num_train_images'] for r in results]
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
        
        # Plot 3: IoU and Dice on same plot vs Number of Images
        ax3 = axes[1, 0]
        ax3.plot(num_images, mean_iou, marker='o', label='IoU', linewidth=2, markersize=8)
        ax3.plot(num_images, mean_dice, marker='s', label='Dice', linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Training Images', fontsize=12)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Performance Metrics vs Number of Training Images', fontsize=14)
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
    train_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Hyperparameters
    epochs = 150
    learning_rate = 0.0001
    input_size = (256, 256)
    with_augmentation = True
    random_seed = 42
    
    # Create and run experiment
    experiment = DataSizeExperiment(images_path, masks_path)
    results = experiment.run_experiment(
        train_percentages=train_percentages,
        epochs=epochs,
        learning_rate=learning_rate,
        input_size=input_size,
        with_augmentation=with_augmentation,
        random_seed=random_seed
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED!")
    print("="*80)
    print(f"Total models trained: {len(results)}")
    print(f"Results saved to: data/experiments/data_size/")


if __name__ == "__main__":
    main()
