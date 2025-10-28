"""
Analysis tool to compare results from multiple data size experiments.

This script helps analyze and compare results from different experiment runs,
useful for investigating reproducibility or comparing different configurations.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class ExperimentAnalyzer:
    def __init__(self, results_dir="data/experiments/data_size"):
        """
        Initialize the analyzer with a directory containing result files.
        
        Args:
            results_dir: Directory containing results_*.txt files
        """
        self.results_dir = results_dir
        self.experiments = []
    
    def load_all_experiments(self):
        """Load all result files from the directory."""
        if not os.path.exists(self.results_dir):
            print(f"Error: Directory {self.results_dir} does not exist")
            return []
        
        result_files = [f for f in os.listdir(self.results_dir) 
                       if f.startswith('results_') and f.endswith('.txt')]
        
        print(f"Found {len(result_files)} result files")
        
        for filename in sorted(result_files):
            filepath = os.path.join(self.results_dir, filename)
            experiment = self.parse_results_file(filepath)
            if experiment:
                self.experiments.append(experiment)
                print(f"  Loaded: {filename}")
        
        return self.experiments
    
    def parse_results_file(self, filepath):
        """Parse a single results file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Extract timestamp from filename
            filename = os.path.basename(filepath)
            timestamp_match = re.search(r'results_(\d+_\d+)\.txt', filename)
            timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
            
            # Parse results
            results = {
                'timestamp': timestamp,
                'filepath': filepath,
                'train_percentages': [],
                'num_images': [],
                'mean_iou': [],
                'std_iou': [],
                'mean_dice': [],
                'std_dice': [],
                'training_times': []
            }
            
            # Parse data lines from summary table
            in_summary = False
            for line in content.split('\n'):
                if 'Train %' in line and '#Images' in line:
                    in_summary = True
                    continue
                
                if in_summary and line.strip() and not line.startswith('-'):
                    parts = line.split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        train_pct = int(parts[0])
                        num_imgs = int(parts[1])
                        
                        # Parse IoU (format: mean±std)
                        iou_str = parts[2]
                        if '±' in iou_str:
                            iou_mean, iou_std = iou_str.split('±')
                            mean_iou = float(iou_mean)
                            std_iou = float(iou_std)
                        else:
                            mean_iou = float(iou_str)
                            std_iou = 0.0
                        
                        # Parse Dice (format: mean±std)
                        dice_str = parts[3]
                        if '±' in dice_str:
                            dice_mean, dice_std = dice_str.split('±')
                            mean_dice = float(dice_mean)
                            std_dice = float(dice_std)
                        else:
                            mean_dice = float(dice_str)
                            std_dice = 0.0
                        
                        training_time = float(parts[4])
                        
                        results['train_percentages'].append(train_pct)
                        results['num_images'].append(num_imgs)
                        results['mean_iou'].append(mean_iou)
                        results['std_iou'].append(std_iou)
                        results['mean_dice'].append(mean_dice)
                        results['std_dice'].append(std_dice)
                        results['training_times'].append(training_time)
                
                if in_summary and line.startswith('='):
                    break
            
            return results
            
        except Exception as e:
            print(f"Error parsing {filepath}: {str(e)}")
            return None
    
    def compare_experiments(self, experiment_indices=None):
        """
        Create comparison plots for multiple experiments.
        
        Args:
            experiment_indices: List of indices to compare, or None for all
        """
        if not self.experiments:
            print("No experiments loaded. Run load_all_experiments() first.")
            return
        
        if experiment_indices is None:
            experiments_to_compare = self.experiments
        else:
            experiments_to_compare = [self.experiments[i] for i in experiment_indices]
        
        if len(experiments_to_compare) == 0:
            print("No experiments to compare")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_to_compare)))
        
        for idx, (exp, color) in enumerate(zip(experiments_to_compare, colors)):
            label = f"Exp {exp['timestamp']}"
            
            # Plot 1: IoU comparison
            ax1 = axes[0, 0]
            ax1.plot(exp['train_percentages'], exp['mean_iou'], 
                    marker='o', label=label, color=color, linewidth=2)
            ax1.set_xlabel('Training Data (%)', fontsize=12)
            ax1.set_ylabel('Mean IoU', fontsize=12)
            ax1.set_title('IoU Comparison', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Plot 2: Dice comparison
            ax2 = axes[0, 1]
            ax2.plot(exp['train_percentages'], exp['mean_dice'], 
                    marker='s', label=label, color=color, linewidth=2)
            ax2.set_xlabel('Training Data (%)', fontsize=12)
            ax2.set_ylabel('Mean Dice', fontsize=12)
            ax2.set_title('Dice Score Comparison', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # Plot 3: Training time comparison
            ax3 = axes[1, 0]
            ax3.plot(exp['train_percentages'], exp['training_times'], 
                    marker='D', label=label, color=color, linewidth=2)
            ax3.set_xlabel('Training Data (%)', fontsize=12)
            ax3.set_ylabel('Training Time (seconds)', fontsize=12)
            ax3.set_title('Training Time Comparison', fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: IoU vs Number of Images
            ax4 = axes[1, 1]
            ax4.plot(exp['num_images'], exp['mean_iou'], 
                    marker='o', label=label, color=color, linewidth=2)
            ax4.set_xlabel('Number of Training Images', fontsize=12)
            ax4.set_ylabel('Mean IoU', fontsize=12)
            ax4.set_title('IoU vs Training Images', fontsize=14)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save comparison plot
        output_file = os.path.join(self.results_dir, 
                                  f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {output_file}")
        
        plt.show()
    
    def print_summary_statistics(self):
        """Print summary statistics for all experiments."""
        if not self.experiments:
            print("No experiments loaded.")
            return
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS ACROSS ALL EXPERIMENTS")
        print("="*80)
        
        for idx, exp in enumerate(self.experiments):
            print(f"\nExperiment {idx + 1}: {exp['timestamp']}")
            print("-" * 80)
            
            # Overall statistics
            all_iou = exp['mean_iou']
            all_dice = exp['mean_dice']
            
            print(f"Data points: {len(exp['train_percentages'])}")
            print(f"Training percentages: {exp['train_percentages']}")
            print(f"\nIoU Statistics:")
            print(f"  Best: {max(all_iou):.4f} at {exp['train_percentages'][all_iou.index(max(all_iou))]}%")
            print(f"  Worst: {min(all_iou):.4f} at {exp['train_percentages'][all_iou.index(min(all_iou))]}%")
            print(f"  Average: {np.mean(all_iou):.4f}")
            
            print(f"\nDice Statistics:")
            print(f"  Best: {max(all_dice):.4f} at {exp['train_percentages'][all_dice.index(max(all_dice))]}%")
            print(f"  Worst: {min(all_dice):.4f} at {exp['train_percentages'][all_dice.index(min(all_dice))]}%")
            print(f"  Average: {np.mean(all_dice):.4f}")
            
            print(f"\nTotal training time: {sum(exp['training_times']):.2f} seconds")
    
    def find_optimal_data_size(self, threshold=0.95, metric='iou'):
        """
        Find the minimum training data size that achieves a certain performance threshold.
        
        Args:
            threshold: Performance threshold as fraction of maximum (default 0.95 = 95%)
            metric: 'iou' or 'dice'
        """
        if not self.experiments:
            print("No experiments loaded.")
            return
        
        print("\n" + "="*80)
        print(f"OPTIMAL DATA SIZE ANALYSIS (Threshold: {threshold*100}% of max {metric.upper()})")
        print("="*80)
        
        for idx, exp in enumerate(self.experiments):
            print(f"\nExperiment {idx + 1}: {exp['timestamp']}")
            
            # Get relevant metric
            if metric.lower() == 'iou':
                scores = exp['mean_iou']
            else:
                scores = exp['mean_dice']
            
            max_score = max(scores)
            target_score = max_score * threshold
            
            # Find first data size that meets threshold
            for i, score in enumerate(scores):
                if score >= target_score:
                    pct = exp['train_percentages'][i]
                    num_imgs = exp['num_images'][i]
                    print(f"  Target score: {target_score:.4f} ({threshold*100}% of {max_score:.4f})")
                    print(f"  Minimum data size: {pct}% ({num_imgs} images)")
                    print(f"  Achieved score: {score:.4f}")
                    
                    # Calculate data efficiency
                    max_pct = exp['train_percentages'][-1]
                    efficiency = (max_pct - pct) / max_pct * 100
                    print(f"  Data efficiency: {efficiency:.1f}% reduction from maximum")
                    break
            else:
                print(f"  No data size achieved {threshold*100}% of maximum score")


def main():
    """Main function to demonstrate analyzer usage."""
    print("="*80)
    print("EXPERIMENT ANALYZER")
    print("="*80)
    
    # Create analyzer
    analyzer = ExperimentAnalyzer("data/experiments/data_size")
    
    # Load all experiments
    experiments = analyzer.load_all_experiments()
    
    if not experiments:
        print("\nNo experiments found. Please run the experiment first.")
        print("Run: python run_data_size_experiment.py")
        return
    
    # Print summary statistics
    analyzer.print_summary_statistics()
    
    # Find optimal data size
    analyzer.find_optimal_data_size(threshold=0.95, metric='iou')
    analyzer.find_optimal_data_size(threshold=0.90, metric='iou')
    
    # Create comparison plots
    print("\n" + "="*80)
    print("Generating comparison plots...")
    analyzer.compare_experiments()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
