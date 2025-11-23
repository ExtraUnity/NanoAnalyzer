"""
Quick Start Configuration for Data Size Experiment

Modify the parameters in this file and run it to execute the experiment.
"""

from data_size_experiment import DataSizeExperiment

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Dataset paths
IMAGES_PATH = "data/medres_images"
MASKS_PATH = "data/medres_masks"
OUTPUT_DIR = "data/experiments/data_size"

# Training data percentages to test
# Example: [0.1, 0.2, 0.3] means train with 10%, 20%, and 30% of available data
TRAIN_PERCENTAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Alternative quick test with fewer percentages:
# TRAIN_PERCENTAGES = [0.2, 0.4, 0.6, 0.8]

# Model hyperparameters
EPOCHS = 50              # Number of training epochs per model
LEARNING_RATE = 0.0001   # Learning rate
INPUT_SIZE = (512, 512)  # Input patch size
WITH_AUGMENTATION = True # Whether to use data augmentation
RANDOM_SEED = 42         # Random seed for reproducibility

# Data split configuration
VAL_SPLIT = 0.2   # 20% of remaining data for validation
TEST_SPLIT = 0.2  # 20% of total data for testing

# ============================================================================
# RUN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("STARTING DATA SIZE EXPERIMENT")
    print("="*80)
    print(f"Dataset: {IMAGES_PATH}")
    print(f"Training percentages: {[int(p*100) for p in TRAIN_PERCENTAGES]}%")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Data augmentation: {WITH_AUGMENTATION}")
    print("="*80)
    
    # Create experiment
    experiment = DataSizeExperiment(
        images_path=IMAGES_PATH,
        masks_path=MASKS_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Run experiment
    results = experiment.run_experiment(
        train_percentages=TRAIN_PERCENTAGES,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        input_size=INPUT_SIZE,
        with_augmentation=WITH_AUGMENTATION,
        random_seed=RANDOM_SEED
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Check results in: {OUTPUT_DIR}")
