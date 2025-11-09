import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, cat, device, cuda, no_grad, save, load
from torch.utils.data import DataLoader
import numpy as np
from threading import Event
import os
from typing import Optional
from torch import autocast, GradScaler

from src.model.PlottingTools import *
from src.model.DiceLoss import DiceLoss, WeightedDiceLoss, ForegroundDiceLoss, FocalLoss, CombinedLoss, TverskyLoss, BoundaryLoss, SizePenaltyLoss

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = F.relu(self.bn(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, input, concat_map):
        x: Tensor = self.upconv(input)

        x = cat((concat_map, x), dim=1)
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
        
class UNet(nn.Module):
    def __init__(self, pre_loaded_model_path: Optional[str] = None, normalizer: Optional[nn.Module] = None):
        super().__init__()

        self.encoder1 = EncoderBlock(1, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bottleneck = EncoderBlock(512, 1024)

        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        self.mappingConvolution = nn.Conv2d(64, 2, 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.optimizer = None
        self.criterion = None
        
        self.device = device("cuda" if cuda.is_available() else "cpu")
        print(f"Using {self.device}")

        self.preferred_input_size = (256, 256)
        self.normalizer = normalizer

        # --- LOAD MODEL (only in eager mode) ---
        if pre_loaded_model_path is not None and not torch.jit.is_scripting():
            from src.model.DataTools import resource_path
            model_path = resource_path(pre_loaded_model_path)
            self.load_model(model_path)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            input = self.normalizer(input)

        e1 = self.encoder1(input)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d1 = self.decoder1(b, e4)
        d2 = self.decoder2(d1, e3)
        d3 = self.decoder3(d2, e2)
        d4 = self.decoder4(d3, e1)

        m = self.mappingConvolution(d4)
        #self._visualize_feature_map(m, "HiSpec_2000_285k", True)
        return m
    
    def _configure_scheduler(self, scheduler_type="plateau"):
        """Configure learning rate scheduler based on type"""
        if scheduler_type in (None, "none"):
            return None
        if scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',           # Reduce LR when validation loss stops decreasing
                factor=0.5,           # Multiply LR by 0.5
                patience=10,          # Wait 10 epochs before reducing
                min_lr=1e-7,          # Don't reduce below this
                verbose=True          # Print when LR is reduced
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=50,            # Period of cosine annealing
                eta_min=1e-7         # Minimum learning rate
            )
        elif scheduler_type == "cosine_restart":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=20,              # Restart every 20 epochs
                T_mult=2,            # Double the restart period each time
                eta_min=1e-7         # Minimum learning rate
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,        # Reduce LR every 30 epochs
                gamma=0.5            # Multiply by 0.5
            )
        elif scheduler_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95           # Multiply by 0.95 each epoch
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def get_current_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def _create_focal_dice_loss(self):
        """Create custom Focal + Dice combination loss"""
        class FocalDiceLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.focal = FocalLoss(alpha=0.25, gamma=2.0)
                self.dice = DiceLoss()
                
            def forward(self, prediction, target):
                focal_loss = self.focal(prediction, target)
                dice_loss = self.dice(prediction, target)
                return 0.6 * focal_loss + 0.4 * dice_loss
        
        return FocalDiceLoss()
    
    def _create_boundary_dice_loss(self):
        """Create custom Boundary + Dice combination loss"""
        class BoundaryDiceLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.boundary = BoundaryLoss(boundary_weight=3.0)
                self.dice = DiceLoss()
                
            def forward(self, prediction, target):
                boundary_loss = self.boundary(prediction, target)
                dice_loss = self.dice(prediction, target)
                return 0.5 * boundary_loss + 0.5 * dice_loss
        
        return BoundaryDiceLoss()
    
    def train_model(self, training_dataloader: DataLoader, validation_dataloader: DataLoader, epochs: int, learningRate: float, model_name: str, with_early_stopping: bool, loss_function: str, scheduler_type: str = "plateau", stop_training_event: Event = None, loss_callback = None):
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), learningRate, weight_decay=1e-4)
        
        # Configure learning rate scheduler
        self.scheduler = self._configure_scheduler(scheduler_type=scheduler_type)
        
        if self.device.type == 'cuda':
            scaler = GradScaler("cuda")

        if loss_function == "dice":
            self.criterion = DiceLoss()
        elif loss_function == "dice2":
            self.criterion = ForegroundDiceLoss()
        elif loss_function == "weighted_dice":
            self.criterion = WeightedDiceLoss(class_weights=[1.0, 2.0])
        elif loss_function == "weighted_cross_entropy":
            self.criterion = nn.CrossEntropyLoss(weight=Tensor([1.0, 2.0], device=self.device))
        elif loss_function == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        
        # NEW LOSS FUNCTIONS
        elif loss_function == "focal":
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif loss_function == "focal_strong":
            self.criterion = FocalLoss(alpha=0.25, gamma=3.0)  # Stronger focus on hard examples
        elif loss_function == "combined":
            self.criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
        elif loss_function == "combined_dice_heavy":
            self.criterion = CombinedLoss(ce_weight=0.3, dice_weight=0.7)  # More emphasis on Dice
        elif loss_function == "tversky":
            self.criterion = TverskyLoss(alpha=0.3, beta=0.7)  # Penalize false negatives more
        elif loss_function == "tversky_balanced":
            self.criterion = TverskyLoss(alpha=0.5, beta=0.5)  # Balanced
        elif loss_function == "boundary":
            self.criterion = BoundaryLoss(boundary_weight=5.0)
        elif loss_function == "size_penalty":
            self.criterion = SizePenaltyLoss(expected_size_range=(50, 500), penalty_weight=0.1)
        
        # COMBINATION APPROACHES
        elif loss_function == "focal_dice":
            # Custom combination of Focal + Dice
            self.criterion = self._create_focal_dice_loss()
        elif loss_function == "boundary_dice":
            # Custom combination of Boundary + Dice  
            self.criterion = self._create_boundary_dice_loss()
        
        else:
            raise ValueError(f"Unknown loss function: {loss_function}. Available options: "
                           f"'dice', 'dice2', 'weighted_dice', 'cross_entropy', 'weighted_cross_entropy', "
                           f"'focal', 'focal_strong', 'combined', 'combined_dice_heavy', 'tversky', "
                           f"'tversky_balanced', 'boundary', 'size_penalty', 'focal_dice', 'boundary_dice'")

        training_loss_values = []
        validation_loss_values = []
        best_loss = np.inf
        no_improvement_epochs = 0
        min_delta = 1e-4  # Minimum improvement threshold
        early_stopping_patience = 25  # Increased patience for better training
        batches_in_epoch = len(training_dataloader.dataset)//training_dataloader.batch_size
        import time
        for epoch in range(epochs):
            start_time = time.perf_counter()
            self.train()
            running_loss = 0.0
            
            for i, data in enumerate(training_dataloader):
                if (stop_training_event is not None) and stop_training_event.is_set():
                    print("Training stopped by user.")
                    self.load_model("data/models/" + model_name)
                    return training_loss_values, validation_loss_values
                
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.long().squeeze(1)

                self.optimizer.zero_grad()

                if self.device.type == 'cuda':
                    with autocast("cuda"):
                        outputs = self(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                
                if self.device.type == 'cuda':
                    scaler.scale(loss).backward()
                    # Add gradient clipping
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                running_loss += loss.item()
            epoch_training_loss = running_loss / len(training_dataloader)
            training_loss_values.append(epoch_training_loss)

            epoch_validation_loss = self.get_validation_loss(validation_dataloader)
            validation_loss_values.append(epoch_validation_loss)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_validation_loss)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'---Epoch {epoch + 1}: Training loss: {epoch_training_loss:.5f}, Validation loss: {epoch_validation_loss:.5f}---')
            if epoch_validation_loss < best_loss - min_delta:
                self.save_model("data/models/", model_name)
                best_loss = epoch_validation_loss
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if with_early_stopping and no_improvement_epochs >= early_stopping_patience:
                    print(f"Early stopping triggered after {no_improvement_epochs} epochs without improvement")
                    break
            
            if loss_callback:
                from src.shared.ModelTrainingStats import ModelTrainingStats

                stats = ModelTrainingStats(training_loss=epoch_training_loss,
                                           val_loss=epoch_validation_loss,
                                           best_loss=best_loss,
                                           epoch=epoch+1,
                                           best_epoch=epoch+1 - no_improvement_epochs)
                loss_callback(stats)
            end_time = time.perf_counter()
            print(f"Time: {end_time - start_time:.4f} seconds")

            
        
        print('Finished Training')
        self.load_model("data/models/" + model_name)
        return training_loss_values, validation_loss_values


    def get_validation_loss(self, validation_dataloader: DataLoader) -> float:
        self.eval()
        running_loss = 0
        with no_grad():
            for i, data in enumerate(validation_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.long().squeeze(1)
                if self.device.type == 'cuda':
                    with autocast("cuda"):
                        outputs = self(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                running_loss += loss.item()

        return running_loss / len(validation_dataloader)

    def save_model(self, folder_path: str, model_name: str):
        path = folder_path + model_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save({
        "model_state_dict": self.state_dict(),
        "normalizer_mean": self.normalizer.mean,
        "normalizer_std": self.normalizer.std,}, path)
        
    def load_model(self, path):
        model = load(path, map_location=self.device, weights_only=True)
        from torchvision.transforms.v2 import Normalize
        self.normalizer = Normalize(mean=model["normalizer_mean"], std=model["normalizer_std"])
        self.load_state_dict(model["model_state_dict"])
    
    def _visualize_feature_map(self, feature_map: Tensor, title: str, is_output: bool = False):
        """
        Helper function to visualize feature maps with a color bar.
        """
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        import torch.nn.functional as F
        from src.model.DataTools import construct_image_from_patches, center_crop
        feature_map = F.softmax(feature_map, dim=1)
        feature_map = feature_map.detach().cpu().numpy()
        num_channels = feature_map.shape[1]
        print(feature_map.shape)
        collected_image = construct_image_from_patches(feature_map, (1072, 1072), (204,204))
        collected_image = center_crop(collected_image, (1024, 1024))
        feature_map = collected_image
        print(feature_map.shape)
        
        # Plot the first few feature maps
        num_to_plot = min(8, num_channels)  
        fig, axes = plt.subplots(1, num_to_plot, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)

        vmin = 0.0
        vmax = 1.0

        for i in range(num_to_plot):
            ax = axes[i]
            im = ax.imshow(feature_map[0, i, :, :], cmap='jet', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if i == 0:
                ax.set_title(f"Class {i + 1} = Background", fontsize=16, weight='bold')
            else:
                ax.set_title(f"Class {i + 1} = Foreground (Nanoparticle)", fontsize=16, weight='bold')

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=14)  # Increase tick label size

        plt.tight_layout()
        
        # Save figure instead of showing (thread-safe)
        import os
        from datetime import datetime
        output_dir = "data/feature_maps"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = os.path.join(output_dir, f"{title.replace(' ', '_')}_{timestamp}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Feature map saved to: {filename}")
        plt.close(fig)
