from torch.utils.data import Dataset
from src.model.SegmentationDataset import RepeatDataset, SegmentationDataset


class DataAugmenter():
    def __init__(self):
        return 
    
    @staticmethod
    def get_transformer(crop: bool, rotate: bool, flip: bool, deform: bool, adjust_brightness: bool, adjust_contrast: bool, blur: bool):
        def transformer(image, mask):
            from torchvision.transforms.v2 import ElasticTransform, RandomCrop, GaussianBlur
            import torchvision.transforms.functional as TF
            from torchvision.transforms import InterpolationMode
            import random

            # Elastic deformation
            if deform:
                params = ElasticTransform.get_params(
                    size=[512, 512],
                    alpha=(20.0, 60),
                    sigma=(4.0, 6.0)
                )
                image = TF.elastic_transform(image, params)
                mask = TF.elastic_transform(mask, params, interpolation=InterpolationMode.NEAREST)

            # Random crop
            if crop:
                i, j, h, w = RandomCrop.get_params(
                    image, output_size=(256, 256))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)

            # Random rotation
            if rotate:
                angle = random.choice([0, 90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # Random horizontal flipping
            if flip and random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Brightness adjustment
            if adjust_brightness:
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)

            # Contrast adjustment
            if adjust_contrast:
                contrast_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_contrast(image, contrast_factor)

            # Random blur
            if blur:
                blur_transform = GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))
                if random.random() < 0.5:
                    image = blur_transform(image)

            return image, mask

        return transformer

    @staticmethod
    def _remove_random_components(mask, drop_fraction: float, min_components_to_keep: int):
        """Remove a random subset of connected components from the mask."""
        if drop_fraction <= 0:
            return mask

        import random
        import numpy as np
        import torch
        import cv2

        mask_np = mask.detach().cpu().numpy().squeeze()
        binary_mask = (mask_np > 0.5).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(binary_mask)
        component_ids = list(range(1, num_labels))
        if not component_ids:
            return mask

        max_removable = max(len(component_ids) - min_components_to_keep, 0)
        requested_removals = int(round(drop_fraction * len(component_ids)))
        remove_count = min(max_removable, requested_removals)
        if remove_count <= 0:
            return mask

        remove_ids = random.sample(component_ids, remove_count)
        binary_mask[np.isin(labels, remove_ids)] = 0

        updated = torch.from_numpy(binary_mask).to(mask.device).type_as(mask)
        if updated.ndim == 2:
            updated = updated.unsqueeze(0)
        return updated

    @staticmethod
    def _inject_random_noise(mask, noise_fraction: float, white_probability: float):
        """Flip a fraction of pixels to random foreground/background values."""
        if noise_fraction <= 0:
            return mask

        import numpy as np
        import torch

        mask_np = mask.detach().cpu().numpy().squeeze()
        total_pixels = mask_np.size
        noise_pixels = min(total_pixels, int(noise_fraction * total_pixels))
        if noise_pixels == 0:
            return mask

        flat = mask_np.reshape(-1)
        indices = np.random.choice(total_pixels, size=noise_pixels, replace=False)
        noise_values = (np.random.rand(noise_pixels) < white_probability).astype(flat.dtype)
        flat[indices] = noise_values

        updated = torch.from_numpy(flat.reshape(mask_np.shape)).to(mask.device).type_as(mask)
        if updated.ndim == 2:
            updated = updated.unsqueeze(0)
        return updated

    @staticmethod
    def corrupt_masks(dataset: Dataset, component_drop_fraction: float = 0.0, min_components_to_keep: int = 0, noise_fraction: float = 0.0, noise_white_probability: float = 0.5, seed: int = None, save_dir: str = None) -> Dataset:
        """
        Permanently corrupt masks by removing connected components and/or injecting random noise.

        Args:
            dataset: SegmentationDataset or Subset referencing one.
            component_drop_fraction: fraction of components to remove per mask (0-1).
            min_components_to_keep: lower bound on remaining components (prevents empty masks).
            noise_fraction: fraction of pixels to flip to random foreground/background.
            noise_white_probability: probability that a flipped pixel becomes foreground.
            seed: optional seed for reproducibility.
            save_dir: optional directory to save corrupted masks for inspection.
        """
        if component_drop_fraction <= 0 and noise_fraction <= 0:
            return dataset

        import numpy as np
        import random
        import os

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            target_dataset = dataset.dataset
            target_indices = dataset.indices
        else:
            target_dataset = dataset
            target_indices = range(len(dataset))

        for idx in target_indices:
            mask = target_dataset.masks[idx]
            mask = DataAugmenter._remove_random_components(mask, component_drop_fraction, min_components_to_keep)
            mask = DataAugmenter._inject_random_noise(mask, noise_fraction, noise_white_probability)
            target_dataset.masks[idx] = mask
            if save_dir:
                import numpy as np
                from PIL import Image

                mask_np = mask.detach().cpu().numpy().squeeze()
                mask_np = (mask_np > 0.5).astype(np.uint8) * 255
                filename = None
                if hasattr(target_dataset, "image_filenames") and target_dataset.image_filenames:
                    filename = target_dataset.image_filenames[idx]
                if not filename:
                    filename = f"mask_{idx}.png"
                base, ext = os.path.splitext(filename)
                if not ext:
                    ext = ".png"
                save_path = os.path.join(save_dir, f"{base}_corrupted{ext}")
                Image.fromarray(mask_np).save(save_path)

        return dataset

    
    def augment_dataset(self, dataset: Dataset, augmentations=[True,True,False,False,True,True,False]) -> Dataset:
        new_images = []
        new_masks = []
        new_filenames = []
        for i in range(len(dataset)):
            image, mask = dataset[i]
            # Handle both regular datasets and Subset datasets
            if hasattr(dataset, 'indices'):
                # This is a Subset, get the filename from the original dataset
                original_idx = dataset.indices[i]
                filename = dataset.dataset.image_filenames[original_idx]
            else:
                # This is a regular dataset
                filename = dataset.image_filenames[i]
            
            new_images.extend(image.unsqueeze(0))
            new_masks.extend(mask.unsqueeze(0))
            new_filenames.append(filename)
        
        return RepeatDataset(dataset=SegmentationDataset.from_image_set(new_images, new_masks, new_filenames, transforms=DataAugmenter.get_transformer(*augmentations)), repeat_factor=10 if augmentations[0] else 20)
    
