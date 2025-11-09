from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import cv2


class RepeatDataset(Dataset):
    def __init__(self, dataset, repeat_factor):
        self.dataset = dataset
        self.repeat_factor = repeat_factor

    def __len__(self):
        return len(self.dataset) * self.repeat_factor

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]
    
    @property
    def image_filenames(self):
        """Expose the underlying dataset's image filenames, repeated as needed."""
        if hasattr(self.dataset, 'image_filenames'):
            base_filenames = self.dataset.image_filenames
            # Repeat the filenames list for each repeat factor
            repeated_filenames = []
            for _ in range(self.repeat_factor):
                repeated_filenames.extend(base_filenames)
            return repeated_filenames
        else:
            return None

class SegmentationDataset(Dataset):
    def __init__(self, image_dir=None, mask_dir=None, transform=None):

        self.images = []
        self.masks = []
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        if not image_dir or not mask_dir:
            return
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform
        
        if len(self.image_filenames) != len(self.mask_filenames):
            raise ValueError("The number of images and masks must be the same.")

        import torchvision.transforms.functional as TF
        from src.model.DataTools import tiff_force_8bit

        for index in range(len(self.image_filenames)):
            img_path = os.path.join(self.image_dir, self.image_filenames[index])
            mask_path = os.path.join(self.mask_dir, self.mask_filenames[index])

            image = Image.open(img_path)
            image = tiff_force_8bit(image).convert("L")
            mask = Image.open(mask_path).convert("L")
            
            # Validate dimensions
            self._validate_dimensions(image, mask, self.image_filenames[index], self.mask_filenames[index])
            
            # Fix mask binarization
            mask = self._fix_mask_binarization(mask)
            
            # Validate mask is properly binarized
            self._validate_mask_binarization(mask, self.mask_filenames[index])
            
            # Resize images preserving aspect ratio (max 1024x1024)
            image, mask = self._resize_pair(image, mask)

            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

            self.images.append(image)
            self.masks.append(mask)
        
    
    def _validate_dimensions(self, image, mask, img_filename, mask_filename):
        """Validate that image and mask have the same dimensions."""
        if image.size != mask.size:
            raise ValueError(
                f"Image and mask dimensions don't match:\n"
                f"  Image '{img_filename}': {image.size}\n"
                f"  Mask '{mask_filename}': {mask.size}\n"
                f"  All image-mask pairs must have identical dimensions."
            )
    
    def _fix_mask_binarization(self, mask):
        """Fix near-binary values by flooring/ceiling to 0/255."""
        mask_array = np.array(mask)
        
        # Apply ceiling/flooring for near-binary values
        mask_array[mask_array <= 10] = 0      # Floor values ≤10 to 0
        mask_array[mask_array >= 245] = 255   # Ceil values ≥245 to 255
        
        # Create new mask with fixed values
        return Image.fromarray(mask_array, mode='L')
    
    def _validate_mask_binarization(self, mask, mask_filename):
        """Validate that mask contains only binary values (0 and/or 255)."""
        mask_array = np.array(mask)
        unique_values = np.unique(mask_array)
        
        valid_combinations = [
            np.array([0]),          # All background
            np.array([255]),        # All foreground
            np.array([0, 255])      # Proper binary
        ]
        
        is_valid = any(np.array_equal(unique_values, valid) for valid in valid_combinations)
        if not is_valid:
            raise ValueError(
                f"Mask '{mask_filename}' contains non-binary values after auto-fix.\n"
                f"  Values found: {unique_values}\n"
                f"  Expected only: [0], [255], or [0, 255]"
            )
        
    def _resize_pair(self, image, mask):
        """Resize image and mask to maximum 1024x1024 preserving aspect ratio."""
        target_size = (1024, 1024)
        
        # Only resize if image is larger than target
        if image.size[0] > target_size[0] or image.size[1] > target_size[1]:
            # Use thumbnail to preserve aspect ratio
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            mask.thumbnail(target_size, Image.Resampling.NEAREST)  # Use NEAREST for masks to preserve binary values        
        return image, mask

    @classmethod
    def from_image_set(cls, images, masks, file_names=None, transforms=None):
        res = cls()
        res.images = images
        res.masks = masks
        res.image_filenames = file_names
        res.transform = transforms
        return res

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        if self.transform:
            image, mask = self.transform(self.images[index], self.masks[index])
        else:
            image = self.images[index]
            mask = self.masks[index]
        
        return image, mask
    
    @staticmethod
    def apply_clahe(image):
        img = np.array(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(img)
        return equalized