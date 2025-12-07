import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
from torch import Tensor
import numpy as np
import sys
from src.model.DataAugmenter import DataAugmenter
from src.model.dmFileReader import dmFileReader
from src.shared.IOFunctions import is_dm_format
from src.model.SegmentationDataset import SegmentationDataset
from src.shared.ModelConfig import ModelConfig

def _get_safe_num_workers(desired_workers: int) -> int:
    """
    Avoid spawning extra GUI instances when running the frozen PyInstaller build.
    Torch DataLoader workers spin up new Python processes; in a frozen app those
    processes relaunch the executable unless worker count is zero.
    """
    if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
        return 0
    return desired_workers

class ImagePreprocessor:
    """Handles all image preprocessing operations for the segmentation model."""
    
    def __init__(self, model_input_size):
        self.model_input_size = model_input_size
        
    def prepare_image_patches(self, pil_image, device):
        """Prepare image patches for segmentation by converting to tensor and extracting patches."""
        import torchvision.transforms.functional as TF
        
        tensor = TF.to_tensor(pil_image).unsqueeze(0).to(device)
        stride_length = self.model_input_size[0]*4//5
        tensor_mirror_filled = mirror_fill(tensor, self.model_input_size, (stride_length,stride_length))
        patches = extract_slices(tensor_mirror_filled, self.model_input_size, (stride_length,stride_length))
        
        return tensor, tensor_mirror_filled, patches, stride_length
    
    def post_process_segmentation(self, segmentations, tensor_mirror_filled, tensor, stride_length):
        """Post-process the segmentation output."""
        segmented_image = construct_image_from_patches(
            segmentations, 
            tensor_mirror_filled.shape[2:], 
            (stride_length, stride_length)
        )
        segmented_image = center_crop(segmented_image, (tensor.shape[2], tensor.shape[3]))
        segmented_image = binarize_segmentation_output(segmented_image)
        return to_2d_image_array(segmented_image)

def slice_dataset_in_four(dataset, input_size=(256, 256)):
    images = []
    masks = []
    filenames = []
    for (img, mask), filename in zip(dataset, dataset.image_filenames):
        width = img.shape[-1]
        height = img.shape[-2]
        if width <= input_size[0] or height <= input_size[1]:
            images.append(img)
            masks.append(mask)
            continue
        new_width = width // 2
        new_height = height // 2

        image_slices = [
            img[:, :new_height, :new_width],
            img[:, new_height:, :new_width],
            img[:, :new_height, new_width:],
            img[:, new_height:, new_width:]
        ]
        mask_slices = [
            mask[:, :new_height, :new_width],
            mask[:, new_height:, :new_width],
            mask[:, :new_height, new_width:],
            mask[:, new_height:, new_width:]
        ]
        filename_slices = [
            filename + " TOP LEFT",
            filename + " BOTTOM LEFT",
            filename + " TOP RIGHT",
            filename + " BOTTOM RIGHT"
        ]
        images.extend(image_slices)
        masks.extend(mask_slices)
        filenames.extend(filename_slices)
    return SegmentationDataset.from_image_set(images, masks, filenames)

def process_no_slice(data_subset):
    images = []
    masks = []
    filenames = []
    
    for (img, mask), idx in zip(data_subset, data_subset.indices):
        img = img.unsqueeze(0) if img.dim() == 3 else img
        mask = mask.unsqueeze(0) if mask.dim() == 3 else mask

        images.extend(img)
        masks.extend(mask)
        
        # Get the base filename for this image
        base_filename = data_subset.dataset.image_filenames[idx]
        filenames.append(base_filename)

    # Create list of (image, mask) tensors
    return SegmentationDataset.from_image_set(images, masks, filenames)

# Helper to process val/test with mirror_fill and extract_slices
def process_and_slice(data_subset, input_size=(256, 256)):
    images = []
    masks = []
    filenames = []
    
    for (img, mask), idx in zip(data_subset, data_subset.indices):
        img = img.unsqueeze(0) if img.dim() == 3 else img
        mask = mask.unsqueeze(0) if mask.dim() == 3 else mask

        filled_image = mirror_fill(img, patch_size=input_size, stride_size=input_size)
        filled_mask = mirror_fill(mask, patch_size=input_size, stride_size=input_size)

        sliced_images = extract_slices(filled_image, patch_size=input_size, stride_size=input_size)
        sliced_masks = extract_slices(filled_mask, patch_size=input_size, stride_size=input_size)
        
        if isinstance(sliced_masks[0], np.ndarray):
            sliced_images = [torch.from_numpy(img) for img in sliced_images]
        if isinstance(sliced_masks[0], np.ndarray):
            sliced_masks = [torch.from_numpy(mask) for mask in sliced_masks]
        images.extend(sliced_images)
        masks.extend(sliced_masks)
        
        # Get the base filename for this image
        base_filename = data_subset.dataset.image_filenames[idx]
        
        # Convert tensors to list and add to results
        num_slices = len(sliced_images)
        slice_filenames = []
        for i in range(num_slices):
            slice_filename = f"{base_filename}_slice_{i:03d}"
            slice_filenames.append(slice_filename)

        filenames.extend(slice_filenames)


    # Create list of (image, mask) tensors
    return SegmentationDataset.from_image_set(images, masks, filenames)

def log_data_split_info(dataset, train_data, val_data, test_data=None, log_file_path=None):
    """
    Log data split information including indices and filenames.
    
    Args:
        dataset: The original dataset containing image_filenames
        train_data: Training data subset
        val_data: Validation data subset  
        test_data: Test data subset (optional)
        log_file_path: Optional path for file logging
    """
    if log_file_path is None:
        return
    train_filenames = sorted([dataset.image_filenames[i] for i in train_data.indices])
    val_filenames = sorted([dataset.image_filenames[i] for i in val_data.indices])
    test_filenames = sorted([dataset.image_filenames[i] for i in test_data.indices]) if test_data is not None else None


    import os
    log_dir = os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else "data/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    suffix = "with_testset" if test_data is not None else "without_testset"
    split_log_path = os.path.join(log_dir, f"data_splits_{suffix}.txt")
    
    with open(split_log_path, 'w', encoding='utf-8') as f:
        title = "Data Split Information" + (" (With Test Set)" if test_data is not None else " (Without Test Set)")
        f.write(f"{title}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Train files ({len(train_filenames)}):\n")
        for filename in train_filenames:
            f.write(f"  {filename}\n")
        
        f.write(f"\nValidation files ({len(val_filenames)}):\n")
        for filename in val_filenames:
            f.write(f"  {filename}\n")
        
        if test_filenames is not None:
            f.write(f"\nTest files ({len(test_filenames)}):\n")
            for filename in test_filenames:
                f.write(f"  {filename}\n")
    
    print(f"Data split information logged to: {split_log_path}")

def get_dataloaders(dataset: Dataset, model_config: ModelConfig, input_size: tuple[int, int], log_file_path = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    # Set parameters:
    dataset = slice_dataset_in_four(dataset, input_size)
    train_data, val_data, test_data = get_data_splits(dataset, model_config, input_size, log_file_path)    
    
    if torch.cuda.is_available():
        batch_size = 32
        worker_count = _get_safe_num_workers(12)
    else:
        batch_size = 8
        worker_count = _get_safe_num_workers(2)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=worker_count)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=1)
    return (train_dataloader, val_dataloader, test_dataloader)

def get_data_splits(dataset, model_config, input_size, log_file_path=None):
    train_data, val_data, test_data = split_dataset(dataset, model_config)
    log_data_split_info(dataset, train_data, val_data, test_data, log_file_path)

    # Augment training data
    data_augmenter = DataAugmenter()
    if model_config.with_data_augmentation:
        train_data = data_augmenter.augment_dataset(train_data)
    else:
        train_data = data_augmenter.augment_dataset(train_data, [False, False, False, False, False, False, False])
    val_data = process_and_slice(val_data, input_size)
    test_data = process_no_slice(test_data)
    return train_data, val_data, test_data


def split_dataset(dataset, model_config):
    if model_config.test_images_path and model_config.test_masks_path:
        train_size = model_config.train_subset_size + model_config.validation_subset_size
        val_size = 1-train_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        test_data = SegmentationDataset(model_config.test_images_path, model_config.test_masks_path)
    else:
        train_size = model_config.train_subset_size
        val_size = model_config.validation_subset_size
        test_size = 1 - (train_size + val_size)
        train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    return train_data, val_data, test_data

def get_dataloaders_kfold_already_split(train_data, val_data, batch_size, input_size, augmentations=[True,True,False,False,True,True,False]):
    data_augmenter = DataAugmenter()
    print(augmentations)
    if not augmentations[0]: # No random cropping
        train_data = process_and_slice(train_data, input_size)
    train_data = data_augmenter.augment_dataset(train_data, augmentations)

    val_data = process_and_slice(val_data, input_size)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=_get_safe_num_workers(24),
    )
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True)
    return (train_dataloader, val_dataloader)

def center_crop(image, target_size: tuple[int, int]):
    _, _, h, w = image.shape
    th, tw = target_size

    start_h = (h-th) // 2
    start_w = (w-tw) // 2
    return image[:, :, start_h:start_h + th, start_w:start_w + tw]

def resize_and_save_images(folder_path, output_size=(256, 256), is_masks=False):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.tif')):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            img = tiff_force_8bit(img)
            img = img.convert("L")  
            if img.width == output_size[0] and img.height == output_size[1]:
                continue
            img = img.resize(output_size)
            if is_masks:
                img_binary = img.point(lambda p: 255 if p > 20 else 0)
                img_binary.save(os.path.join(folder_path,"new"+filename))
            else:    
                img.save(os.path.join(folder_path,"new"+filename))  
            print(image_path)

def tensor_from_image_no_resize(image_path: str):
    import torchvision.transforms.functional as TF
    image = Image.open(image_path).convert("L")
    image = TF.to_tensor(image).unsqueeze(0)
    return image

def tensor_from_image(image_path: str, resize=(256,256)) -> Tensor:
    import torchvision.transforms.functional as TF

    image = Image.open(image_path).convert("L")
    image.thumbnail(resize)
    image = TF.to_tensor(image).unsqueeze(0)
    return image

def to_2d_image_array(array: np.ndarray) -> np.ndarray:
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return (np.squeeze(array) * 255).astype(np.uint8)

def load_image_as_tensor(image_path: str):
    import torchvision.transforms.functional as TF

    reader = dmFileReader()
    tensor = None
    if is_dm_format(image_path):
        tensor = reader.get_tensor_from_dm_file(image_path)
    else:
        tensor = tensor_from_image_no_resize(image_path)
    if tensor.shape[-1] > 1024 or tensor.shape[-2] > 1024:
        tensor = TF.resize(tensor, 1024)
    return tensor

def binarize_segmentation_output(segmented_image, high_thresh=0.85):
    """
    Post-process U-Net probabilities by seeding on high-confidence pixels
    and growing into lower-confidence regions.

    Args:
        segmented_image (numpy.ndarray): U-Net logits [1, 2, H, W].
        high_thresh (float): Seed threshold for confident foreground.
        mean_prob_thresh (float): Min avg probability for final region.

    Returns:
        numpy.ndarray: Binary mask [1, H, W].
    """    
    # Step 1: Extract probabilities and compute confidence
    fg_prob, confidence_margin = _extract_probabilities(segmented_image)
    
    # Step 2: Generate seeds and candidate regions
    seeds, candidates = _generate_regions(fg_prob, segmented_image, high_thresh)
    
    # Step 3: Find candidate regions that contain seeds
    valid_candidate_ids = _find_seeded_candidates(seeds, candidates)
    
    # Step 4: Filter candidates by confidence margin
    approved_regions = _filter_by_confidence(valid_candidate_ids, candidates, confidence_margin)
    
    # Step 5: Create final binary mask
    return _create_final_mask(candidates, approved_regions, seeds.shape)


def _extract_probabilities(segmented_image):
    """Extract foreground probabilities and confidence margin."""
    from scipy.special import softmax
    
    if torch.is_tensor(segmented_image):
        segmented_image = segmented_image.detach().cpu().numpy()

    probs = softmax(segmented_image, axis=1)
    fg_prob = probs[0, 1]  # Foreground probability
    bg_prob = probs[0, 0]  # Background probability
    confidence_margin = fg_prob - bg_prob
    
    return fg_prob, confidence_margin


def _generate_regions(fg_prob, segmented_image, high_thresh):
    """Generate seed pixels and candidate regions."""
    from skimage.measure import label
    
    # High-confidence seed pixels
    seeds = label(fg_prob > high_thresh)
    
    # All positive predictions as candidates
    candidates = label(segmented_image.argmax(axis=1)[0])
    
    return seeds, candidates


def _find_seeded_candidates(seeds, candidates):
    """Find candidate regions that contain at least one seed pixel."""
    import numpy as np
    
    # Map seed pixels to their candidate regions
    seed_pixels = seeds > 0
    seed_candidate_map = np.where(seed_pixels, candidates, 0)
    
    # Get unique candidate region IDs that contain seeds
    valid_candidate_ids = np.unique(seed_candidate_map[seed_candidate_map > 0])
    
    return valid_candidate_ids


def _filter_by_confidence(valid_candidate_ids, candidates, confidence_margin):
    """Filter candidate regions by average confidence margin."""
    import numpy as np
    
    if len(valid_candidate_ids) == 0:
        return []
    
    # Vectorized calculation of average confidence for each region
    flat_labels = candidates.ravel()
    flat_margins = confidence_margin.ravel()
    max_id = valid_candidate_ids.max()
    
    # Sum margins and count pixels for each candidate region
    margin_sums = np.bincount(flat_labels, weights=flat_margins, minlength=max_id + 1)
    region_counts = np.bincount(flat_labels, minlength=max_id + 1)
    
    # Calculate average margins (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        margin_means = margin_sums[valid_candidate_ids] / region_counts[valid_candidate_ids]
    
    # Return regions with positive average confidence
    return valid_candidate_ids[margin_means > 0]


def _create_final_mask(candidates, approved_regions, shape):
    """Create the final binary mask from approved regions."""
    import numpy as np
    
    final_mask = np.zeros(shape, dtype=bool)
    
    if len(approved_regions) > 0:
        final_mask[np.isin(candidates, approved_regions)] = True
    
    return np.expand_dims(final_mask.astype(np.uint8), axis=0)

# Made with help from https://stackoverflow.com/questions/65754703/pillow-converting-a-tiff-from-greyscale-16-bit-to-8-bit-results-in-fully-white
def tiff_force_8bit(image, **kwargs):
    if image.format == 'TIFF' and image.mode in ('I;16', 'I;16B', 'I;16L'):
        array = np.array(image)
        normalized = (array.astype(np.uint16) - array.min()) * 255.0 / (array.max() - array.min())
        range_val = array.max() - array.min()
        if range_val == 0:
            normalized = np.zeros_like(array, dtype=np.uint8)
        else:
            normalized = (array.astype(np.uint16) - array.min()) * 255.0 / range_val
        image = Image.fromarray(normalized.astype(np.uint8))

    return image

# Made with help from https://www.programmersought.com/article/15316517340/
def mirror_fill(images: Tensor, patch_size: tuple, stride_size: tuple) -> Tensor:
    # images: (B, C, H, W)
    _, _, img_height, img_width = images.shape
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride_size

    remaining_width = (img_width - patch_width) % stride_width
    remaining_height = (img_height - patch_height) % stride_height

    needed_padding_width = (stride_width - remaining_width) % stride_width
    needed_padding_height = (stride_height - remaining_height) % stride_height

    if needed_padding_height == 0 and needed_padding_width == 0:
        return images

    pad_left = needed_padding_width // 2
    pad_right = needed_padding_width - pad_left
    pad_top = needed_padding_height // 2
    pad_bottom = needed_padding_height - pad_top

    return F.pad(
        images,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="reflect",
    )
    


# Made with help from https://www.programmersought.com/article/15316517340/
def extract_slices(images: Tensor, patch_size: tuple, stride_size: tuple):
    # images: (B, C, H, W)
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride_size

    unfolded = images.unfold(2, patch_height, stride_height).unfold(3, patch_width, stride_width)
    # Shape: (B, C, n_h, n_w, patch_height, patch_width)
    b, c, n_h, n_w, _, _ = unfolded.shape
    patches = unfolded.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, patch_height, patch_width).contiguous()
    return patches

def construct_image_from_patches(patches: np.ndarray, img_size: tuple, stride_size: tuple):
    """
    Reconstruct an image from patches using torch.fold with proper overlap averaging.
    Accepts torch tensors or numpy arrays; returns a torch tensor on the same device.
    """
    if not torch.is_tensor(patches):
        patches = torch.as_tensor(patches)

    img_height, img_width = img_size
    stride_height, stride_width = stride_size
    n_patches_total, channels, patch_height, patch_width = patches.shape

    n_patches_y = (img_height - patch_height) // stride_height + 1
    n_patches_x = (img_width - patch_width) // stride_width + 1
    n_patches_per_image = n_patches_x * n_patches_y

    if n_patches_total % n_patches_per_image != 0:
        raise ValueError("Patches count is not divisible by patches per image.")

    batch_size = n_patches_total // n_patches_per_image

    patches = patches.view(batch_size, n_patches_per_image, channels, patch_height, patch_width)
    patches = patches.permute(0, 2, 3, 4, 1).reshape(
        batch_size, channels * patch_height * patch_width, n_patches_per_image
    )

    output = F.fold(
        patches,
        output_size=(img_height, img_width),
        kernel_size=(patch_height, patch_width),
        stride=(stride_height, stride_width),
    )

    ones = torch.ones_like(patches)
    weights = F.fold(
        ones,
        output_size=(img_height, img_width),
        kernel_size=(patch_height, patch_width),
        stride=(stride_height, stride_width),
    )
    output = output / weights
    return output

def get_normalizer(dataset):
    """Calculate normalization statistics for images of potentially different sizes."""
    import torch
    from torchvision.transforms.v2 import Normalize
    
    # Collect and flatten all images
    all_pixels = []
    for image in dataset.images:
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        
        # Standardize to CHW format
        if image.dim() == 2:
            image = image.unsqueeze(0)
        elif image.dim() == 3 and image.shape[0] > image.shape[2]:
            image = image.permute(2, 0, 1)
        
        all_pixels.append(image.reshape(image.shape[0], -1).float())
    
    # Concatenate all pixels and calculate statistics
    combined = torch.cat(all_pixels, dim=1)
    mu = combined.mean(dim=1).tolist()
    std = combined.std(dim=1, unbiased=True).tolist()
    
    return Normalize(mean=mu, std=std)

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS  
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def normalizeTensorToPixels(tensor: Tensor) -> Tensor:
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = tensor * 255
    return tensor
        
def calculate_class_imbalance(masks_dir: str) -> dict:
    class_counts = {}
    for filename in os.listdir(masks_dir):
        if filename.endswith(('.tif')):
            mask_path = os.path.join(masks_dir, filename)
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)
            unique, counts = np.unique(mask_np, return_counts=True)
            for u, c in zip(unique, counts):
                if int(u) not in class_counts:
                    class_counts[int(u)] = 0
                class_counts[int(u)] += int(c)
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} pixels")
    print("Total pixels:", sum(class_counts.values()))
    print("Class imbalance ratio:", {k: v / sum(class_counts.values()) for k, v in class_counts.items()})
    
    return class_counts


if __name__ == '__main__':
    folder_path = 'data/medres_masks/'
    #resize_and_save_images(folder_path, is_masks=True, output_size=(1024, 1024))
