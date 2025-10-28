import torch
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

class ImagePreprocessor:
    """Handles all image preprocessing operations for the segmentation model."""
    
    def __init__(self, model_input_size):
        self.model_input_size = model_input_size
        
    def prepare_image_patches(self, pil_image, device):
        """Prepare image patches for segmentation by converting to tensor and extracting patches."""
        import torchvision.transforms.functional as TF
        
        tensor = TF.to_tensor(pil_image).unsqueeze(0)
        tensor = tensor.to(device)
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

def get_dataloaders(dataset: Dataset, train_data_size: float, validation_data_size: float, input_size: tuple[int, int], with_data_augmentation: bool, log_file_path = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_augmenter = DataAugmenter()
    dataset = slice_dataset_in_four(dataset, input_size)
    train_data, val_data, test_data = random_split(dataset, [train_data_size, validation_data_size, 1-train_data_size-validation_data_size])
    
    # Log data split information
    log_data_split_info(dataset, train_data, val_data, test_data, log_file_path)
    
    if with_data_augmentation:
        train_data = data_augmenter.augment_dataset(train_data, input_size)
    else:
        train_data = data_augmenter.augment_dataset(train_data, input_size, [False, False, False, False, False, False, False])
    val_data = process_and_slice(val_data, input_size)
    test_data = process_and_slice(test_data, input_size)

    if torch.cuda.is_available():
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True, num_workers=24)
    else:
        train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=1)
    return (train_dataloader, val_dataloader, test_dataloader)


def get_dataloaders_without_testset(dataset: Dataset, train_data_size: float, input_size: tuple[int, int], with_data_augmentation: bool, log_file_path = None) -> tuple[DataLoader, DataLoader]:
    data_augmenter = DataAugmenter()
    dataset = slice_dataset_in_four(dataset, input_size)
    train_data, val_data = random_split(dataset, [train_data_size, 1-train_data_size])
    
    # Log data split information
    log_data_split_info(dataset, train_data, val_data, None, log_file_path)
    
    if with_data_augmentation:
        train_data = data_augmenter.augment_dataset(train_data, input_size)
    else:
        train_data = data_augmenter.augment_dataset(train_data, input_size, [False, False, False, False, False, False, False])
    val_data = process_and_slice(val_data, input_size)

    if torch.cuda.is_available():
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True, num_workers=24)
    else:
        train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True)
    return (train_dataloader, val_dataloader)


def get_dataloaders_kfold(dataset: Dataset, train_data_size: float, batch_size: int, input_size: tuple[int, int], random_cropping: bool) -> tuple[DataLoader, DataLoader]:
    data_augmenter = DataAugmenter()
    train_data, val_data = random_split(dataset, [train_data_size, 1-train_data_size])
    if random_cropping:
        train_data = data_augmenter.augment_dataset(train_data, input_size)
    else:
        train_data = data_augmenter.augment_dataset(train_data, input_size, [False, True, True, False, False, False, False])
    val_data = process_and_slice(val_data, input_size)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=24)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True)
    return (train_dataloader, val_dataloader)

def get_dataloaders_kfold_already_split(train_data, val_data, batch_size, input_size, augmentations=[True,True,False,False,True,True,False]):
    data_augmenter = DataAugmenter()
    print(augmentations)
    if not augmentations[0]: # No random cropping
        train_data = process_and_slice(train_data, input_size)
    train_data = data_augmenter.augment_dataset(train_data, input_size, augmentations)
  
    val_data = process_and_slice(val_data, input_size)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=24)
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

def binarize_segmentation_output(segmented_image, high_thresh=0.85, mean_prob_thresh=0.5):
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
    images_np = images.cpu().numpy()
    batch_size, channels, img_width, img_height = images_np.shape
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride_size

    remaining_width = (img_width - patch_width) % stride_width
    remaining_height = (img_height - patch_height) % stride_height

    needed_padding_width = (stride_width - remaining_width) % stride_width
    needed_padding_height = (stride_height - remaining_height) % stride_height

    if needed_padding_width:

        padded_images = np.empty(
            (batch_size, channels, img_width + needed_padding_width, img_height), 
            dtype=images_np.dtype
        )

        start_x = needed_padding_width // 2
        end_x = start_x + img_width

        for i, img in enumerate(images_np):
            padded_images[i, :, start_x:end_x, :] = img
            padded_images[i, :, :start_x, :] = np.flip(img[:, :start_x, :], axis=1)
            padded_images[i, :, end_x:, :] = np.flip(img[:, img_width - (needed_padding_width - needed_padding_width // 2):, :], axis=1)
        
        images_np = padded_images
    
    if needed_padding_height:
        img_width = images_np.shape[2]
        padded_images = np.empty(
            (batch_size, channels, img_width, img_height + needed_padding_height), 
            dtype=images_np.dtype
        )
        start_y = needed_padding_height // 2
        end_y = start_y + img_height

        for i, img in enumerate(images_np):
            padded_images[i, :, :, start_y:end_y] = img
            padded_images[i, :, :, :start_y] = np.flip(img[:, :, :start_y], axis=2)
            padded_images[i, :, :, end_y:] = np.flip(img[:, :, img_height - (needed_padding_height - needed_padding_height // 2):], axis=2)
        
        images_np = padded_images
    
    return torch.tensor(images_np, dtype=images.dtype, device=images.device)
    


# Made with help from https://www.programmersought.com/article/15316517340/
def extract_slices(images: Tensor, patch_size: tuple, stride_size: tuple) -> np.ndarray:
    images_np = images.cpu().numpy()
    batch_size, channels, img_width, img_height = images.shape
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride_size


    n_patches_y = (img_height - patch_height) // stride_height + 1
    n_patches_x = (img_width - patch_width) // stride_width + 1

    n_patches_per_image = n_patches_x * n_patches_y

    n_patches_total = n_patches_per_image * batch_size

    patches = np.empty((n_patches_total, channels, patch_width, patch_height), dtype=images_np.dtype)

    patch_idx = 0

    for img in images_np:
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                start_x = j * stride_width
                start_y = i * stride_height
                end_x = start_x + patch_width
                end_y = start_y + patch_height

                patches[patch_idx] = img[:, start_x:end_x, start_y:end_y]
                patch_idx += 1
    return patches

def construct_image_from_patches(patches: np.ndarray, img_size: tuple, stride_size: tuple):
    img_width, img_height = img_size
    stride_width, stride_height = stride_size
    n_patches_total, channels, patch_width, patch_height = patches.shape
    
    n_patches_y = (img_height - patch_height) // stride_height + 1
    n_patches_x = (img_width - patch_width) // stride_width + 1

    n_patches_per_image = n_patches_x * n_patches_y

    batch_size = n_patches_total // n_patches_per_image

    images = np.zeros((batch_size, channels, img_width, img_height))
    weights = np.zeros_like(images)

    for img_idx, (img, weights) in enumerate(zip(images, weights)):
        start = img_idx * n_patches_per_image

        for i in range(n_patches_y):
            for j in range(n_patches_x):
                start_x = j * stride_width
                start_y = i * stride_height
                end_x = start_x + patch_width
                end_y = start_y + patch_height
                patch_idx = start + i * n_patches_x + j
                img[:, start_x:end_x, start_y:end_y] += patches[patch_idx]
                weights[:, start_x:end_x, start_y:end_y] += 1
    images /= weights

    return images

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
    
def showTensor(tensor: Tensor) -> None:
    import torchvision.transforms.functional as TF
    if tensor.dim == 4:
        tensor = tensor.squeeze(1)
    if tensor.size(0) == 1:
        pixels = normalizeTensorToPixels(tensor[0, :, :])
    
        img = TF.to_pil_image(pixels.byte())
        img.show()
        
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

def bootstrap_compare(model_A: list, model_B: list, n_bootstraps=100_000, ci=(2.5, 97.5)):
    A = np.array(model_A)
    B = np.array(model_B)
    obs_diff = B.mean() - A.mean()

    # Center both samples under H0
    pooled_mean = np.concatenate((A, B)).mean()
    A0 = A - A.mean() + pooled_mean
    B0 = B - B.mean() + pooled_mean

    boot_diffs = np.empty(n_bootstraps)
    null_diffs = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = np.random.choice(len(A), size=len(A), replace=True)
        boot_diffs[i] = B[idx].mean() - A[idx].mean()
        null_diffs[i] = B0[idx].mean() - A0[idx].mean()

    p_value = np.mean(np.abs(null_diffs) >= np.abs(obs_diff))
    ci_lower, ci_upper = np.percentile(boot_diffs, ci)
    
    print(f"Mean performance of model A:        {A.mean():.4f}")
    print(f"Mean performance of model B:        {B.mean():.4f}")
    print(f"Mean difference:                    {obs_diff:.4f}")
    print(f"p_value:                            {p_value:.4f}")
    print(f"{ci[1]-ci[0]}% Confidence interval: [{ci_lower:.4f}; {ci_upper:.4f}]")

if __name__ == '__main__':
    folder_path = 'data/medres_masks/'
    #print(calculate_class_imbalance(folder_path))
    # from scipy.stats import shapiro, wilcoxon, ttest_rel, ttest_ind
    # import pylab
    # A = np.array([0.8189010620117188, 0.8571428656578064, 0.9178795218467712, 0.8751553893089294, 0.8745369911193848, 0.9351348280906677, 0.949617862701416, 0.9185584783554077, 0.8370370268821716, 0.6106142401695251, 0.7116121649742126, 0.5505999326705933, 0.824397623538971, 0.7843194603919983, 0.8063369393348694, 0.9159950613975525, 2.3148147842988465e-09, 0.9436193704605103, 0.8439226150512695, 0.8252636790275574, 0.5228062868118286, 0.647161602973938, 0.7481721043586731, 0.7512163519859314, 0.812365710735321, 0.7645303010940552, 0.8095420002937317, 0.9050678014755249, 0.9098699688911438, 0.9076755046844482, 0.4877781867980957, 0.8024998903274536, 0.7866934537887573, 0.6343388557434082, 0.8860703110694885, 0.9109057188034058, 0.9286608695983887, 0.8974626660346985, 0.6606569886207581, 0.6980690360069275, 0.7837166786193848, 0.6875, 0.8505190014839172, 0.9488529562950134, 0.928041398525238, 0.9150332808494568, 0.9279829263687134, 1.1363636254202447e-08, 0.9575209617614746, 0.6052687764167786, 0.6553806066513062, 0.6585831046104431])
    # B = np.array([0.8070696592330933, 0.8209963440895081, 0.9127664566040039, 0.8677415251731873, 0.8510182499885559, 0.9271577596664429, 0.919836699962616, 0.9304184913635254, 0.8230254054069519, 0.6788084506988525, 0.6632747054100037, 0.5800628066062927, 0.8833523392677307, 0.8297789692878723, 0.8240399360656738, 0.9260881543159485, 1.1481056105822063e-09, 0.9449887275695801, 0.8440329432487488, 0.801849901676178, 0.6655553579330444, 0.6905125975608826, 0.7500243186950684, 0.7326681017875671, 0.8172498345375061, 0.7755703330039978, 0.8015865683555603, 0.9526355862617493, 0.9298364520072937, 0.9053589105606079, 0.4888246953487396, 0.782210111618042, 0.7829706072807312, 0.6859729886054993, 0.905697226524353, 0.9105371236801147, 0.925720751285553, 0.9109379053115845, 0.6998302936553955, 0.7029004096984863, 0.7522885203361511, 0.6602857112884521, 0.9188163876533508, 0.9345554709434509, 0.9188467860221863, 0.9108409285545349, 0.9042534828186035, 7.6335879839462e-09, 0.8919678330421448, 0.664997935295105, 0.7005658149719238, 0.7098470330238342])
    # C = np.array([0.7156996726989746, 0.839595377445221, 0.9103406071662903, 0.8813445568084717, 0.8152257204055786, 0.9477984309196472, 0.9035064578056335, 0.8436450362205505, 0.8591357469558716, 0.5535863637924194, 0.6164604425430298, 0.6764838099479675, 0.856521725654602, 0.7836874723434448, 0.8168814778327942, 0.8962214589118958, 6.1728395728266605e-09, 0.9371362924575806, 0.8391126990318298, 0.8221358060836792, 0.5752031803131104, 0.6559829711914062, 0.6941782236099243, 0.6896812915802002, 0.7720627784729004, 0.7810831665992737, 0.7844630479812622, 0.395327091217041, 0.9033476710319519, 0.857358992099762, 0.4650120735168457, 0.71390300989151, 0.8016790151596069, 0.6760180592536926, 0.8919995427131653, 0.8774086833000183, 0.9108430743217468, 0.8615129590034485, 0.6782435774803162, 0.7333087921142578, 0.6742063760757446, 0.6467403769493103, 0.5136589407920837, 0.935197114944458, 0.9048972725868225, 0.8930467367172241, 0.9188039302825928, 7.518797140448896e-09, 0.9241149425506592, 0.7254098653793335, 0.7375807762145996, 0.6626594662666321])
    # D = np.array([0.8148669004440308, 0.8528174757957458, 0.921642005443573, 0.8759924173355103, 0.8771366477012634, 0.9136196374893188, 0.9228909015655518, 0.930431067943573, 0.8147156238555908, 0.6361990571022034, 0.7129687070846558, 0.5857410430908203, 0.8603967428207397, 0.7876034379005432, 0.8301846385002136, 0.9160687923431396, 4.0064102035941573e-10, 0.9469137787818909, 0.8480867147445679, 0.8656420707702637, 0.6600008606910706, 0.6921167969703674, 0.7422634959220886, 0.749715268611908, 0.7979646325111389, 0.8208861351013184, 0.815980851650238, 0.9149709343910217, 0.919501543045044, 0.8878215551376343, 0.3271441161632538, 0.6956005096435547, 0.7772523164749146, 0.6407861709594727, 0.9288687705993652, 0.9092661142349243, 0.9308812618255615, 0.8961268663406372, 0.7224286198616028, 0.6190782189369202, 0.6241026520729065, 0.6147547364234924, 0.9298921227455139, 0.9476283192634583, 0.9380788207054138, 0.9315861463546753, 0.9183943271636963, 6.097561122686557e-09, 0.9554773569107056, 0.7182522416114807, 0.7422594428062439, 0.6767650246620178])
    # A = np.array([0.7170665860176086, 0.7668022513389587, 0.7926896214485168, 0.7779718637466431, 0.786379873752594])
    # B = np.array([0.7599958181381226, 0.7802650332450867, 0.8206750750541687, 0.7711340188980103, 0.7873132824897766])
    # print(A.mean())
    # print(B.mean())
    # #bootstrap_compare(A, C)
    # #print(shapiro(B-A))
    # print(wilcoxon(A, B))
    # #print(wilcoxon(A, C))
    # #print(wilcoxon(A, D))
    # #print(wilcoxon(B, C))
    # #print(wilcoxon(B, D))
    # #print(wilcoxon(C, D))
    # #print(ttest_ind(A, B))
    # print(ttest_rel(A, B))
    #print(ttest_rel(A, C))
    #print(ttest_rel(A, D))
    #print(ttest_rel(B, C))
    #print(ttest_rel(B, D))
    #print(ttest_rel(C, D))
    
    
    #resize_and_save_images(folder_path, is_masks=True, output_size=(1024, 1024))
    # tensor = tensor_from_image('data/W. sample_0011.tif', (256, 256))
    # tensor = mirror_fill(tensor, (100,100), (100,100))
    # patches = extract_slices(tensor, (100,100), (100,100))
    # showTensor(tensor)
    # reconstructed = construct_image_from_patches(patches, (300,300), (100,100))
    # showTensor(reconstructed)
