import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchvision.transforms.functional as TF

def plot_paired_bar(iou_A, iou_B, dice_A, dice_B):
    A_label = "No rotation"
    B_label = "Rotation"
    diff_iou = iou_B - iou_A
    diff_dice = dice_B - dice_A

    indices = np.arange(1, len(diff_iou)+1)

    plt.figure(figsize=(10,4))

    plt.subplot(1, 2, 1)
    plt.bar(indices, diff_iou, color='skyblue', edgecolor='black')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f'Paired Differences: IOU ({B_label} - {A_label})')
    plt.xlabel('Fold')
    plt.ylabel('Difference')
    plt.xticks(indices)
    plt.ylim(-max(np.abs(diff_iou))*1.2, max(np.abs(diff_iou))*1.2)

    plt.subplot(1, 2, 2)
    plt.bar(indices, diff_dice, color='salmon', edgecolor='black')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f'Paired Differences: Dice ({B_label} - {A_label})')
    plt.xlabel('Fold')
    plt.ylabel('Difference')
    plt.xticks(indices)
    plt.ylim(-max(np.abs(diff_dice))*1.2, max(np.abs(diff_dice))*1.2)

    plt.tight_layout()
    plt.show()

def plot_paired_scatter(iou_A, iou_B, dice_A, dice_B):
    
    diff_iou = iou_B - iou_A
    diff_dice = dice_B - dice_A

    plt.figure(figsize=(10,4))

    plt.subplot(1, 2, 1)
    plt.scatter(range(1, len(diff_iou)+1), diff_iou, color='skyblue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Paired Differences: IOU (B - A)')
    plt.xlabel('Sample')
    plt.ylabel('Difference')
    plt.ylim(min(diff_iou)*1.2, max(diff_iou)*1.2)

    plt.subplot(1, 2, 2)
    plt.scatter(range(1, len(diff_dice)+1), diff_dice, color='salmon')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Paired Differences: Dice (B - A)')
    plt.xlabel('Sample')
    plt.ylabel('Difference')
    plt.ylim(min(diff_dice)*1.2, max(diff_dice)*1.2)

    plt.tight_layout()
    plt.show()

def plot_paired_histogram(iou_A, iou_B, dice_A, dice_B):
    # Calculate paired differences
    diff_iou = iou_B - iou_A
    diff_dice = dice_B - dice_A

    # Plot histograms
    plt.figure(figsize=(12, 5))
    max_diff = max(np.max(np.abs(diff_iou)), np.max(np.abs(diff_dice)))
    plt.subplot(1, 2, 1)
    plt.hist(diff_iou, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Paired Differences: IOU')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.xlim(-max_diff, max_diff)  # symmetric around zero


    plt.subplot(1, 2, 2)
    plt.hist(diff_dice, bins=20, color='salmon', edgecolor='black')
    plt.title('Histogram of Paired Differences: Dice')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.xlim(-max_diff, max_diff)  # symmetric around zero


    plt.tight_layout()
    plt.show()

def plot_iou_dice_boxplot(iou_A, iou_B, dice_A, dice_B):
    A_label = "No augmentation"
    B_label = "Horizontal flip"

    # Data for seaborn (it accepts dicts just like DataFrames)
    data = {
        'Score': np.concatenate([iou_A, iou_B, dice_A, dice_B]),
        'Metric': (['IoU'] * len(iou_A) + ['IoU'] * len(iou_B) +
                   ['Dice'] * len(dice_A) + ['Dice'] * len(dice_B)),
        'Model': ([A_label] * len(iou_A) + [B_label] * len(iou_B) +
                  [A_label] * len(dice_A) + [B_label] * len(dice_B)),
    }

    # Helper structure to replace df[(df[...] == ...)]
    grouped_scores = {
        ('IoU', A_label): iou_A,
        ('IoU', B_label): iou_B,
        ('Dice', A_label): dice_A,
        ('Dice', B_label): dice_B,
    }

    sns.set_theme(style='whitegrid')
    palette = {A_label: '#1f77b4', B_label: '#ff7f0e'}

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        x='Metric',
        y='Score',
        hue='Model',
        data=data,
        palette=palette,
        width=0.6,
        linewidth=1.5,
        whis=[0, 100],
        showfliers=False
    )

    means = {
        ('IoU', A_label): np.mean(iou_A),
        ('IoU', B_label): np.mean(iou_B),
        ('Dice', A_label): np.mean(dice_A),
        ('Dice', B_label): np.mean(dice_B),
    }

    x_offsets = {
        ('IoU', A_label): -0.15,
        ('IoU', B_label): 0.15,
        ('Dice', A_label): 0.85,
        ('Dice', B_label): 1.15,
    }

    for key, mean in means.items():
        metric, model = key
        x = x_offsets[key]

        # Replace df[...] filtering:
        y_max = np.max(grouped_scores[key])

        ax.text(
            x,
            y_max + 0.01,
            f"Mean = {mean:.3f}",
            ha='center',
            fontsize=10,
            color='dimgray',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='grey',
                boxstyle='round,pad=0.3'
            )
        )

    ax.set_title(f'Comparison of IoU and Dice Scores\n {A_label} vs. {B_label}',
                 fontsize=14, weight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(title='Model', title_fontsize=12, fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_iou_dice_boxplot3(iou_A, iou_B, iou_C, dice_A, dice_B, dice_C):
    A_label = "No augmentation"
    B_label = "Horizontal flip"
    C_label = "Horizontal flip + rotation"

    # Data for seaborn
    data = {
        'Score': np.concatenate([iou_A, iou_B, iou_C, dice_A, dice_B, dice_C]),
        'Metric': (['IoU'] * len(iou_A) + ['IoU'] * len(iou_B) + ['IoU'] * len(iou_C) +
                   ['Dice'] * len(dice_A) + ['Dice'] * len(dice_B) + ['Dice'] * len(dice_C)),
        'Model': ([A_label] * len(iou_A) + [B_label] * len(iou_B) + [C_label] * len(iou_C) +
                  [A_label] * len(dice_A) + [B_label] * len(dice_B) + [C_label] * len(dice_C)),
    }

    # Replace DataFrame filtering with direct dict lookups
    grouped_scores = {
        ('IoU', A_label): iou_A,
        ('IoU', B_label): iou_B,
        ('IoU', C_label): iou_C,
        ('Dice', A_label): dice_A,
        ('Dice', B_label): dice_B,
        ('Dice', C_label): dice_C,
    }

    sns.set_theme(style='whitegrid')
    palette = {A_label: '#1f77b4', B_label: '#ff7f0e', C_label: '#2ca02c'}

    plt.figure(figsize=(10, 7))
    ax = sns.boxplot(
        x='Metric',
        y='Score',
        hue='Model',
        data=data,
        palette=palette,
        width=0.6,
        linewidth=1.5,
        whis=[0, 100],
        showfliers=False
    )

    means = {
        ('IoU', A_label): np.mean(iou_A),
        ('IoU', B_label): np.mean(iou_B),
        ('IoU', C_label): np.mean(iou_C),
        ('Dice', A_label): np.mean(dice_A),
        ('Dice', B_label): np.mean(dice_B),
        ('Dice', C_label): np.mean(dice_C),
    }

    x_offsets = {
        ('IoU', A_label): -0.2,
        ('IoU', B_label): 0.0,
        ('IoU', C_label): 0.2,
        ('Dice', A_label): 0.8,
        ('Dice', B_label): 1.0,
        ('Dice', C_label): 1.2,
    }

    for key, mean in means.items():
        metric, model = key
        x = x_offsets[key]

        # Replace df[...] with list lookups:
        y_max = np.max(grouped_scores[key])

        ax.text(
            x,
            y_max + 0.005,
            f"Mean = {mean:.3f}",
            ha='center',
            fontsize=10,
            color='dimgray',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='grey',
                boxstyle='round,pad=0.3'
            )
        )

    ax.set_title(f'Comparison of IoU and Dice Scores\n {A_label} vs. {B_label} vs. {C_label}',
                 fontsize=14, weight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(title='Model', title_fontsize=12, fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.show()
    

def show_annotation(image, mask):
    """
    Show the image alongside the mask.
    """
    # Convert tensors to numpy arrays
    image_np = to_2d_image_array(image.numpy())
    mask_np = to_2d_image_array(mask.numpy())
    # Normalize the image for display
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Display the original image
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title("Original Image")
    
    # Display the ground truth mask
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Ground Truth Mask")

    plt.tight_layout()
    plt.show()


def show_labkit_annotation(image, labkit, mask):
    """
    Show the image alongside the mask.
    """
    # Convert tensors to numpy arrays
    image_np = to_2d_image_array(image.numpy())
    labkit_np = to_2d_image_array(labkit.numpy())
    mask_np = to_2d_image_array(mask.numpy())
    # Normalize the image for display
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    # Display the original image
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title("Original Image")
    
    # Display the original image
    axes[1].imshow(labkit_np, cmap='gray')
    axes[1].set_title("Labkit Ground Truth")

    # Display the ground truth mask
    axes[2].imshow(mask_np, cmap='gray')
    axes[2].set_title("Ground Truth Mask")

    plt.tight_layout()
    plt.show()


def bootstrap_compare(model_A: list, model_B: list, n_bootstraps=100_00, ci=(2.5, 97.5)):
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
def to_2d_image_array(array: np.ndarray) -> np.ndarray:
    return (np.squeeze(array) * 255).astype(np.uint8)

def show_example_images(images):
    """
    Show a few example images from the dataset.
    """
    fig, axes = plt.subplots(2, len(images)//2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, image in zip(axes, images):
        # Convert tensor to numpy array
        image_np = to_2d_image_array(image)
        # Normalize the image for display
        image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
        ax.imshow(image_np, cmap='gray')
        ax.axis('on')
    fig.suptitle("Example TEM Images", fontsize=16)
    plt.tight_layout()  # Leave space for the title
    plt.show()

def remove_border_artifacts(image, mask, crop_size=448):
    # Center crop after affine to remove black borders
    i, j, h, w = TF.center_crop.get_params(image, output_size=(crop_size, crop_size))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Resize back to original size if needed
    image = TF.resize(image, [512, 512], interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, [512, 512], interpolation=TF.InterpolationMode.NEAREST)

    return image, mask

def random_affine(image, mask):
    import random
    # Parameters for affine transformation
    angle = random.uniform(-10, 10)            # Small rotation
    translate = (random.uniform(-0.05, 0.05),  # Max 5% shift
                 random.uniform(-0.05, 0.05))
    scale = random.uniform(0.95, 1.05)         # Mild scaling
    shear = random.uniform(-5, 5)              # Slight shearing
    import torchvision.transforms.functional as TF

    # Apply the same transform to both image and mask
    image = TF.affine(image, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.affine(mask, angle, translate, scale, shear, interpolation=TF.InterpolationMode.NEAREST)

    return image, mask
def tensor_from_image_no_resize(image_path: str):
    from PIL import Image
    import torchvision.transforms.functional as TF

    image = Image.open(image_path).convert("L")
    image = TF.to_tensor(image).unsqueeze(0)
    return image

def compute_collected_histogram(image_paths):
    total_histogram = np.zeros(256, dtype=np.uint64)
    from PIL import Image
    for path in image_paths:
        img = Image.open(path).convert("L")  # Ensure grayscale
        hist = np.array(img.histogram(), dtype=np.uint64)
        total_histogram += hist
    
    return total_histogram

def get_image_paths(directory, extensions={'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

def get_sorted_image_mask_paths(image_dir, mask_dir):
    import os
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])

    assert len(image_files) == len(mask_files), "Number of images and masks must match."

    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

    return image_paths, mask_paths

def apply_clahe(pil_image):
    import cv2
    import numpy as np
    from PIL import Image
    img = np.array(pil_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img)
    return Image.fromarray(equalized)

def contrast_stretch(image, low=0.01, high=0.99):
    """
    Apply contrast stretching to the image.
    """
    import numpy as np
    from PIL import Image

    # Convert to numpy array
    img_array = np.array(image)

    # Compute low and high percentiles
    p_low = np.percentile(img_array, low * 100)
    p_high = np.percentile(img_array, high * 100)

    # Stretch the contrast
    stretched = (img_array - p_low) / (p_high - p_low) * 255.0
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    return Image.fromarray(stretched)

def compute_foreground_background_histograms(image_paths, mask_paths):
    import numpy as np
    from PIL import Image
    fg_hist = np.zeros(256, dtype=np.uint64)
    bg_hist = np.zeros(256, dtype=np.uint64)

    for img_path, mask_path in zip(image_paths, mask_paths):
        img = np.array(Image.open(img_path).convert("L"))
        img = np.array(apply_clahe(img))
        mask = np.array(Image.open(mask_path).convert("1"))  # Binary mask (0 or 255)

        fg_pixels = img[mask == 1]
        bg_pixels = img[mask == 0]

        fg_hist += np.bincount(fg_pixels.astype(np.uint8), minlength=256).astype(np.uint64)
        bg_hist += np.bincount(bg_pixels.astype(np.uint8), minlength=256).astype(np.uint64)

    return fg_hist, bg_hist

def compute_foreground_background_histograms2(images, masks):
    import numpy as np
    from PIL import Image
    fg_hist = np.zeros(256, dtype=np.uint64)
    bg_hist = np.zeros(256, dtype=np.uint64)

    for img_path, mask_path in zip(image_paths, mask_paths):
        img = np.array(Image.open(img_path).convert("L"))
        img = np.array(apply_clahe(img))
        mask = np.array(Image.open(mask_path).convert("1"))  # Binary mask (0 or 255)

        fg_pixels = img[mask == 1]
        bg_pixels = img[mask == 0]

        fg_hist += np.bincount(fg_pixels.astype(np.uint8), minlength=256).astype(np.uint64)
        bg_hist += np.bincount(bg_pixels.astype(np.uint8), minlength=256).astype(np.uint64)

    return fg_hist, bg_hist

def array_to_tensor(array: np.ndarray):
    """
    Convert a 2D numpy array to a PyTorch tensor.
    """
    import torch
    return torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

def plot_memory_usage(memory_usages, labels, title='Memory Usage by Batch Size and Precision'):
    """
    Plots a bar graph of memory usage.

    Parameters:
    - memory_usages (list of int): Memory usage values in MB.
    - labels (list of str): Labels corresponding to each memory usage value.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, memory_usages, color=['skyblue', 'orange', 'green'])

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 300, f'{yval} MB', ha='center', va='bottom')

    plt.title(title)
    plt.ylabel('Memory Usage (MB)')
    plt.tight_layout()
    plt.show()

def plot_cpu_gpu_times(cpu_seg_time, cpu_post_process_time, gpu_seg_time, gpu_post_process_time):
    n = len(cpu_seg_time)
    ind = np.arange(n)  # x locations for groups
    width = 0.2  # narrower bars to fit all 4 side by side

    fig, ax = plt.subplots(figsize=(14, 7))

    # Bar positions
    cpu_seg_pos = ind - 1.5 * width - 0.02
    cpu_post_pos = ind - 0.5 * width - 0.02
    gpu_seg_pos = ind + 0.5 * width + 0.02
    gpu_post_pos = ind + 1.5 * width + 0.02

    # Plot bars
    ax.bar(cpu_seg_pos, cpu_seg_time, width, label='Grid Crop IOU', color='steelblue')
    ax.bar(gpu_seg_pos, cpu_post_process_time, width, label='Random Crop IOU', color='lightblue')
    ax.bar(cpu_post_pos, gpu_seg_time, width, label='Grid Crop Dice', color='darkorange')
    ax.bar(gpu_post_pos, gpu_post_process_time, width, label='Random Crop Dice', color='peachpuff')

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Time (seconds)')
    ax.set_ylim(0.6, 0.9)
    #ax.set_yscale('log')  # Log scale for better visibility of differences
    ax.set_title('CPU vs GPU Segmentation and Post-Processing Times')
    ax.set_xticks(ind)
    ax.set_xticklabels([str(i+1) for i in range(n)])
    ax.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    #show_example_images(images[0:6])

    image_paths = get_image_paths("data/highres_images/")
    histogram = compute_collected_histogram(image_paths)
    # Print or plot the histogram
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.plot(histogram)
    plt.title("Collected Pixel Value Histogram")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency")
    plt.grid(True)

    image_paths, mask_paths = get_sorted_image_mask_paths("data/highres_images", "data/highres_masks")
    
    fg_hist, bg_hist = compute_foreground_background_histograms(image_paths, mask_paths)
    # # Plotting the histograms
    plt.subplot(1, 2, 2)
    plt.plot(fg_hist, label='Foreground', color='red')
    plt.plot(bg_hist, label='Background', color='blue')
    plt.title("Foreground vs Background Pixel Histograms")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()



    # image = tensor_from_image_no_resize("data/highres_images/E. sample_0008_4k.tif")
    # mask = tensor_from_image_no_resize("data/highres_masks/E. sample_0008_mask_4k.tif")
    # labkit = tensor_from_image_no_resize("data/labkit_segmentation.tif")
    # show_labkit_annotation(image, labkit, mask)
    # image2, mask2 = random_affine(image, mask)
    # show_example_images([image, mask, image2, mask2])
    # show_annotation(image, mask)
    from scipy import stats
    grid_crop = np.array([0.6528146886825562, 0.6797138452529907, 0.7251459360122681, 0.6453444671630859, 0.7232711219787598])
    grid_crop_dice = np.array([0.7642716979980469, 0.7461620569229126, 0.7954027056694031, 0.7549303579330444, 0.8166839981079102])
    random_crop = np.array([0.668975293636322, 0.8001164793968201, 0.7264196920394897, 0.7317933440208435, 0.7347612380981445])
    random_crop_dice = np.array([0.7676895260810852, 0.8609291315078735, 0.7848815321922302, 0.8258540034294128, 0.8281547427177429])
    no_rotation = np.array([0.762157678604126, 0.704014003276825, 0.719645619392395, 0.7562171220779419, 0.7786809206008911])
    no_rotation_dice = np.array([0.8458858132362366, 0.7899280786514282, 0.7921444773674011, 0.8396288752555847, 0.8582870364189148])
    rotation = np.array([0.796130895614624, 0.7645872235298157, 0.7844747304916382, 0.7975218892097473, 0.8107856512069702])
    rotation_dice = np.array([0.8698691725730896, 0.8418214321136475, 0.8482317924499512, 0.8769534826278687, 0.8891916275024414])

    no_bright = np.array([0.6516720056533813, 0.7034207582473755, 0.7970498204231262, 0.7814952731132507, 0.7757821083068848])
    bright = np.array([0.6945510506629944, 0.7286107540130615, 0.7784311771392822, 0.7878152132034302, 0.7485595345497131])

    no_flip = np.array([0.7106041312217712, 0.7450249195098877, 0.7225610613822937, 0.7385781407356262, 0.7855268716812134])
    no_flip_dice = np.array([0.7884710431098938, 0.8103492856025696, 0.8125852346420288, 0.8322637677192688, 0.8616784811019897])
    flip = np.array([0.7667941451072693, 0.6404618620872498, 0.7248525619506836, 0.7824148535728455, 0.8072212338447571])
    flip_dice = np.array([0.8451998233795166, 0.7093654274940491, 0.8108140826225281, 0.8679991960525513, 0.8851615786552429])
    print(no_rotation.mean(), rotation.mean())
    print(no_rotation_dice.mean(), rotation_dice.mean())
    print(no_flip.mean(), flip.mean())
    print(no_flip_dice.mean(), flip_dice.mean())
    print(stats.shapiro(grid_crop-random_crop))
    print(stats.shapiro(random_crop_dice-grid_crop_dice))
    print(stats.shapiro(no_rotation - rotation))
    print(stats.shapiro(no_rotation_dice - rotation_dice))
    print(stats.shapiro(no_flip - flip))
    print(stats.shapiro(no_flip_dice - flip_dice))

    print(stats.ttest_rel(grid_crop, random_crop))
    print(stats.ttest_rel(grid_crop_dice, random_crop_dice))
    print(stats.ttest_rel(no_rotation, rotation))
    print(stats.ttest_rel(no_rotation_dice, rotation_dice))
    print(stats.ttest_rel(no_bright, bright))

    print()

    lr_03 = np.array([0.8283874988555908, 0.7682378888130188, 0.7941887378692627, 0.7041546106338501, 0.7252908945083618])
    lr_04 = np.array([0.8291698694229126, 0.70115727186203, 0.7721849679946899, 0.727835476398468, 0.7304939031600952])
    lr_05 = np.array([0.8114704489707947, 0.742527186870575, 0.7933081388473511, 0.7155537605285645, 0.7669943571090698])
    print(stats.shapiro(lr_03 - lr_04))
    print(stats.shapiro(lr_04 - lr_05))
    print(stats.shapiro(lr_03 - lr_05))
    print(stats.ttest_rel(lr_03, lr_04))
    print(stats.ttest_rel(lr_04, lr_05))
    print(stats.ttest_rel(lr_03, lr_05))

    # print(stats.ttest_rel(no_flip, flip))
    # print(stats.ttest_rel(no_flip_dice, flip_dice))
    # iou_A = np.array([0.7106041312217712, 0.7450249195098877, 0.7225610613822937, 0.7385781407356262, 0.7855268716812134])
    # iou_B = np.array([0.7667941451072693, 0.6404618620872498, 0.7248525619506836, 0.7824148535728455, 0.8072212338447571] )
    # iou_C = np.array([0.7660102844238281, 0.786454975605011, 0.749537467956543, 0.759172797203064, 0.7851026654243469])
    # dice_A = np.array([0.7884710431098938, 0.8103492856025696, 0.8125852346420288, 0.8322637677192688, 0.8616784811019897])
    # dice_B = np.array([0.8451998233795166, 0.7093654274940491, 0.8108140826225281, 0.8679991960525513, 0.8851615786552429])
    # dice_C = np.array([0.8413566946983337, 0.8447999954223633, 0.8332952260971069, 0.8431856036186218, 0.8538570404052734])
    # #plot_iou_dice_boxplot(iou_A, iou_B, dice_A, dice_B) 
    # plot_iou_dice_boxplot(iou_A, iou_B, dice_A, dice_B)
    plot_paired_bar(no_rotation, rotation, no_rotation_dice, rotation_dice)
    plot_cpu_gpu_times(grid_crop, grid_crop_dice, random_crop, random_crop_dice)
    # plot_paired_bar(iou_B, iou_C, dice_B, dice_C)
    # diff = iou_B - iou_A
    # bootstrap_compare(iou_A, iou_B)

    
    # images = []
    # masks = []
    # from PIL import Image
    # image_dir = "data/highres_images/"
    # masks_dir = "data/highres_masks/"
    # image_filenames = sorted(os.listdir(image_dir))
    # mask_filenames = sorted(os.listdir(masks_dir))
    # indices = [4, 0, 11]#, 9, 2, 10]#[0, 2, 4, 9, 10, 11]
    # selected_filenames = selected_items = [image_filenames[i] for i in indices]
    # import torchvision.transforms.functional as TF
    # for image_name, mask_name in zip(image_filenames, mask_filenames):
    #     img_path = os.path.join(image_dir, image_name)
    #     mask_path = os.path.join(masks_dir, mask_name)


    #     image = Image.open(img_path).convert("L")
    #     image2 = contrast_stretch(image)
    #     mask = Image.open(mask_path).convert("L")
    
    #     #image = TF.to_tensor(image)
    #     #image2 = TF.to_tensor(image2)
    #     #images.append(image)
    #     images.append(image2)
    #     masks.append(mask)
    # #show_example_images(images)
    # fg_hist, bg_hist = compute_foreground_background_histograms2(images, masks)
    # plt.plot(fg_hist, label='Foreground', color='red')
    # plt.plot(bg_hist, label='Background', color='blue')
    # plt.title("Foreground vs Background Pixel Histograms")
    # plt.xlabel("Pixel Value (0-255)")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # startup_times_before = [12.9041, 12.2768, 12.1893, 12.1012, 12.4508, 11.6724, 11.5952, 11.6920, 11.6894, 12.0990]
    # startup_times_after = [0.6642, 0.4200, 0.4553, 0.4453, 0.4193, 0.4202, 0.4153, 0.4380, 0.4465, 0.4230]
    # print(np.mean(startup_times_before), np.mean(startup_times_after))

    cpu_seg_time = np.array([10.1272, 10.4156, 9.9943, 9.9314, 9.7927, 9.9107, 9.6871, 9.9212, 9.9013, 9.7655])
    cpu_post_process_time = np.array([0.1420, 0.1515, 0.2147, 0.1513, 0.1493, 0.1472, 0.1504, 0.1561, 0.1345, 0.1676])

    gpu_seg_time = np.array([0.1400, 0.1271, 0.1183, 0.1162, 0.1238, 0.1303, 0.1261, 0.1280, 0.1269, 0.1258])
    gpu_post_process_time = np.array([0.0984, 0.1040, 0.1679, 0.0938, 0.1020, 0.0990, 0.1028, 0.1065, 0.0876, 0.1200])

    print(np.mean(cpu_seg_time), np.mean(cpu_post_process_time), np.mean(cpu_post_process_time + cpu_seg_time))
    print(np.mean(gpu_seg_time), np.mean(gpu_post_process_time), np.mean(gpu_post_process_time + gpu_seg_time))


    #plot_cpu_gpu_times(cpu_seg_time, cpu_post_process_time, gpu_seg_time, gpu_post_process_time)

    # memory_usages = [5119, 18359, 9083]
    # labels = ['Batch size 8', 'Batch size 32', 'Batch size 32 (mixed precision)']
    # plot_memory_usage(memory_usages, labels, title='Memory Usage by Batch Size and Precision')

    cpu_epoch_time = [452.4048, 466.7603, 459.6591, 471.3772, 479.2589, 500.8772, 460.6193, 454.4951, 453.2173, 454.0637]
    gpu_epoch_time = [4.2034, 3.9112, 2.6794, 3.8027, 3.8488, 4.7915, 5.5026, 4.4560, 4.1945, 5.4300]
    print(np.mean(cpu_epoch_time), np.mean(gpu_epoch_time))

    