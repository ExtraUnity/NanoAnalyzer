"""
Compare U-Net nanoparticle segmentation against classical thresholding baselines.

Example:
    python experiments/compare_thresholding_baselines.py ^
        --images-dir data/medres_images ^
        --masks-dir data/medres_masks ^
        --model-path data/models/UNet_best.pt
"""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.ParticleMetrics import compute_size_stratified_metrics


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    threshold: str
    smoothing: str | None = None
    use_morphology: bool = False
    adaptive_block_size: int = 51
    adaptive_offset: float = 5.0
    gaussian_sigma: float = 1.0
    median_kernel: int = 3
    min_object_area: int = 12
    hole_size: int = 32
    morphology_radius: int = 1
    watershed: bool = False


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a 2D uint8 mask with values 0 and 1."""
    mask = np.asarray(mask)
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}.")
    return (mask > 0).astype(np.uint8)


def load_dataset_arrays(images_dir: str, masks_dir: str) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Load image/mask pairs with the same binarization and max-size policy as the dataset class."""
    from PIL import Image

    image_filenames = sorted(os.listdir(images_dir))
    mask_filenames = sorted(os.listdir(masks_dir))
    if len(image_filenames) != len(mask_filenames):
        raise ValueError("The number of images and masks must be the same.")

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for image_file_name, mask_file_name in zip(image_filenames, mask_filenames):
        image = _load_grayscale_image(os.path.join(images_dir, image_file_name))
        mask = Image.open(os.path.join(masks_dir, mask_file_name)).convert("L")

        if image.size != mask.size:
            raise ValueError(
                f"Image and mask dimensions do not match: {image_file_name} {image.size} vs {mask_file_name} {mask.size}"
            )

        if image.width > 1024 or image.height > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            mask.thumbnail((1024, 1024), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.uint8)
        mask_array = np.array(mask, dtype=np.uint8, copy=True)
        mask_array[mask_array <= 10] = 0
        mask_array[mask_array >= 245] = 255

        images.append(image_array)
        masks.append(ensure_binary_mask(mask_array))

    return images, masks, image_filenames


def _load_grayscale_image(path: str):
    from PIL import Image

    image = Image.open(path)
    if image.format == "TIFF" and image.mode in ("I;16", "I;16B", "I;16L"):
        array = np.asarray(image)
        value_range = array.max() - array.min()
        if value_range == 0:
            normalized = np.zeros_like(array, dtype=np.uint8)
        else:
            normalized = (array.astype(np.float32) - array.min()) * 255.0 / value_range
        image = Image.fromarray(normalized.astype(np.uint8))
    return image.convert("L")


def apply_smoothing(image: np.ndarray, config: BaselineConfig) -> np.ndarray:
    if config.smoothing == "gaussian":
        sigma = max(float(config.gaussian_sigma), 0.0)
        ksize = max(3, int(round(sigma * 6)) | 1)
        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)
    if config.smoothing == "median":
        kernel = max(3, int(config.median_kernel) | 1)
        return cv2.medianBlur(image, kernel)
    return image


def run_otsu_baseline(image: np.ndarray, foreground: str, config: BaselineConfig) -> np.ndarray:
    smoothed = apply_smoothing(image, config)
    threshold, _ = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if foreground == "bright":
        mask = smoothed > threshold
    else:
        mask = smoothed < threshold
    return mask.astype(np.uint8)


def run_adaptive_threshold_baseline(image: np.ndarray, foreground: str, config: BaselineConfig) -> np.ndarray:
    smoothed = apply_smoothing(image, config)
    block_size = max(3, int(config.adaptive_block_size) | 1)
    threshold_type = cv2.THRESH_BINARY if foreground == "bright" else cv2.THRESH_BINARY_INV
    mask = cv2.adaptiveThreshold(
        smoothed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        block_size,
        float(config.adaptive_offset),
    )
    return ensure_binary_mask(mask)


def apply_morphological_postprocessing(mask: np.ndarray, config: BaselineConfig) -> np.ndarray:
    """Apply a simple connected-component and morphology cleanup workflow."""
    mask_bool = ensure_binary_mask(mask).astype(bool)

    try:
        from scipy import ndimage as ndi
        from skimage import morphology, segmentation

        mask_bool = morphology.remove_small_objects(mask_bool, min_size=max(1, int(config.min_object_area)))
        mask_bool = morphology.remove_small_holes(mask_bool, area_threshold=max(1, int(config.hole_size)))

        radius = max(0, int(config.morphology_radius))
        if radius > 0:
            footprint = morphology.disk(radius)
            mask_bool = morphology.binary_opening(mask_bool, footprint)
            mask_bool = morphology.binary_closing(mask_bool, footprint)

        if config.watershed and np.any(mask_bool):
            distance = ndi.distance_transform_edt(mask_bool)
            local_max = morphology.local_maxima(distance)
            markers, _ = ndi.label(local_max)
            labels = segmentation.watershed(-distance, markers, mask=mask_bool)
            mask_bool = labels > 0

        return mask_bool.astype(np.uint8)
    except Exception:
        cleaned = mask_bool.astype(np.uint8)
        radius = max(0, int(config.morphology_radius))
        if radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        output = np.zeros_like(cleaned)
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] >= config.min_object_area:
                output[labels == label_id] = 1
        return output


def run_baseline(image: np.ndarray, foreground: str, config: BaselineConfig) -> np.ndarray:
    if config.threshold == "otsu":
        mask = run_otsu_baseline(image, foreground, config)
    elif config.threshold == "adaptive":
        mask = run_adaptive_threshold_baseline(image, foreground, config)
    else:
        raise ValueError(f"Unknown threshold method: {config.threshold}")

    if config.use_morphology:
        mask = apply_morphological_postprocessing(mask, config)
    return ensure_binary_mask(mask)


def compute_pixel_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    prediction = ensure_binary_mask(prediction)
    ground_truth = ensure_binary_mask(ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Prediction shape {prediction.shape} != ground truth shape {ground_truth.shape}")

    tp = int(np.logical_and(prediction == 1, ground_truth == 1).sum())
    fp = int(np.logical_and(prediction == 1, ground_truth == 0).sum())
    fn = int(np.logical_and(prediction == 0, ground_truth == 1).sum())
    union = tp + fp + fn
    pred_sum = int(prediction.sum())
    gt_sum = int(ground_truth.sum())

    iou = 1.0 if union == 0 else tp / union
    dice = 1.0 if pred_sum + gt_sum == 0 else 2 * tp / (pred_sum + gt_sum)
    precision = 1.0 if pred_sum == 0 and gt_sum == 0 else (tp / pred_sum if pred_sum > 0 else 0.0)
    recall = 1.0 if gt_sum == 0 and pred_sum == 0 else (tp / gt_sum if gt_sum > 0 else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def get_row(rows: list[dict], row_type: str, size_bin: str) -> dict:
    for row in rows:
        if row["row_type"] == row_type and row["size_bin"] == size_bin:
            return row
    return {}


def value_or_nan(value) -> float:
    if value == "" or value is None:
        return float("nan")
    return float(value)


def evaluate_method_on_dataset(
    method_name: str,
    predictions: list[np.ndarray],
    ground_truths: list[np.ndarray],
    file_names: list[str],
    object_iou_threshold: float,
) -> tuple[dict, list[dict], list[dict]]:
    """Evaluate one segmentation method at pixel, object, and size-stratified levels."""
    per_image_rows: list[dict] = []
    pixel_metric_names = ["iou", "dice", "precision", "recall", "f1"]

    for file_name, prediction, ground_truth in zip(file_names, predictions, ground_truths):
        pixel_metrics = compute_pixel_metrics(prediction, ground_truth)
        single_size_metrics = compute_size_stratified_metrics(
            ground_truths=[ground_truth],
            predictions=[prediction],
            file_names=[file_name],
            iou_threshold=object_iou_threshold,
        )
        gt_overall = get_row(single_size_metrics.rows, "ground_truth_overall", "overall")
        pred_overall = get_row(single_size_metrics.rows, "predicted_overall", "overall")

        per_image_rows.append(
            {
                "method": method_name,
                "file_name": file_name,
                **pixel_metrics,
                "object_precision": value_or_nan(pred_overall.get("precision")),
                "object_recall": value_or_nan(gt_overall.get("recall")),
                "object_f1": value_or_nan(gt_overall.get("f1_score")),
                "particle_count_error": value_or_nan(gt_overall.get("particle_count_error")),
                "relative_particle_count_error": value_or_nan(gt_overall.get("relative_particle_count_error")),
                "mean_relative_ecd_error": value_or_nan(gt_overall.get("mean_relative_ecd_error")),
                "mean_absolute_relative_ecd_error": value_or_nan(gt_overall.get("mean_absolute_relative_ecd_error")),
            }
        )

    size_metrics = compute_size_stratified_metrics(
        ground_truths=ground_truths,
        predictions=predictions,
        file_names=file_names,
        iou_threshold=object_iou_threshold,
    )
    gt_overall = get_row(size_metrics.rows, "ground_truth_overall", "overall")
    pred_overall = get_row(size_metrics.rows, "predicted_overall", "overall")

    aggregate = {
        "method": method_name,
        "num_images": len(ground_truths),
        "object_iou_threshold": object_iou_threshold,
    }
    for metric_name in pixel_metric_names:
        aggregate[metric_name] = float(np.nanmean([row[metric_name] for row in per_image_rows]))

    aggregate.update(
        {
            "object_precision": value_or_nan(pred_overall.get("precision")),
            "object_recall": value_or_nan(gt_overall.get("recall")),
            "object_f1": value_or_nan(gt_overall.get("f1_score")),
            "particle_count_error": value_or_nan(gt_overall.get("particle_count_error")),
            "relative_particle_count_error": value_or_nan(gt_overall.get("relative_particle_count_error")),
            "mean_ecd_error_px": value_or_nan(gt_overall.get("mean_ecd_error_px")),
            "median_ecd_error_px": value_or_nan(gt_overall.get("median_ecd_error_px")),
            "mean_relative_ecd_error": value_or_nan(gt_overall.get("mean_relative_ecd_error")),
            "median_relative_ecd_error": value_or_nan(gt_overall.get("median_relative_ecd_error")),
            "mean_absolute_relative_ecd_error": value_or_nan(gt_overall.get("mean_absolute_relative_ecd_error")),
            "median_absolute_relative_ecd_error": value_or_nan(gt_overall.get("median_absolute_relative_ecd_error")),
        }
    )

    size_rows = []
    for row in size_metrics.rows:
        size_rows.append({"method": method_name, **row})

    return aggregate, per_image_rows, size_rows


def default_baseline_configs() -> list[BaselineConfig]:
    return [
        BaselineConfig(name="otsu", threshold="otsu"),
        BaselineConfig(name="otsu_morph", threshold="otsu", use_morphology=True),
        BaselineConfig(name="adaptive", threshold="adaptive"),
        BaselineConfig(name="adaptive_morph", threshold="adaptive", use_morphology=True),
        BaselineConfig(name="gaussian_otsu", threshold="otsu", smoothing="gaussian", gaussian_sigma=1.0),
        BaselineConfig(name="median_otsu", threshold="otsu", smoothing="median", median_kernel=3),
    ]


def config_to_dict(config: BaselineConfig) -> dict:
    return {
        "name": config.name,
        "threshold": config.threshold,
        "smoothing": config.smoothing,
        "use_morphology": config.use_morphology,
        "adaptive_block_size": config.adaptive_block_size,
        "adaptive_offset": config.adaptive_offset,
        "gaussian_sigma": config.gaussian_sigma,
        "median_kernel": config.median_kernel,
        "min_object_area": config.min_object_area,
        "hole_size": config.hole_size,
        "morphology_radius": config.morphology_radius,
        "watershed": config.watershed,
    }


def tune_baseline_parameters(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    foreground: str,
    base_configs: list[BaselineConfig],
) -> list[BaselineConfig]:
    """Tune a small parameter grid on validation data using mean pixel IoU."""
    tuned_configs: list[BaselineConfig] = []

    for base_config in base_configs:
        candidates = [base_config]
        if base_config.threshold == "adaptive":
            candidates = [
                BaselineConfig(
                    **{
                        **config_to_dict(base_config),
                        "adaptive_block_size": block_size,
                        "adaptive_offset": offset,
                    }
                )
                for block_size, offset in itertools.product([31, 51, 71], [0.0, 5.0, 10.0])
            ]
        elif base_config.threshold == "otsu" and base_config.use_morphology:
            candidates = [
                BaselineConfig(
                    **{
                        **config_to_dict(base_config),
                        "min_object_area": min_area,
                        "morphology_radius": radius,
                    }
                )
                for min_area, radius in itertools.product([6, 12, 24], [1, 2])
            ]

        best_config = base_config
        best_score = -1.0
        for candidate in candidates:
            predictions = [run_baseline(image, foreground, candidate) for image in images]
            mean_iou = float(np.mean([compute_pixel_metrics(pred, gt)["iou"] for pred, gt in zip(predictions, masks)]))
            if mean_iou > best_score:
                best_score = mean_iou
                best_config = candidate

        tuned_configs.append(best_config)

    return tuned_configs


def load_prediction_masks(predictions_dir: str, file_names: list[str]) -> list[np.ndarray]:
    from PIL import Image

    predictions = []
    for file_name in file_names:
        stem = Path(file_name).stem
        candidates = [
            Path(predictions_dir) / file_name,
            Path(predictions_dir) / f"{stem}.png",
            Path(predictions_dir) / f"{stem}.tif",
            Path(predictions_dir) / f"{stem}.tiff",
        ]
        prediction_path = next((path for path in candidates if path.exists()), None)
        if prediction_path is None:
            raise FileNotFoundError(f"No saved prediction found for {file_name} in {predictions_dir}")
        predictions.append(ensure_binary_mask(np.asarray(Image.open(prediction_path).convert("L"))))
    return predictions


def run_unet_predictions(model_path: str, images: list[np.ndarray]) -> list[np.ndarray]:
    import torch

    from src.model.UNet import UNet

    unet = UNet()
    checkpoint = torch.load(model_path, map_location=unet.device, weights_only=True)
    unet.load_state_dict(checkpoint["model_state_dict"])
    unet.normalizer = _SimpleNormalize(checkpoint.get("normalizer_mean"), checkpoint.get("normalizer_std"))
    unet.to(unet.device).eval()

    predictions = []
    with torch.inference_mode():
        for image in images:
            tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(unet.device)
            padded, original_shape = _pad_tensor_to_multiple(tensor, multiple=16)
            logits = unet(padded)
            logits = logits[:, :, : original_shape[0], : original_shape[1]]
            mask = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            predictions.append(ensure_binary_mask(mask))
    return predictions


class _SimpleNormalize:
    def __init__(self, mean, std):
        super().__init__()
        mean = [0.0] if mean is None else mean
        std = [1.0] if std is None else std
        self.register_buffer("mean", torch_tensor(mean))
        self.register_buffer("std", torch_tensor(std))

    def forward(self, tensor):
        return (tensor - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)


def torch_tensor(values):
    import torch

    return torch.tensor(values, dtype=torch.float32)


def _pad_tensor_to_multiple(tensor, multiple: int):
    import torch.nn.functional as F

    height, width = tensor.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height == 0 and pad_width == 0:
        return tensor, (height, width)
    return F.pad(tensor, (0, pad_width, 0, pad_height), mode="reflect"), (height, width)


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir: str, aggregate_rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    methods = [row["method"] for row in aggregate_rows]
    metrics = [
        ("iou", "Pixel IoU"),
        ("dice", "Dice"),
        ("object_f1", "Object F1"),
        ("relative_particle_count_error", "Relative Count Error"),
        ("mean_absolute_relative_ecd_error", "Mean |Relative ECD Error|"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    for axis, (key, title) in zip(axes, metrics):
        values = [row[key] for row in aggregate_rows]
        axis.bar(methods, values)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "baseline_comparison_plots.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_example_panels(
    output_dir: str,
    images: list[np.ndarray],
    masks: list[np.ndarray],
    predictions_by_method: dict[str, list[np.ndarray]],
    file_names: list[str],
    num_examples: int,
) -> None:
    if num_examples <= 0:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    example_dir = os.path.join(output_dir, "examples")
    os.makedirs(example_dir, exist_ok=True)
    methods = list(predictions_by_method.keys())

    for index in range(min(num_examples, len(images))):
        n_cols = 2 + len(methods)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.2))
        axes[0].imshow(images[index], cmap="gray")
        axes[0].set_title("Image")
        axes[1].imshow(masks[index], cmap="gray")
        axes[1].set_title("Ground truth")

        for axis, method in zip(axes[2:], methods):
            axis.imshow(predictions_by_method[method][index], cmap="gray")
            axis.set_title(method)

        for axis in axes:
            axis.axis("off")

        fig.tight_layout()
        stem = Path(file_names[index]).stem.replace(" ", "_")
        fig.savefig(os.path.join(example_dir, f"{index:03d}_{stem}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def print_summary_table(rows: list[dict]) -> None:
    headers = [
        "Method",
        "IoU",
        "Dice",
        "Object precision",
        "Object recall",
        "Object F1",
        "Relative count error",
        "Mean relative ECD error",
    ]
    print("\n" + " | ".join(headers))
    print("-" * 130)
    for row in rows:
        print(
            f"{row['method']} | "
            f"{row['iou']:.4f} | "
            f"{row['dice']:.4f} | "
            f"{row['object_precision']:.4f} | "
            f"{row['object_recall']:.4f} | "
            f"{row['object_f1']:.4f} | "
            f"{row['relative_particle_count_error']:.4f} | "
            f"{row['mean_absolute_relative_ecd_error']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare U-Net segmentation with thresholding baselines.")
    parser.add_argument("--images-dir", default="data/medres_images")
    parser.add_argument("--masks-dir", default="data/medres_masks")
    parser.add_argument("--model-path", default=None, help="Optional trained U-Net checkpoint path.")
    parser.add_argument("--unet-predictions-dir", default=None, help="Optional folder with saved U-Net prediction masks.")
    parser.add_argument("--validation-images-dir", default=None, help="Optional validation images for tuning baselines.")
    parser.add_argument("--validation-masks-dir", default=None, help="Optional validation masks for tuning baselines.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--foreground", choices=["bright", "dark"], default="bright")
    parser.add_argument("--object-iou-threshold", type=float, default=0.3)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--watershed", action="store_true", help="Enable watershed in morphology baselines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("data", "experiments", "baseline_comparison", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    images, masks, file_names = load_dataset_arrays(args.images_dir, args.masks_dir)
    baseline_configs = default_baseline_configs()
    if args.watershed:
        baseline_configs = [
            BaselineConfig(**{**config_to_dict(config), "watershed": config.use_morphology})
            for config in baseline_configs
        ]

    tuning_used = False
    if args.validation_images_dir and args.validation_masks_dir:
        validation_images, validation_masks, _ = load_dataset_arrays(args.validation_images_dir, args.validation_masks_dir)
        baseline_configs = tune_baseline_parameters(validation_images, validation_masks, args.foreground, baseline_configs)
        tuning_used = True

    predictions_by_method: dict[str, list[np.ndarray]] = {}
    for config in baseline_configs:
        predictions_by_method[config.name] = [run_baseline(image, args.foreground, config) for image in images]

    if args.unet_predictions_dir:
        predictions_by_method["unet"] = load_prediction_masks(args.unet_predictions_dir, file_names)
    elif args.model_path:
        predictions_by_method["unet"] = run_unet_predictions(args.model_path, images)

    aggregate_rows: list[dict] = []
    per_image_rows: list[dict] = []
    size_rows: list[dict] = []

    for method_name, predictions in predictions_by_method.items():
        aggregate, per_image, size_stratified = evaluate_method_on_dataset(
            method_name=method_name,
            predictions=predictions,
            ground_truths=masks,
            file_names=file_names,
            object_iou_threshold=args.object_iou_threshold,
        )
        aggregate_rows.append(aggregate)
        per_image_rows.extend(per_image)
        size_rows.extend(size_stratified)

    write_csv(os.path.join(output_dir, "baseline_comparison_metrics.csv"), aggregate_rows)
    write_csv(os.path.join(output_dir, "baseline_comparison_per_image.csv"), per_image_rows)
    write_csv(os.path.join(output_dir, "baseline_comparison_size_stratified.csv"), size_rows)

    parameters = {
        "foreground": args.foreground,
        "object_iou_threshold": args.object_iou_threshold,
        "parameter_tuning": "validation_grid_search" if tuning_used else "documented_defaults",
        "baselines": {config.name: config_to_dict(config) for config in baseline_configs},
        "unet": {
            "model_path": args.model_path,
            "predictions_dir": args.unet_predictions_dir,
        },
    }
    with open(os.path.join(output_dir, "baseline_parameters.json"), "w", encoding="utf-8") as json_file:
        json.dump(parameters, json_file, indent=2)

    save_plots(output_dir, aggregate_rows)
    save_example_panels(output_dir, images, masks, predictions_by_method, file_names, args.num_examples)
    print_summary_table(aggregate_rows)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
