import csv
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ParticleProperties:
    image_index: int
    image_name: str
    label_id: int
    area_px: int
    ecd_px: float
    area: float
    ecd: float


@dataclass(frozen=True)
class ParticleMatch:
    image_index: int
    image_name: str
    gt_label: int
    pred_label: int
    iou: float
    intersection_px: int
    union_px: int


@dataclass(frozen=True)
class SizeBin:
    index: int
    name: str
    lower: float
    upper: float


@dataclass
class SizeStratifiedMetricsResult:
    rows: list[dict]
    unit: str
    bin_edges: list[float]
    bin_labels: list[str]
    iou_threshold: float


def label_particles(mask: np.ndarray) -> np.ndarray:
    """Label connected components in a binary particle mask."""
    binary_mask = np.asarray(mask)
    while binary_mask.ndim > 2 and binary_mask.shape[0] == 1:
        binary_mask = binary_mask[0]
    binary_mask = (binary_mask > 0).astype(np.uint8)
    _, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    return labels


def compute_particle_properties(
    label_image: np.ndarray,
    scale_factor: float | None = None,
    unit: str = "pixel",
    image_index: int = 0,
    image_name: str = "",
) -> dict[int, ParticleProperties]:
    """Compute per-particle area and equivalent circular diameter (ECD)."""
    flattened = np.asarray(label_image, dtype=np.int32).ravel()
    counts = np.bincount(flattened)
    properties: dict[int, ParticleProperties] = {}

    for label_id in range(1, len(counts)):
        area_px = int(counts[label_id])
        if area_px <= 0:
            continue

        ecd_px = 2.0 * math.sqrt(area_px / math.pi)
        if scale_factor is None:
            area = float(area_px)
            ecd = float(ecd_px)
        else:
            area = float(area_px) * scale_factor * scale_factor
            ecd = float(ecd_px) * scale_factor

        properties[label_id] = ParticleProperties(
            image_index=image_index,
            image_name=image_name,
            label_id=label_id,
            area_px=area_px,
            ecd_px=float(ecd_px),
            area=area,
            ecd=ecd,
        )

    return properties


def match_particles(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    iou_threshold: float = 0.3,
) -> dict[str, object]:
    """
    Match predicted particles to ground-truth particles using one-to-one IoU matching.

    The matching maximizes the number of matched pairs. When several matches are
    possible, higher-IoU edges are preferred deterministically.
    """
    gt_labels = np.asarray(gt_labels, dtype=np.int32)
    pred_labels = np.asarray(pred_labels, dtype=np.int32)

    if gt_labels.shape != pred_labels.shape:
        raise ValueError("Ground-truth and prediction labels must have the same shape.")

    gt_flat = gt_labels.ravel()
    pred_flat = pred_labels.ravel()
    gt_areas = np.bincount(gt_flat)
    pred_areas = np.bincount(pred_flat)

    gt_ids = set(range(1, len(gt_areas)))
    pred_ids = set(range(1, len(pred_areas)))

    overlap_mask = (gt_flat > 0) & (pred_flat > 0)
    adjacency: dict[int, list[tuple[int, float]]] = {}
    iou_lookup: dict[tuple[int, int], tuple[float, int, int]] = {}

    if np.any(overlap_mask):
        pred_base = int(pred_flat.max()) + 1
        pair_codes = gt_flat[overlap_mask] * pred_base + pred_flat[overlap_mask]
        pair_counts = np.bincount(pair_codes)
        pair_ids = np.flatnonzero(pair_counts)

        for pair_code in pair_ids:
            intersection = int(pair_counts[pair_code])
            gt_id = int(pair_code // pred_base)
            pred_id = int(pair_code % pred_base)
            union = int(gt_areas[gt_id] + pred_areas[pred_id] - intersection)
            iou = (intersection / union) if union > 0 else 0.0

            if iou >= iou_threshold:
                adjacency.setdefault(gt_id, []).append((pred_id, float(iou)))
                iou_lookup[(gt_id, pred_id)] = (float(iou), intersection, union)

    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: (-item[1], item[0]))

    gt_order = sorted(adjacency, key=lambda gt_id: (-adjacency[gt_id][0][1], gt_id))
    pred_to_gt: dict[int, int] = {}

    def _try_match(gt_id: int, seen_preds: set[int]) -> bool:
        for pred_id, _ in adjacency.get(gt_id, []):
            if pred_id in seen_preds:
                continue
            seen_preds.add(pred_id)

            current_gt = pred_to_gt.get(pred_id)
            if current_gt is None or _try_match(current_gt, seen_preds):
                pred_to_gt[pred_id] = gt_id
                return True
        return False

    for gt_id in gt_order:
        _try_match(gt_id, set())

    matches = [
        ParticleMatch(
            image_index=0,
            image_name="",
            gt_label=gt_id,
            pred_label=pred_id,
            iou=iou_lookup[(gt_id, pred_id)][0],
            intersection_px=iou_lookup[(gt_id, pred_id)][1],
            union_px=iou_lookup[(gt_id, pred_id)][2],
        )
        for pred_id, gt_id in pred_to_gt.items()
    ]
    matches.sort(key=lambda match: (match.gt_label, match.pred_label))

    matched_gt_ids = {match.gt_label for match in matches}
    matched_pred_ids = {match.pred_label for match in matches}

    return {
        "matches": matches,
        "matched_gt_ids": matched_gt_ids,
        "matched_pred_ids": matched_pred_ids,
        "unmatched_gt_ids": gt_ids - matched_gt_ids,
        "unmatched_pred_ids": pred_ids - matched_pred_ids,
    }


def compute_size_stratified_metrics(
    ground_truths: list[np.ndarray],
    predictions: list[np.ndarray],
    file_names: list[str] | None = None,
    file_infos: list | None = None,
    iou_threshold: float = 0.3,
    bin_edges: list[float] | None = None,
    bin_labels: list[str] | None = None,
) -> SizeStratifiedMetricsResult:
    """
    Evaluate object-level segmentation quality stratified by particle size.

    Ground-truth rows focus on recall and size estimation error.
    Predicted-size rows focus on false positives and precision.
    """
    if len(ground_truths) != len(predictions):
        raise ValueError("Ground-truth and prediction lists must have the same length.")

    if file_names is None:
        file_names = [f"image_{idx:03d}" for idx in range(len(ground_truths))]

    gt_particles: list[ParticleProperties] = []
    pred_particles: list[ParticleProperties] = []
    matches: list[ParticleMatch] = []
    match_by_gt_key: dict[tuple[int, int], ParticleMatch] = {}
    match_by_pred_key: dict[tuple[int, int], ParticleMatch] = {}

    resolved_unit = _resolve_unit(file_infos)

    for image_index, (ground_truth, prediction) in enumerate(zip(ground_truths, predictions)):
        image_name = file_names[image_index] if image_index < len(file_names) else f"image_{image_index:03d}"
        scale_factor = _resolve_scale_factor(file_infos, image_index)

        gt_labels = label_particles(ground_truth)
        pred_labels = label_particles(prediction)

        gt_props = compute_particle_properties(
            gt_labels,
            scale_factor=scale_factor,
            unit=resolved_unit,
            image_index=image_index,
            image_name=image_name,
        )
        pred_props = compute_particle_properties(
            pred_labels,
            scale_factor=scale_factor,
            unit=resolved_unit,
            image_index=image_index,
            image_name=image_name,
        )

        gt_particles.extend(gt_props.values())
        pred_particles.extend(pred_props.values())

        match_result = match_particles(gt_labels, pred_labels, iou_threshold=iou_threshold)
        for base_match in match_result["matches"]:
            enriched_match = ParticleMatch(
                image_index=image_index,
                image_name=image_name,
                gt_label=base_match.gt_label,
                pred_label=base_match.pred_label,
                iou=base_match.iou,
                intersection_px=base_match.intersection_px,
                union_px=base_match.union_px,
            )
            matches.append(enriched_match)
            match_by_gt_key[(image_index, enriched_match.gt_label)] = enriched_match
            match_by_pred_key[(image_index, enriched_match.pred_label)] = enriched_match

    gt_by_key = {(particle.image_index, particle.label_id): particle for particle in gt_particles}
    pred_by_key = {(particle.image_index, particle.label_id): particle for particle in pred_particles}

    rows: list[dict] = []
    rows.extend(_build_overall_rows(gt_particles, pred_particles, matches, gt_by_key, pred_by_key, resolved_unit, iou_threshold))

    gt_ecds = [particle.ecd for particle in gt_particles]
    bins = _build_bins(gt_ecds, bin_edges=bin_edges, bin_labels=bin_labels)
    if not bins:
        return SizeStratifiedMetricsResult(
            rows=rows,
            unit=resolved_unit,
            bin_edges=[],
            bin_labels=[],
            iou_threshold=iou_threshold,
        )

    for size_bin in bins:
        gt_bin_particles = [particle for particle in gt_particles if _particle_in_bin(particle.ecd, size_bin)]
        pred_bin_particles = [particle for particle in pred_particles if _particle_in_bin(particle.ecd, size_bin)]

        gt_bin_matches = [
            match_by_gt_key[(particle.image_index, particle.label_id)]
            for particle in gt_bin_particles
            if (particle.image_index, particle.label_id) in match_by_gt_key
        ]
        pred_bin_matches = [
            match_by_pred_key[(particle.image_index, particle.label_id)]
            for particle in pred_bin_particles
            if (particle.image_index, particle.label_id) in match_by_pred_key
        ]

        rows.append(
            _build_ground_truth_row(
                size_bin=size_bin,
                unit=resolved_unit,
                iou_threshold=iou_threshold,
                gt_particles=gt_bin_particles,
                gt_matches=gt_bin_matches,
                gt_by_key=gt_by_key,
                pred_by_key=pred_by_key,
            )
        )
        rows.append(
            _build_predicted_row(
                size_bin=size_bin,
                unit=resolved_unit,
                iou_threshold=iou_threshold,
                pred_particles=pred_bin_particles,
                pred_matches=pred_bin_matches,
                gt_by_key=gt_by_key,
                pred_by_key=pred_by_key,
            )
        )

    return SizeStratifiedMetricsResult(
        rows=rows,
        unit=resolved_unit,
        bin_edges=[size_bin.lower for size_bin in bins] + [bins[-1].upper],
        bin_labels=[size_bin.name for size_bin in bins],
        iou_threshold=iou_threshold,
    )


def save_size_stratified_metrics(
    result: SizeStratifiedMetricsResult,
    csv_path: str,
    summary_path: str | None = None,
    plot_path: str | None = None,
) -> dict[str, str]:
    """Persist size-stratified metrics to CSV and optional TXT/PNG artifacts."""
    output_paths: dict[str, str] = {}
    _ensure_parent_dir(csv_path)

    fieldnames = [
        "row_type",
        "size_bin",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "unit",
        "iou_threshold",
        "ground_truth_particles",
        "predicted_particles",
        "matched_particles",
        "true_positives",
        "false_negatives",
        "false_positives",
        "precision",
        "recall",
        "f1_score",
        "mean_iou",
        "mean_ecd_error",
        "median_ecd_error",
        "mean_absolute_ecd_error",
        "median_absolute_ecd_error",
        "mean_ecd_error_px",
        "median_ecd_error_px",
        "mean_absolute_ecd_error_px",
        "median_absolute_ecd_error_px",
        "median_relative_ecd_error",
        "mean_absolute_relative_ecd_error",
        "median_absolute_relative_ecd_error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            writer.writerow({field: row.get(field) for field in fieldnames})

    output_paths["csv"] = csv_path

    confusion_csv_path = _append_to_stem(csv_path, "_confusion_matrices")
    output_paths["confusion_csv"] = save_size_stratified_confusion_matrices(
        result,
        csv_path=confusion_csv_path,
    )["csv"]

    if summary_path is not None:
        _ensure_parent_dir(summary_path)
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            summary_file.write("Particle Size-Stratified Evaluation\n")
            summary_file.write("=================================\n")
            summary_file.write(f"IoU threshold: {result.iou_threshold:.2f}\n")
            summary_file.write(f"ECD unit: {result.unit}\n\n")

            summary_file.write(
                f"{'Row Type':<14}{'Bin':<12}{'Range':<26}{'GT':>6}{'Pred':>6}{'TP':>6}{'FN':>6}{'FP':>6}"
                f"{'Prec':>10}{'Recall':>10}{'F1':>10}{'MeanIoU':>10}{'Rel|ECD|':>12}\n"
            )
            for row in result.rows:
                summary_file.write(
                    f"{row['row_type']:<14}{row['size_bin']:<12}"
                    f"{_format_range(row['bin_lower'], row['bin_upper'], row['unit']):<26}"
                    f"{_format_int(row['ground_truth_particles']):>6}{_format_int(row['predicted_particles']):>6}"
                    f"{_format_int(row['true_positives']):>6}{_format_int(row['false_negatives']):>6}"
                    f"{_format_int(row['false_positives']):>6}{_format_float(row['precision']):>10}"
                    f"{_format_float(row['recall']):>10}{_format_float(row['f1_score']):>10}"
                    f"{_format_float(row['mean_iou']):>10}"
                    f"{_format_float(row['mean_absolute_relative_ecd_error']):>12}\n"
                )

        output_paths["summary"] = summary_path

        confusion_summary_path = _append_to_stem(summary_path, "_confusion_matrices")
        save_size_stratified_confusion_matrices(
            result,
            csv_path=confusion_csv_path,
            summary_path=confusion_summary_path,
        )
        output_paths["confusion_summary"] = confusion_summary_path

    if plot_path is not None:
        plot_file = plot_size_stratified_metrics(result, plot_path)
        if plot_file is not None:
            output_paths["plot"] = plot_file

        confusion_plot_path = _append_to_stem(plot_path, "_confusion_matrices")
        confusion_plot_file = plot_size_stratified_confusion_matrices(result, confusion_plot_path)
        if confusion_plot_file is not None:
            output_paths["confusion_plot"] = confusion_plot_file

    return output_paths


def save_size_stratified_confusion_matrices(
    result: SizeStratifiedMetricsResult,
    csv_path: str,
    summary_path: str | None = None,
) -> dict[str, str]:
    """Save object-detection confusion matrices for each size bin."""
    matrices = _build_confusion_matrices(result)
    output_paths: dict[str, str] = {}
    _ensure_parent_dir(csv_path)

    fieldnames = [
        "size_bin",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "unit",
        "true_positives",
        "false_negatives",
        "false_positives",
        "true_negatives",
        "note",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for matrix in matrices:
            writer.writerow({field: matrix.get(field) for field in fieldnames})

    output_paths["csv"] = csv_path

    if summary_path is not None:
        _ensure_parent_dir(summary_path)
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            summary_file.write("Particle Size-Stratified Confusion Matrices\n")
            summary_file.write("==========================================\n")
            summary_file.write("True negatives are not defined for object-level particle detection.\n\n")

            for matrix in matrices:
                summary_file.write(
                    f"{matrix['size_bin']} "
                    f"{_format_range(matrix['bin_lower'], matrix['bin_upper'], matrix['unit'])}\n"
                )
                summary_file.write(f"{'':<18}{'Predicted particle':>20}{'Missed':>12}\n")
                summary_file.write(
                    f"{'GT particle':<18}{matrix['true_positives']:>20}{matrix['false_negatives']:>12}\n"
                )
                summary_file.write(
                    f"{'No matching GT':<18}{matrix['false_positives']:>20}{'N/A':>12}\n\n"
                )

        output_paths["summary"] = summary_path

    return output_paths


def plot_size_stratified_confusion_matrices(
    result: SizeStratifiedMetricsResult,
    plot_path: str,
) -> str | None:
    """Save compact object-level confusion matrices for each size bin."""
    matrices = _build_confusion_matrices(result)
    if not matrices:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    _ensure_parent_dir(plot_path)

    n_matrices = len(matrices)
    n_cols = min(4, n_matrices)
    n_rows = int(math.ceil(n_matrices / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.5 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#eeeeee")

    for axis, matrix in zip(axes, matrices):
        values = np.asarray(
            [
                [matrix["true_positives"], matrix["false_negatives"]],
                [matrix["false_positives"], np.nan],
            ],
            dtype=float,
        )
        axis.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0)
        axis.set_title(matrix["size_bin"])
        axis.set_xticks([0, 1], labels=["Detected", "Missed"])
        axis.set_yticks([0, 1], labels=["GT particle", "No matching GT"])

        for y in range(2):
            for x in range(2):
                label = "N/A" if np.isnan(values[y, x]) else str(int(values[y, x]))
                axis.text(x, y, label, ha="center", va="center", color="black")

    for axis in axes[n_matrices:]:
        axis.axis("off")

    fig.suptitle("Object-Level Confusion Matrices by Particle Size", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_size_stratified_metrics(result: SizeStratifiedMetricsResult, plot_path: str) -> str | None:
    """Save a compact figure with recall, precision, bin IoU, and relative ECD error by size bin."""
    gt_rows = [row for row in result.rows if row["row_type"] == "ground_truth_bin"]
    pred_rows = [row for row in result.rows if row["row_type"] == "predicted_bin"]
    if not gt_rows:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    _ensure_parent_dir(plot_path)

    labels = [row["size_bin"] for row in gt_rows]
    recalls = [_plot_value(row["recall"]) for row in gt_rows]
    bin_ious = [_plot_value(row["mean_iou"]) for row in gt_rows]
    mean_abs_relative_errors = [_plot_value(row["mean_absolute_relative_ecd_error"]) for row in gt_rows]
    precisions = [_plot_value(row["precision"]) for row in pred_rows]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))

    axes[0].bar(labels, recalls, color="steelblue")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Recall by GT Size Bin")
    axes[0].set_ylabel("Recall")

    axes[1].bar(labels, precisions[: len(labels)], color="darkorange")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Precision by Pred Size Bin")
    axes[1].set_ylabel("Precision")

    axes[2].bar(labels, bin_ious, color="mediumpurple")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Mean IoU by GT Size Bin")
    axes[2].set_ylabel("Pixel IoU within bin")

    axes[3].bar(labels, mean_abs_relative_errors, color="seagreen")
    axes[3].set_title("Mean Relative |ECD Error|")
    axes[3].set_ylabel("Relative ECD error")

    for axis in axes:
        axis.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _build_overall_rows(
    gt_particles: list[ParticleProperties],
    pred_particles: list[ParticleProperties],
    matches: list[ParticleMatch],
    gt_by_key: dict[tuple[int, int], ParticleProperties],
    pred_by_key: dict[tuple[int, int], ParticleProperties],
    unit: str,
    iou_threshold: float,
) -> list[dict]:
    gt_count = len(gt_particles)
    pred_count = len(pred_particles)
    tp = len(matches)
    fn = gt_count - tp
    fp = pred_count - tp
    precision = _safe_ratio(tp, pred_count)
    recall = _safe_ratio(tp, gt_count)
    f1_score = _f1_score(precision, recall)
    ecd_errors = _collect_ecd_errors(matches, gt_by_key, pred_by_key)
    ecd_pixel_errors = _collect_ecd_pixel_errors(matches, gt_by_key, pred_by_key)
    relative_ecd_errors = _collect_relative_ecd_errors(matches, gt_by_key, pred_by_key)
    gt_mean_iou = _mean_iou_for_ground_truth_particles(gt_particles, matches)
    pred_mean_iou = _mean_iou_for_predicted_particles(pred_particles, matches)

    return [
        {
            "row_type": "ground_truth_overall",
            "size_bin": "overall",
            "bin_index": -1,
            "bin_lower": float("-inf"),
            "bin_upper": float("inf"),
            "unit": unit,
            "iou_threshold": iou_threshold,
            "ground_truth_particles": gt_count,
            "predicted_particles": "",
            "matched_particles": tp,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": "",
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "mean_iou": gt_mean_iou,
            "mean_ecd_error": _safe_stat(ecd_errors, np.mean),
            "median_ecd_error": _safe_stat(ecd_errors, np.median),
            "mean_absolute_ecd_error": _safe_stat(np.abs(ecd_errors), np.mean),
            "median_absolute_ecd_error": _safe_stat(np.abs(ecd_errors), np.median),
            "mean_ecd_error_px": _safe_stat(ecd_pixel_errors, np.mean),
            "median_ecd_error_px": _safe_stat(ecd_pixel_errors, np.median),
            "mean_absolute_ecd_error_px": _safe_stat(np.abs(ecd_pixel_errors), np.mean),
            "median_absolute_ecd_error_px": _safe_stat(np.abs(ecd_pixel_errors), np.median),
            "median_relative_ecd_error": _safe_stat(relative_ecd_errors, np.median),
            "mean_absolute_relative_ecd_error": _safe_stat(np.abs(relative_ecd_errors), np.mean),
            "median_absolute_relative_ecd_error": _safe_stat(np.abs(relative_ecd_errors), np.median),
        },
        {
            "row_type": "predicted_overall",
            "size_bin": "overall",
            "bin_index": -1,
            "bin_lower": float("-inf"),
            "bin_upper": float("inf"),
            "unit": unit,
            "iou_threshold": iou_threshold,
            "ground_truth_particles": "",
            "predicted_particles": pred_count,
            "matched_particles": tp,
            "true_positives": tp,
            "false_negatives": "",
            "false_positives": fp,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "mean_iou": pred_mean_iou,
            "mean_ecd_error": _safe_stat(ecd_errors, np.mean),
            "median_ecd_error": _safe_stat(ecd_errors, np.median),
            "mean_absolute_ecd_error": _safe_stat(np.abs(ecd_errors), np.mean),
            "median_absolute_ecd_error": _safe_stat(np.abs(ecd_errors), np.median),
            "mean_ecd_error_px": _safe_stat(ecd_pixel_errors, np.mean),
            "median_ecd_error_px": _safe_stat(ecd_pixel_errors, np.median),
            "mean_absolute_ecd_error_px": _safe_stat(np.abs(ecd_pixel_errors), np.mean),
            "median_absolute_ecd_error_px": _safe_stat(np.abs(ecd_pixel_errors), np.median),
            "median_relative_ecd_error": _safe_stat(relative_ecd_errors, np.median),
            "mean_absolute_relative_ecd_error": _safe_stat(np.abs(relative_ecd_errors), np.mean),
            "median_absolute_relative_ecd_error": _safe_stat(np.abs(relative_ecd_errors), np.median),
        },
    ]


def _build_ground_truth_row(
    size_bin: SizeBin,
    unit: str,
    iou_threshold: float,
    gt_particles: list[ParticleProperties],
    gt_matches: list[ParticleMatch],
    gt_by_key: dict[tuple[int, int], ParticleProperties],
    pred_by_key: dict[tuple[int, int], ParticleProperties],
) -> dict:
    gt_count = len(gt_particles)
    tp = len(gt_matches)
    fn = gt_count - tp
    ecd_errors = _collect_ecd_errors(gt_matches, gt_by_key, pred_by_key)
    ecd_pixel_errors = _collect_ecd_pixel_errors(gt_matches, gt_by_key, pred_by_key)
    relative_ecd_errors = _collect_relative_ecd_errors(gt_matches, gt_by_key, pred_by_key)
    mean_iou = _mean_iou_for_ground_truth_particles(gt_particles, gt_matches)

    return {
        "row_type": "ground_truth_bin",
        "size_bin": size_bin.name,
        "bin_index": size_bin.index,
        "bin_lower": size_bin.lower,
        "bin_upper": size_bin.upper,
        "unit": unit,
        "iou_threshold": iou_threshold,
        "ground_truth_particles": gt_count,
        "predicted_particles": "",
        "matched_particles": tp,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": "",
        "precision": "",
        "recall": _safe_ratio(tp, gt_count),
        "f1_score": "",
        "mean_iou": mean_iou,
        "mean_ecd_error": _safe_stat(ecd_errors, np.mean),
        "median_ecd_error": _safe_stat(ecd_errors, np.median),
        "mean_absolute_ecd_error": _safe_stat(np.abs(ecd_errors), np.mean),
        "median_absolute_ecd_error": _safe_stat(np.abs(ecd_errors), np.median),
        "mean_ecd_error_px": _safe_stat(ecd_pixel_errors, np.mean),
        "median_ecd_error_px": _safe_stat(ecd_pixel_errors, np.median),
        "mean_absolute_ecd_error_px": _safe_stat(np.abs(ecd_pixel_errors), np.mean),
        "median_absolute_ecd_error_px": _safe_stat(np.abs(ecd_pixel_errors), np.median),
        "median_relative_ecd_error": _safe_stat(relative_ecd_errors, np.median),
        "mean_absolute_relative_ecd_error": _safe_stat(np.abs(relative_ecd_errors), np.mean),
        "median_absolute_relative_ecd_error": _safe_stat(np.abs(relative_ecd_errors), np.median),
    }


def _build_predicted_row(
    size_bin: SizeBin,
    unit: str,
    iou_threshold: float,
    pred_particles: list[ParticleProperties],
    pred_matches: list[ParticleMatch],
    gt_by_key: dict[tuple[int, int], ParticleProperties],
    pred_by_key: dict[tuple[int, int], ParticleProperties],
) -> dict:
    pred_count = len(pred_particles)
    tp = len(pred_matches)
    fp = pred_count - tp
    ecd_errors = _collect_ecd_errors(pred_matches, gt_by_key, pred_by_key)
    ecd_pixel_errors = _collect_ecd_pixel_errors(pred_matches, gt_by_key, pred_by_key)
    relative_ecd_errors = _collect_relative_ecd_errors(pred_matches, gt_by_key, pred_by_key)
    mean_iou = _mean_iou_for_predicted_particles(pred_particles, pred_matches)

    return {
        "row_type": "predicted_bin",
        "size_bin": size_bin.name,
        "bin_index": size_bin.index,
        "bin_lower": size_bin.lower,
        "bin_upper": size_bin.upper,
        "unit": unit,
        "iou_threshold": iou_threshold,
        "ground_truth_particles": "",
        "predicted_particles": pred_count,
        "matched_particles": tp,
        "true_positives": tp,
        "false_negatives": "",
        "false_positives": fp,
        "precision": _safe_ratio(tp, pred_count),
        "recall": "",
        "f1_score": "",
        "mean_iou": mean_iou,
        "mean_ecd_error": _safe_stat(ecd_errors, np.mean),
        "median_ecd_error": _safe_stat(ecd_errors, np.median),
        "mean_absolute_ecd_error": _safe_stat(np.abs(ecd_errors), np.mean),
        "median_absolute_ecd_error": _safe_stat(np.abs(ecd_errors), np.median),
        "mean_ecd_error_px": _safe_stat(ecd_pixel_errors, np.mean),
        "median_ecd_error_px": _safe_stat(ecd_pixel_errors, np.median),
        "mean_absolute_ecd_error_px": _safe_stat(np.abs(ecd_pixel_errors), np.mean),
        "median_absolute_ecd_error_px": _safe_stat(np.abs(ecd_pixel_errors), np.median),
        "median_relative_ecd_error": _safe_stat(relative_ecd_errors, np.median),
        "mean_absolute_relative_ecd_error": _safe_stat(np.abs(relative_ecd_errors), np.mean),
        "median_absolute_relative_ecd_error": _safe_stat(np.abs(relative_ecd_errors), np.median),
    }


def _build_bins(
    gt_ecds: list[float],
    bin_edges: list[float] | None = None,
    bin_labels: list[str] | None = None,
) -> list[SizeBin]:
    if not gt_ecds:
        return []

    if bin_edges is None:
        q1, q2 = np.quantile(np.asarray(gt_ecds, dtype=float), [1 / 3, 2 / 3])
        edges = [float("-inf"), float(q1), float(q2), float("inf")]
        labels = bin_labels or ["small", "medium", "large"]
    else:
        if len(bin_edges) < 2:
            raise ValueError("Explicit bin_edges must contain at least two values.")
        edges = [float(edge) for edge in bin_edges]
        default_labels = [f"bin_{index + 1}" for index in range(len(edges) - 1)]
        labels = bin_labels or default_labels

    if len(labels) != len(edges) - 1:
        raise ValueError("The number of bin labels must match the number of bins.")

    return [
        SizeBin(index=index, name=labels[index], lower=edges[index], upper=edges[index + 1])
        for index in range(len(labels))
    ]


def _particle_in_bin(value: float, size_bin: SizeBin) -> bool:
    if math.isinf(size_bin.lower):
        return value <= size_bin.upper
    if math.isinf(size_bin.upper):
        return value > size_bin.lower
    return size_bin.lower < value <= size_bin.upper


def _collect_ecd_errors(
    matches: list[ParticleMatch],
    gt_by_key: dict[tuple[int, int], ParticleProperties],
    pred_by_key: dict[tuple[int, int], ParticleProperties],
) -> np.ndarray:
    errors = []
    for match in matches:
        gt_particle = gt_by_key[(match.image_index, match.gt_label)]
        pred_particle = pred_by_key[(match.image_index, match.pred_label)]
        errors.append(pred_particle.ecd - gt_particle.ecd)
    return np.asarray(errors, dtype=float)


def _collect_ecd_pixel_errors(
    matches: list[ParticleMatch],
    gt_by_key: dict[tuple[int, int], ParticleProperties],
    pred_by_key: dict[tuple[int, int], ParticleProperties],
) -> np.ndarray:
    errors = []
    for match in matches:
        gt_particle = gt_by_key[(match.image_index, match.gt_label)]
        pred_particle = pred_by_key[(match.image_index, match.pred_label)]
        errors.append(pred_particle.ecd_px - gt_particle.ecd_px)
    return np.asarray(errors, dtype=float)


def _collect_relative_ecd_errors(
    matches: list[ParticleMatch],
    gt_by_key: dict[tuple[int, int], ParticleProperties],
    pred_by_key: dict[tuple[int, int], ParticleProperties],
) -> np.ndarray:
    errors = []
    for match in matches:
        gt_particle = gt_by_key[(match.image_index, match.gt_label)]
        pred_particle = pred_by_key[(match.image_index, match.pred_label)]
        if gt_particle.ecd_px == 0:
            continue
        errors.append((pred_particle.ecd_px - gt_particle.ecd_px) / gt_particle.ecd_px)
    return np.asarray(errors, dtype=float)


def _mean_iou_for_ground_truth_particles(
    particles: list[ParticleProperties],
    matches: list[ParticleMatch],
) -> float | str:
    match_by_gt_key = {(match.image_index, match.gt_label): match for match in matches}
    intersection_sum = 0
    union_sum = 0

    for particle in particles:
        match = match_by_gt_key.get((particle.image_index, particle.label_id))
        if match is None:
            union_sum += particle.area_px
        else:
            intersection_sum += match.intersection_px
            union_sum += match.union_px

    return _safe_ratio(intersection_sum, union_sum)


def _mean_iou_for_predicted_particles(
    particles: list[ParticleProperties],
    matches: list[ParticleMatch],
) -> float | str:
    match_by_pred_key = {(match.image_index, match.pred_label): match for match in matches}
    intersection_sum = 0
    union_sum = 0

    for particle in particles:
        match = match_by_pred_key.get((particle.image_index, particle.label_id))
        if match is None:
            union_sum += particle.area_px
        else:
            intersection_sum += match.intersection_px
            union_sum += match.union_px

    return _safe_ratio(intersection_sum, union_sum)


def _build_confusion_matrices(result: SizeStratifiedMetricsResult) -> list[dict]:
    matrices = []

    overall_gt = _find_row(result.rows, "ground_truth_overall", "overall")
    overall_pred = _find_row(result.rows, "predicted_overall", "overall")
    if overall_gt is not None and overall_pred is not None:
        matrices.append(_confusion_matrix_from_rows(overall_gt, overall_pred))

    gt_rows = [row for row in result.rows if row["row_type"] == "ground_truth_bin"]
    for gt_row in gt_rows:
        pred_row = _find_row(result.rows, "predicted_bin", gt_row["size_bin"])
        if pred_row is not None:
            matrices.append(_confusion_matrix_from_rows(gt_row, pred_row))

    return matrices


def _find_row(rows: list[dict], row_type: str, size_bin: str) -> dict | None:
    for row in rows:
        if row["row_type"] == row_type and row["size_bin"] == size_bin:
            return row
    return None


def _confusion_matrix_from_rows(gt_row: dict, pred_row: dict) -> dict:
    return {
        "size_bin": gt_row["size_bin"],
        "bin_index": gt_row["bin_index"],
        "bin_lower": gt_row["bin_lower"],
        "bin_upper": gt_row["bin_upper"],
        "unit": gt_row["unit"],
        "true_positives": _count_value(gt_row["true_positives"]),
        "false_negatives": _count_value(gt_row["false_negatives"]),
        "false_positives": _count_value(pred_row["false_positives"]),
        "true_negatives": "not_applicable",
        "note": "True negatives are not defined for object-level particle detection.",
    }


def _resolve_scale_factor(file_infos: list | None, image_index: int) -> float | None:
    if not file_infos or image_index >= len(file_infos):
        return None

    file_info = file_infos[image_index]
    if file_info is None:
        return None

    unit = getattr(file_info, "unit", None)
    pixel_width = getattr(file_info, "pixel_width", None)
    downsize_factor = getattr(file_info, "downsize_factor", 1.0)
    if unit is None or str(unit).strip() in ("", "pixel") or pixel_width in (None, 0):
        return None

    return float(downsize_factor) * float(pixel_width)


def _resolve_unit(file_infos: list | None) -> str:
    if not file_infos:
        return "pixel"
    for file_info in file_infos:
        unit = getattr(file_info, "unit", None) if file_info is not None else None
        if unit is not None and str(unit).strip() not in ("", "pixel"):
            return unit
    return "pixel"


def _safe_ratio(numerator: float, denominator: float) -> float | str:
    if denominator == 0:
        return ""
    return float(numerator) / float(denominator)


def _f1_score(precision: float | str, recall: float | str) -> float | str:
    if precision == "" or recall == "":
        return ""
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


def _safe_stat(values: np.ndarray, reducer) -> float | str:
    if values.size == 0:
        return ""
    return float(reducer(values))


def _safe_penalized_mean(values: np.ndarray, denominator: int) -> float | str:
    if denominator == 0:
        return ""
    return float(np.sum(values)) / float(denominator)


def _format_float(value) -> str:
    if value == "":
        return ""
    return f"{value:.3f}"


def _format_int(value) -> str:
    if value == "":
        return ""
    return str(int(value))


def _count_value(value) -> int:
    if value == "":
        return 0
    return int(value)


def _format_range(lower: float, upper: float, unit: str) -> str:
    lower_text = "-inf" if math.isinf(lower) and lower < 0 else f"{lower:.2f}"
    upper_text = "inf" if math.isinf(upper) and upper > 0 else f"{upper:.2f}"
    return f"({lower_text}, {upper_text}] {unit}"


def _ensure_parent_dir(path: str) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _append_to_stem(path: str, suffix: str) -> str:
    root, extension = os.path.splitext(path)
    return f"{root}{suffix}{extension}"


def _plot_value(value):
    if value == "":
        return np.nan
    return value
