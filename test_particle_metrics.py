import unittest
import csv
import os
import tempfile

import numpy as np

from src.model.ParticleMetrics import (
    compute_size_stratified_metrics,
    label_particles,
    match_particles,
    save_size_stratified_metrics,
)


class ParticleMetricsTests(unittest.TestCase):
    def _get_row(self, rows, row_type, size_bin):
        for row in rows:
            if row["row_type"] == row_type and row["size_bin"] == size_bin:
                return row
        self.fail(f"Missing row_type={row_type}, size_bin={size_bin}")

    def test_perfect_match(self):
        gt = np.zeros((8, 8), dtype=np.uint8)
        gt[1:3, 1:3] = 1
        pred = gt.copy()

        result = compute_size_stratified_metrics(
            ground_truths=[gt],
            predictions=[pred],
            bin_edges=[0.0, 10.0],
            bin_labels=["all"],
        )

        overall_gt = self._get_row(result.rows, "ground_truth_overall", "overall")
        overall_pred = self._get_row(result.rows, "predicted_overall", "overall")
        all_gt = self._get_row(result.rows, "ground_truth_bin", "all")
        all_pred = self._get_row(result.rows, "predicted_bin", "all")

        self.assertEqual(overall_gt["true_positives"], 1)
        self.assertEqual(overall_gt["false_negatives"], 0)
        self.assertEqual(overall_pred["false_positives"], 0)
        self.assertAlmostEqual(overall_gt["precision"], 1.0)
        self.assertAlmostEqual(overall_gt["recall"], 1.0)
        self.assertAlmostEqual(overall_gt["f1_score"], 1.0)
        self.assertAlmostEqual(overall_gt["mean_iou"], 1.0)
        self.assertAlmostEqual(all_gt["mean_absolute_ecd_error"], 0.0)
        self.assertAlmostEqual(all_pred["precision"], 1.0)
        self.assertAlmostEqual(all_pred["mean_iou"], 1.0)

    def test_missed_small_particle(self):
        gt = np.zeros((12, 12), dtype=np.uint8)
        gt[1:3, 1:3] = 1
        gt[6:10, 6:10] = 1

        pred = np.zeros_like(gt)
        pred[6:10, 6:10] = 1

        result = compute_size_stratified_metrics(
            ground_truths=[gt],
            predictions=[pred],
            bin_edges=[0.0, 3.0, 10.0],
            bin_labels=["small", "large"],
        )

        small_gt = self._get_row(result.rows, "ground_truth_bin", "small")
        large_gt = self._get_row(result.rows, "ground_truth_bin", "large")

        self.assertEqual(small_gt["ground_truth_particles"], 1)
        self.assertEqual(small_gt["true_positives"], 0)
        self.assertEqual(small_gt["false_negatives"], 1)
        self.assertEqual(small_gt["recall"], 0.0)
        self.assertAlmostEqual(small_gt["mean_iou"], 0.0)
        self.assertEqual(large_gt["true_positives"], 1)
        self.assertEqual(large_gt["false_negatives"], 0)
        self.assertAlmostEqual(large_gt["mean_iou"], 1.0)

    def test_false_positive_particle(self):
        gt = np.zeros((12, 12), dtype=np.uint8)
        gt[2:5, 2:5] = 1

        pred = gt.copy()
        pred[7:9, 7:9] = 1

        result = compute_size_stratified_metrics(
            ground_truths=[gt],
            predictions=[pred],
            bin_edges=[0.0, 3.0, 10.0],
            bin_labels=["small", "large"],
        )

        small_pred = self._get_row(result.rows, "predicted_bin", "small")
        large_pred = self._get_row(result.rows, "predicted_bin", "large")

        self.assertEqual(small_pred["predicted_particles"], 1)
        self.assertEqual(small_pred["false_positives"], 1)
        self.assertEqual(small_pred["precision"], 0.0)
        self.assertAlmostEqual(small_pred["mean_iou"], 0.0)
        self.assertEqual(large_pred["predicted_particles"], 1)
        self.assertEqual(large_pred["false_positives"], 0)
        self.assertAlmostEqual(large_pred["precision"], 1.0)
        self.assertAlmostEqual(large_pred["mean_iou"], 1.0)

    def test_ecd_error_is_reported_in_pixels_and_relative_to_gt_size(self):
        gt = np.zeros((12, 12), dtype=np.uint8)
        gt[1:4, 1:4] = 1

        pred = np.zeros_like(gt)
        pred[1:5, 1:5] = 1

        result = compute_size_stratified_metrics(
            ground_truths=[gt],
            predictions=[pred],
            bin_edges=[0.0, 10.0],
            bin_labels=["all"],
        )

        all_gt = self._get_row(result.rows, "ground_truth_bin", "all")
        expected_gt_ecd_px = 2.0 * np.sqrt(9.0 / np.pi)
        expected_pred_ecd_px = 2.0 * np.sqrt(16.0 / np.pi)
        expected_error_px = expected_pred_ecd_px - expected_gt_ecd_px

        self.assertAlmostEqual(all_gt["mean_iou"], 9.0 / 16.0)
        self.assertAlmostEqual(all_gt["mean_ecd_error_px"], expected_error_px)
        self.assertAlmostEqual(all_gt["mean_absolute_ecd_error_px"], abs(expected_error_px))
        self.assertAlmostEqual(all_gt["mean_relative_ecd_error"], 1.0 / 3.0)
        self.assertAlmostEqual(all_gt["mean_absolute_relative_ecd_error"], 1.0 / 3.0)

    def test_confusion_matrix_artifacts_are_saved(self):
        gt = np.zeros((12, 12), dtype=np.uint8)
        gt[1:3, 1:3] = 1
        gt[6:10, 6:10] = 1

        pred = np.zeros_like(gt)
        pred[6:10, 6:10] = 1
        pred[1:3, 8:10] = 1

        result = compute_size_stratified_metrics(
            ground_truths=[gt],
            predictions=[pred],
            bin_edges=[0.0, 3.0, 10.0],
            bin_labels=["small", "large"],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = save_size_stratified_metrics(
                result,
                csv_path=os.path.join(tmp_dir, "metrics.csv"),
                summary_path=os.path.join(tmp_dir, "metrics.txt"),
            )

            self.assertTrue(os.path.exists(paths["confusion_csv"]))
            self.assertTrue(os.path.exists(paths["confusion_summary"]))

            with open(paths["confusion_csv"], newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            small_row = next(row for row in rows if row["size_bin"] == "small")
            large_row = next(row for row in rows if row["size_bin"] == "large")

            self.assertEqual(small_row["true_positives"], "0")
            self.assertEqual(small_row["false_negatives"], "1")
            self.assertEqual(small_row["false_positives"], "1")
            self.assertEqual(small_row["true_negatives"], "not_applicable")
            self.assertEqual(large_row["true_positives"], "1")

    def test_one_prediction_overlapping_multiple_ground_truth_particles(self):
        gt = np.zeros((8, 10), dtype=np.uint8)
        gt[2:4, 1:3] = 1
        gt[2:4, 5:7] = 1

        pred = np.zeros_like(gt)
        pred[2:4, 1:6] = 1

        gt_labels = label_particles(gt)
        pred_labels = label_particles(pred)
        match_result = match_particles(gt_labels, pred_labels, iou_threshold=0.3)

        self.assertEqual(len(match_result["matches"]), 1)
        self.assertEqual(len(match_result["unmatched_gt_ids"]), 1)
        self.assertEqual(len(match_result["unmatched_pred_ids"]), 0)

    def test_multiple_predictions_overlapping_one_ground_truth_particle(self):
        gt = np.zeros((8, 10), dtype=np.uint8)
        gt[2:4, 1:6] = 1

        pred = np.zeros_like(gt)
        pred[2:4, 1:3] = 1
        pred[2:4, 4:6] = 1

        gt_labels = label_particles(gt)
        pred_labels = label_particles(pred)
        match_result = match_particles(gt_labels, pred_labels, iou_threshold=0.3)

        self.assertEqual(len(match_result["matches"]), 1)
        self.assertEqual(len(match_result["unmatched_gt_ids"]), 0)
        self.assertEqual(len(match_result["unmatched_pred_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
