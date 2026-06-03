import os
import tempfile
import unittest

import numpy as np

from experiments.compare_thresholding_baselines import (
    AUTO_FOREGROUND_MEDIAN_THRESHOLD,
    BaselineConfig,
    compute_pixel_metrics,
    ensure_binary_mask,
    evaluate_method_on_dataset,
    infer_foreground_from_image,
    run_baseline,
    write_csv,
)


class ThresholdingBaselineTests(unittest.TestCase):
    def test_otsu_baseline_returns_binary_mask(self):
        image = np.zeros((32, 32), dtype=np.uint8)
        image[10:20, 10:20] = 220

        prediction = run_baseline(image, "bright", BaselineConfig(name="otsu", threshold="otsu"))

        self.assertEqual(prediction.shape, image.shape)
        self.assertTrue(np.isin(prediction, [0, 1]).all())
        self.assertGreater(prediction.sum(), 0)

    def test_auto_foreground_infers_polarity_from_image_median(self):
        bright_particles = np.zeros((16, 16), dtype=np.uint8)
        bright_particles[4:8, 4:8] = 220
        dark_particles = np.full((16, 16), 140, dtype=np.uint8)
        dark_particles[4:8, 4:8] = 20

        self.assertEqual(infer_foreground_from_image(bright_particles), "bright")
        self.assertEqual(infer_foreground_from_image(dark_particles), "dark")
        self.assertEqual(AUTO_FOREGROUND_MEDIAN_THRESHOLD, 63.0)

    def test_auto_foreground_runs_baseline_for_dark_particles(self):
        image = np.full((32, 32), 140, dtype=np.uint8)
        image[10:20, 10:20] = 20

        prediction = run_baseline(image, "auto", BaselineConfig(name="adaptive", threshold="adaptive"))

        self.assertEqual(prediction.shape, image.shape)
        self.assertTrue(np.isin(prediction, [0, 1]).all())
        self.assertGreater(prediction[10:20, 10:20].sum(), 0)

    def test_empty_masks_do_not_crash_metrics(self):
        prediction = np.zeros((16, 16), dtype=np.uint8)
        ground_truth = np.zeros((16, 16), dtype=np.uint8)

        pixel_metrics = compute_pixel_metrics(prediction, ground_truth)
        aggregate, per_image, size_rows = evaluate_method_on_dataset(
            method_name="empty",
            predictions=[prediction],
            ground_truths=[ground_truth],
            file_names=["empty.tif"],
            object_iou_threshold=0.3,
        )

        self.assertEqual(pixel_metrics["iou"], 1.0)
        self.assertEqual(aggregate["iou"], 1.0)
        self.assertEqual(len(per_image), 1)
        self.assertGreaterEqual(len(size_rows), 1)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_pixel_metrics(np.zeros((8, 8), dtype=np.uint8), np.zeros((9, 8), dtype=np.uint8))

    def test_output_csv_is_written(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "metrics.csv")
            write_csv(output_path, [{"method": "otsu", "iou": 1.0}])

            self.assertTrue(os.path.exists(output_path))

    def test_ensure_binary_mask_accepts_channel_dimension(self):
        mask = np.zeros((1, 8, 8), dtype=np.uint8)
        mask[:, 2:4, 2:4] = 255

        binary = ensure_binary_mask(mask)

        self.assertEqual(binary.shape, (8, 8))
        self.assertTrue(np.isin(binary, [0, 1]).all())


if __name__ == "__main__":
    unittest.main()
