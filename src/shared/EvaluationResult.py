class EvaluationResult:
    def __init__(
        self,
        iou_scores,
        dice_scores,
        size_stratified_metrics=None,
        size_stratified_paths=None,
    ):
        import numpy as np
        self.iou_scores = iou_scores
        self.dice_scores = dice_scores
        self.size_stratified_metrics = size_stratified_metrics
        self.size_stratified_paths = size_stratified_paths or {}

        self.mean_iou = np.mean(iou_scores)
        self.mean_dice = np.mean(dice_scores)
        self.min_iou = np.min(iou_scores)
        self.min_dice = np.min(dice_scores)
        self.max_iou = np.max(iou_scores)
        self.max_dice = np.max(dice_scores)
        size_summary = self._extract_overall_size_summary()
        self.object_precision = size_summary["precision"]
        self.object_recall = size_summary["recall"]
        self.mean_absolute_relative_ecd_error = size_summary["mean_absolute_relative_ecd_error"]

    def __len__(self):
        return len(self.iou_scores)

    def __iter__(self):
        yield self.mean_iou
        yield self.mean_dice

    def _extract_overall_size_summary(self):
        import numpy as np

        default_summary = {
            "precision": np.nan,
            "recall": np.nan,
            "mean_absolute_relative_ecd_error": np.nan,
        }
        rows = getattr(self.size_stratified_metrics, "rows", None)
        if not rows:
            return default_summary

        overall_row = next(
            (
                row
                for row in rows
                if row.get("row_type") == "ground_truth_overall"
                and row.get("size_bin") == "overall"
            ),
            None,
        )
        if overall_row is None:
            return default_summary

        return {
            "precision": self._metric_to_float(overall_row.get("precision")),
            "recall": self._metric_to_float(overall_row.get("recall")),
            "mean_absolute_relative_ecd_error": self._metric_to_float(
                overall_row.get("mean_absolute_relative_ecd_error")
            ),
        }

    @staticmethod
    def _metric_to_float(value):
        import numpy as np

        if value in ("", None):
            return np.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan
