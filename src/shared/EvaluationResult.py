class EvaluationResult:
    def __init__(
        self,
        iou_scores,
        dice_scores,
        precision_scores=None,
        recall_scores=None,
        size_stratified_metrics=None,
        size_stratified_paths=None,
    ):
        import numpy as np
        self.iou_scores = iou_scores
        self.dice_scores = dice_scores
        self.precision_scores = precision_scores or []
        self.recall_scores = recall_scores or []
        self.size_stratified_metrics = size_stratified_metrics
        self.size_stratified_paths = size_stratified_paths or {}

        self.mean_iou = np.mean(iou_scores)
        self.mean_dice = np.mean(dice_scores)
        self.mean_precision = np.mean(self.precision_scores) if self.precision_scores else np.nan
        self.mean_recall = np.mean(self.recall_scores) if self.recall_scores else np.nan
        self.min_iou = np.min(iou_scores)
        self.min_dice = np.min(dice_scores)
        self.min_precision = np.min(self.precision_scores) if self.precision_scores else np.nan
        self.min_recall = np.min(self.recall_scores) if self.recall_scores else np.nan
        self.max_iou = np.max(iou_scores)
        self.max_dice = np.max(dice_scores)
        self.max_precision = np.max(self.precision_scores) if self.precision_scores else np.nan
        self.max_recall = np.max(self.recall_scores) if self.recall_scores else np.nan

    def __len__(self):
        return len(self.iou_scores)

    def __iter__(self):
        yield self.mean_iou
        yield self.mean_dice
