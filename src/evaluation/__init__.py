"""Detection and verification evaluation utilities."""

from src.evaluation.gt_matching import (
    GreedyGtMatch,
    best_gt_overlap,
    greedy_match_bboxes_to_gt,
    greedy_match_detections_to_gt,
)

__all__ = [
    "GreedyGtMatch",
    "best_gt_overlap",
    "greedy_match_bboxes_to_gt",
    "greedy_match_detections_to_gt",
]
