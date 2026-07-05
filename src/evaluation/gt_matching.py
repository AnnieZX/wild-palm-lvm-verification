"""Greedy one-to-one matching between detections and ground-truth boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.yolo.predictions_io import iou_xywh


@dataclass(frozen=True)
class GreedyGtMatch:
    """GT assignment for one detection after greedy matching."""

    gt_bbox: tuple[float, float, float, float] | None
    gt_index: int | None
    max_iou: float
    matched_gt: bool


def greedy_match_detections_to_gt(
    detections: list[dict[str, Any]],
    gt_bboxes: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """
    Greedy one-to-one matching between detections and GT boxes.

    Pairs are considered in descending IoU order (Pascal VOC / COCO style).
    Each detection and each GT box may be matched at most once.
    """
    candidates: list[tuple[float, int, int]] = []
    for det_index, detection in enumerate(detections):
        det_bbox = detection["bbox"]
        for gt_index, gt_bbox in enumerate(gt_bboxes):
            overlap = iou_xywh(det_bbox, gt_bbox)
            if overlap >= iou_threshold:
                candidates.append((overlap, det_index, gt_index))

    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_detections: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for overlap, det_index, gt_index in candidates:
        if det_index in matched_detections or gt_index in matched_gt:
            continue
        matched_detections.add(det_index)
        matched_gt.add(gt_index)
        matches.append((det_index, gt_index, overlap))

    return matches, matched_detections, matched_gt


def best_gt_overlap(
    detection_bbox: tuple[float, float, float, float],
    gt_bboxes: list[tuple[float, float, float, float]],
) -> tuple[int | None, float, tuple[float, float, float, float] | None]:
    """Return the highest-IoU GT box for one detection, ignoring assignment."""
    best_iou = 0.0
    best_index: int | None = None
    best_bbox: tuple[float, float, float, float] | None = None

    for gt_index, gt_bbox in enumerate(gt_bboxes):
        overlap = iou_xywh(detection_bbox, gt_bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_index = gt_index
            best_bbox = gt_bbox

    return best_index, best_iou, best_bbox


def greedy_match_bboxes_to_gt(
    detection_bboxes: list[tuple[float, float, float, float]],
    gt_bboxes: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> list[GreedyGtMatch]:
    """
    Assign each detection bbox to at most one GT box using greedy matching.

    Matched detections receive their assigned GT and IoU. Unmatched detections
    still report the best available GT overlap for debugging.
    """
    detections = [{"bbox": bbox} for bbox in detection_bboxes]
    matches, _, _ = greedy_match_detections_to_gt(detections, gt_bboxes, iou_threshold)
    match_by_detection = {
        det_index: (gt_index, overlap)
        for det_index, gt_index, overlap in matches
    }

    results: list[GreedyGtMatch] = []
    for det_index, detection_bbox in enumerate(detection_bboxes):
        if det_index in match_by_detection:
            gt_index, overlap = match_by_detection[det_index]
            results.append(
                GreedyGtMatch(
                    gt_bbox=gt_bboxes[gt_index],
                    gt_index=gt_index,
                    max_iou=overlap,
                    matched_gt=True,
                )
            )
            continue

        gt_index, overlap, gt_bbox = best_gt_overlap(detection_bbox, gt_bboxes)
        results.append(
            GreedyGtMatch(
                gt_bbox=gt_bbox,
                gt_index=gt_index,
                max_iou=overlap,
                matched_gt=False,
            )
        )

    return results
