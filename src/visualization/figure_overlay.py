"""YOLO vs ground-truth overlay figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from src.preprocessing.verification_dataset import resolve_patch_image
from src.utils.labelme_paths import resolve_labelme_json
from src.visualization.drawing import (
    add_bbox_legend,
    annotate_points_legend,
    draw_bbox,
    draw_center,
    draw_endpoints,
    load_image_rgb,
    show_image,
)
from src.visualization.experiment_data import ExperimentContext, SampleRecord
from src.visualization.gt_geometry import extract_matched_palm_geometry
from src.visualization.publication_style import DPI, apply_publication_style

logger = logging.getLogger(__name__)


def save_yolo_gt_overlay(
    context: ExperimentContext,
    record: SampleRecord,
    output_path: Path,
) -> bool:
    """Generate Visualization 1: YOLO vs GT overlay on the original patch."""
    image_path = resolve_patch_image(context.images_root, record.image_name)
    if image_path is None:
        logger.warning("Missing raw patch for %s", record.sample_id)
        return False

    image = load_image_rgb(image_path)
    if image is None:
        logger.warning("Could not read image for %s", record.sample_id)
        return False

    json_path = resolve_labelme_json(context.annotations_root, record.image_name)
    reference_bbox = record.gt_bbox if record.gt_bbox is not None else record.yolo_bbox
    geometry = extract_matched_palm_geometry(json_path, reference_bbox) if json_path else None
    gt_bbox = record.gt_bbox or (geometry.gt_bbox if geometry else None)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=DPI)
    show_image(ax, image)

    if gt_bbox is not None:
        draw_bbox(ax, gt_bbox, (0.15, 0.72, 0.25), "GT")
    if record.yolo_bbox is not None:
        draw_bbox(ax, record.yolo_bbox, (0.86, 0.18, 0.18), "YOLO")
    if geometry and geometry.center is not None:
        draw_center(ax, geometry.center)
    if geometry:
        draw_endpoints(ax, geometry.endpoints)

    confidence = (
        f"{record.yolo_confidence:.3f}"
        if record.yolo_confidence is not None
        else "n/a"
    )
    title = (
        f"{record.sample_id}\n"
        f"YOLO conf: {confidence}   IoU: {record.max_iou:.3f}   "
        f"LVM: {record.decision or 'n/a'}   GT label: {record.gt_label}"
    )
    fig.suptitle(title, fontsize=12, color="black", y=0.98)
    add_bbox_legend(fig, y=0.02)
    annotate_points_legend(fig, y=0.05)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved overlay figure: %s", output_path)
    return True
