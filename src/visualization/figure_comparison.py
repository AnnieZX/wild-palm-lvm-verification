"""Side-by-side verification comparison figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from src.preprocessing.verification_dataset import resolve_patch_image
from src.visualization.drawing import (
    add_bbox_legend,
    draw_bbox,
    load_image_rgb,
    render_on_image,
    show_image,
)
from src.visualization.experiment_data import (
    ExperimentContext,
    SampleRecord,
    overlay_dataset_image_path,
)
from src.visualization.publication_style import (
    DPI,
    apply_publication_style,
    wrap_text,
)

logger = logging.getLogger(__name__)


def save_verification_comparison(
    context: ExperimentContext,
    record: SampleRecord,
    output_path: Path,
) -> bool:
    """Generate Visualization 2: multi-panel verification comparison."""
    original_path = resolve_patch_image(context.images_root, record.image_name)
    overlay_path = overlay_dataset_image_path(context, record.sample_id)

    original = load_image_rgb(original_path) if original_path else None
    overlay = load_image_rgb(overlay_path)

    if original is None and overlay is None:
        logger.warning("No images available for comparison: %s", record.sample_id)
        return False

    if original is None and overlay is not None:
        original = overlay.copy()
    if overlay is None and original is not None:
        overlay = original.copy()

    assert original is not None and overlay is not None

    overlay_gt = render_on_image(overlay, gt_bbox=record.gt_bbox, draw_yolo=False)
    overlay_yolo = render_on_image(overlay, yolo_bbox=record.yolo_bbox, draw_gt=False)

    apply_publication_style()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), dpi=DPI)

    show_image(axes[0], original, "Original")
    show_image(axes[1], overlay, "Overlay")
    show_image(axes[2], overlay_gt, "Overlay + GT")
    show_image(axes[3], overlay_yolo, "Overlay + YOLO")

    if record.gt_bbox is not None:
        draw_bbox(axes[2], record.gt_bbox, (0.15, 0.72, 0.25))
    if record.yolo_bbox is not None:
        draw_bbox(axes[3], record.yolo_bbox, (0.86, 0.18, 0.18))

    axes[4].set_axis_off()
    axes[4].set_facecolor("white")
    confidence = (
        f"{record.yolo_confidence:.3f}"
        if record.yolo_confidence is not None
        else "n/a"
    )
    reasoning = wrap_text(
        record.visual_reasoning or record.confidence_reasoning,
        width=34,
    )
    panel_text = (
        f"Decision:\n{record.decision or 'n/a'}\n\n"
        f"Model:\n{context.model_label}\n\n"
        f"Confidence:\n{confidence}\n\n"
        f"Reasoning:\n{reasoning}"
    )
    axes[4].text(
        0.05,
        0.95,
        panel_text,
        transform=axes[4].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="black",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    axes[4].set_title("Verification result", fontsize=11, pad=8)

    fig.suptitle(record.sample_id, fontsize=12, color="black", y=1.02)
    add_bbox_legend(fig, y=-0.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved comparison figure: %s", output_path)
    return True
