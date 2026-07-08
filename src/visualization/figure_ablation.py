"""Five-ablation comparison figures for a single sample."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.paths import ABLATION_CODES
from src.visualization.drawing import load_image_rgb, show_image
from src.visualization.experiment_data import (
    ExperimentContext,
    SampleRecord,
    ablation_image_path,
    load_verification_result,
    verification_results_dir,
)
from src.visualization.publication_style import (
    COLOR_HIGHLIGHT,
    DPI,
    apply_publication_style,
    wrap_text,
)

logger = logging.getLogger(__name__)


def save_ablation_comparison(
    context: ExperimentContext,
    record: SampleRecord,
    output_path: Path,
    *,
    best_ablation_code: str,
) -> bool:
    """Generate Visualization 3: A1–A5 ablation row for one sample."""
    apply_publication_style()
    fig, axes = plt.subplots(1, len(ABLATION_CODES), figsize=(22, 5.5), dpi=DPI)

    for axis, ablation_code in zip(axes, ABLATION_CODES):
        image_path = ablation_image_path(context, ablation_code, record.sample_id)
        image = load_image_rgb(image_path)
        if image is None:
            image = load_image_rgb(
                context.dataset_dir / "images" / f"{record.sample_id}.png"
            )
        if image is None:
            axis.set_facecolor("white")
            axis.text(0.5, 0.5, "Missing", ha="center", va="center")
            axis.set_axis_off()
            continue

        show_image(axis, image, ablation_code)
        if ablation_code == best_ablation_code:
            axis.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=axis.transAxes,
                    fill=False,
                    edgecolor=COLOR_HIGHLIGHT,
                    linewidth=4,
                )
            )
            axis.set_title(f"{ablation_code} (best)", fontsize=11, color=COLOR_HIGHLIGHT, pad=8)

        result = load_verification_result(
            verification_results_dir(context, ablation_code),
            record.sample_id,
        )
        decision = str(result.get("decision", "n/a") or "n/a")
        confidence = record.yolo_confidence
        confidence_text = f"{confidence:.3f}" if confidence is not None else "n/a"
        reasoning = wrap_text(
            str(result.get("visual_reasoning", "") or result.get("confidence_reasoning", "")),
            width=28,
        )
        caption = f"Decision: {decision}\nConfidence: {confidence_text}\n{reasoning}"
        axis.text(
            0.5,
            -0.22,
            caption,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="black",
        )

    fig.suptitle(
        f"{record.sample_id} — Ablation comparison",
        fontsize=12,
        color="black",
        y=1.03,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved ablation comparison: %s", output_path)
    return True
