#!/usr/bin/env python3
"""
Evaluation Protocol v2 tests for LabelMe palm GT extraction.

Unit tests run anywhere. Integration tests use the cluster LabelMe root
(src.paths.RAW_PATCHES_ROOT) and the canonical 5,747-sample index, and skip
when those are unavailable.

    pytest tests/test_gt_palm_bboxes.py -v
    python -m unittest tests.test_gt_palm_bboxes -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.paths import (  # noqa: E402
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    VERIFICATION_DATASET_INDEX_CSV,
)
from src.preprocessing.gt_palm_bboxes import (  # noqa: E402
    EVALUATION_PROTOCOL_VERSION,
    axis_aligned_bbox_from_points,
    extract_gt_palm_bboxes,
    is_palm_label,
)
from src.preprocessing.json_parser import load_json  # noqa: E402

V1_GT_BOXES = 5367
V2_GT_BOXES = 5853
V1_GT_POSITIVE, V1_GT_NEGATIVE = 4685, 1062
V2_GT_POSITIVE, V2_GT_NEGATIVE = 5109, 638
V1_TO_V2_FLIPS = 424


def _v1_extract(json_path: Path) -> list[tuple[float, float, float, float]]:
    """Protocol v1 extractor (exact case-sensitive "palm"), kept for regression only."""
    bboxes = []
    for shape in load_json(json_path).get("shapes", []):
        if isinstance(shape, dict) and shape.get("label") == "palm":
            bbox = axis_aligned_bbox_from_points(shape.get("points", []))
            if bbox is not None:
                bboxes.append(bbox)
    return bboxes


def _write_labelme(directory: Path, shapes: list[dict[str, Any]]) -> Path:
    path = directory / "patch.json"
    path.write_text(
        json.dumps({"shapes": shapes, "imageWidth": 912, "imageHeight": 912}),
        encoding="utf-8",
    )
    return path


class PalmLabelTests(unittest.TestCase):
    def test_protocol_version(self) -> None:
        self.assertEqual(EVALUATION_PROTOCOL_VERSION, "v2")

    def test_accepted_labels(self) -> None:
        for label in ("palm", "Palm", "PALM", " PALM ", "pAlM", "palm\n", "\tPalm "):
            with self.subTest(label=label):
                self.assertTrue(is_palm_label(label))

    def test_rejected_labels(self) -> None:
        for label in ("center", "end", "", " ", "palms", "palm tree", "p alm", None, 1, 0.0,
                      ["palm"], {"label": "palm"}, b"palm"):
            with self.subTest(label=label):
                self.assertFalse(is_palm_label(label))


class BoundingBoxTests(unittest.TestCase):
    def test_rotated_box_axis_aligned_envelope(self) -> None:
        points = [[10.0, 0.0], [20.0, 10.0], [10.0, 20.0], [0.0, 10.0]]
        self.assertEqual(axis_aligned_bbox_from_points(points), (0.0, 0.0, 20.0, 20.0))

    def test_rectangle_two_and_four_points(self) -> None:
        self.assertEqual(axis_aligned_bbox_from_points([[5, 7], [15, 27]]), (5.0, 7.0, 10.0, 20.0))
        four = [[5, 7], [15, 7], [15, 27], [5, 27]]
        self.assertEqual(axis_aligned_bbox_from_points(four), (5.0, 7.0, 10.0, 20.0))

    def test_point_shape_yields_zero_area_box(self) -> None:
        self.assertEqual(axis_aligned_bbox_from_points([[3.5, 4.5]]), (3.5, 4.5, 0.0, 0.0))

    def test_invalid_points(self) -> None:
        self.assertIsNone(axis_aligned_bbox_from_points([]))
        self.assertIsNone(axis_aligned_bbox_from_points([[1.0]]))


class ExtractorTests(unittest.TestCase):
    def _extract(self, shapes: list[Any]) -> list[tuple[float, float, float, float]]:
        with tempfile.TemporaryDirectory() as tmp:
            return extract_gt_palm_bboxes(_write_labelme(Path(tmp), shapes))

    def test_mixed_casing_and_shape_types(self) -> None:
        shapes = [
            {"label": "palm", "shape_type": "rotation",
             "points": [[10, 0], [20, 10], [10, 20], [0, 10]]},
            {"label": "Palm", "shape_type": "rectangle",
             "points": [[100, 100], [150, 100], [150, 180], [100, 180]]},
            {"label": " PALM ", "shape_type": "rectangle", "points": [[1, 2], [3, 4]]},
            {"label": "palm", "shape_type": "point", "points": [[6.5, 7.5]]},
            {"label": "center", "shape_type": "point", "points": [[10, 10]]},
            {"label": "end", "shape_type": "rotation", "points": [[0, 0], [1, 1], [0, 1], [1, 0]]},
            {"label": None, "points": [[0, 0], [1, 1]]},
            "not-a-shape",
        ]
        self.assertEqual(
            self._extract(shapes),
            [
                (0.0, 0.0, 20.0, 20.0),
                (100.0, 100.0, 50.0, 80.0),
                (1.0, 2.0, 2.0, 2.0),
                (6.5, 7.5, 0.0, 0.0),
            ],
        )

    def test_only_label_normalization_differs_from_v1(self) -> None:
        shapes = [
            {"label": "palm", "shape_type": "rotation",
             "points": [[10, 0], [20, 10], [10, 20], [0, 10]]},
            {"label": "palm", "shape_type": "point", "points": [[6.5, 7.5]]},
            {"label": "Palm", "shape_type": "rectangle", "points": [[1, 2], [3, 4]]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_labelme(Path(tmp), shapes)
            v1 = _v1_extract(path)
            v2 = extract_gt_palm_bboxes(path)
        self.assertEqual(v2[: len(v1)], v1)
        self.assertEqual(v2[len(v1):], [(1.0, 2.0, 2.0, 2.0)])


@unittest.skipUnless(RAW_PATCHES_ROOT.is_dir(), "LabelMe Raw_Patches root unavailable")
class RawPatchesIntegrationTests(unittest.TestCase):
    def test_gt_box_counts(self) -> None:
        json_files = sorted(RAW_PATCHES_ROOT.glob("*.json"))
        self.assertEqual(len(json_files), 880)
        v1 = sum(len(_v1_extract(path)) for path in json_files)
        v2 = sum(len(extract_gt_palm_bboxes(path)) for path in json_files)
        self.assertEqual(v1, V1_GT_BOXES)
        self.assertEqual(v2, V2_GT_BOXES)


@unittest.skipUnless(
    RAW_PATCHES_ROOT.is_dir()
    and VERIFICATION_DATASET_INDEX_CSV.is_file()
    and PREDICTIONS_FULL_JSON.is_file(),
    "Raw_Patches, verification index, or YOLO predictions unavailable",
)
class VerificationLabelIntegrationTests(unittest.TestCase):
    def test_canonical_5747_labels(self) -> None:
        import pandas as pd

        import evaluate_verification_against_groundtruth as evaluator
        from src.yolo.predictions_io import group_predictions_by_image, load_predictions

        index_df = pd.read_csv(VERIFICATION_DATASET_INDEX_CSV)
        self.assertEqual(len(index_df), 5747)
        predictions = group_predictions_by_image(load_predictions(PREDICTIONS_FULL_JSON))

        v2 = evaluator.compute_greedy_matches_for_index(
            index_df, predictions, RAW_PATCHES_ROOT, {}, evaluator.IOU_THRESHOLD,
        )
        original = evaluator.extract_gt_palm_bboxes
        evaluator.extract_gt_palm_bboxes = _v1_extract
        try:
            v1 = evaluator.compute_greedy_matches_for_index(
                index_df, predictions, RAW_PATCHES_ROOT, {}, evaluator.IOU_THRESHOLD,
            )
        finally:
            evaluator.extract_gt_palm_bboxes = original

        ids = list(index_df["sample_id"].astype(str))
        v1_pos = sum(v1[i].matched_gt for i in ids)
        v2_pos = sum(v2[i].matched_gt for i in ids)
        self.assertEqual((v1_pos, len(ids) - v1_pos), (V1_GT_POSITIVE, V1_GT_NEGATIVE))
        self.assertEqual((v2_pos, len(ids) - v2_pos), (V2_GT_POSITIVE, V2_GT_NEGATIVE))

        lost = [i for i in ids if v1[i].matched_gt and not v2[i].matched_gt]
        gained = [i for i in ids if not v1[i].matched_gt and v2[i].matched_gt]
        self.assertEqual(lost, [])
        self.assertEqual(len(gained), V1_TO_V2_FLIPS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
