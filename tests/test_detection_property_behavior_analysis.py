#!/usr/bin/env python3
"""
Tests for the ground-truth-independent detection-property behavior analysis
(scripts/analysis/detection_property_behavior_analysis.py,
docs/DETECTION_PROPERTY_BEHAVIOR_ANALYSIS.md).

Unit tests use small synthetic inputs and run anywhere. Integration tests check the
generated outputs under outputs/detection_property_behavior_analysis/ and skip when
they have not been generated.

    python -m unittest tests.test_detection_property_behavior_analysis -v
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

import a1_a5_behavior_analysis as behavior  # noqa: E402
import detection_property_behavior_analysis as dp  # noqa: E402

R, U, X = behavior.LABELS


def synthetic_index() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        # sample_id, image, x, y, w, h, conf
        ("s1", "img_a_1", 0.0, 10.0, 20.0, 10.0, 0.9),     # touches left edge, elongated 2:1
        ("s2", "img_a_1", 40.0, 40.0, 20.0, 20.0, 0.6),    # centered square
        ("s3", "img_b_2", 70.0, 80.0, 30.0, 20.0, 0.75),   # touches right and bottom edges
    ]
    index = pd.DataFrame(rows, columns=["sample_id", "image_name", "bbox_x", "bbox_y",
                                        "bbox_width", "bbox_height", "confidence"])
    index["bbox_area"] = index["bbox_width"] * index["bbox_height"]
    index["center_x"] = index["bbox_x"] + index["bbox_width"] / 2
    index["center_y"] = index["bbox_y"] + index["bbox_height"] / 2
    sizes = pd.DataFrame({"sample_id": index["sample_id"], "image_width": 100, "image_height": 100})
    return index, sizes


class PropertyTests(unittest.TestCase):
    def setUp(self):
        index, sizes = synthetic_index()
        self.props = dp.build_property_table(index, sizes, expected_n=None).set_index("sample_id")

    def test_derived_values(self):
        s1, s2, s3 = (self.props.loc[s] for s in ("s1", "s2", "s3"))
        self.assertAlmostEqual(s1["norm_area"], 200 / 10000)
        self.assertAlmostEqual(s1["aspect_ratio"], 2.0)
        self.assertAlmostEqual(s1["elongation"], math.log(2.0))
        self.assertAlmostEqual(s2["elongation"], 0.0)
        self.assertAlmostEqual(s2["center_x_norm"], 0.5)
        self.assertAlmostEqual(s2["center_y_norm"], 0.5)
        self.assertEqual((s1["touches_edge"], s2["touches_edge"], s3["touches_edge"]), (1, 0, 1))
        self.assertAlmostEqual(s2["edge_gap_norm"], 0.40)
        self.assertAlmostEqual(s2["center_edge_dist_norm"], 0.5)
        self.assertAlmostEqual(s3["center_edge_dist_norm"], min(85, 90, 15, 10) / 100)
        self.assertEqual(s1["detections_in_image"], 2)
        self.assertEqual(s3["detections_in_image"], 1)
        self.assertEqual(s1["parent_frame"], "img_a")
        # A5 crop: padded by 15 px and clamped to the image.
        self.assertEqual((s1["a5_crop_width"], s1["a5_crop_height"]), (35, 35))
        self.assertEqual((s3["a5_crop_width"], s3["a5_crop_height"]), (45, 35))
        self.assertAlmostEqual(s2["a5_upscale"], 512 / 50)

    def test_invalid_properties_rejected(self):
        index, sizes = synthetic_index()
        base = dp.build_property_table(index, sizes, expected_n=None)
        cases = {
            "duplicate": pd.concat([base, base.iloc[[0]]]),
            "missing": base.assign(confidence=[0.9, np.nan, 0.7]),
            "low confidence": base.assign(confidence=[0.4, 0.6, 0.7]),
            "outside image": base.assign(bbox_x=[90.0, 40.0, 70.0]),
            "bad area": base.assign(bbox_area=[1.0, 400.0, 600.0]),
        }
        for name, frame in cases.items():
            with self.subTest(name), self.assertRaises(AssertionError):
                dp.validate_property_table(frame, expected_n=None)

    def test_forbidden_columns(self):
        for column in ("gt_label", "iou", "semantic_palm", "is_tp", "matched_gt"):
            with self.subTest(column), self.assertRaises(AssertionError):
                dp.assert_no_forbidden_columns([column])
        dp.assert_no_forbidden_columns(["confidence", "norm_area", "toward_rejection", "touches_edge"])


class OutcomeTests(unittest.TestCase):
    def make_decisions(self) -> pd.DataFrame:
        a1 = [R, R, R, U, U, X, X, R]
        a2 = [R, R, U, U, X, X, R, R]
        a5 = [R, U, X, R, X, U, X, U]
        frame = pd.DataFrame({"sample_id": [f"s{i}" for i in range(len(a1))]})
        for key in dp.MODELS:
            for code, values in zip(behavior.CONDITIONS, (a1, a2, a2, a2, a5)):
                frame[f"{key}__{code}"] = values
        return frame

    def test_outcomes_match_behavior_definitions(self):
        decisions = self.make_decisions()
        outcomes = dp.build_outcomes(decisions)
        pairwise = []
        for key in dp.MODELS:
            for a, b in dp.COMPARISONS:
                row = behavior.transition_summary(
                    decisions[f"{key}__{a}"].set_axis(decisions["sample_id"]),
                    decisions[f"{key}__{b}"].set_axis(decisions["sample_id"]))
                pairwise.append({"model_key": key, "comparison": f"{a}->{b}", **row})
        pairwise = pd.DataFrame(pairwise)
        dp.validate_against_behavior(outcomes, pairwise, expected_n=len(decisions))
        tampered = pairwise.copy()
        tampered.loc[0, "reliable_to_uncertain"] += 1
        with self.assertRaises(AssertionError):
            dp.validate_against_behavior(outcomes, tampered, expected_n=len(decisions))

    def test_event_masks(self):
        outcomes = dp.build_outcomes(self.make_decisions())
        group = outcomes[(outcomes["model_key"] == "qwen3_vl") & (outcomes["comparison"] == "A1->A5")]
        self.assertEqual(group["transition"].tolist(), ["R->R", "R->U", "R->Ur", "U->R", "U->Ur", "Ur->U", "Ur->Ur", "R->U"])
        den, num = dp.event_masks(group, "R->U")
        self.assertEqual((den.sum(), num.sum()), (4, 2))
        den, num = dp.event_masks(group, "into_U")
        self.assertEqual((den.sum(), num.sum()), (6, 3))
        den, num = dp.event_masks(group, "toward_rejection")
        self.assertEqual((den.sum(), num.sum()), (8, 4))
        den, num = dp.event_masks(group, "toward_acceptance")
        self.assertEqual(num.sum(), 2)
        den, num = dp.event_masks(group, "changed")
        self.assertEqual(num.sum(), 6)


class BootstrapTests(unittest.TestCase):
    def test_weights_are_whole_cluster_multiplicities(self):
        w = dp.cluster_bootstrap_weights(7, n_boot=50, seed=3)
        self.assertEqual(w.shape, (50, 7))
        self.assertTrue((w.sum(axis=1) == 7).all())
        self.assertTrue(np.issubdtype(w.dtype, np.integer))
        np.testing.assert_array_equal(w, dp.cluster_bootstrap_weights(7, n_boot=50, seed=3))

    def test_weighted_statistics_equal_explicit_image_resample(self):
        rng = np.random.default_rng(0)
        codes = np.repeat(np.arange(6), [1, 3, 2, 5, 1, 4])
        values = rng.normal(size=codes.size)
        event = rng.random(codes.size) < 0.4
        den = np.ones(codes.size, dtype=bool)
        weights = dp.cluster_bootstrap_weights(6, n_boot=20, seed=11)
        ratio = dp.boot_ratio(weights, dp.cluster_counts(codes, event, 6), dp.cluster_counts(codes, den, 6))
        medians = dp.boot_weighted_median(values, codes, weights)
        for b in range(weights.shape[0]):
            # Explicit resample: every detection of image j appears weights[b, j] times.
            take = np.concatenate([np.flatnonzero(codes == j).repeat(1) for j in range(6)
                                   for _ in range(weights[b, j])])
            self.assertAlmostEqual(ratio[b], event[take].mean())
            explicit = np.sort(values[take])
            self.assertAlmostEqual(medians[b], explicit[math.ceil(len(explicit) / 2) - 1])

    def test_percentile_ci_requires_mostly_finite(self):
        self.assertTrue(all(math.isnan(v) for v in dp.percentile_ci(np.array([np.nan] * 10 + [1.0]))))
        lo, hi = dp.percentile_ci(np.arange(1000, dtype=float))
        self.assertLess(lo, hi)


class SafetyTests(unittest.TestCase):
    def test_output_dir_guard(self):
        for path in (*behavior.PROTECTED_DIRS, dp.BEHAVIOR_DIR, dp.BEHAVIOR_DIR / "sub", dp.OUTPUTS_DIR):
            with self.subTest(str(path)), self.assertRaises(SystemExit):
                dp.assert_safe_output_dir(path)
        dp.assert_safe_output_dir(dp.DEFAULT_OUT_DIR)

    def test_inputs_are_gt_free(self):
        allowed = {behavior.VERIFICATION_DATASET_INDEX_CSV, dp.PER_DETECTION_CSV, dp.PAIRWISE_CSV}
        self.assertEqual(set(dp.INPUT_FILES), allowed)
        source = Path(dp.__file__).read_text(encoding="utf-8")
        for token in ("RAW_PATCHES_ROOT", "gt_palm_bboxes", "semantic_gt", "EVALUATION_PROTOCOL_V2_ROOT",
                      "evaluation_protocol_v2", "semantic_gt_review", "gt_matching", "yolo_gt"):
            self.assertNotIn(token, source)
        self.assertNotIn("labelme", {c.lower() for c in dp.INDEX_COLUMNS})


OUT = dp.DEFAULT_OUT_DIR


@unittest.skipUnless((OUT / "analysis_info.json").exists(), "detection-property outputs not generated")
class GeneratedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.props = pd.read_csv(OUT / "property_table.csv")
        cls.outcomes = pd.read_csv(OUT / "detection_outcomes.csv")
        cls.contrasts = pd.read_csv(OUT / "event_contrasts.csv")
        cls.bins = pd.read_csv(OUT / "event_rates_by_bin.csv")
        cls.clusters = pd.read_csv(OUT / "clustering_summary.csv")
        cls.info = json.loads((OUT / "analysis_info.json").read_text(encoding="utf-8"))

    def test_property_table(self):
        dp.validate_property_table(self.props)
        self.assertEqual(self.props["sample_id"].tolist(),
                         behavior.load_cohort(behavior.VERIFICATION_DATASET_INDEX_CSV))
        self.assertTrue(self.props["center_x_norm"].between(0, 1).all())
        self.assertTrue(self.props["center_y_norm"].between(0, 1).all())

    def test_transition_counts_match_behavior_analysis(self):
        dp.validate_against_behavior(self.outcomes, pd.read_csv(dp.PAIRWISE_CSV))
        self.assertEqual(len(self.outcomes), dp.EXPECTED_N * len(dp.MODELS) * len(dp.COMPARISONS))
        self.assertEqual(set(self.outcomes["sample_id"]), set(self.props["sample_id"]))

    def test_overall_event_counts(self):
        overall = self.contrasts[self.contrasts["property"] == "(overall)"].set_index(
            ["model_key", "comparison", "event"])
        for (key, comparison), group in self.outcomes.groupby(["model_key", "comparison"]):
            for event in dp.events_for(key):
                den, num = dp.event_masks(group, event)
                row = overall.loc[(key, comparison, event)]
                self.assertEqual((row["n_denominator"], row["n_event"]), (den.sum(), num.sum()))
                self.assertLessEqual(row["ci_low"], row["rate"])
                self.assertGreaterEqual(row["ci_high"], row["rate"])

    def test_bins_partition_denominators(self):
        overall = self.contrasts[self.contrasts["property"] == "(overall)"].set_index(
            ["model_key", "comparison", "event"])["n_denominator"]
        sums = self.bins.groupby(["model_key", "comparison", "event", "property"])["n_denominator"].sum()
        for (key, comparison, event, _), total in sums.items():
            self.assertEqual(total, overall.loc[(key, comparison, event)])

    def test_clustering(self):
        image = self.clusters.set_index("level").loc["source_image"]
        self.assertEqual(image["clusters"], self.props["image_name"].nunique())
        self.assertEqual(image["detections"], dp.EXPECTED_N)
        self.assertEqual(self.info["bootstrap"]["seed"], dp.SEED)
        self.assertEqual(self.info["bootstrap"]["unit"], "source image (image_name)")

    def test_no_gt_columns_in_outputs(self):
        for path in OUT.glob("*.csv"):
            columns = pd.read_csv(path, nrows=0, index_col=False).columns
            dp.assert_no_forbidden_columns([c for c in columns if not str(c).startswith("Unnamed")])

    def test_inputs_unchanged_since_run(self):
        for rel, digest in self.info["input_sha256"].items():
            self.assertEqual(dp.sha256_file(PROJECT_ROOT / rel), digest, rel)


if __name__ == "__main__":
    unittest.main()
