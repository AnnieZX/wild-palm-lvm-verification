#!/usr/bin/env python3
"""
Tests for the semantic-GT evaluation path (docs/SEMANTIC_GT_EVALUATION.md).

Unit tests use a tiny synthetic dataset with hand-computed TP/TN/FP/FN and run anywhere.
Integration tests use the canonical outputs and skip when they are unavailable. The
protected-output hash check is slow (~2 min) and runs only with
SEMANTIC_GT_VERIFY_SNAPSHOT=1.

    python -m unittest tests.test_semantic_gt_evaluation -v
    SEMANTIC_GT_VERIFY_SNAPSHOT=1 python -m unittest tests.test_semantic_gt_evaluation -v
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import compute_verification_metrics as metrics_mod  # noqa: E402
import evaluate_semantic_gt as sem_eval  # noqa: E402
from src.evaluation.semantic_gt import (  # noqa: E402
    EXPECTED_TOTAL,
    EXPECTED_V2_NEGATIVE,
    EXPECTED_V2_POSITIVE,
    INHERITED_POSITIVE,
    MANUALLY_REVIEWED_NEGATIVE_POOL,
    PROTOCOL_V2_REFERENCE_CSV,
    REFERENCE_COLUMNS,
    REVIEW_MANIFEST_COLUMNS,
    SEMANTIC_GT_ROOT,
    SEMANTIC_LABELS,
    SEMANTIC_REVIEW_MANIFEST_CSV,
    UNREVIEWED_NEGATIVE_POOL,
    SemanticGtError,
    build_semantic_gt_table,
    load_reference,
    read_str_csv,
    semantic_status,
    validate_reference,
    validate_review_manifest,
)
from src.paths import EVALUATION_PROTOCOL_V2_ROOT, OUTPUTS_DIR, VERIFICATION_DATASET_INDEX_CSV  # noqa: E402

BASELINE_SNAPSHOT = SEMANTIC_GT_ROOT / "provenance" / "protected_outputs_baseline.json"

# Synthetic world: s1-s5 Protocol v2 GT+, s6-s8 GT-.
V2_GT = {"s1": True, "s2": True, "s3": True, "s4": True, "s5": True,
         "s6": False, "s7": False, "s8": False}
DECISIONS = {"s1": "Reliable", "s2": "Reliable", "s3": "Unreliable", "s4": "Uncertain",
             "s5": "Reliable", "s6": "Unreliable", "s7": "Unreliable", "s8": "Reliable"}
REVIEW = {"s6": ("palm", "missing_annotation"), "s7": ("non_palm", "true_non_palm"),
          "s8": ("ambiguous", "ambiguous")}


def synthetic_reference() -> pd.DataFrame:
    rows = []
    for i, (sample_id, matched) in enumerate(V2_GT.items()):
        rows.append({
            "sample_id": sample_id, "image_id": f"img_{i // 4}", "image_path": f"/x/img_{i // 4}.png",
            "yolo_bbox_xywh": f"[{i}.0, 0.0, 10.0, 10.0]", "yolo_confidence": "0.7",
            "max_iou": "0.8" if matched else "0.1", "matched_gt_index": "0" if matched else "",
            "nearest_gt_index": "0", "nearest_gt_bbox_xywh": "[0.0, 0.0, 10.0, 10.0]",
            "nearest_gt_owner_sample_id": "",
            "original_protocol_v2_gt": "positive" if matched else "negative",
        })
    return pd.DataFrame(rows, columns=REFERENCE_COLUMNS)


def synthetic_manifest(reference: pd.DataFrame, review: dict[str, tuple[str, str]]) -> pd.DataFrame:
    negatives = reference[reference["original_protocol_v2_gt"] == "negative"].reset_index(drop=True)
    manifest = pd.DataFrame({c: negatives[c] for c in REVIEW_MANIFEST_COLUMNS if c in negatives})
    manifest.insert(0, "review_index", [str(i + 1) for i in range(len(manifest))])
    manifest["iou_ge_050_gt_taken_by_other"] = "false"
    manifest["audit_panel_path"] = ""
    manifest["semantic_gt"] = [review.get(s, ("", ""))[0] for s in manifest["sample_id"]]
    manifest["review_reason"] = [review.get(s, ("", ""))[1] for s in manifest["sample_id"]]
    manifest["reviewer_notes"] = ""
    return manifest[REVIEW_MANIFEST_COLUMNS]


def write_synthetic_run(root: Path, reference: pd.DataFrame) -> argparse.Namespace:
    """Create v2_root/m/e/A1/{A1_evaluation.csv,A1_metrics.json} and verification_root/m/e/A1/sample_*.json."""
    v2_dir = root / "v2" / "m" / "e" / "A1"
    pred_dir = root / "verification" / "m" / "e" / "A1"
    v2_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    eval_df = pd.DataFrame({
        "image_name": reference["image_id"], "sample_id": reference["sample_id"], "ablation": "A1",
        "yolo_bbox": reference["yolo_bbox_xywh"], "gt_bbox": "", "max_iou": reference["max_iou"],
        "matched_gt": [V2_GT[s] for s in reference["sample_id"]],
        "yolo_confidence": reference["yolo_confidence"],
        "verification_label": [DECISIONS[s] for s in reference["sample_id"]],
    })
    eval_df.to_csv(v2_dir / "A1_evaluation.csv", index=False)
    metrics = metrics_mod.compute_metrics(pd.read_csv(v2_dir / "A1_evaluation.csv"), "A1")
    (v2_dir / "A1_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    for sample_id, decision in DECISIONS.items():
        (pred_dir / f"sample_{sample_id}.json").write_text(
            json.dumps({"sample_id": sample_id, "decision": decision}), encoding="utf-8")
    return argparse.Namespace(v2_root=root / "v2", verification_root=root / "verification")


def find_row(rows: list[dict], gt_definition: str, scope_prefix: str) -> dict:
    matches = [r for r in rows if r["gt_definition"] == gt_definition
               and r["scope"].startswith(scope_prefix)]
    assert len(matches) == 1, matches
    return matches[0]


class ManifestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = synthetic_reference()
        self.manifest = synthetic_manifest(self.reference, REVIEW)

    def assert_invalid(self, manifest: pd.DataFrame, fragment: str) -> None:
        with self.assertRaises(SemanticGtError) as ctx:
            validate_review_manifest(manifest, self.reference)
        self.assertIn(fragment, str(ctx.exception))

    def test_valid_and_blank_manifests_pass(self) -> None:
        validate_review_manifest(self.manifest, self.reference)
        validate_review_manifest(synthetic_manifest(self.reference, {}), self.reference)

    def test_duplicate_ids_rejected(self) -> None:
        self.assert_invalid(pd.concat([self.manifest, self.manifest.iloc[[0]]]), "duplicate sample_id")

    def test_unknown_id_rejected(self) -> None:
        bad = self.manifest.copy()
        bad.loc[0, "sample_id"] = "s999"
        self.assert_invalid(bad, "not in canonical detection set")

    def test_positive_row_rejected(self) -> None:
        bad = self.manifest.copy()
        bad.loc[0, "sample_id"] = "s1"
        self.assert_invalid(bad, "not Protocol v2 GT-")

    def test_missing_negative_rejected(self) -> None:
        self.assert_invalid(self.manifest.iloc[1:], "missing from manifest")

    def test_disallowed_label_rejected(self) -> None:
        for value in ("negative", "unmatched", "0", "maybe"):
            bad = self.manifest.copy()
            bad.loc[0, "semantic_gt"] = value
            self.assert_invalid(bad, "semantic_gt values not allowed")

    def test_inconsistent_reason_rejected(self) -> None:
        for label, reason in (("palm", "true_non_palm"), ("non_palm", "missing_annotation"),
                              ("palm", "matched_existing_gt"), ("palm", ""), ("", "other")):
            bad = self.manifest.copy()
            bad.loc[0, ["semantic_gt", "review_reason"]] = [label, reason]
            with self.assertRaises(SemanticGtError):
                validate_review_manifest(bad, self.reference)

    def test_identity_column_edit_rejected(self) -> None:
        bad = self.manifest.copy()
        bad.loc[0, "yolo_bbox_xywh"] = "[1, 2, 3, 4]"
        self.assert_invalid(bad, "yolo_bbox_xywh differs")

    def test_case_and_whitespace_normalized(self) -> None:
        manifest = self.manifest.copy()
        manifest.loc[0, ["semantic_gt", "review_reason"]] = [" Palm ", "Missing_Annotation"]
        self.assertEqual(validate_review_manifest(manifest, self.reference).loc[0, "semantic_gt"], "palm")

    def test_reference_count_check(self) -> None:
        with self.assertRaises(SemanticGtError):
            validate_reference(self.reference)
        validate_reference(self.reference, expected_counts=(5, 3))


class SemanticTableTests(unittest.TestCase):
    def test_provenance(self) -> None:
        reference = synthetic_reference()
        table = build_semantic_gt_table(reference, synthetic_manifest(reference, {"s6": REVIEW["s6"]}))
        by_id = table.set_index("sample_id")
        for s in ("s1", "s2", "s3", "s4", "s5"):
            self.assertEqual(by_id.loc[s, "semantic_gt"], "palm")
            self.assertEqual(by_id.loc[s, "review_reason"], "matched_existing_gt")
            self.assertEqual(by_id.loc[s, "label_provenance"], INHERITED_POSITIVE)
        self.assertEqual(by_id.loc["s6", "label_provenance"], MANUALLY_REVIEWED_NEGATIVE_POOL)
        for s in ("s7", "s8"):
            self.assertEqual(by_id.loc[s, "semantic_gt"], "")
            self.assertEqual(by_id.loc[s, "label_provenance"], UNREVIEWED_NEGATIVE_POOL)
        status = semantic_status(table)
        self.assertFalse(status["semantic_gt_complete"])
        self.assertEqual(status["semantic_gt_status"], "semantic GT incomplete")
        self.assertEqual((status["n_unreviewed"], status["n_negative_pool_reviewed"]), (2, 1))
        self.assertEqual(list(table["sample_id"]), list(reference["sample_id"]))


class SemanticMetricTests(unittest.TestCase):
    def test_hand_computed_metrics(self) -> None:
        # Semantic evaluable s1-s7 (s8 ambiguous). s4 Uncertain excluded.
        # TP: s1 s2 s5; FN: s3 s6; TN: s7; FP: none.
        df = pd.DataFrame({
            "sample_id": list(DECISIONS), "verification_label": list(DECISIONS.values()),
            "semantic_gt": ["palm"] * 6 + ["non_palm", "ambiguous"],
            "max_iou": 0.5, "yolo_confidence": 0.7,
        })
        m = sem_eval.semantic_binary_metrics(df, "A1")
        self.assertEqual((m["true_positive"], m["true_negative"], m["false_positive"],
                          m["false_negative"]), (3, 1, 0, 2))
        self.assertAlmostEqual(m["accuracy"], round(4 / 6, 4))
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 0.6)
        self.assertAlmostEqual(m["specificity"], 1.0)
        self.assertAlmostEqual(m["f1"], 0.75)
        self.assertEqual((m["reliable_count"], m["uncertain_count"], m["unreliable_count"]), (3, 1, 3))

    def test_ambiguous_and_unreviewed_excluded(self) -> None:
        df = pd.DataFrame({
            "verification_label": ["Reliable", "Unreliable", "Unreliable", "Reliable"],
            "semantic_gt": ["palm", "non_palm", "", "ambiguous"],
            "max_iou": 0.5, "yolo_confidence": 0.7,
        })
        m = sem_eval.semantic_binary_metrics(df, "A1")
        self.assertEqual((m["true_positive"], m["true_negative"], m["false_positive"],
                          m["false_negative"]), (1, 1, 0, 0))

    def test_zero_denominator_is_undefined(self) -> None:
        df = pd.DataFrame({"verification_label": ["Reliable"], "semantic_gt": ["palm"],
                           "max_iou": 0.5, "yolo_confidence": 0.7})
        m = sem_eval.semantic_binary_metrics(df, "A1")
        self.assertTrue(math.isnan(m["specificity"]))
        self.assertTrue(math.isnan(m["balanced_accuracy"]))


class SyntheticEndToEndTests(unittest.TestCase):
    def run_synthetic(self, review: dict[str, tuple[str, str]]) -> dict:
        reference = synthetic_reference()
        with tempfile.TemporaryDirectory() as tmp:
            args = write_synthetic_run(Path(tmp), reference)
            table = build_semantic_gt_table(reference, synthetic_manifest(reference, review))
            run_dir = args.v2_root / "m" / "e" / "A1"
            return sem_eval.evaluate_run(run_dir, args, reference, table, semantic_status(table))

    def test_complete_review(self) -> None:
        rows = self.run_synthetic(REVIEW)["comparison"]
        v2 = find_row(rows, "protocol_v2_annotation_alignment", "all_5747")
        self.assertEqual((v2["tp"], v2["tn"], v2["fp"], v2["fn"]), (3, 2, 1, 1))
        sem = find_row(rows, "semantic_object_presence", "all_detections")
        self.assertEqual((sem["tp"], sem["tn"], sem["fp"], sem["fn"]), (3, 1, 0, 2))
        self.assertEqual(sem["n_scope_rows"], 7)
        self.assertEqual(sem["semantic_gt_status"], "complete")
        v2_same = find_row(rows, "protocol_v2_annotation_alignment", "all_detections")
        self.assertEqual((v2_same["tp"], v2_same["tn"], v2_same["fp"], v2_same["fn"]), (3, 2, 0, 1))
        self.assertFalse(any(r["semantic_gt_status"] == "semantic GT incomplete" for r in rows))

    def test_incomplete_review_never_counts_unreviewed_as_negative(self) -> None:
        review = {k: v for k, v in REVIEW.items() if k != "s6"}
        result = self.run_synthetic(review)
        rows = result["comparison"]
        full = find_row(rows, "semantic_object_presence", "all_5747")
        self.assertEqual(full["semantic_gt_status"], "semantic GT incomplete")
        self.assertNotIn("tp", full)
        sem = find_row(rows, "semantic_object_presence", "reviewed_subset")
        # s6 (Unreliable, unreviewed) would be TN if treated as negative.
        self.assertEqual((sem["tp"], sem["tn"], sem["fp"], sem["fn"]), (3, 1, 0, 1))
        self.assertEqual(sem["n_scope_rows"], 6)
        categories = {r["semantic_category"]: r for r in result["pool"]}
        self.assertEqual(categories["unreviewed"]["unreliable"], 1)

    def test_prediction_mismatch_detected(self) -> None:
        reference = synthetic_reference()
        with tempfile.TemporaryDirectory() as tmp:
            args = write_synthetic_run(Path(tmp), reference)
            pred = args.verification_root / "m" / "e" / "A1" / "sample_s1.json"
            pred.write_text(json.dumps({"sample_id": "s1", "decision": "Unreliable"}), encoding="utf-8")
            table = build_semantic_gt_table(reference, synthetic_manifest(reference, REVIEW))
            with self.assertRaises(SemanticGtError):
                sem_eval.evaluate_run(args.v2_root / "m" / "e" / "A1", args, reference, table,
                                      semantic_status(table))


@unittest.skipUnless(
    VERIFICATION_DATASET_INDEX_CSV.is_file() and PROTOCOL_V2_REFERENCE_CSV.is_file()
    and SEMANTIC_REVIEW_MANIFEST_CSV.is_file(),
    "canonical outputs unavailable (run scripts/build_semantic_review_manifest.py)",
)
class CanonicalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = pd.read_csv(VERIFICATION_DATASET_INDEX_CSV, dtype=str)
        cls.reference = load_reference()
        cls.manifest = read_str_csv(SEMANTIC_REVIEW_MANIFEST_CSV)

    def test_canonical_detection_count(self) -> None:
        self.assertEqual(len(self.index), EXPECTED_TOTAL)
        self.assertTrue(self.index["sample_id"].is_unique)
        self.assertEqual(list(self.reference["sample_id"]), list(self.index["sample_id"]))

    def test_protocol_v2_counts(self) -> None:
        counts = self.reference["original_protocol_v2_gt"].value_counts().to_dict()
        self.assertEqual(counts, {"positive": EXPECTED_V2_POSITIVE, "negative": EXPECTED_V2_NEGATIVE})
        with (EVALUATION_PROTOCOL_V2_ROOT / "PROTOCOL.json").open(encoding="utf-8") as file:
            protocol = json.load(file)
        self.assertEqual((protocol["canonical_gt_positive"], protocol["canonical_gt_negative"]),
                         (EXPECTED_V2_POSITIVE, EXPECTED_V2_NEGATIVE))

    def test_manifest_rows_unique_and_join_once(self) -> None:
        self.assertEqual(len(self.manifest), EXPECTED_V2_NEGATIVE)
        self.assertTrue(self.manifest["sample_id"].is_unique)
        joined = self.manifest.merge(self.reference, on="sample_id", how="left",
                                     validate="one_to_one", suffixes=("", "_ref"), indicator=True)
        self.assertTrue((joined["_merge"] == "both").all())
        self.assertTrue((joined["original_protocol_v2_gt_ref"] == "negative").all())

    def test_manifest_labels_allowed(self) -> None:
        validated = validate_review_manifest(self.manifest, self.reference)
        self.assertTrue(set(validated["semantic_gt"]) <= {*SEMANTIC_LABELS, ""})

    def test_semantic_table_consistent(self) -> None:
        table = build_semantic_gt_table(self.reference, self.manifest)
        self.assertEqual(len(table), EXPECTED_TOTAL)
        inherited = table["label_provenance"] == INHERITED_POSITIVE
        self.assertEqual(int(inherited.sum()), EXPECTED_V2_POSITIVE)
        self.assertTrue((table.loc[inherited, "semantic_gt"] == "palm").all())

    def test_stored_protocol_v2_metrics_reproducible(self) -> None:
        runs = sem_eval.discover_runs(EVALUATION_PROTOCOL_V2_ROOT)
        self.assertGreater(len(runs), 0)
        for run_dir in runs:
            with self.subTest(run=str(run_dir.relative_to(EVALUATION_PROTOCOL_V2_ROOT))):
                sem_eval.load_run(run_dir, EVALUATION_PROTOCOL_V2_ROOT,
                                  OUTPUTS_DIR / "verification", self.reference)


@unittest.skipUnless(os.environ.get("SEMANTIC_GT_VERIFY_SNAPSHOT") == "1" and BASELINE_SNAPSHOT.is_file(),
                     "set SEMANTIC_GT_VERIFY_SNAPSHOT=1 to hash protected outputs")
class ProtectedOutputsTests(unittest.TestCase):
    def test_protected_outputs_unchanged(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "diagnostics" / "snapshot_protected_outputs.py"),
             "--verify", str(BASELINE_SNAPSHOT)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
