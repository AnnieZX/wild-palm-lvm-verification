#!/usr/bin/env python3
"""
Tests for the ground-truth-independent A1-A5 behavior analysis
(scripts/analysis/a1_a5_behavior_analysis.py, docs/A1_A5_BEHAVIOR_ANALYSIS.md).

Unit tests use small synthetic prediction trees and run anywhere. Integration tests
check the generated outputs under outputs/a1_a5_behavior_analysis/ and skip when
they have not been generated.

    python -m unittest tests.test_a1_a5_behavior_analysis -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

import a1_a5_behavior_analysis as ba  # noqa: E402

R, U, X = ba.LABELS


def write_condition(condition_dir: Path, decisions: dict[str, str], *, index: bool = True,
                    status: str = "ok", overrides: dict | None = None) -> None:
    condition_dir.mkdir(parents=True)
    for sample_id, decision in decisions.items():
        record = {"sample_id": sample_id, "decision": decision, "parse_error": "", "inference_error": ""}
        record.update((overrides or {}).get(sample_id, {}))
        (condition_dir / f"{sample_id}.json").write_text(json.dumps(record), encoding="utf-8")
    if index:
        pd.DataFrame({
            "sample_id": list(decisions),
            "result_path": [f"{s}.json" for s in decisions],
            "status": status,
        }).to_csv(condition_dir / "results_index.csv", index=False)


class TransitionTests(unittest.TestCase):
    def test_hand_computed_matrix(self):
        ids = [f"sample_{i:06d}" for i in range(1, 8)]
        src = pd.Series([R, R, R, U, U, X, X], index=ids)
        dst = pd.Series([R, U, X, R, X, U, X], index=ids)
        row = ba.transition_summary(src, dst)
        self.assertEqual(row["n"], 7)
        self.assertEqual(row["reliable_to_reliable"], 1)
        self.assertEqual(row["reliable_to_uncertain"], 1)
        self.assertEqual(row["reliable_to_unreliable"], 1)
        self.assertEqual(row["uncertain_to_reliable"], 1)
        self.assertEqual(row["uncertain_to_unreliable"], 1)
        self.assertEqual(row["unreliable_to_uncertain"], 1)
        self.assertEqual(row["unreliable_to_unreliable"], 1)
        self.assertEqual(row["unchanged"], 2)
        self.assertEqual(row["changed"], 5)
        self.assertEqual(row["toward_rejection"], 3)
        self.assertEqual(row["toward_acceptance"], 2)
        self.assertEqual(row["net_toward_rejection"], 1)
        self.assertEqual(row["changed_into_reliable"], 1)
        self.assertEqual(row["changed_into_uncertain"], 2)
        self.assertEqual(row["changed_into_unreliable"], 2)

    def test_direction_sets_partition_off_diagonal(self):
        off_diagonal = {(a, b) for a in ba.LABELS for b in ba.LABELS if a != b}
        rejection = {(a, b) for a, b in off_diagonal if ba.REJECTION_RANK[b] > ba.REJECTION_RANK[a]}
        acceptance = {(a, b) for a, b in off_diagonal if ba.REJECTION_RANK[b] < ba.REJECTION_RANK[a]}
        self.assertEqual(rejection, {(R, U), (R, X), (U, X)})
        self.assertEqual(acceptance, {(X, U), (X, R), (U, R)})
        self.assertFalse(rejection & acceptance)
        self.assertEqual(rejection | acceptance, off_diagonal)

    def test_reverse_comparison_swaps_directions(self):
        ids = [f"s{i}" for i in range(6)]
        a = pd.Series([R, R, U, X, X, U], index=ids)
        b = pd.Series([X, U, R, R, X, U], index=ids)
        fwd, rev = ba.transition_summary(a, b), ba.transition_summary(b, a)
        self.assertEqual(fwd["toward_rejection"], rev["toward_acceptance"])
        self.assertEqual(fwd["toward_acceptance"], rev["toward_rejection"])
        self.assertEqual(fwd["unchanged"], rev["unchanged"])

    def test_misaligned_index_rejected(self):
        with self.assertRaises(AssertionError):
            ba.transition_summary(pd.Series([R], index=["a"]), pd.Series([R], index=["b"]))

    def test_inconsistent_row_rejected(self):
        ids = ["a", "b"]
        row = ba.transition_summary(pd.Series([R, U], index=ids), pd.Series([U, U], index=ids))
        row["toward_rejection"] += 1
        with self.assertRaises(AssertionError):
            ba.validate_transition_row(row)


class DistributionAndStabilityTests(unittest.TestCase):
    def test_distribution(self):
        row = ba.decision_distribution(pd.Series([R, R, U, X]))
        self.assertEqual((row["n"], row["reliable"], row["uncertain"], row["unreliable"]), (4, 2, 1, 1))
        self.assertEqual(row["decided"], 3)
        self.assertAlmostEqual(row["abstention_rate"], 0.25)
        self.assertAlmostEqual(row["reliable_pct"] + row["uncertain_pct"] + row["unreliable_pct"], 100.0)

    def test_stability(self):
        matrix = pd.DataFrame(
            [[R, R, R, R, R], [R, U, R, U, R], [R, U, X, R, R], [X, X, X, X, U]],
            columns=list(ba.CONDITIONS),
        )
        per = ba.per_detection_stability(matrix)
        self.assertEqual(per["n_distinct_labels"].tolist(), [1, 2, 3, 2])
        self.assertEqual(per["adjacent_changes"].tolist(), [0, 4, 3, 1])
        summary = ba.stability_summary(matrix)
        self.assertEqual(summary["identical_all_5"], 1)
        self.assertEqual(summary["exactly_2_labels"], 2)
        self.assertEqual(summary["all_3_labels"], 1)
        self.assertEqual(summary["adjacent_changes_total"], 8)
        self.assertEqual(summary["always_reliable"], 1)
        self.assertEqual(sum(summary[f"detections_with_{k}_adjacent_changes"] for k in range(5)), 4)


class LoaderValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cohort = [f"sample_{i:06d}" for i in range(1, 5)]

    def tearDown(self):
        self.tmp.cleanup()

    def _experiment(self, per_condition: dict[str, dict[str, str]] | None = None, **kwargs) -> Path:
        exp = self.root / "exp"
        for code in ba.CONDITIONS:
            decisions = (per_condition or {}).get(code, {s: R for s in self.cohort})
            write_condition(exp / code, decisions, **(kwargs if code == "A3" else {}))
        return exp

    def test_valid_experiment(self):
        matrix = ba.build_decision_matrix(self._experiment(), self.cohort)
        self.assertEqual(list(matrix.columns), list(ba.CONDITIONS))
        self.assertEqual(list(matrix.index), self.cohort)

    def test_missing_detection(self):
        exp = self._experiment({"A4": {s: R for s in self.cohort[:-1]}})
        with self.assertRaises(ba.CohortError):
            ba.build_decision_matrix(exp, self.cohort)

    def test_extra_detection(self):
        exp = self._experiment({"A2": {s: R for s in [*self.cohort, "sample_999999"]}})
        with self.assertRaises(ba.CohortError):
            ba.build_decision_matrix(exp, self.cohort)

    def test_invalid_label(self):
        exp = self._experiment({"A5": {**{s: R for s in self.cohort}, self.cohort[0]: "Maybe"}})
        with self.assertRaises(ba.CohortError):
            ba.build_decision_matrix(exp, self.cohort)

    def test_empty_label(self):
        exp = self._experiment({"A1": {**{s: R for s in self.cohort}, self.cohort[1]: ""}})
        with self.assertRaises(ba.CohortError):
            ba.build_decision_matrix(exp, self.cohort)

    def test_sample_id_filename_mismatch(self):
        exp = self._experiment(overrides={self.cohort[0]: {"sample_id": self.cohort[1]}})
        with self.assertRaises(ba.CohortError):
            ba.build_decision_matrix(exp, self.cohort)

    def test_parse_error_rejected(self):
        exp = self._experiment(overrides={self.cohort[0]: {"parse_error": "bad json"}})
        with self.assertRaises(ba.CohortError):
            ba.build_decision_matrix(exp, self.cohort)

    def test_results_index_status(self):
        exp = self._experiment(status="failed")
        with self.assertRaises(ba.CohortError):
            ba.build_decision_matrix(exp, self.cohort)

    def test_missing_condition_dir(self):
        exp = self._experiment()
        for path in (exp / "A5").iterdir():
            path.unlink()
        (exp / "A5").rmdir()
        with self.assertRaises(ba.CohortError):
            ba.build_decision_matrix(exp, self.cohort)

    def test_duplicate_cohort_ids(self):
        index = self.root / "index.csv"
        pd.DataFrame({"sample_id": ["a", "a", "b"]}).to_csv(index, index=False)
        with self.assertRaises(ba.CohortError):
            ba.load_cohort(index, expected_n=None)

    def test_cohort_size(self):
        index = self.root / "index.csv"
        pd.DataFrame({"sample_id": ["a", "b"]}).to_csv(index, index=False)
        with self.assertRaises(ba.CohortError):
            ba.load_cohort(index)

    def test_protected_output_dirs_refused(self):
        for protected in ba.PROTECTED_DIRS:
            with self.assertRaises(SystemExit):
                ba.assert_safe_output_dir(protected / "sub")
        with self.assertRaises(SystemExit):
            ba.assert_safe_output_dir(ba.OUTPUTS_DIR)
        ba.assert_safe_output_dir(ba.DEFAULT_OUT_DIR)


OUT = ba.DEFAULT_OUT_DIR


@unittest.skipUnless((OUT / "analysis_info.json").exists(), "behavior analysis outputs not generated")
class GeneratedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = pd.read_csv(OUT / "table_b_decision_distribution.csv")
        cls.c = pd.read_csv(OUT / "table_c_transitions_a1_reference.csv")
        cls.p = pd.read_csv(OUT / "pairwise_transitions.csv")
        cls.s = pd.read_csv(OUT / "stability_summary.csv")
        cls.per = pd.read_csv(OUT / "per_detection_decisions.csv", dtype=str)
        cls.info = json.loads((OUT / "analysis_info.json").read_text(encoding="utf-8"))

    def test_models_and_conditions_complete(self):
        self.assertEqual(sorted(self.b["model_key"].unique()), sorted(ba.MODEL_RUNS))
        for _, group in self.b.groupby("model_key"):
            self.assertEqual(sorted(group["condition"]), list(ba.CONDITIONS))

    def test_table_b_sums(self):
        self.assertTrue((self.b["n"] == ba.EXPECTED_N).all())
        self.assertTrue((self.b[["reliable", "uncertain", "unreliable"]].sum(axis=1) == ba.EXPECTED_N).all())
        self.assertTrue((self.b["decided"] == self.b["reliable"] + self.b["unreliable"]).all())

    def test_transition_tables(self):
        self.assertEqual(sorted(self.c["comparison"].unique()), [f"A1->{c}" for c in ba.CONDITIONS[1:]])
        self.assertEqual(len(self.p), len(ba.MODEL_RUNS) * 10)
        for table in (self.c, self.p):
            self.assertTrue((table[list(ba.TRANSITION_CELLS)].sum(axis=1) == ba.EXPECTED_N).all())
            self.assertTrue((table["unchanged"] + table["changed"] == ba.EXPECTED_N).all())
            self.assertTrue((table["toward_rejection"] + table["toward_acceptance"] == table["changed"]).all())

    def test_transition_margins_match_table_b(self):
        b = self.b.set_index(["model_key", "condition"])
        for _, row in self.p.iterrows():
            for label in ba.LABELS:
                out_of = sum(row[ba.cell_name(label, d)] for d in ba.LABELS)
                into = sum(row[ba.cell_name(a, label)] for a in ba.LABELS)
                self.assertEqual(out_of, b.loc[(row["model_key"], row["from_condition"]), label.lower()])
                self.assertEqual(into, b.loc[(row["model_key"], row["to_condition"]), label.lower()])

    def test_per_detection_cohort(self):
        self.assertEqual(len(self.per), ba.EXPECTED_N)
        self.assertFalse(self.per["sample_id"].duplicated().any())
        cohort = ba.load_cohort(ba.VERIFICATION_DATASET_INDEX_CSV)
        self.assertEqual(sorted(self.per["sample_id"]), cohort)
        for key in ba.MODEL_RUNS:
            for code in ba.CONDITIONS:
                self.assertTrue(self.per[f"{key}__{code}"].isin(ba.LABELS).all())

    def test_stability_consistent_with_per_detection(self):
        for _, row in self.s.iterrows():
            distinct = self.per[f"{row['model_key']}__n_distinct_labels"].astype(int)
            self.assertEqual(row["identical_all_5"], int((distinct == 1).sum()))
            self.assertEqual(row["exactly_2_labels"] + row["all_3_labels"] + row["identical_all_5"], ba.EXPECTED_N)
            adjacent = self.per[f"{row['model_key']}__adjacent_changes"].astype(int)
            self.assertEqual(row["adjacent_changes_total"], int(adjacent.sum()))

    def test_no_gt_columns(self):
        forbidden = ("gt", "iou", "tp", "fp", "tn", "fn", "accuracy", "sensitivity", "specificity",
                     "precision", "recall", "f1", "matched")
        for table in (self.b, self.c, self.p, self.s, self.per):
            for column in table.columns:
                tokens = column.lower().replace("__", "_").split("_")
                self.assertFalse(set(tokens) & set(forbidden), column)


if __name__ == "__main__":
    unittest.main()
