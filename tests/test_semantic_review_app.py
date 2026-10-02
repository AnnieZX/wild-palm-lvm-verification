#!/usr/bin/env python3
"""
Tests for the blind semantic review app (scripts/semantic_review_app.py).

Unit tests use a synthetic manifest/reference/images in a temp dir. Integration tests
read the canonical 638-row manifest and 100-row positive QC sample (read only) and skip
when unavailable. The full protected-output hash check (~2 min) runs only with
SEMANTIC_GT_VERIFY_SNAPSHOT=1.

    python3 -m unittest tests.test_semantic_review_app -v
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.semantic_gt import (  # noqa: E402
    PROTOCOL_V2_REFERENCE_CSV,
    REFERENCE_COLUMNS,
    REVIEW_MANIFEST_COLUMNS,
    SEMANTIC_GT_ROOT,
    SEMANTIC_REVIEW_MANIFEST_CSV,
    read_str_csv,
)
from src.paths import OUTPUTS_DIR, RAW_PATCHES_ROOT  # noqa: E402
from src.semantic_review import render  # noqa: E402
from src.semantic_review.server import BLIND_ITEM_FIELDS, WEB_DIR, ReviewApp, make_server  # noqa: E402
from src.semantic_review.store import (  # noqa: E402
    CONFIDENCE_PILOT_MANIFEST_CSV,
    DEFAULT_OUTPUT,
    IDENTITY_COLUMNS,
    LABEL_COLUMNS,
    MODE_CONFIDENCE_PILOT,
    MODE_POSITIVE_QC,
    MODE_UNMATCHED,
    POSITIVE_QC_MANIFEST_CSV,
    SEALED_PILOT_REFERENCE_CSV,
    LabelConflict,
    ReviewDataError,
    ReviewStore,
    build_items,
    check_output_path,
    load_confidence_pilot_rows,
    load_positive_qc_rows,
    load_unmatched_rows,
    refuse_sealed_reference,
)

BASELINE_SNAPSHOT = SEMANTIC_GT_ROOT / "provenance" / "protected_outputs_baseline.json"
DIAGNOSTIC_TERMS = ("confidence", "iou", "nearest", "protocol", "negative", "positive",
                    "matched", "image_path", "bbox", "prediction", "model", "gt")


PILOT_LEAK_PATTERN = re.compile(
    r"confidence|conf_|score|\biou\b|max_iou|labelme|nearest|geo_|protocol|matched|stratum|\bvlm\b|"
    r"prediction|image_path|bbox|reference|\bbin\b|bin_|data-(conf|score|bin|iou)"
)
COMPLETED_LABEL_FILES = (DEFAULT_OUTPUT[MODE_UNMATCHED], DEFAULT_OUTPUT[MODE_POSITIVE_QC])

_opened_files: list[str] | None = None


def _record_opens(event: str, args: tuple) -> None:
    if _opened_files is not None and event == "open" and args:
        _opened_files.append(os.fsdecode(args[0]) if isinstance(args[0], (str, bytes, os.PathLike)) else "")


sys.addaudithook(_record_opens)


class RecordOpens:
    """Collect every file path opened in this process while active."""

    def __enter__(self) -> list[str]:
        global _opened_files
        _opened_files = []
        return _opened_files

    def __exit__(self, *exc) -> None:
        global _opened_files
        _opened_files = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_state(path: Path) -> str | None:
    return sha256(path) if path.exists() else None


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class SyntheticWorld:
    """3 GT- detections (s4-s6) and 3 GT+ (s1-s3) on two 200x200 images."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.images = root / "images"
        self.images.mkdir()
        for name in ("img_a", "img_b"):
            Image.new("RGB", (200, 200), (30, 120, 40)).save(self.images / f"{name}.png")
        boxes = {"s1": [10, 10, 40, 40], "s2": [60, 60, 30, 20], "s3": [100, 20, 50, 50],
                 "s4": [5, 150, 40, 45], "s5": [120, 120, 60, 70], "s6": [0, 0, 20, 20]}
        self.reference_rows = []
        for i, (key, box) in enumerate(boxes.items()):
            image = "img_a" if i % 2 == 0 else "img_b"
            self.reference_rows.append({
                "sample_id": key, "image_id": image, "image_path": str(self.images / f"{image}.png"),
                "yolo_bbox_xywh": json.dumps(box), "yolo_confidence": "0.61", "max_iou": "0.2",
                "matched_gt_index": "", "nearest_gt_index": "1", "nearest_gt_bbox_xywh": "[1, 1, 5, 5]",
                "nearest_gt_owner_sample_id": "",
                "original_protocol_v2_gt": "positive" if key in ("s1", "s2", "s3") else "negative",
            })
        self.reference = root / "reference.csv"
        write_csv(self.reference, REFERENCE_COLUMNS, self.reference_rows)
        self.manifest = root / "manifest.csv"
        self.write_manifest([r for r in self.reference_rows if r["original_protocol_v2_gt"] == "negative"])
        self.qc_manifest = root / "qc.csv"
        write_csv(self.qc_manifest, ["blind_id", "sample_id", "blind_visualization_path", "semantic",
                                     "reviewer_confidence", "notes"],
                  [{"blind_id": f"qc_00{i + 1}", "sample_id": key, "blind_visualization_path": "",
                    "semantic": "", "reviewer_confidence": "", "notes": ""}
                   for i, key in enumerate(("s3", "s1", "s2"))])

    def write_manifest(self, ref_rows: list[dict]) -> None:
        rows = []
        for index, ref in enumerate(ref_rows, start=1):
            row = {c: "" for c in REVIEW_MANIFEST_COLUMNS}
            row.update({k: ref[k] for k in ref if k in row})
            row["review_index"] = str(index)
            row["iou_ge_050_gt_taken_by_other"] = "false"
            rows.append(row)
        write_csv(self.manifest, REVIEW_MANIFEST_COLUMNS, rows)

    def unmatched_items(self):
        rows = load_unmatched_rows(self.manifest, self.reference, expected_count=3, reference_counts=None)
        return build_items(MODE_UNMATCHED, rows, self.images)

    def store(self, output: Path | None = None, reviewer: str = "tester", **kwargs) -> ReviewStore:
        return ReviewStore(MODE_UNMATCHED, self.unmatched_items(), output or self.root / "out" / "human_review.csv",
                           reviewer, **kwargs)


class SyntheticTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.world = SyntheticWorld(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestLoadingAndValidation(SyntheticTestCase):
    def test_loads_items_in_review_order(self) -> None:
        items = self.world.unmatched_items()
        self.assertEqual([i.sample_id for i in items], ["s4", "s5", "s6"])
        self.assertEqual(items[0].bbox, (5.0, 150.0, 40.0, 45.0))

    def test_duplicate_sample_id_fails(self) -> None:
        negatives = [r for r in self.world.reference_rows if r["original_protocol_v2_gt"] == "negative"]
        self.world.write_manifest([negatives[0], negatives[1], negatives[1]])
        with self.assertRaisesRegex(ReviewDataError, "duplicate sample_id"):
            self.world.unmatched_items()

    def test_positive_row_in_manifest_fails(self) -> None:
        rows = [r for r in self.world.reference_rows if r["sample_id"] in ("s1", "s5", "s6")]
        self.world.write_manifest(rows)
        with self.assertRaisesRegex(ReviewDataError, "not Protocol v2 GT-"):
            self.world.unmatched_items()

    def test_identity_differs_from_reference_fails(self) -> None:
        negatives = [dict(r) for r in self.world.reference_rows if r["original_protocol_v2_gt"] == "negative"]
        negatives[0]["yolo_bbox_xywh"] = "[6, 150, 40, 45]"
        self.world.write_manifest(negatives)
        with self.assertRaisesRegex(ReviewDataError, "yolo_bbox_xywh differs"):
            self.world.unmatched_items()

    def test_missing_image_fails(self) -> None:
        (self.world.images / "img_b.png").unlink()
        with self.assertRaisesRegex(ReviewDataError, "image not found"):
            self.world.unmatched_items()

    def test_image_outside_root_fails(self) -> None:
        rows = load_unmatched_rows(self.world.manifest, self.world.reference, 3, None)
        with self.assertRaisesRegex(ReviewDataError, "image outside"):
            build_items(MODE_UNMATCHED, rows, self.world.root / "elsewhere")

    def test_invalid_bbox_fails(self) -> None:
        rows = load_unmatched_rows(self.world.manifest, self.world.reference, 3, None)
        rows[0]["yolo_bbox_xywh"] = "[5, 150, 0, 45]"
        with self.assertRaisesRegex(ReviewDataError, "invalid bbox"):
            build_items(MODE_UNMATCHED, rows, self.world.images)
        rows[0]["yolo_bbox_xywh"] = "[180, 150, 40, 45]"
        with self.assertRaisesRegex(ReviewDataError, "outside image"):
            build_items(MODE_UNMATCHED, rows, self.world.images)

    def test_positive_qc_rows(self) -> None:
        rows = load_positive_qc_rows(self.world.qc_manifest, self.world.reference, 3, None)
        self.assertEqual([r["blind_id"] for r in rows], ["qc_001", "qc_002", "qc_003"])
        self.assertEqual([r["sample_id"] for r in rows], ["s3", "s1", "s2"])
        items = build_items(MODE_POSITIVE_QC, rows, self.world.images)
        self.assertEqual(items[0].display_id, "qc_001")

    def test_positive_qc_rejects_negative_sample(self) -> None:
        write_csv(self.world.qc_manifest, ["blind_id", "sample_id"],
                  [{"blind_id": "qc_001", "sample_id": "s4"}])
        with self.assertRaisesRegex(ReviewDataError, "not a Protocol v2 GT"):
            load_positive_qc_rows(self.world.qc_manifest, self.world.reference, 1, None)


class TestStore(SyntheticTestCase):
    def test_new_file_is_blank_and_has_only_identity_and_label_columns(self) -> None:
        store = self.world.store()
        frame = read_str_csv(store.output_path)
        self.assertEqual(list(frame.columns), IDENTITY_COLUMNS[MODE_UNMATCHED] + LABEL_COLUMNS)
        self.assertTrue((frame["semantic_label"] == "").all())
        self.assertEqual(store.first_unlabeled(), 0)

    def test_reviewer_required(self) -> None:
        with self.assertRaises(ReviewDataError):
            self.world.store(reviewer="  ")

    def test_valid_and_invalid_labels(self) -> None:
        store = self.world.store()
        for bad in ("PALM ", "tree", "positive", "Palm"):
            with self.assertRaises(LabelConflict):
                store.set_label(0, "s4", bad, "")
        store.set_label(0, "s4", "PALM".lower(), "")
        store.set_label(1, "s5", "non_palm", "")
        store.set_label(2, "s6", "ambiguous", "")
        self.assertEqual(store.labels, ["palm", "non_palm", "ambiguous"])

    def test_save_is_atomic_and_complete(self) -> None:
        store = self.world.store()
        store.set_label(1, "s5", "palm", "")
        frame = read_str_csv(store.output_path)
        row = frame.iloc[1]
        self.assertEqual((row["sample_id"], row["semantic_label"], row["reviewer"]), ("s5", "palm", "tester"))
        self.assertTrue(row["review_timestamp"])
        self.assertEqual(frame.iloc[0]["semantic_label"], "")
        leftovers = [p.name for p in store.output_path.parent.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        events = [json.loads(line) for line in store.log_path.read_text().splitlines()]
        self.assertEqual(events[-1]["previous"], "")
        self.assertEqual(events[-1]["new"], "palm")

    def test_resume(self) -> None:
        store = self.world.store()
        store.set_label(0, "s4", "palm", "")
        store.set_label(1, "s5", "ambiguous", "")
        resumed = self.world.store(reviewer="someone_else")
        self.assertEqual(resumed.labels, ["palm", "ambiguous", ""])
        self.assertEqual(resumed.reviewers[:2], ["tester", "tester"])
        self.assertEqual(resumed.first_unlabeled(), 2)
        self.assertEqual(resumed.completed(), 2)

    def test_explicit_change_required(self) -> None:
        store = self.world.store()
        store.set_label(0, "s4", "palm", "")
        with self.assertRaisesRegex(LabelConflict, "explicit change"):
            store.set_label(0, "s4", "non_palm", "palm")
        with self.assertRaisesRegex(LabelConflict, "stale"):
            store.set_label(0, "s4", "non_palm", "", confirm_change=True)
        with self.assertRaisesRegex(LabelConflict, "explicit change"):
            store.set_label(0, "s4", "", "palm")
        self.assertEqual(store.set_label(0, "s4", "palm", "palm"), {"changed": False, "label": "palm"})
        store.set_label(0, "s4", "non_palm", "palm", confirm_change=True)
        self.assertEqual(read_str_csv(store.output_path).iloc[0]["semantic_label"], "non_palm")
        store.set_label(0, "s4", "", "non_palm", confirm_change=True)
        row = read_str_csv(store.output_path).iloc[0]
        self.assertEqual((row["semantic_label"], row["reviewer"], row["review_timestamp"]), ("", "", ""))

    def test_wrong_display_id_refused(self) -> None:
        store = self.world.store()
        with self.assertRaisesRegex(LabelConflict, "not s5"):
            store.set_label(0, "s5", "palm", "")

    def test_identity_mismatch_in_existing_file_fails_without_repair(self) -> None:
        store = self.world.store()
        store.set_label(0, "s4", "palm", "")
        text = store.output_path.read_text().replace("s5", "s9")
        store.output_path.write_text(text)
        before = store.output_path.read_bytes()
        with self.assertRaisesRegex(ReviewDataError, "disagrees with the source"):
            self.world.store()
        self.assertEqual(store.output_path.read_bytes(), before)

    def test_invalid_label_in_existing_file_fails(self) -> None:
        store = self.world.store()
        frame = read_str_csv(store.output_path)
        frame.loc[0, ["semantic_label", "reviewer", "review_timestamp"]] = ["PALM", "x", "t"]
        frame.to_csv(store.output_path, index=False)
        with self.assertRaisesRegex(ReviewDataError, "invalid semantic_label"):
            self.world.store()

    def test_periodic_backups_are_rotated(self) -> None:
        store = self.world.store(backup_every=1, backups_kept=2)
        for label in ("palm", "non_palm", "ambiguous"):
            current = store.labels[0]
            store.set_label(0, "s4", label, current, confirm_change=bool(current))
        self.assertEqual(len(list(store.backup_dir.glob("human_review.*.csv"))), 2)

    def test_source_manifest_untouched(self) -> None:
        before = (sha256(self.world.manifest), sha256(self.world.reference))
        store = self.world.store()
        store.set_label(0, "s4", "palm", "")
        self.assertEqual((sha256(self.world.manifest), sha256(self.world.reference)), before)

    def test_output_path_guard(self) -> None:
        for protected in ("verification", "evaluation_protocol_v2", "evaluation", "diagnostics",
                          "semantic_gt_evaluation"):
            with self.assertRaises(ReviewDataError):
                check_output_path(OUTPUTS_DIR / protected / "human_review.csv", [])
        with self.assertRaises(ReviewDataError):
            check_output_path(self.world.manifest, [self.world.manifest])
        check_output_path(OUTPUTS_DIR / "semantic_gt_review" / "human_review.csv", [])


class TestRender(SyntheticTestCase):
    def test_crop_window_inside_image_and_deterministic(self) -> None:
        self.assertEqual(render.crop_window((0, 0, 20, 20), 912, 912), (0, 0, 160, 160))
        self.assertEqual(render.crop_window((800, 850, 100, 62), 912, 912), (612, 612, 912, 912))
        left, top, right, bottom = render.crop_window((400, 400, 300, 300), 912, 912)
        self.assertEqual((right - left, bottom - top), (900, 900))
        image = self.world.images / "img_a.png"
        for kind in render.KINDS:
            self.assertEqual(render.render(image, (5, 150, 40, 45), kind),
                             render.render(image, (5, 150, 40, 45), kind))


class TestServer(SyntheticTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.store = self.world.store()
        self.app = ReviewApp(self.store, token="test-token")
        self.server = make_server(self.app, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def request(self, method: str, path: str, body: dict | None = None, token: str | None = "test-token",
                host: str = "localhost") -> tuple[int, bytes, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Host": f"{host}:{self.port}"}
        if token:
            headers["X-Review-Token"] = token
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        return response.status, payload, dict(response.getheaders())

    def test_refuses_non_loopback_bind(self) -> None:
        with self.assertRaises(ValueError):
            make_server(self.app, "0.0.0.0", 0)

    def test_token_and_host_required(self) -> None:
        self.assertEqual(self.request("GET", "/api/state", token=None)[0], 403)
        self.assertEqual(self.request("GET", "/api/state", token="wrong")[0], 403)
        self.assertEqual(self.request("GET", "/image/context/0.jpg", token=None)[0], 403)
        self.assertEqual(self.request("GET", "/api/state", host="evil.example")[0], 403)
        self.assertEqual(self.request("GET", "/", token=None)[0], 403)
        status, _, headers = self.request("GET", "/?token=test-token", token=None)
        self.assertEqual(status, 303)
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertEqual(self.request("GET", "/")[0], 200)

    def test_image_endpoint_serves_jpeg_by_position_only(self) -> None:
        status, body, headers = self.request("GET", "/image/crop/1.jpg")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/jpeg")
        self.assertTrue(body.startswith(b"\xff\xd8"))
        for path in ("/image/context/3.jpg", "/image/context/-1.jpg", "/image/context/../../etc/passwd",
                     "/image/context/%2e%2e%2fetc%2fpasswd", "/image/context//etc/passwd",
                     "/image/raw/0.jpg", f"/image/context/{self.world.images}/img_a.png",
                     "/static/../store.py", "/static/semantic_review.py", "/etc/passwd",
                     "/api/item?position=../../etc/passwd", "/api/item?position=99"):
            status, body, _ = self.request("GET", path)
            self.assertIn(status, (403, 404), path)
            self.assertNotIn(b"root:", body)

    def test_blind_api_exposes_no_diagnostics(self) -> None:
        status, body, _ = self.request("GET", "/api/item?position=0")
        self.assertEqual(status, 200)
        item = json.loads(body)
        self.assertEqual(tuple(item), BLIND_ITEM_FIELDS)
        status, state_body, _ = self.request("GET", "/api/state")
        state = json.loads(state_body)
        self.assertEqual(set(state), {"mode_title", "reviewer", "total", "completed", "first_unlabeled"})
        for text in (body.decode().lower(), state_body.decode().lower()):
            for term in DIAGNOSTIC_TERMS:
                self.assertNotIn(term, text.replace("display_id", ""), term)
            for value in ("0.61", "150", "img_a", str(self.world.images).lower()):
                self.assertNotIn(value, text)
        self.store.set_label(1, "s5", "non_palm", "")
        item = json.loads(self.request("GET", "/api/item?position=0")[1])
        self.assertNotIn("non_palm", json.dumps(item))

    def test_label_post_flow(self) -> None:
        body = {"position": 0, "display_id": "s4", "label": "palm", "expected_current": ""}
        status, payload, _ = self.request("POST", "/api/label", body)
        self.assertEqual(status, 200)
        result = json.loads(payload)
        self.assertEqual((result["human_label"], result["next_unlabeled"], result["completed"]), ("palm", 1, 1))
        status, _, _ = self.request("POST", "/api/label", {**body, "label": "non_palm", "expected_current": "palm"})
        self.assertEqual(status, 409)
        status, _, _ = self.request("POST", "/api/label", {**body, "label": "non_palm", "expected_current": "palm",
                                                          "confirm_change": True})
        self.assertEqual(status, 200)
        self.assertEqual(self.store.labels[0], "non_palm")
        self.assertEqual(self.request("POST", "/api/label", {**body, "label": "tree"})[0], 409)
        self.assertEqual(self.request("POST", "/api/label", {"position": 0})[0], 400)
        self.assertEqual(self.request("POST", "/api/label", body, token=None)[0], 403)


class TestCli(unittest.TestCase):
    def test_reviewer_required(self) -> None:
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "semantic_review_app.py")],
                                capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--reviewer is required", result.stderr)


@unittest.skipUnless(SEMANTIC_REVIEW_MANIFEST_CSV.is_file() and PROTOCOL_V2_REFERENCE_CSV.is_file()
                     and RAW_PATCHES_ROOT.is_dir(), "canonical data unavailable")
class TestCanonicalData(unittest.TestCase):
    def test_loads_638_unmatched_with_images(self) -> None:
        before = sha256(SEMANTIC_REVIEW_MANIFEST_CSV)
        rows = load_unmatched_rows()
        items = build_items(MODE_UNMATCHED, rows, RAW_PATCHES_ROOT)
        self.assertEqual(len(items), 638)
        self.assertEqual(len({i.sample_id for i in items}), 638)
        self.assertEqual(sha256(SEMANTIC_REVIEW_MANIFEST_CSV), before)

    @unittest.skipUnless(POSITIVE_QC_MANIFEST_CSV.is_file(), "positive QC sample unavailable")
    def test_loads_existing_100_positive_qc(self) -> None:
        before = sha256(POSITIVE_QC_MANIFEST_CSV)
        rows = load_positive_qc_rows()
        items = build_items(MODE_POSITIVE_QC, rows, RAW_PATCHES_ROOT)
        self.assertEqual(len(items), 100)
        self.assertEqual(items[0].display_id, "qc_001")
        self.assertEqual(sha256(POSITIVE_QC_MANIFEST_CSV), before)

    def test_unmatched_and_qc_sets_are_disjoint(self) -> None:
        unmatched = {r["sample_id"] for r in load_unmatched_rows()}
        qc = {r["sample_id"] for r in load_positive_qc_rows()}
        self.assertFalse(unmatched & qc)


def http_request(port: int, method: str, path: str, body: dict | None = None, token: str | None = "test-token",
                 host: str = "localhost") -> tuple[int, bytes, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Host": f"{host}:{port}"}
    if token:
        headers["X-Review-Token"] = token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=headers)
    response = conn.getresponse()
    payload = response.read()
    conn.close()
    return response.status, payload, dict(response.getheaders())


PILOT_COLUMNS = IDENTITY_COLUMNS[MODE_CONFIDENCE_PILOT]


class PilotWorld:
    """5-row blind pilot manifest on two 200x200 images, with a sealed reference next to it."""

    BOXES = ([10, 10, 40, 40], [60, 60, 30, 20], [100, 20, 50, 50], [5, 150, 40, 45], [120, 120, 60, 70])

    def __init__(self, root: Path) -> None:
        self.root = root
        self.images = root / "images"
        self.images.mkdir()
        for name in ("img_a", "img_b"):
            Image.new("RGB", (200, 200), (30, 120, 40)).save(self.images / f"{name}.png")
        self.rows = []
        for index, box in enumerate(self.BOXES, start=1):
            image = "img_a" if index % 2 else "img_b"
            self.rows.append({"review_index": str(index), "pilot_id": f"cp_{index:03d}", "image_id": image,
                              "image_path": str(self.images / f"{image}.png"), "yolo_bbox_xywh": json.dumps(box)})
        self.audit = root / "audit"
        self.audit.mkdir()
        self.manifest = self.audit / "semantic_pilot_manifest.csv"
        write_csv(self.manifest, PILOT_COLUMNS, self.rows)
        self.reference = self.audit / "semantic_pilot_reference.csv"
        write_csv(self.reference, ["pilot_id", "yolo_confidence", "confidence_bin", "geo_max_iou_labelme"],
                  [{"pilot_id": r["pilot_id"], "yolo_confidence": "0.1373", "confidence_bin": "[0.10,0.20)",
                    "geo_max_iou_labelme": "0.4321"} for r in self.rows])
        self.output = root / "review" / "human_confidence_pilot.csv"

    def write_manifest(self, rows: list[dict], columns: list[str] = PILOT_COLUMNS) -> None:
        write_csv(self.manifest, columns, rows)

    def items(self):
        return build_items(MODE_CONFIDENCE_PILOT, load_confidence_pilot_rows(self.manifest, expected_count=5),
                           self.images)

    def store(self, reviewer: str = "tester", **kwargs) -> ReviewStore:
        return ReviewStore(MODE_CONFIDENCE_PILOT, self.items(), self.output, reviewer, **kwargs)


class PilotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.world = PilotWorld(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestConfidencePilotLoading(PilotTestCase):
    def test_loads_in_manifest_order_with_neutral_ids(self) -> None:
        items = self.world.items()
        self.assertEqual([i.display_id for i in items], [f"cp_{n:03d}" for n in range(1, 6)])
        self.assertEqual([i.identity for i in items], self.world.rows)
        self.assertEqual(items[3].bbox, (5.0, 150.0, 40.0, 45.0))

    def test_wrong_row_count_fails(self) -> None:
        with self.assertRaisesRegex(ReviewDataError, "expected 400"):
            load_confidence_pilot_rows(self.world.manifest)

    def test_forbidden_diagnostic_columns_fail_closed(self) -> None:
        for extra in ("yolo_confidence", "confidence_bin", "score", "geo_max_iou_labelme", "max_iou",
                      "matched", "nearest_gt_index", "prediction_index", "original_protocol_v2_gt", "Conf"):
            rows = [{**r, extra: "0.3"} for r in self.world.rows]
            self.world.write_manifest(rows, PILOT_COLUMNS + [extra])
            with self.assertRaisesRegex(ReviewDataError, "forbidden diagnostic columns", msg=extra):
                self.world.items()

    def test_unexpected_or_missing_columns_fail(self) -> None:
        rows = [{**r, "notes": ""} for r in self.world.rows]
        self.world.write_manifest(rows, PILOT_COLUMNS + ["notes"])
        with self.assertRaisesRegex(ReviewDataError, "columns"):
            self.world.items()
        self.world.write_manifest([{c: r[c] for c in PILOT_COLUMNS[1:]} for r in self.world.rows], PILOT_COLUMNS[1:])
        with self.assertRaisesRegex(ReviewDataError, "columns"):
            self.world.items()

    def test_reordered_manifest_fails_without_resorting(self) -> None:
        rows = list(self.world.rows)
        rows[0], rows[1] = rows[1], rows[0]
        self.world.write_manifest(rows)
        with self.assertRaisesRegex(ReviewDataError, "review_index"):
            self.world.items()

    def test_non_neutral_or_duplicate_ids_fail(self) -> None:
        for bad in ("cp_1", "sample_000123", "cp_002"):
            rows = [dict(r) for r in self.world.rows]
            rows[0]["pilot_id"] = bad
            self.world.write_manifest(rows)
            with self.assertRaises(ReviewDataError, msg=bad):
                self.world.items()

    def test_review_index_must_be_1_to_n(self) -> None:
        rows = [dict(r) for r in self.world.rows]
        rows[4]["review_index"] = "7"
        self.world.write_manifest(rows)
        with self.assertRaisesRegex(ReviewDataError, "review_index"):
            self.world.items()

    def test_duplicate_target_fails(self) -> None:
        rows = [dict(r) for r in self.world.rows]
        rows[2]["image_path"], rows[2]["image_id"], rows[2]["yolo_bbox_xywh"] = (
            rows[0]["image_path"], rows[0]["image_id"], rows[0]["yolo_bbox_xywh"])
        self.world.write_manifest(rows)
        with self.assertRaisesRegex(ReviewDataError, "duplicate image/box"):
            self.world.items()

    def test_missing_image_and_invalid_box_fail(self) -> None:
        rows = [dict(r) for r in self.world.rows]
        rows[1]["yolo_bbox_xywh"] = "[180, 150, 40, 45]"
        self.world.write_manifest(rows)
        with self.assertRaisesRegex(ReviewDataError, "outside image"):
            self.world.items()
        self.world.write_manifest(self.world.rows)
        (self.world.images / "img_b.png").unlink()
        with self.assertRaisesRegex(ReviewDataError, "image not found"):
            self.world.items()

    def test_sealed_reference_is_refused_by_path(self) -> None:
        for path in (self.world.reference, SEALED_PILOT_REFERENCE_CSV):
            with self.assertRaisesRegex(ReviewDataError, "sealed"):
                refuse_sealed_reference(path)
            with self.assertRaisesRegex(ReviewDataError, "sealed"):
                load_confidence_pilot_rows(path, expected_count=5)


class TestConfidencePilotStore(PilotTestCase):
    def test_output_has_only_identity_and_label_columns(self) -> None:
        store = self.world.store()
        store.set_label(0, "cp_001", "palm", "")
        frame = read_str_csv(self.world.output)
        self.assertEqual(list(frame.columns), PILOT_COLUMNS + LABEL_COLUMNS)
        text = self.world.output.read_text().lower()
        for term in ("confidence", "0.1373", "0.4321", "[0.10", "max_iou", "geo_"):
            self.assertNotIn(term, text)
        self.assertEqual(store.log_path.name, "human_confidence_pilot.log.jsonl")

    def test_labels_resume_and_explicit_change(self) -> None:
        store = self.world.store()
        store.set_label(0, "cp_001", "palm", "")
        store.set_label(1, "cp_002", "non_palm", "")
        store.set_label(2, "cp_003", "ambiguous", "")
        with self.assertRaises(LabelConflict):
            store.set_label(3, "cp_004", "Palm", "")
        with self.assertRaisesRegex(LabelConflict, "not cp_004"):
            store.set_label(4, "cp_004", "palm", "")
        resumed = self.world.store(reviewer="someone_else")
        self.assertEqual(resumed.labels, ["palm", "non_palm", "ambiguous", "", ""])
        self.assertEqual(resumed.first_unlabeled(), 3)
        with self.assertRaisesRegex(LabelConflict, "explicit change"):
            resumed.set_label(1, "cp_002", "palm", "non_palm")
        resumed.set_label(1, "cp_002", "palm", "non_palm", confirm_change=True)
        row = read_str_csv(self.world.output).iloc[1]
        self.assertEqual((row["pilot_id"], row["semantic_label"], row["reviewer"]), ("cp_002", "palm", "someone_else"))
        events = [json.loads(line) for line in resumed.log_path.read_text().splitlines()]
        self.assertEqual((events[-1]["display_id"], events[-1]["previous"], events[-1]["new"]),
                         ("cp_002", "non_palm", "palm"))

    def test_backups_use_separate_names(self) -> None:
        store = self.world.store(backup_every=1)
        store.set_label(0, "cp_001", "palm", "")
        names = [p.name for p in store.backup_dir.iterdir()]
        self.assertTrue(names)
        self.assertTrue(all(n.startswith("human_confidence_pilot.") for n in names), names)

    def test_identity_mismatch_in_existing_output_fails_without_repair(self) -> None:
        store = self.world.store()
        store.set_label(0, "cp_001", "palm", "")
        text = self.world.output.read_text().replace("[60, 60, 30, 20]", "[61, 60, 30, 20]")
        self.world.output.write_text(text)
        before = self.world.output.read_bytes()
        with self.assertRaisesRegex(ReviewDataError, "disagrees with the source"):
            self.world.store()
        self.assertEqual(self.world.output.read_bytes(), before)

    def test_invalid_existing_label_fails(self) -> None:
        self.world.store()
        frame = read_str_csv(self.world.output)
        frame.loc[0, LABEL_COLUMNS] = ["tree", "x", "t"]
        frame.to_csv(self.world.output, index=False)
        with self.assertRaisesRegex(ReviewDataError, "invalid semantic_label"):
            self.world.store()

    def test_sources_untouched(self) -> None:
        before = (sha256(self.world.manifest), sha256(self.world.reference))
        self.world.store().set_label(0, "cp_001", "palm", "")
        self.assertEqual((sha256(self.world.manifest), sha256(self.world.reference)), before)

    def test_output_guard_keeps_modes_and_audit_dir_separate(self) -> None:
        for target in COMPLETED_LABEL_FILES:
            with self.assertRaisesRegex(ReviewDataError, "label file"):
                check_output_path(target, [], MODE_CONFIDENCE_PILOT)
        with self.assertRaisesRegex(ReviewDataError, "protected"):
            check_output_path(CONFIDENCE_PILOT_MANIFEST_CSV.parent / "labels.csv", [], MODE_CONFIDENCE_PILOT)
        with self.assertRaisesRegex(ReviewDataError, "label file"):
            check_output_path(DEFAULT_OUTPUT[MODE_CONFIDENCE_PILOT], [], MODE_UNMATCHED)
        self.assertEqual(check_output_path(DEFAULT_OUTPUT[MODE_CONFIDENCE_PILOT], [], MODE_CONFIDENCE_PILOT),
                         DEFAULT_OUTPUT[MODE_CONFIDENCE_PILOT].resolve())


class TestConfidencePilotServer(PilotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.store = self.world.store()
        self.app = ReviewApp(self.store, token="test-token")
        self.server = make_server(self.app, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def request(self, *args, **kwargs):
        return http_request(self.port, *args, **kwargs)

    def assert_no_leak(self, text: str) -> None:
        lowered = text.lower()
        self.assertIsNone(PILOT_LEAK_PATTERN.search(lowered), PILOT_LEAK_PATTERN.search(lowered))
        for value in ("0.1373", "0.4321", "[0.10", "img_a", "img_b", str(self.world.root).lower()):
            self.assertNotIn(value, lowered)

    def test_api_exposes_only_blind_fields(self) -> None:
        status, state_body, _ = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        state = json.loads(state_body)
        self.assertEqual(set(state), {"mode_title", "reviewer", "total", "completed", "first_unlabeled"})
        self.assertEqual((state["mode_title"], state["total"]), ("Pilot review", 5))
        self.assert_no_leak(state_body.decode())
        for position in range(5):
            status, body, _ = self.request("GET", f"/api/item?position={position}")
            item = json.loads(body)
            self.assertEqual(tuple(item), BLIND_ITEM_FIELDS)
            self.assertEqual(item["display_id"], f"cp_{position + 1:03d}")
            self.assert_no_leak(body.decode())

    def test_label_flow_has_no_class_totals(self) -> None:
        body = {"position": 0, "display_id": "cp_001", "label": "non_palm", "expected_current": ""}
        status, payload, _ = self.request("POST", "/api/label", body)
        self.assertEqual(status, 200)
        result = json.loads(payload)
        self.assertEqual((result["human_label"], result["completed"], result["next_unlabeled"]), ("non_palm", 1, 1))
        self.assertEqual(set(result), set(BLIND_ITEM_FIELDS) | {"changed", "label"})
        self.assert_no_leak(payload.decode())
        status, _, _ = self.request("POST", "/api/label", {**body, "label": "palm", "expected_current": "non_palm"})
        self.assertEqual(status, 409)
        other = json.loads(self.request("GET", "/api/item?position=1")[1])
        self.assertNotIn("non_palm", json.dumps(other))

    def test_html_js_css_leak_nothing(self) -> None:
        status, html, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        static = [self.request("GET", path, token=None)[1].decode()
                  for path in ("/static/semantic_review.js", "/static/semantic_review.css")]
        for text in [html.decode(), *static]:
            self.assert_no_leak(text)
        self.assertNotIn("console.", static[0])
        self.assertIn("<title>Semantic GT Review</title>", html.decode())

    def test_image_urls_are_positional_and_traversal_blocked(self) -> None:
        status, body, headers = self.request("GET", "/image/context/0.jpg")
        self.assertEqual((status, headers["Content-Type"]), (200, "image/jpeg"))
        self.assertTrue(body.startswith(b"\xff\xd8"))
        self.assertEqual(self.request("GET", "/image/crop/4.jpg")[0], 200)
        for path in ("/image/context/5.jpg", "/image/context/../../etc/passwd", "/image/context/%2e%2e%2f",
                     f"/image/context/{self.world.reference}", "/static/../store.py",
                     "/semantic_pilot_reference.csv", "/api/item?position=../../etc/passwd"):
            status, body, _ = self.request("GET", path)
            self.assertIn(status, (403, 404), path)
            self.assertNotIn(b"0.1373", body)

    def test_token_host_and_loopback_enforced(self) -> None:
        self.assertEqual(self.request("GET", "/api/state", token=None)[0], 403)
        self.assertEqual(self.request("GET", "/api/state", token="wrong")[0], 403)
        self.assertEqual(self.request("GET", "/image/context/0.jpg", token=None)[0], 403)
        self.assertEqual(self.request("GET", "/api/state", host="evil.example")[0], 403)
        self.assertEqual(self.request("POST", "/api/label", {"position": 0}, token=None)[0], 403)
        for host in ("0.0.0.0", "::", "10.0.0.1"):
            with self.assertRaises(ValueError):
                make_server(self.app, host, 0)


class TestConfidencePilotSealedReference(PilotTestCase):
    def run_full_session(self) -> None:
        store = self.world.store()
        app = ReviewApp(store, token="test-token")
        server = make_server(app, "127.0.0.1", 0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for path in ("/", "/api/state", "/api/item?position=0", "/image/context/0.jpg", "/image/crop/0.jpg",
                         "/image/crop_raw/0.jpg"):
                self.assertEqual(http_request(port, "GET", path)[0], 200, path)
            body = {"position": 0, "display_id": "cp_001", "label": "palm", "expected_current": ""}
            self.assertEqual(http_request(port, "POST", "/api/label", body)[0], 200)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(self.world.store().labels[0], "palm")

    def test_reference_never_opened(self) -> None:
        with RecordOpens() as opened:
            self.run_full_session()
        self.assertTrue(any(p.endswith("semantic_pilot_manifest.csv") for p in opened))
        self.assertFalse([p for p in opened if "reference" in p], opened)

    def test_review_works_with_reference_removed_or_renamed(self) -> None:
        renamed = self.world.reference.with_name("sealed_elsewhere.csv")
        self.world.reference.rename(renamed)
        self.run_full_session()
        renamed.unlink()
        self.world.output.unlink()
        self.run_full_session()


@unittest.skipUnless(CONFIDENCE_PILOT_MANIFEST_CSV.is_file() and RAW_PATCHES_ROOT.is_dir(),
                     "confidence pilot manifest unavailable")
class TestCanonicalConfidencePilot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.output = Path(self._tmp.name) / "human_confidence_pilot.csv"
        self.guarded = [CONFIDENCE_PILOT_MANIFEST_CSV, SEALED_PILOT_REFERENCE_CSV, *COMPLETED_LABEL_FILES,
                        DEFAULT_OUTPUT[MODE_CONFIDENCE_PILOT]]
        self.before = {path: file_state(path) for path in self.guarded}

    def tearDown(self) -> None:
        self.assertEqual({path: file_state(path) for path in self.guarded}, self.before)
        self._tmp.cleanup()

    def test_loads_400_in_unchanged_order(self) -> None:
        with RecordOpens() as opened:
            rows = load_confidence_pilot_rows()
            items = build_items(MODE_CONFIDENCE_PILOT, rows, RAW_PATCHES_ROOT)
        self.assertFalse([p for p in opened if "reference" in p])
        with CONFIDENCE_PILOT_MANIFEST_CSV.open(newline="", encoding="utf-8") as file:
            raw = list(csv.DictReader(file))
        self.assertEqual(len(items), 400)
        self.assertEqual([i.identity for i in items], raw)
        self.assertEqual([i.display_id for i in items], [f"cp_{n:03d}" for n in range(1, 401)])
        self.assertEqual(len({i.display_id for i in items}), 400)

    def test_store_and_server_on_real_pilot_use_temp_output_only(self) -> None:
        with RecordOpens() as opened:
            items = build_items(MODE_CONFIDENCE_PILOT, load_confidence_pilot_rows(), RAW_PATCHES_ROOT)
            store = ReviewStore(MODE_CONFIDENCE_PILOT, items, self.output, "tester")
            app = ReviewApp(store, token="test-token")
            server = make_server(app, "127.0.0.1", 0)
            port = server.server_address[1]
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                state = json.loads(http_request(port, "GET", "/api/state")[1])
                self.assertEqual((state["total"], state["completed"], state["mode_title"]), (400, 0, "Pilot review"))
                status, body, _ = http_request(port, "GET", "/api/item?position=0")
                self.assertEqual(json.loads(body)["display_id"], "cp_001")
                self.assertIsNone(PILOT_LEAK_PATTERN.search(body.decode().lower()))
                for kind in ("context", "crop", "crop_raw"):
                    self.assertEqual(http_request(port, "GET", f"/image/{kind}/0.jpg")[0], 200)
                post = {"position": 0, "display_id": "cp_001", "label": "ambiguous", "expected_current": ""}
                self.assertEqual(http_request(port, "POST", "/api/label", post)[0], 200)
            finally:
                server.shutdown()
                server.server_close()
            resumed = ReviewStore(MODE_CONFIDENCE_PILOT, items, self.output, "tester")
        self.assertEqual((resumed.completed(), resumed.labels[0]), (1, "ambiguous"))
        self.assertFalse([p for p in opened if "reference" in p])
        opened_paths = {Path(p).resolve() for p in opened if p}
        for path in (DEFAULT_OUTPUT[MODE_CONFIDENCE_PILOT], *COMPLETED_LABEL_FILES):
            self.assertNotIn(path.resolve(), opened_paths)


class TestCliConfidencePilot(unittest.TestCase):
    def test_mode_is_offered(self) -> None:
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "semantic_review_app.py"), "--help"],
                                capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(result.returncode, 0)
        self.assertIn("confidence-pilot", result.stdout)

    def test_refuses_completed_label_file_as_output(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "semantic_review_app.py"), "--reviewer", "tester",
             "--mode", "confidence-pilot", "--port", "0", "--output", str(DEFAULT_OUTPUT[MODE_UNMATCHED])],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STARTUP VALIDATION FAILED", result.stderr)


@unittest.skipUnless(os.environ.get("SEMANTIC_GT_VERIFY_SNAPSHOT") == "1" and BASELINE_SNAPSHOT.is_file(),
                     "set SEMANTIC_GT_VERIFY_SNAPSHOT=1 to hash protected outputs (~2 min)")
class TestProtectedOutputs(unittest.TestCase):
    def test_protected_outputs_unchanged(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "diagnostics" / "snapshot_protected_outputs.py"),
             "--verify", str(BASELINE_SNAPSHOT)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
