#!/usr/bin/env python3
"""
Tests for model-expansion integration: size-specific keys/configs, family adapter
settings, shell runtime helpers, and the paired-probe qualification evaluator.

No model weights are loaded; verifiers are replaced by fakes.

    python -m unittest tests.test_model_expansion_integration -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_paired_probe_qualification as probe_eval  # noqa: E402
from src.config.model_config import normalize_model_key  # noqa: E402
from src.verification.registry import get_registered_models, resolve_registry_key  # noqa: E402
from src.lvm import qwen3_vl_verification_adapter as qwen3_adapter  # noqa: E402
from src.lvm.parsers.cleanup import normalize_raw_response  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "configs" / "models"
RUNTIME_SH = PROJECT_ROOT / "jobs" / "lib" / "model_runtime.sh"

QWEN3_SIZES = {
    "qwen3_vl_2b": ("Qwen/Qwen3-VL-2B-Instruct", "2B", "L40S"),
    "qwen3_vl_4b": ("Qwen/Qwen3-VL-4B-Instruct", "4B", "L40S"),
    "qwen3_vl_32b": ("Qwen/Qwen3-VL-32B-Instruct", "32B", "H200"),
}

EXISTING_KEYS = {
    "qwen": "qwen2_5_vl",
    "qwen2_5_vl": "qwen2_5_vl",
    "qwen3_vl": "qwen3_vl",
    "qwen3_vl_8b": "qwen3_vl",
    "qwen3vl": "qwen3_vl",
    "internvl3_5_hf": "internvl3_5_hf",
    "phi4": "phi4_multimodal",
    "glm46v_flash": "glm_4_6v_flash",
    "molmo2": "molmo2_8b",
    "minicpm45": "minicpm_v4_5",
    "gemma4_12b": "gemma4",
}


def run_shell(snippet: str) -> str:
    result = subprocess.run(
        ["bash", "-c", f"source {RUNTIME_SH}; {snippet}"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


class FakeVerifier:
    instances: list["FakeVerifier"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = mock.Mock()
        self.model.config._attn_implementation = kwargs.get("attn_implementation") or "sdpa"
        FakeVerifier.instances.append(self)

    def generate_response(self, **_kwargs):
        return '{"decision": "Reliable", "confidence_reasoning": "", "visual_reasoning": "x"}'


class TestKeysAndConfigs(unittest.TestCase):
    def test_existing_keys_unchanged(self):
        for raw, canonical in EXISTING_KEYS.items():
            self.assertEqual(normalize_model_key(raw), canonical, raw)

    def test_size_keys_have_own_namespace(self):
        registered = get_registered_models()
        for key in QWEN3_SIZES:
            self.assertIn(key, registered)
            self.assertEqual(resolve_registry_key(key), key)
            self.assertEqual(normalize_model_key(key), key)
        self.assertEqual(resolve_registry_key("qwen3_vl_8b"), "qwen3_vl")

    def test_size_configs_pinned_and_distinct(self):
        paths = set()
        for key, (repo, size, gpu) in QWEN3_SIZES.items():
            cfg = yaml.safe_load((CONFIG_DIR / f"{key}.yaml").read_text())
            self.assertEqual(cfg["registry_key"], key)
            self.assertEqual(cfg["hf_repo"], repo)
            self.assertEqual(cfg["parameter_size"], size)
            self.assertEqual(cfg["family"], "qwen3_vl")
            self.assertEqual(cfg["gpu_class"], gpu)
            self.assertRegex(cfg["revision"], r"^[0-9a-f]{40}$")
            self.assertEqual(cfg["attn_implementation"], "sdpa")
            self.assertEqual(cfg["dtype"], "bfloat16")
            self.assertFalse(cfg["trust_remote_code"])
            self.assertIn(repo.split("/")[1], cfg["model_path"])
            paths.add(cfg["model_path"])
        anchor = yaml.safe_load((CONFIG_DIR / "qwen3_vl.yaml").read_text())
        paths.add(anchor["model_path"])
        self.assertEqual(len(paths), 4)

    def test_anchor_config_unchanged(self):
        anchor = yaml.safe_load((CONFIG_DIR / "qwen3_vl.yaml").read_text())
        self.assertIsNone(anchor["attn_implementation"])
        self.assertEqual(anchor["model_label"], "Qwen3-VL-8B-Instruct")
        self.assertNotIn("revision", anchor)

    def test_qwen3_sizes_use_identity_normalizer(self):
        raw = '{"decision": "Reliable"}\n'
        for key in list(QWEN3_SIZES) + ["qwen3_vl"]:
            self.assertEqual(normalize_raw_response(raw, model_key=key), raw)


class TestQwen3FamilyAdapter(unittest.TestCase):
    def setUp(self):
        FakeVerifier.instances.clear()
        patcher = mock.patch.object(qwen3_adapter, "Qwen3VlVerifier", FakeVerifier)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def checkpoint(self, revision: str | None) -> str:
        path = Path(self.tmp.name) / "ckpt"
        path.mkdir(exist_ok=True)
        if revision:
            (path / "DOWNLOAD_META.txt").write_text(f"repo_id=x\nrevision={revision}\n")
        return str(path)

    def test_size_settings_flow_from_config(self):
        cfg = yaml.safe_load((CONFIG_DIR / "qwen3_vl_4b.yaml").read_text())
        adapter = qwen3_adapter.build_qwen3_vl_adapter(
            model_name=self.checkpoint(cfg["revision"]), model_key="qwen3_vl_4b"
        )
        self.assertEqual(adapter.model_label, "Qwen3-VL-4B-Instruct")
        self.assertEqual(FakeVerifier.instances[-1].kwargs["attn_implementation"], "sdpa")
        self.assertEqual(FakeVerifier.instances[-1].kwargs["dtype"], "bfloat16")

    def test_anchor_settings_unchanged(self):
        adapter = qwen3_adapter.build_qwen3_vl_adapter(
            model_name=self.checkpoint(None), model_key="qwen3_vl"
        )
        self.assertEqual(adapter.model_label, "Qwen3-VL-8B-Instruct")
        self.assertIsNone(FakeVerifier.instances[-1].kwargs["attn_implementation"])
        self.assertEqual(FakeVerifier.instances[-1].kwargs["dtype"], "bfloat16")

    def test_revision_mismatch_refused(self):
        with self.assertRaises(ValueError):
            qwen3_adapter.build_qwen3_vl_adapter(
                model_name=self.checkpoint("0" * 40), model_key="qwen3_vl_4b"
            )

    def test_record_carries_provenance(self):
        cfg = yaml.safe_load((CONFIG_DIR / "qwen3_vl_2b.yaml").read_text())
        ckpt = self.checkpoint(cfg["revision"])
        with mock.patch.dict("os.environ", {"WILD_PALM_GIT_COMMIT": "abc123"}):
            adapter = qwen3_adapter.build_qwen3_vl_adapter(model_name=ckpt, model_key="qwen3_vl_2b")
        prompt = Path(self.tmp.name) / "p.txt"
        prompt.write_text("frozen prompt")
        job = mock.Mock(sample_id="sample_000001", image_path=Path("x.png"), prompt_path=prompt)
        outcome = adapter.verify(job)
        self.assertEqual(outcome.status, "ok")
        generation = outcome.record["generation"]
        self.assertEqual(generation["attn_implementation"], "sdpa")
        self.assertEqual(generation["attn_implementation_resolved"], "sdpa")
        self.assertEqual(generation["checkpoint_revision"], cfg["revision"])
        self.assertEqual(generation["git_commit"], "abc123")
        self.assertEqual(outcome.record["model_key"], "qwen3_vl_2b")


class TestShellRuntime(unittest.TestCase):
    def test_existing_shell_keys_unchanged(self):
        self.assertEqual(run_shell("canonicalize_model_key qwen3_vl_8b"), "qwen3_vl")
        self.assertEqual(run_shell("model_venv_path qwen3_vl"), "/deac/csc/yangGrp/luoz23/envs/wild-palm-qwen3vl")
        self.assertEqual(run_shell("model_venv_path internvl3_5_hf"), "")
        self.assertEqual(run_shell("model_default_batch_size qwen2_5_vl"), "4")
        self.assertEqual(
            run_shell("model_default_checkpoint qwen3_vl"),
            "/deac/csc/yangGrp/luoz23/models/Qwen3-VL-8B-Instruct",
        )

    def test_size_keys_share_family_env_not_checkpoint(self):
        checkpoints = set()
        for key in QWEN3_SIZES:
            self.assertEqual(run_shell(f"canonicalize_model_key {key}"), key)
            self.assertEqual(run_shell(f"model_family {key}"), "qwen3_vl")
            self.assertEqual(run_shell(f"model_venv_path {key}"), "/deac/csc/yangGrp/luoz23/envs/wild-palm-qwen3vl")
            self.assertEqual(run_shell(f"model_default_batch_size {key}"), "1")
            cfg = yaml.safe_load((CONFIG_DIR / f"{key}.yaml").read_text())
            checkpoint = run_shell(f"model_default_checkpoint {key}")
            self.assertEqual(checkpoint, cfg["model_path"])
            checkpoints.add(checkpoint)
        self.assertEqual(len(checkpoints), len(QWEN3_SIZES))


def write_probe(root: Path, ids: list[str]) -> Path:
    probe_root = root / "probe_inputs"
    for folder in probe_eval.CONDITION_FOLDERS.values():
        (probe_root / folder).mkdir(parents=True)
        pd.DataFrame({"sample_id": ids}).to_csv(probe_root / folder / "prompt_index.csv", index=False)
    return probe_root


def write_records(condition_dir: Path, records: dict[str, dict]) -> None:
    condition_dir.mkdir(parents=True)
    for sample_id, overrides in records.items():
        record = {
            "sample_id": sample_id, "decision": "Reliable", "raw_response": '{"decision": "Reliable"}',
            "parse_error": "", "inference_error": "", "runtime_seconds": 1.0,
            "generation": {"attn_implementation_resolved": "sdpa", "checkpoint_revision": "r" * 40},
        }
        record.update(overrides)
        (condition_dir / f"{sample_id}.json").write_text(json.dumps(record))


class TestPairedProbeEvaluator(unittest.TestCase):
    CONFIG = {"registry_key": "m", "revision": "r" * 40, "attn_implementation": "sdpa"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = [f"sample_{i:06d}" for i in range(1, 41)]
        self.probe_root = write_probe(self.root, self.ids)
        self.exp = self.root / "exp"

    def build(self, per_condition: dict[str, dict[str, dict]] | None = None):
        per_condition = per_condition or {}
        for code in probe_eval.CONDITIONS:
            records = {s: {} for s in self.ids}
            records.update(per_condition.get(code, {}))
            write_records(self.exp / code, records)
        write_records(self.exp / "determinism_A1", {s: {} for s in self.ids[:5]})
        return probe_eval.evaluate(self.exp, self.probe_root, self.CONFIG, None)

    def test_single_label_collapse_still_passes(self):
        report = self.build()
        self.assertEqual(report["verdict"], "PASS", report["failures"])
        self.assertIn("Uncertain unused", report["behavior"]["notes"])
        self.assertIn("no decision changes across A1-A5", report["behavior"]["notes"])
        self.assertEqual(report["determinism"]["raw_identical"], 5)

    def test_one_parse_failure_in_40_fails_95_gate(self):
        bad = {"decision": "", "parse_error": "bad json", "raw_response": '{"decision": "Reliable", "x": "a\nb"}'}
        report = self.build({"A3": {self.ids[0]: bad, self.ids[1]: bad, self.ids[2]: bad}})
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(report["per_condition"]["A3"]["categories"]["parse_error"], 3)

    def test_inference_error_fails(self):
        report = self.build({"A2": {self.ids[0]: {"decision": "", "inference_error": "CUDA OOM"}}})
        self.assertEqual(report["verdict"], "FAIL")

    def test_garbage_fails(self):
        report = self.build({"A5": {self.ids[0]: {"decision": "", "parse_error": "x", "raw_response": "!!!!"}}})
        self.assertEqual(report["verdict"], "FAIL")
        self.assertTrue(any("garbage" in f for f in report["failures"]))

    def test_truncation_is_unusable_but_not_garbage(self):
        trunc = {"decision": "", "parse_error": "x", "raw_response": '{"decision": "Reli'}
        report = self.build({"A1": {self.ids[0]: trunc}})
        self.assertEqual(report["per_condition"]["A1"]["categories"]["truncated"], 1)
        self.assertEqual(report["per_condition"]["A1"]["usable_rate"], 0.975)
        self.assertEqual(report["verdict"], "PASS")

    def test_invalid_label_fails(self):
        report = self.build({"A4": {self.ids[0]: {"decision": "Maybe"}}})
        self.assertEqual(report["verdict"], "FAIL")

    def test_revision_mismatch_fails(self):
        wrong = {"generation": {"attn_implementation_resolved": "sdpa", "checkpoint_revision": "x" * 40}}
        report = self.build({"A1": {self.ids[0]: wrong}})
        self.assertEqual(report["verdict"], "FAIL")

    def test_transitions_reported(self):
        flips = {s: {"decision": "Unreliable"} for s in self.ids[:10]}
        report = self.build({"A5": flips})
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["behavior"]["paired_transitions"]["A1->A5"]["changed"], 10)
        self.assertEqual(report["behavior"]["responsive_samples"], 10)


class TestDownloadVerifier(unittest.TestCase):
    def test_detects_missing_and_size_mismatch(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))
        import download_pinned_model as dl

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.safetensors").write_bytes(b"12345")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"w": "a.safetensors", "v": "b.safetensors"}})
            )
            problems = dl.verify(root, {"a.safetensors": 5, "config.json": 3})
            self.assertIn("missing: config.json", problems)
            self.assertTrue(any("b.safetensors" in p for p in problems))
            self.assertEqual(dl.verify(root, {"a.safetensors": 6})[0].startswith("size mismatch"), True)


if __name__ == "__main__":
    unittest.main()
