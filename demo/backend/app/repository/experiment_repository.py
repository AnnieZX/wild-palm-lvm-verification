"""Filesystem-backed read-only access to verification experiment outputs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.models import (
    AblationCode,
    BoundingBox,
    ExperimentSummary,
    GroundTruthLabel,
    ModelInfo,
    SampleDetail,
    SampleSummary,
    StatisticsResponse,
    VerificationDecision,
)

from app.repository.constants import (
    ABLATION_CODE_TO_CONDITION,
    ALL_ABLATION_CODES,
    METRICS_JSON_SUFFIX,
    MODEL_DESCRIPTIONS,
    MODEL_DISPLAY_NAMES,
)
from app.repository.metrics import (
    compute_metrics,
    metrics_to_ablation_statistics,
    normalize_matched_gt,
    normalize_verification_label,
    safe_float,
)
from app.repository.paths import (
    canonical_model_key,
    discover_ablation_inputs_dir,
    discover_evaluation_csv,
    evaluation_condition_dir,
    list_experiment_dirs,
    list_verification_model_dirs,
    verification_condition_dir,
)


class ExperimentRepository:
    """Discover and serve experiment artifacts from outputs/."""

    def __init__(self, outputs_root: Path) -> None:
        self.outputs_root = outputs_root.resolve()
        self._catalog: Optional[List[ModelInfo]] = None
        self._evaluation_rows_cache: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
        self._verification_json_cache: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        self._overlay_path_cache: Dict[Tuple[str, str], Optional[Path]] = {}
        self._ablation_inputs_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Defaults for API query parameters
    # ------------------------------------------------------------------

    @property
    def default_model_key(self) -> str:
        models = self.list_models()
        if not models:
            return ""
        return self._model_with_latest_experiment(models).model_key

    @property
    def default_experiment_id(self) -> str:
        model = self.resolve_model(self.default_model_key)
        if model is None or not model.experiments:
            return ""
        return model.experiments[0].experiment_id

    @staticmethod
    def _model_with_latest_experiment(models: List[ModelInfo]) -> ModelInfo:
        def latest_created_at(model: ModelInfo) -> str:
            if not model.experiments:
                return ""
            return model.experiments[0].created_at

        return max(models, key=latest_created_at)

    @property
    def default_ablation(self) -> AblationCode:
        model = self.resolve_model(self.default_model_key)
        if model is None or not model.experiments:
            return AblationCode.A1
        experiment = model.experiments[0]
        return experiment.primary_ablation

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def list_models(self) -> List[ModelInfo]:
        if self._catalog is None:
            self._catalog = self._build_catalog()
        return self._catalog

    def resolve_model(self, model_key: str) -> Optional[ModelInfo]:
        canonical = canonical_model_key(model_key)
        return next((model for model in self.list_models() if model.model_key == canonical), None)

    def resolve_experiment(self, model: ModelInfo, experiment_id: str) -> Optional[ExperimentSummary]:
        return next(
            (experiment for experiment in model.experiments if experiment.experiment_id == experiment_id),
            None,
        )

    def _build_catalog(self) -> List[ModelInfo]:
        experiments_by_model: Dict[str, Dict[str, ExperimentSummary]] = {}

        for dir_name, model_dir in list_verification_model_dirs(self.outputs_root):
            api_key = canonical_model_key(dir_name)
            experiments_by_model.setdefault(api_key, {})

            for experiment_dir in list_experiment_dirs(model_dir):
                experiment_id = experiment_dir.name
                ablations = self._discover_ablations(experiment_dir)
                if not ablations:
                    continue

                sample_count = self._experiment_sample_count(api_key, experiment_id, ablations[0])
                primary_ablation = self._resolve_primary_ablation(api_key, experiment_id, ablations)
                summary = ExperimentSummary(
                    experiment_id=experiment_id,
                    sample_count=sample_count,
                    ablations=ablations,
                    primary_ablation=primary_ablation,
                    created_at=self._experiment_created_at(experiment_dir, experiment_id),
                )
                existing = experiments_by_model[api_key].get(experiment_id)
                if existing is None or self._is_newer_experiment_summary(summary, existing):
                    experiments_by_model[api_key][experiment_id] = summary

        models: List[ModelInfo] = []
        for model_key in sorted(experiments_by_model):
            experiments = sorted(
                experiments_by_model[model_key].values(),
                key=lambda item: item.created_at,
                reverse=True,
            )
            models.append(
                ModelInfo(
                    model_key=model_key,
                    display_name=MODEL_DISPLAY_NAMES.get(
                        model_key,
                        model_key.replace("_", " ").title(),
                    ),
                    description=MODEL_DESCRIPTIONS.get(model_key, ""),
                    experiments=experiments,
                )
            )
        return models

    @staticmethod
    def _is_newer_experiment_summary(
        candidate: ExperimentSummary,
        existing: ExperimentSummary,
    ) -> bool:
        if candidate.sample_count != existing.sample_count:
            return candidate.sample_count > existing.sample_count
        if len(candidate.ablations) != len(existing.ablations):
            return len(candidate.ablations) > len(existing.ablations)
        return candidate.created_at >= existing.created_at

    def _discover_ablations(self, experiment_dir: Path) -> List[AblationCode]:
        discovered: List[AblationCode] = []
        for ablation in ALL_ABLATION_CODES:
            condition_dir = experiment_dir / ablation.value
            if not condition_dir.is_dir():
                continue
            if (
                list(condition_dir.glob("sample_*.json"))
                or (condition_dir / "results_index.csv").exists()
            ):
                discovered.append(ablation)
        return discovered

    def _experiment_sample_count(
        self,
        model_key: str,
        experiment_id: str,
        ablation: AblationCode,
    ) -> int:
        rows = self._load_evaluation_rows(model_key, experiment_id, ablation.value)
        if rows:
            return len(rows)
        verification_dir = verification_condition_dir(
            self.outputs_root,
            model_key,
            experiment_id,
            ablation.value,
        )
        return len(list(verification_dir.glob("sample_*.json"))) if verification_dir.exists() else 0

    def _resolve_primary_ablation(
        self,
        model_key: str,
        experiment_id: str,
        ablations: List[AblationCode],
    ) -> AblationCode:
        best_code = ablations[0]
        best_f1 = -1.0
        for ablation in ablations:
            metrics_path = (
                evaluation_condition_dir(
                    self.outputs_root,
                    model_key,
                    experiment_id,
                    ablation.value,
                )
                / f"{ablation.value}{METRICS_JSON_SUFFIX}"
            )
            if not metrics_path.exists():
                continue
            with metrics_path.open(encoding="utf-8") as file:
                metrics = json.load(file)
            f1 = float(metrics.get("f1", 0.0))
            if f1 >= best_f1:
                best_f1 = f1
                best_code = ablation
        return best_code

    def _experiment_created_at(self, experiment_dir: Path, experiment_id: str) -> str:
        try:
            parsed = datetime.strptime(experiment_id, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
            return parsed.isoformat().replace("+00:00", "Z")
        except ValueError:
            pass

        latest_mtime = experiment_dir.stat().st_mtime
        for json_path in experiment_dir.rglob("sample_*.json"):
            latest_mtime = max(latest_mtime, json_path.stat().st_mtime)
        return datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(
        self,
        model_key: str,
        experiment_id: str,
        ablation: AblationCode,
    ) -> Optional[StatisticsResponse]:
        model = self.resolve_model(model_key)
        if model is None:
            return None
        experiment = self.resolve_experiment(model, experiment_id)
        if experiment is None or ablation not in experiment.ablations:
            return None

        evaluation_dir = evaluation_condition_dir(
            self.outputs_root,
            model_key,
            experiment_id,
            ablation.value,
        )
        metrics_path = evaluation_dir / f"{ablation.value}{METRICS_JSON_SUFFIX}"
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as file:
                metrics = json.load(file)
        else:
            rows = self._load_evaluation_rows(model_key, experiment_id, ablation.value)
            if not rows:
                return None
            metrics = compute_metrics(rows, ablation.value)

        statistics = metrics_to_ablation_statistics(metrics, ablation)
        return StatisticsResponse(
            model_key=canonical_model_key(model_key),
            experiment_id=experiment_id,
            statistics=statistics,
        )

    # ------------------------------------------------------------------
    # Samples
    # ------------------------------------------------------------------

    def list_sample_summaries(
        self,
        *,
        model_key: str,
        experiment_id: str,
        ablation: AblationCode,
        decision: Optional[VerificationDecision] = None,
        matched_gt: Optional[bool] = None,
        gt_label: Optional[GroundTruthLabel] = None,
    ) -> List[SampleSummary]:
        rows = self._load_evaluation_rows(model_key, experiment_id, ablation.value)
        summaries = [self._row_to_summary(row, model_key, experiment_id, ablation) for row in rows]

        if decision is not None:
            summaries = [item for item in summaries if item.decision == decision]
        if matched_gt is not None:
            summaries = [item for item in summaries if item.matched_gt == matched_gt]
        if gt_label is not None:
            summaries = [item for item in summaries if item.gt_label == gt_label]
        return summaries

    def get_sample_detail(
        self,
        sample_id: str,
        *,
        model_key: str,
        experiment_id: str,
        ablation: AblationCode,
    ) -> Optional[SampleDetail]:
        row = self._find_evaluation_row(model_key, experiment_id, ablation.value, sample_id)
        if row is None:
            return None

        summary = self._row_to_summary(row, model_key, experiment_id, ablation)
        verification = self._load_verification_json(model_key, experiment_id, ablation.value, sample_id)

        return SampleDetail(
            **summary.model_dump(),
            yolo_bbox=self._parse_bbox(row.get("yolo_bbox")),
            gt_bbox=self._parse_bbox(row.get("gt_bbox")),
            confidence_reasoning=str(verification.get("confidence_reasoning", "") or ""),
            visual_reasoning=str(verification.get("visual_reasoning", "") or ""),
            image_path=f"/api/v1/image/{sample_id}",
        )

    def get_overlay_image_path(
        self,
        sample_id: str,
        *,
        model_key: str,
        experiment_id: str,
        ablation: AblationCode,
    ) -> Optional[Path]:
        cache_key = (ablation.value, sample_id)
        if cache_key in self._overlay_path_cache:
            cached = self._overlay_path_cache[cache_key]
            return cached if cached is not None and cached.exists() else None

        condition_name = ABLATION_CODE_TO_CONDITION[ablation.value]
        ablation_inputs_dir = self._resolve_ablation_inputs_dir(model_key, experiment_id)
        if ablation_inputs_dir is None:
            self._overlay_path_cache[cache_key] = None
            return None

        condition_dir = ablation_inputs_dir / condition_name
        prompt_index_path = condition_dir / "prompt_index.csv"
        candidate_paths: List[Path] = []

        if prompt_index_path.exists():
            image_rel = self._lookup_prompt_index_image(prompt_index_path, sample_id)
            if image_rel:
                candidate_paths.append((condition_dir / image_rel).resolve())

        candidate_paths.append((condition_dir / "images" / f"{sample_id}.png").resolve())

        for path in candidate_paths:
            if path.exists():
                self._overlay_path_cache[cache_key] = path
                return path

        self._overlay_path_cache[cache_key] = None
        return None

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _resolve_ablation_inputs_dir(self, model_key: str, experiment_id: str) -> Optional[Path]:
        if self._ablation_inputs_dir is not None and self._ablation_inputs_dir.exists():
            return self._ablation_inputs_dir

        sample_count = self._experiment_sample_count(
            model_key,
            experiment_id,
            AblationCode.A1,
        )
        discovered = discover_ablation_inputs_dir(self.outputs_root, sample_count or None)
        if discovered is not None:
            self._ablation_inputs_dir = discovered
        return discovered

    def _load_evaluation_rows(
        self,
        model_key: str,
        experiment_id: str,
        ablation_code: str,
    ) -> List[Dict[str, str]]:
        cache_key = (canonical_model_key(model_key), experiment_id, ablation_code)
        if cache_key in self._evaluation_rows_cache:
            return self._evaluation_rows_cache[cache_key]

        evaluation_dir = evaluation_condition_dir(
            self.outputs_root,
            model_key,
            experiment_id,
            ablation_code,
        )
        csv_path = discover_evaluation_csv(evaluation_dir)
        if csv_path is None:
            self._evaluation_rows_cache[cache_key] = []
            return []

        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self._evaluation_rows_cache[cache_key] = rows
        return rows

    def _find_evaluation_row(
        self,
        model_key: str,
        experiment_id: str,
        ablation_code: str,
        sample_id: str,
    ) -> Optional[Dict[str, str]]:
        for row in self._load_evaluation_rows(model_key, experiment_id, ablation_code):
            if str(row.get("sample_id", "")).strip() == sample_id:
                return row
        return None

    def _load_verification_json(
        self,
        model_key: str,
        experiment_id: str,
        ablation_code: str,
        sample_id: str,
    ) -> Dict[str, Any]:
        cache_key = (canonical_model_key(model_key), experiment_id, ablation_code, sample_id)
        if cache_key in self._verification_json_cache:
            return self._verification_json_cache[cache_key]

        json_path = (
            verification_condition_dir(
                self.outputs_root,
                model_key,
                experiment_id,
                ablation_code,
            )
            / f"{sample_id}.json"
        )
        if not json_path.exists():
            self._verification_json_cache[cache_key] = {}
            return {}

        with json_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        result = payload if isinstance(payload, dict) else {}
        self._verification_json_cache[cache_key] = result
        return result

    def _lookup_prompt_index_image(self, prompt_index_path: Path, sample_id: str) -> Optional[str]:
        with prompt_index_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if str(row.get("sample_id", "")).strip() == sample_id:
                    image_path = str(row.get("image_path", "")).strip()
                    return image_path or None
        return None

    def _row_to_summary(
        self,
        row: Dict[str, str],
        model_key: str,
        experiment_id: str,
        ablation: AblationCode,
    ) -> SampleSummary:
        sample_id = str(row.get("sample_id", "")).strip()
        verification = self._load_verification_json(model_key, experiment_id, ablation.value, sample_id)

        label = normalize_verification_label(
            row.get("verification_label") or verification.get("decision")
        )
        decision = self._label_to_decision(label)

        matched_gt = normalize_matched_gt(row.get("matched_gt"))
        max_iou = safe_float(row.get("max_iou")) or 0.0
        yolo_confidence = safe_float(row.get("yolo_confidence"))

        return SampleSummary(
            sample_id=sample_id,
            image_name=str(row.get("image_name", "")).strip(),
            ablation=ablation,
            decision=decision,
            matched_gt=matched_gt,
            max_iou=max_iou,
            yolo_confidence=yolo_confidence,
            gt_label=self._derive_gt_label(matched_gt, max_iou),
        )

    @staticmethod
    def _label_to_decision(label: str) -> Optional[VerificationDecision]:
        if not label:
            return None
        for decision in VerificationDecision:
            if decision.value.lower() == label.lower():
                return decision
        return None

    @staticmethod
    def _derive_gt_label(matched_gt: bool, max_iou: float) -> GroundTruthLabel:
        if matched_gt and max_iou >= 0.5:
            return GroundTruthLabel.POSITIVE
        if not matched_gt and max_iou < 0.3:
            return GroundTruthLabel.NEGATIVE
        return GroundTruthLabel.UNCERTAIN

    @staticmethod
    def _parse_bbox(value: Any) -> Optional[BoundingBox]:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            if isinstance(value, str):
                data = json.loads(value)
            else:
                data = value
            if not isinstance(data, (list, tuple)) or len(data) != 4:
                return None
            x, y, width, height = (float(item) for item in data)
            return BoundingBox(x=x, y=y, width=width, height=height)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
