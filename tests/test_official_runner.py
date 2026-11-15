from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

import industrial_evolution.cli as cli
from industrial_evolution.cli import build_parser
from industrial_evolution.data.official_jsonl import flatten_inputs
from industrial_evolution.models.official_adapter import A_CLASSES, OfficialBaselineAdapter
from industrial_evolution.official_runner import evaluate_public_package, predict_jsonl
from industrial_evolution.tracking.local_tracker import LocalTracker
from industrial_evolution.tracking.swanlab_tracker import CompositeTracker


def _row(identity: str, batch: int) -> dict:
    return {
        "id": identity,
        "inputs": {"features": [float(index) for index in range(128)]},
        "answer": "Ethanol",
        "meta": {"source_batch": batch, "source_index": 1},
    }


def _make_pack(root: Path) -> tuple[Path, list[str]]:
    config_dir = root / "baselines" / "configs"
    config_dir.mkdir(parents=True)
    (root / "checkpoint.npz").write_bytes(b"baseline-checkpoint")
    checkpoint_hash = "sha256:" + hashlib.sha256((root / "checkpoint.npz").read_bytes()).hexdigest()
    (root / "baseline-info.json").write_text(
        json.dumps({"dataset_key": "A", "checkpoint_sha256": checkpoint_hash}),
        encoding="utf-8",
    )
    (config_dir / "A.json").write_text(
        json.dumps({"profile": "A", "task": "classification", "classes": A_CLASSES}),
        encoding="utf-8",
    )
    (root / "training.json").write_text(
        json.dumps(
            {
                "profile": "A",
                "task": "classification",
                "purpose": "development_reference",
                "official_scoring_ready": False,
                "checkpoint_sha256": checkpoint_hash,
                "label_provenance": None,
            }
        ),
        encoding="utf-8",
    )
    paths, _ = flatten_inputs({"features": [float(index) for index in range(128)]})
    return root, list(paths)


def _write_dataset(path: Path, row: dict) -> Path:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


class _FakeModel:
    def __init__(self, paths: list[str], candidate: bool = False):
        self.meta_json = json.dumps({"paths": paths})
        self.task = "classification"
        self.classes = A_CLASSES
        self.candidate = candidate
        self.fit_called = False

    def validate_profile(self):
        return None

    def official_metric(self, X, y):
        return {"macro_f1": 0.6 if self.candidate else 0.5, "accuracy": 0.7}

    def new_unfitted(self):
        return _FakeModel(json.loads(self.meta_json)["paths"], candidate=True)

    def fit(self, X, y):
        self.fit_called = True
        return self

    def save_checkpoint(self, path):
        path.write_bytes(b"candidate")
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    def predict_proba(self, X):
        return np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


def test_official_evaluate_accepts_tracking_config_and_swanlab_overrides():
    args = build_parser().parse_args(
        [
            "official-evaluate",
            "--pack-root",
            "pack",
            "--profile",
            "A",
            "--train",
            "train.jsonl",
            "--validation",
            "validation.jsonl",
            "--config",
            "configs/llm-local.yaml",
            "--swanlab",
            "--swanlab-mode",
            "offline",
        ]
    )

    assert args.config == Path("configs/llm-local.yaml")
    assert args.swanlab is True
    assert args.swanlab_mode == "offline"


def test_official_evaluate_builds_and_passes_configured_tracker(tmp_path: Path, monkeypatch):
    captured = {}
    tracker = object()

    def build_tracker(config):
        captured["tracking"] = config.tracking
        return tracker

    def evaluate_public_package(**kwargs):
        captured["kwargs"] = kwargs
        return {"run_status": "complete"}

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "industrial-evolution",
            "official-evaluate",
            "--pack-root",
            "pack",
            "--profile",
            "A",
            "--train",
            "train.jsonl",
            "--validation",
            "validation.jsonl",
            "--config",
            "configs/official.yaml",
            "--run-root",
            str(tmp_path / "runs"),
            "--swanlab",
            "--swanlab-mode",
            "online",
        ],
    )
    monkeypatch.setattr(cli, "_build_tracker", build_tracker)
    monkeypatch.setattr(cli, "evaluate_public_package", evaluate_public_package)

    assert cli.main() == 0
    assert captured["tracking"].enable_swanlab is True
    assert captured["tracking"].swanlab_mode == "online"
    assert captured["tracking"].run_root == (tmp_path / "runs").resolve()
    assert captured["kwargs"]["tracker"] is tracker


def test_public_candidate_is_warm_started_and_saved_only_after_baseline_gate(
    tmp_path: Path, monkeypatch
):
    pack, paths = _make_pack(tmp_path / "pack")
    train = _write_dataset(tmp_path / "train.jsonl", _row("train", 1))
    validation = _write_dataset(tmp_path / "validation.jsonl", _row("validation", 2))
    baseline = _FakeModel(paths)
    monkeypatch.setattr(
        OfficialBaselineAdapter,
        "from_pack",
        classmethod(lambda cls, *args, **kwargs: baseline),
    )

    result = evaluate_public_package(
        pack_root=pack,
        profile="A",
        train_path=train,
        validation_path=validation,
        run_root=tmp_path / "runs",
        epochs=1,
    )

    assert result["candidate_status"] == "accepted"
    assert result["baseline_metrics"]["macro_f1"] == 0.5
    assert Path(result["candidate_checkpoint"]).read_bytes() == b"candidate"
    assert result["network_used"] is False
    assert result["official_scoring_ready"] is False


def test_public_evaluation_mirrors_events_metrics_and_summary(tmp_path: Path, monkeypatch):
    class RecordingMirror:
        def __init__(self):
            self.events = []
            self.metrics = []
            self.summaries = []

        def log_event(self, event_type, payload):
            self.events.append((event_type, payload))

        def log_metrics(self, row):
            self.metrics.append(row)

        def log_hypothesis(self, record):
            pass

        def close(self, summary):
            self.summaries.append(summary)

    pack, paths = _make_pack(tmp_path / "pack")
    train = _write_dataset(tmp_path / "train.jsonl", _row("train", 1))
    validation = _write_dataset(tmp_path / "validation.jsonl", _row("validation", 2))
    model = _FakeModel(paths)
    monkeypatch.setattr(
        OfficialBaselineAdapter,
        "from_pack",
        classmethod(lambda cls, *args, **kwargs: model),
    )
    local = LocalTracker(tmp_path / "runs")
    mirror = RecordingMirror()
    tracker = CompositeTracker(local, [mirror])

    result = evaluate_public_package(
        pack_root=pack,
        profile="A",
        train_path=train,
        validation_path=validation,
        run_root=tmp_path / "unused-runs",
        tracker=tracker,
    )

    assert result["run_id"] == local.run_id
    assert result["run_dir"] == str(local.run_dir)
    assert (local.run_dir / "summary.json").is_file()
    assert [event for event, _ in mirror.events] == ["public_baseline_evaluated"]
    assert mirror.metrics[0]["stage"] == "baseline"
    assert mirror.summaries[0]["run_id"] == local.run_id


def test_predict_jsonl_outputs_only_reference_predictions(tmp_path: Path, monkeypatch):
    pack, paths = _make_pack(tmp_path / "pack")
    model = _FakeModel(paths)
    monkeypatch.setattr(
        OfficialBaselineAdapter,
        "from_pack",
        classmethod(lambda cls, *args, **kwargs: model),
    )
    request = json.dumps({"task": "classification", "inputs": _row("x", 1)["inputs"]})
    output = io.StringIO()

    predict_jsonl(pack_root=pack, profile="A", stdin=io.StringIO(request + "\n"), stdout=output)

    parsed = json.loads(output.getvalue())
    assert parsed["outputs"][0] == {"label": "Ethanol", "score": 1.0}


def test_predict_jsonl_rejects_answers_in_requests(tmp_path: Path, monkeypatch):
    pack, paths = _make_pack(tmp_path / "pack")
    model = _FakeModel(paths)
    monkeypatch.setattr(
        OfficialBaselineAdapter,
        "from_pack",
        classmethod(lambda cls, *args, **kwargs: model),
    )
    request = json.dumps(
        {"task": "classification", "inputs": _row("x", 1)["inputs"], "answer": "Ethanol"}
    )

    with pytest.raises(ValueError, match="只能包含 task 和 inputs"):
        predict_jsonl(pack_root=pack, profile="A", stdin=io.StringIO(request), stdout=io.StringIO())
