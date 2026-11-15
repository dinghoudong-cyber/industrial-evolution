from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from industrial_evolution.config import load_config
from industrial_evolution.data.synthetic import SyntheticDriftStream
from industrial_evolution.domain import DataWindow
from industrial_evolution.models.sklearn_adapter import SklearnModelAdapter
from industrial_evolution.orchestrator import EvolutionOrchestrator
from industrial_evolution.tracking.local_tracker import LocalTracker
from industrial_evolution.tracking.swanlab_tracker import CompositeTracker

CONFIG_ROOT = Path(__file__).parents[1] / "configs"


class _SmallStream:
    """固定种子的小型流：120 条初始样本 + 2 个 30 样本窗口。"""

    def __init__(self, n_windows: int = 2, window_size: int = 30, initial: int = 120) -> None:
        self.n_windows = n_windows
        self.window_size = window_size
        self.initial = initial

    @staticmethod
    def _sample(size: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(rng.normal(size=(size, 4)), columns=[f"f{i}" for i in range(4)])
        labels = (frame.iloc[:, 0] + frame.iloc[:, 1] > 0).astype(int).to_numpy()
        return frame, labels

    def initial_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        return self._sample(self.initial, 7)

    def __iter__(self):
        for index in range(self.n_windows):
            features, labels = self._sample(self.window_size, 100 + index)
            yield DataWindow(index=index, X=features, y=labels, phase="stable")


def _orchestrator(config, stream, tmp_path):
    local = LocalTracker(tmp_path)
    tracker = CompositeTracker(local)
    return EvolutionOrchestrator(config, stream, tracker), local


def test_label_budget_count_only_never_skips_windows(tmp_path):
    config = load_config(CONFIG_ROOT / "default.yaml")
    config.budget.max_labels = 150  # 初始 120 + 两窗 60 = 180，必然超旧上限
    config.tracking.run_root = tmp_path
    config.tracking.enable_swanlab = False
    orchestrator, _ = _orchestrator(config, _SmallStream(), tmp_path)

    result = orchestrator.run()

    assert config.budget.label_budget_mode == "count_only"
    assert result["label_budget_mode"] == "count_only"
    assert result["skipped_windows"] == 0
    assert result["successful_windows"] == 2
    assert result["failures"] == 0
    assert result["budget"]["labels_used"] == 180
    assert result["budget"]["budget_violations"] == 0


def test_batch_decisions_are_written_for_every_window(tmp_path):
    config = load_config(CONFIG_ROOT / "tuned.yaml")
    config.tracking.run_root = tmp_path
    config.tracking.enable_swanlab = False
    orchestrator, local = _orchestrator(
        config, SyntheticDriftStream(config.stream, config.seed), tmp_path
    )

    result = orchestrator.run()
    decisions = [
        json.loads(line)
        for line in (local.run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(decisions) == result["windows_processed"] == 18
    assert [row["batch"] for row in decisions] == list(range(18))
    assert {row["decision"] for row in decisions} <= {"deploy", "no_update"}
    assert sum(row["decision"] == "deploy" for row in decisions) == result["updates_deployed"]
    assert any(row["history_read"] for row in decisions)
    assert result["min_updates_required"] == 3
    assert result["min_updates_met"] is True


def test_configurable_score_metric_macro_f1():
    features, labels = _SmallStream._sample(160, 11)
    adapter = SklearnModelAdapter(task_type="classification", seed=42, score_metric="macro_f1")
    adapter.fit(features, labels)

    assert adapter.metric_name == "macro_f1"
    assert adapter.new_unfitted().metric_name == "macro_f1"
    assert adapter.score(features, labels) == float(
        f1_score(labels, adapter.predict(features), average="macro", zero_division=0)
    )


def test_official_baseline_comparison_requires_matching_metric(tmp_path):
    config = load_config(CONFIG_ROOT / "default.yaml")
    config.budget.max_labels = 150
    config.validation.official_baseline_score = 0.5
    config.validation.official_metric_name = "macro_f1"  # 与默认 balanced_accuracy 不同
    config.tracking.run_root = tmp_path
    config.tracking.enable_swanlab = False
    orchestrator, _ = _orchestrator(config, _SmallStream(), tmp_path)

    result = orchestrator.run()
    comparison = result["vs_official_baseline"]

    assert comparison["comparable"] is False
    assert comparison["delta"] is None
    assert comparison["meets_baseline"] is False
    assert comparison["project_metric"] == "balanced_accuracy"


def test_official_baseline_comparison_when_metrics_match(tmp_path):
    config = load_config(CONFIG_ROOT / "default.yaml")
    config.budget.max_labels = 150
    config.validation.score_metric = "macro_f1"
    config.validation.official_metric_name = "macro_f1"
    config.validation.official_baseline_score = 0.5
    config.tracking.run_root = tmp_path
    config.tracking.enable_swanlab = False
    orchestrator, _ = _orchestrator(config, _SmallStream(), tmp_path)

    result = orchestrator.run()
    comparison = result["vs_official_baseline"]

    assert result["metric_name"] == "macro_f1"
    assert comparison["comparable"] is True
    assert comparison["delta"] is not None
    assert comparison["meets_baseline"] == (result["final_score"] >= 0.5)
