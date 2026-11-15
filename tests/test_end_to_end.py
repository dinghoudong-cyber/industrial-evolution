import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_evolution.config import load_config
from industrial_evolution.data.synthetic import SyntheticDriftStream
from industrial_evolution.domain import Action, DataWindow, Decision
from industrial_evolution.orchestrator import EvolutionOrchestrator
from industrial_evolution.tracking.local_tracker import LocalTracker
from industrial_evolution.tracking.swanlab_tracker import CompositeTracker


class LabelBudgetStream:
    def initial_data(self):
        rng = np.random.default_rng(17)
        X = pd.DataFrame(rng.normal(size=(120, 8)), columns=[f"f{i}" for i in range(8)])
        y = (X["f0"] > 0).astype(int).to_numpy()
        return X, y

    def __init__(self, windows: int = 2):
        self.windows = windows

    def __iter__(self):
        rng = np.random.default_rng(23)
        for index in range(self.windows):
            X = pd.DataFrame(rng.normal(size=(30, 8)), columns=[f"f{i}" for i in range(8)])
            y = (X["f0"] > 0).astype(int).to_numpy()
            start = 120 + index * 30
            yield DataWindow(index, X, y, sample_ids=np.arange(start, start + 30))


class CountingNoUpdatePolicy:
    def __init__(self):
        self.calls = 0

    def decide(self, context):
        self.calls += 1
        return Decision(Action.NO_UPDATE, "测试策略保持模型")

    def mark_update(self, window_index):
        pass

    def snapshot_state(self):
        return {"calls": self.calls}

    def restore_state(self, state):
        self.calls = int(state.get("calls", 0))


def test_full_stream_runs_without_manual_intervention(tmp_path):
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = load_config(config_path)
    config.stream.initial_samples = 320
    config.stream.window_size = 100
    config.stream.n_windows = 14
    config.stream.gradual_drift_start = 4
    config.stream.abrupt_drift_start = 9
    config.training.replay_capacity = 800
    config.training.max_replay_samples = 300
    config.validation.guard_samples = 200
    config.tracking.run_root = tmp_path

    local = LocalTracker(tmp_path)
    tracker = CompositeTracker(local)
    result = EvolutionOrchestrator(
        config,
        SyntheticDriftStream(config.stream, config.seed),
        tracker,
    ).run()

    assert result["windows_processed"] == 14
    assert result["successful_windows"] == 14
    assert result["failures"] == 0
    assert result["updates_attempted"] >= 1
    assert result["budget"]["labels_used"] == 320 + 14 * 100
    assert (local.run_dir / "events.jsonl").exists()
    assert (local.run_dir / "metrics.csv").exists()
    events = [
        json.loads(line)
        for line in (local.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    hypotheses = [event for event in events if event["event_type"] == "hypothesis_recorded"]
    assert len(hypotheses) == 14
    for hypothesis_index, event in enumerate(events):
        if event["event_type"] != "hypothesis_recorded":
            continue
        chain = event["payload"]
        assert {
            "observation",
            "diagnosis",
            "decision",
            "expectation",
            "outcome",
            "verdict",
            "history_read",
        }.issubset(chain)
        window = chain["window"]
        assert all(item["window"] < window for item in chain["history_read"])
        diagnosis_index = next(
            index
            for index, item in enumerate(events[:hypothesis_index])
            if item["event_type"] == "drift_diagnosis" and item["payload"].get("window") == window
        )
        assert diagnosis_index < hypothesis_index
        decision_indices = [
            index
            for index, item in enumerate(events[:hypothesis_index])
            if item["event_type"] == "update_decision" and item["payload"].get("window") == window
        ]
        history_index = next(
            index
            for index, item in enumerate(events[:hypothesis_index])
            if item["event_type"] == "experiment_history_read"
            and item["payload"].get("window") == window
        )
        assert not decision_indices or history_index < max(decision_indices)
        history_event = events[history_index]["payload"]
        assert history_event["entries"] == chain["history_read"]
        assert not decision_indices or max(decision_indices) < hypothesis_index
    summary = json.loads((local.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_model_version"].startswith("v")
    reproducibility = summary["reproducibility"]
    assert reproducibility["source_sha256"].startswith("sha256:")
    assert reproducibility["config_sha256"].startswith("sha256:")
    assert reproducibility["python_version"]
    assert "numpy" in reproducibility["dependency_versions"]
    assert "container_image_digest" in reproducibility


def test_label_budget_skips_window_without_diagnosis_or_failure(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    config.budget.max_labels = 150  # 初始集 + 一个 30 样本窗口恰好达到上限。
    config.budget.label_budget_mode = "enforce"  # 显式启用旧的「标签需申请配额」模拟
    config.tracking.run_root = tmp_path
    policy = CountingNoUpdatePolicy()
    local = LocalTracker(tmp_path)
    orchestrator = EvolutionOrchestrator(
        config,
        LabelBudgetStream(windows=2),
        CompositeTracker(local),
        policy=policy,
    )

    result = orchestrator.run()

    assert result["windows_processed"] == 2
    assert result["successful_windows"] == 1
    assert result["skipped_windows"] == 1
    assert result["failures"] == 0
    assert result["run_status"] == "partial"
    assert result["action_counts"] == {"no_update": 2}
    assert result["budget"]["labels_used"] == 150
    assert result["budget"]["budget_violations"] == 0
    assert policy.calls == 1
    assert not set(range(150, 180)).intersection(orchestrator.replay.sample_ids.tolist())

    events = [
        json.loads(line) for line in local.events_path.read_text(encoding="utf-8").splitlines()
    ]
    skipped_event = next(event for event in events if event["event_type"] == "window_skipped")
    assert skipped_event["payload"]["window"] == 1
    assert skipped_event["payload"]["requested_new_labels"] == 30
    assert not any(
        event["event_type"] == "drift_diagnosis" and event["payload"].get("window") == 1
        for event in events
    )
    assert not any(
        event["event_type"] == "window_failure" and event["payload"].get("window") == 1
        for event in events
    )
    hypothesis = next(
        event["payload"]
        for event in events
        if event["event_type"] == "hypothesis_recorded" and event["payload"]["window"] == 1
    )
    assert hypothesis["outcome"]["status"] == "budget_skipped"
    assert hypothesis["diagnosis"] == "标签预算不足，未执行评分与漂移诊断"
    assert hypothesis["verdict"] == "inconclusive"

    with local.metrics_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    skipped_row = next(row for row in rows if row["window"] == "1")
    assert skipped_row["status"] == "budget_skipped"
    assert skipped_row["action"] == "no_update"
    assert skipped_row["score"] == ""


def test_all_label_budget_skipped_windows_remain_failed_run(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    config.budget.max_labels = 120
    config.budget.label_budget_mode = "enforce"
    config.tracking.run_root = tmp_path
    policy = CountingNoUpdatePolicy()
    local = LocalTracker(tmp_path)
    orchestrator = EvolutionOrchestrator(
        config,
        LabelBudgetStream(windows=1),
        CompositeTracker(local),
        policy=policy,
    )

    result = orchestrator.run()

    assert result["windows_processed"] == 1
    assert result["successful_windows"] == 0
    assert result["skipped_windows"] == 1
    assert result["failures"] == 0
    assert result["run_status"] == "failed"
    assert result["action_counts"] == {"no_update": 1}
    assert result["budget"]["labels_used"] == 120
    assert policy.calls == 0
    assert not set(range(120, 150)).intersection(orchestrator.replay.sample_ids.tolist())


def test_degraded_deployment_is_rolled_back_with_full_state(tmp_path):
    config_path = Path(__file__).parents[1] / "configs" / "tuned.yaml"
    config = load_config(config_path)
    config.policy.rollback_drop = 0.04
    config.tracking.run_root = tmp_path

    local = LocalTracker(tmp_path)
    result = EvolutionOrchestrator(
        config,
        SyntheticDriftStream(config.stream, config.seed),
        CompositeTracker(local),
    ).run()

    assert result["failures"] == 0
    assert result["rollbacks"] >= 1
    events = (local.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"event_type": "model_rolled_back"' in events
    assert "完整监测状态" in events


def test_tuned_profile_preserves_deterministic_leak_free_behavior(tmp_path):
    config_path = Path(__file__).parents[1] / "configs" / "tuned.yaml"
    config = load_config(config_path)
    config.tracking.run_root = tmp_path

    local = LocalTracker(tmp_path)
    result = EvolutionOrchestrator(
        config,
        SyntheticDriftStream(config.stream, config.seed),
        CompositeTracker(local),
    ).run()

    assert result["run_id"] == local.run_id
    assert result["windows_processed"] == 18
    assert result["successful_windows"] == 18
    assert result["updates_attempted"] == 6
    assert result["updates_deployed"] == 4
    assert result["updates_rejected"] == 2
    assert result["rollbacks"] == 0
    assert result["failures"] == 0
    assert result["initial_score"] == 0.734093
    assert result["final_score"] == 0.69906
    assert result["mean_score"] == 0.6762
    assert result["run_status"] == "complete"
    assert result["action_counts"] == {"no_update": 12, "retrain_with_replay": 6}
    assert result["budget"]["labels_used"] == 4200
    assert result["budget"]["budget_violations"] == 0
    assert result["final_model_version"] == "v003"
    assert result["final_score_model_version"] == "v003"
    assert result["final_deployed_model_version"] == "v004"
    # 最后窗口之后上线的版本不会被任何窗口评分，因此单独记录它上线时的未见数据得分。
    assert result["final_deployed_score"] == 0.660055
    assert result["final_deployed_guard_score"] == 0.600167
    assert result["final_deployed_score_source"] == "candidate_holdout_window_17"
    assert result["final_deployed_window"] == 17
    assert result["metric_name"] == "balanced_accuracy"

    with local.metrics_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    deployed = [row for row in rows if row["status"] == "deployed"]
    assert deployed
    assert all(row["scored_model_version"] != row["deployed_model_version"] for row in deployed)


def test_all_invalid_windows_finish_with_failed_status(tmp_path):
    class InvalidLabelStream:
        def initial_data(self):
            X = pd.DataFrame({"x": np.arange(120), "z": np.arange(120) % 3})
            y = np.array([0, 1] * 60)
            return X, y

        def __iter__(self):
            X = pd.DataFrame({"x": np.arange(30), "z": np.arange(30) % 3})
            invalid_y = np.linspace(0.1, 0.9, 30)
            yield DataWindow(0, X, invalid_y, sample_ids=np.arange(120, 150))

    config = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    config.tracking.run_root = tmp_path
    local = LocalTracker(tmp_path)
    result = EvolutionOrchestrator(
        config,
        InvalidLabelStream(),
        CompositeTracker(local),
    ).run()

    assert result["windows_processed"] == 1
    assert result["successful_windows"] == 0
    assert result["failures"] == 1
    assert result["run_status"] == "failed"
    assert result["action_counts"] == {"error": 1}


def test_static_profile_labels_missing_deployment_score_explicitly(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs" / "static.yaml")
    config.tracking.run_root = tmp_path
    local = LocalTracker(tmp_path)

    result = EvolutionOrchestrator(
        config,
        SyntheticDriftStream(config.stream, config.seed),
        CompositeTracker(local),
    ).run()

    assert result["updates_deployed"] == 0
    assert result["final_deployed_model_version"] == result["final_score_model_version"]
    # 从未通过候选评估上线，就不该编造一个部署得分。
    assert result["final_deployed_score"] is None
    assert result["final_deployed_score_source"].startswith("none:")
    events = (local.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"event_type": "final_deployment_score_missing"' in events
