from __future__ import annotations

import json

import pytest

from industrial_evolution.config import TrackingConfig
from industrial_evolution.tracking.local_tracker import LocalTracker
from industrial_evolution.tracking.swanlab_tracker import CompositeTracker, SwanLabTracker


class FlakyMirror:
    """前 n 次调用失败，之后恢复——用来验证镜像不会被一次抖动永久踢掉。"""

    def __init__(self, fail_times: int) -> None:
        self.remaining = fail_times
        self.calls = 0
        self.hypotheses: list[dict] = []

    def log_event(self, event_type, payload):
        self.calls += 1
        if self.remaining:
            self.remaining -= 1
            raise RuntimeError("network hiccup")

    def log_metrics(self, row):
        self.calls += 1

    def log_hypothesis(self, record):
        self.calls += 1
        self.hypotheses.append(record)

    def close(self, summary):
        self.calls += 1


class BrokenMirror(FlakyMirror):
    def __init__(self) -> None:
        super().__init__(fail_times=10_000)


def test_hypothesis_lands_in_local_events(tmp_path):
    tracker = LocalTracker(tmp_path, run_id="hypothesis-test")
    tracker.log_hypothesis(
        {
            "window": 3,
            "phase": "gradual",
            "decision": {
                "action": "error",
                "proposed_action": "retrain_with_replay",
                "fallback_reason": None,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "reasoning_tokens": 0,
                    "total_tokens": 120,
                },
            },
            "verdict": "supported",
        }
    )

    events = [
        json.loads(line) for line in tracker.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["event_type"] == "hypothesis_recorded"
    assert events[0]["payload"]["verdict"] == "supported"
    assert events[0]["payload"]["decision"]["proposed_action"] == "retrain_with_replay"
    assert events[0]["payload"]["decision"]["usage"]["total_tokens"] == 120


def test_hypothesis_reaches_mirror_through_composite(tmp_path):
    primary = LocalTracker(tmp_path, run_id="mirror-hypothesis")
    mirror = FlakyMirror(fail_times=0)
    tracker = CompositeTracker(primary, [mirror])

    tracker.log_hypothesis({"window": 0, "verdict": "refuted"})

    assert mirror.hypotheses == [{"window": 0, "verdict": "refuted"}]


def test_mirror_survives_transient_failures(tmp_path):
    primary = LocalTracker(tmp_path, run_id="transient")
    mirror = FlakyMirror(fail_times=2)
    tracker = CompositeTracker(primary, [mirror])

    tracker.log_event("a", {})
    tracker.log_event("b", {})
    tracker.log_event("c", {})

    events = primary.events_path.read_text(encoding="utf-8")
    assert events.count("tracker_mirror_failure") == 2
    assert "tracker_mirror_disabled" not in events
    assert mirror.calls == 3


def test_mirror_disabled_after_repeated_failures(tmp_path):
    primary = LocalTracker(tmp_path, run_id="persistent")
    mirror = BrokenMirror()
    tracker = CompositeTracker(primary, [mirror])
    limit = CompositeTracker.MIRROR_FAILURE_LIMIT

    for index in range(limit + 2):
        tracker.log_event(f"e{index}", {})

    events = primary.events_path.read_text(encoding="utf-8")
    assert events.count("tracker_mirror_failure") == limit
    assert "tracker_mirror_disabled" in events
    # 停用后不再调用镜像，避免无意义的超时等待
    assert mirror.calls == limit


def test_close_writes_local_summary_before_mirror_upload(tmp_path):
    primary = LocalTracker(tmp_path, run_id="order")
    mirror = BrokenMirror()
    tracker = CompositeTracker(primary, [mirror])

    tracker.close({"run_status": "complete"})

    assert (primary.run_dir / "summary.json").exists()


def test_online_mode_downgrades_without_credential(monkeypatch):
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    monkeypatch.setattr(SwanLabTracker, "_has_saved_credential", classmethod(lambda cls: False))

    mode, note = SwanLabTracker._resolve_mode("online")

    assert mode == "offline"
    assert note and "swanlab login" in note


def test_online_mode_kept_with_saved_credential(monkeypatch):
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    monkeypatch.setattr(SwanLabTracker, "_has_saved_credential", classmethod(lambda cls: True))

    mode, note = SwanLabTracker._resolve_mode("online")

    assert mode == "online"
    assert note is None


def test_online_mode_kept_with_api_key(monkeypatch):
    monkeypatch.setenv("SWANLAB_API_KEY", "dummy")

    mode, note = SwanLabTracker._resolve_mode("online")

    assert mode == "online"
    assert note is None


@pytest.mark.parametrize("mode", ["local", "offline", "disabled"])
def test_non_online_modes_pass_through(mode):
    assert SwanLabTracker._resolve_mode(mode) == (mode, None)


def test_tracking_config_defaults_are_offline_safe():
    config = TrackingConfig(run_root="runs")

    assert config.enable_swanlab is False
    assert config.swanlab_mode == "online"
    assert config.swanlab_upload_artifacts is True
    assert config.swanlab_tags == []


class FakeRun:
    """替代 swanlab.Run，捕获真正发出去的 payload。"""

    def __init__(self) -> None:
        self.logs: list[tuple[dict, int | None]] = []
        self.texts: list[dict] = []

    def log(self, data, step=None):
        self.logs.append((data, step))

    def log_text(self, **kwargs):
        self.texts.append(kwargs)


def _bare_tracker(run: FakeRun) -> SwanLabTracker:
    """绕开 swanlab.init，只测发送逻辑。"""
    tracker = object.__new__(SwanLabTracker)
    tracker._run = run
    tracker._auto_step = 0
    tracker._event_counts = {}
    tracker._failure_count = 0
    tracker._supports_text = True
    tracker._supports_save = False
    return tracker


def test_metrics_payload_is_numeric_only():
    """SwanLab 0.10 会拒绝字符串标量并在内部吞掉错误，列会静默变空。"""
    run = FakeRun()
    tracker = _bare_tracker(run)

    tracker.log_metrics(
        {
            "window": 5,
            "score": 0.66,
            "phase": "gradual",
            "action": "retrain_with_replay",
            "status": "deployed",
            "drift_kind": "concept_suspected",
            "decision_source": "llm",
            "reason": "很长的一段中文理由",
            "llm_model": "local-model",
        }
    )

    payload, step = run.logs[0]
    assert step == 5
    assert all(isinstance(value, (int, float)) for value in payload.values())
    # 关键状态仍可筛选，只是换成了数值码
    assert payload["action_code"] == 1
    assert payload["status_code"] == 1
    assert payload["drift_kind_code"] == 3
    assert payload["decision_source_code"] == 2


def test_unknown_status_value_is_dropped_not_crashed():
    run = FakeRun()
    tracker = _bare_tracker(run)

    tracker.log_metrics({"window": 0, "status": "从未见过的状态"})

    # 全部字段都无法映射时不发空包，避免在看板上造出一列空数据
    assert run.logs == []


def test_hypothesis_text_keeps_readable_labels():
    run = FakeRun()
    tracker = _bare_tracker(run)

    tracker.log_hypothesis(
        {
            "window": 3,
            "phase": "gradual",
            "observation": {"score": 0.6, "baseline": 0.7, "delta": -0.1},
            "diagnosis": "滚动性能下降 0.07",
            "decision": {"action": "retrain_with_replay", "source": "rule", "reason": "漂移确认"},
            "expectation": "提升 ≥ 0.008",
            "outcome": {"status": "rejected", "candidate_improvement": -0.01},
            "verdict": "refuted",
            "verdict_reason": "候选无提升",
        }
    )

    text = run.texts[0]["data"]
    assert "retrain_with_replay" in text
    assert "假设被否定" in text
