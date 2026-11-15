from __future__ import annotations

import hashlib
import json
import os
import platform
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .budget import BudgetExceededError, BudgetLedger
from .config import AppConfig
from .domain import (
    Action,
    DataWindow,
    Decision,
    DriftReport,
    LLMUsage,
    PolicyContext,
    RunSummary,
    StringEnum,
)
from .evaluation.champion_challenger import ChampionChallengerEvaluator
from .interfaces import ExperimentTracker, StreamSource, UpdatePolicy
from .llm.factory import build_provider
from .memory.replay_buffer import ReplayBuffer
from .models.registry import ModelRegistry
from .models.sklearn_adapter import SklearnModelAdapter
from .monitoring.diagnosis import DriftDiagnoser
from .monitoring.feature_drift import FeatureDriftDetector
from .monitoring.performance_drift import PerformanceDriftDetector
from .policy.hybrid_policy import HybridPolicy
from .policy.rule_policy import RuleBasedPolicy
from .training.trainer import CandidateTrainer, TrainingOutcome


class RunState(StringEnum):
    INIT = "init"
    MONITOR = "monitor"
    DIAGNOSE = "diagnose"
    DECIDE = "decide"
    TRAIN = "train"
    EVALUATE = "evaluate"
    DEPLOY = "deploy"
    REJECT = "reject"
    ROLLBACK = "rollback"
    SAFE_FALLBACK = "safe_fallback"
    COMPLETE = "complete"


class DataQualityError(ValueError):
    """窗口特征不足以进行可靠监测。"""


MAX_POLICY_HISTORY = 8


def _source_digest() -> str:
    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source_path in sorted(source_root.rglob("*.py")):
        if source_path.is_symlink() or not source_path.is_file():
            continue
        digest.update(source_path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _run_provenance(config: AppConfig) -> dict[str, Any]:
    configuration = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scikit-learn", "pydantic"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "source_sha256": _source_digest(),
        "config_sha256": f"sha256:{hashlib.sha256(configuration).hexdigest()}",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": versions,
        "container_image_digest": os.environ.get("CONTAINER_IMAGE_DIGEST"),
    }


@dataclass(slots=True)
class _DeploymentSnapshot:
    feature_state: dict[str, Any]
    performance_state: dict[str, Any]
    diagnosis_state: dict[str, Any]
    policy_state: dict[str, Any]
    deployment_score: float | None


@dataclass(slots=True)
class _RuntimeContext:
    champion: SklearnModelAdapter
    feature_detector: FeatureDriftDetector
    performance_detector: PerformanceDriftDetector
    diagnoser: DriftDiagnoser
    summary: RunSummary
    guard_X: pd.DataFrame
    guard_y: np.ndarray
    guard_ids: np.ndarray
    scores: list[float] = field(default_factory=list)
    last_deployment_score: float | None = None
    regression_streak: int = 0
    last_scored_model_version: str | None = None
    deployment_evaluations: dict[str, dict[str, Any]] = field(default_factory=dict)
    deployment_states: dict[str, _DeploymentSnapshot] = field(default_factory=dict)
    experiment_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class _WindowRecord:
    scored_model_version: str | None
    deployed_model_version: str | None
    score: float = float("nan")
    report: DriftReport | None = None
    action: Action = Action.NO_UPDATE
    status: str = "no_update"
    reason: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)
    decision: Decision | None = None
    baseline_score: float | None = None
    history_read: list[dict[str, Any]] = field(default_factory=list)


class EvolutionOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        stream: StreamSource,
        tracker: ExperimentTracker,
        policy: UpdatePolicy | None = None,
    ) -> None:
        self.config = config
        self.stream = stream
        self.tracker = tracker
        self.budget = BudgetLedger(config.budget)
        self.registry = ModelRegistry(tracker.run_dir / "models")
        self.replay = ReplayBuffer(
            capacity=config.training.replay_capacity,
            seed=config.seed,
            task_type=config.task_type,
        )
        self.policy = policy or self._build_policy()
        self.trainer = CandidateTrainer(config.training)
        self.evaluator = ChampionChallengerEvaluator(config.validation)
        self.state = RunState.INIT
        self._next_sample_id = 0
        self._decisions_path = tracker.run_dir / "decisions.jsonl"

    def _build_policy(self) -> UpdatePolicy:
        rule_policy = RuleBasedPolicy(self.config.policy, self.config.monitoring)
        if not self.config.llm.enabled:
            return rule_policy
        provider = build_provider(self.config.llm)
        return HybridPolicy(rule_policy, provider, self.config.llm)

    def run(self) -> dict[str, Any]:
        runtime = self._initialize_runtime()
        for window in self.stream:
            self._process_window(window, runtime)
        return self._finalize(runtime)

    def _initialize_runtime(self) -> _RuntimeContext:
        X_initial, y_initial = self.stream.initial_data()
        initial_ids = np.arange(len(y_initial), dtype=np.int64)
        self._next_sample_id = len(initial_ids)
        self.budget.acquire_labels(initial_ids)
        (
            X_train,
            y_train,
            train_ids,
            X_valid,
            y_valid,
            valid_ids,
        ) = self._split_initial_data(X_initial, y_initial, initial_ids)

        can_start, reason = self.budget.can_start_initial_training()
        if not can_start:
            raise BudgetExceededError(reason)
        champion = SklearnModelAdapter(
            self.config.task_type,
            self.config.seed,
            score_metric=self.config.validation.score_metric,
        )
        started = perf_counter()
        try:
            champion.fit(X_train, y_train)
        except Exception:
            elapsed = perf_counter() - started
            self.budget.record_initial_training(elapsed)
            raise
        training_seconds = perf_counter() - started
        self.budget.record_initial_training(training_seconds)
        initial_score = champion.score(X_valid, y_valid)

        version = self.registry.register(
            champion,
            {
                "source": "initial",
                "score": round(initial_score, 6),
                "training_seconds": round(training_seconds, 6),
            },
        )
        self.replay.update(X_train, y_train, train_ids)
        guard_count = min(self.config.validation.guard_samples, len(y_valid))
        guard_X = X_valid.iloc[:guard_count].reset_index(drop=True)
        guard_y = np.asarray(y_valid[:guard_count]).copy()
        guard_ids = np.asarray(valid_ids[:guard_count], dtype=object).copy()

        runtime = _RuntimeContext(
            champion=champion,
            feature_detector=FeatureDriftDetector(X_train, self.config.monitoring),
            performance_detector=PerformanceDriftDetector(
                baseline_score=initial_score,
                drop_threshold=self.config.monitoring.performance_drop_threshold,
            ),
            diagnoser=DriftDiagnoser(
                self.config.monitoring,
                self._label_distribution(y_train),
            ),
            summary=RunSummary(
                run_id=self.tracker.run_id,
                initial_score=round(initial_score, 6),
            ),
            guard_X=guard_X,
            guard_y=guard_y,
            guard_ids=guard_ids,
            last_scored_model_version=version,
        )
        runtime.deployment_states[version] = self._snapshot_runtime(runtime)
        self.tracker.log_event(
            "initial_model_ready",
            {
                "version": version,
                "score": initial_score,
                "training_samples": len(y_train),
                "validation_samples": len(y_valid),
                "guard_samples": len(guard_y),
                "training_seconds": training_seconds,
                "budget": self.budget.snapshot().to_dict(),
            },
        )
        return runtime

    @staticmethod
    def _split_initial_data(
        X: pd.DataFrame,
        y: np.ndarray,
        sample_ids: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
        if len(y) < 2:
            raise ValueError("初始数据至少需要两条样本")
        split = min(len(y) - 1, max(1, int(len(y) * 0.8)))
        return (
            X.iloc[:split].reset_index(drop=True),
            np.asarray(y[:split]),
            np.asarray(sample_ids[:split]),
            X.iloc[split:].reset_index(drop=True),
            np.asarray(y[split:]),
            np.asarray(sample_ids[split:]),
        )

    def _process_window(self, window: DataWindow, runtime: _RuntimeContext) -> None:
        current_version = self.registry.current_version()
        record = _WindowRecord(
            scored_model_version=current_version,
            deployed_model_version=current_version,
            # 窗口开始时的部署基线：本窗口若上线新模型，这个值仍代表"打分时"的参照，
            # 否则 delta 会被同窗口的部署覆盖成 0，漂移证据就丢失了。
            baseline_score=runtime.last_deployment_score,
        )
        self._ensure_window_ids(window)
        assert window.sample_ids is not None
        can_acquire, label_reason, requested_labels = self.budget.can_acquire_labels(
            window.sample_ids
        )
        if not can_acquire:
            self._transition(RunState.MONITOR, window.index)
            record.action = Action.NO_UPDATE
            record.status = "budget_skipped"
            record.reason = f"标签预算拒绝本窗口；{label_reason}，需要新增标签 {requested_labels}"
            runtime.summary.skipped_windows += 1
            self.tracker.log_event(
                "window_skipped",
                {
                    "window": window.index,
                    "status": record.status,
                    "reason": record.reason,
                    "requested_new_labels": requested_labels,
                    "budget": self.budget.snapshot().to_dict(),
                },
            )
            self._record_window(window, runtime, record)
            return

        try:
            record.score, record.report = self._monitor_and_diagnose(window, runtime)
            runtime.summary.successful_windows += 1
            rolled_back = self._try_rollback(window, runtime, record)
            if not rolled_back:
                self._decide_and_update(window, runtime, record)
        except Exception as exc:
            self._handle_window_failure(window, runtime, record, exc)

        if record.report is not None:
            self.replay.update(window.X, window.y, window.sample_ids)
        self._record_window(window, runtime, record)

    def _ensure_window_ids(self, window: DataWindow) -> None:
        if window.sample_ids is None:
            window.sample_ids = np.arange(
                self._next_sample_id,
                self._next_sample_id + len(window.y),
                dtype=np.int64,
            )
        if len(window.sample_ids) != len(window.y):
            raise ValueError("窗口样本ID与标签长度不一致")
        if len(window.sample_ids):
            numeric_ids = [int(value) for value in window.sample_ids if str(value).isdigit()]
            if numeric_ids:
                self._next_sample_id = max(self._next_sample_id, max(numeric_ids) + 1)

    def _monitor_and_diagnose(
        self,
        window: DataWindow,
        runtime: _RuntimeContext,
    ) -> tuple[float, DriftReport]:
        self._transition(RunState.MONITOR, window.index)
        feature_result = runtime.feature_detector.detect(window.X)
        if not feature_result.data_quality_valid:
            raise DataQualityError("；".join(feature_result.data_quality_issues))
        assert window.sample_ids is not None
        self.budget.acquire_labels(window.sample_ids)
        score = runtime.champion.score(window.X, window.y)
        if not np.isfinite(score):
            raise DataQualityError("模型评分不是有限值")
        runtime.scores.append(score)
        runtime.last_scored_model_version = self.registry.current_version()
        performance_result = runtime.performance_detector.update(score)

        self._transition(RunState.DIAGNOSE, window.index)
        report = runtime.diagnoser.diagnose(
            feature_result,
            performance_result,
            self._label_distribution(window.y),
        )
        self.tracker.log_event(
            "drift_diagnosis",
            {
                "window": window.index,
                "phase": window.phase,
                "scored_model_version": runtime.last_scored_model_version,
                "report": asdict(report),
            },
        )
        return score, report

    def _try_rollback(
        self,
        window: DataWindow,
        runtime: _RuntimeContext,
        record: _WindowRecord,
    ) -> bool:
        if runtime.last_deployment_score is None:
            return False

        rollback_threshold = runtime.last_deployment_score - self.config.policy.rollback_drop
        if record.score < rollback_threshold:
            runtime.regression_streak += 1
        else:
            runtime.regression_streak = 0

        should_rollback = (
            runtime.regression_streak >= self.config.policy.rollback_patience
            and self.registry.can_rollback()
        )
        if not should_rollback:
            return False

        self._transition(RunState.ROLLBACK, window.index)
        current_version = self.registry.current_version()
        target = self.registry.rollback_target()
        target_state = runtime.deployment_states.get(target)
        if target_state is None:
            raise RuntimeError(f"模型版本 {target} 缺少完整运行状态快照")
        target_model = self.registry.load(target)
        current_state = self._snapshot_runtime(runtime)
        current_model = runtime.champion
        try:
            self._restore_runtime(runtime, target_state)
            runtime.champion = target_model
            self.registry.activate(target)
        except Exception:
            self._restore_runtime(runtime, current_state)
            runtime.champion = current_model
            if current_version is not None:
                self.registry.activate(current_version)
            raise

        rolled_score = runtime.champion.score(window.X, window.y)
        runtime.last_deployment_score = target_state.deployment_score
        runtime.regression_streak = 0
        runtime.summary.rollbacks += 1
        record.deployed_model_version = target
        record.action = Action.ROLLBACK
        record.reason = "新模型连续窗口性能显著下降，已恢复模型及完整监测状态"
        record.status = "rolled_back"
        self.tracker.log_event(
            "model_rolled_back",
            {
                "window": window.index,
                "from_version": current_version,
                "version": target,
                "score_after_rollback": rolled_score,
                "reason": record.reason,
            },
        )
        return True

    def _decide_and_update(
        self,
        window: DataWindow,
        runtime: _RuntimeContext,
        record: _WindowRecord,
    ) -> None:
        if record.report is None:
            raise RuntimeError("漂移诊断结果缺失")

        self._transition(RunState.DECIDE, window.index)
        train_size = self._training_sample_count(len(window.y))
        can_train, budget_reason = self.budget.can_train()
        history = tuple(runtime.experiment_history[-MAX_POLICY_HISTORY:])
        record.history_read = [dict(item) for item in history]
        self.tracker.log_event(
            "experiment_history_read",
            {
                "window": window.index,
                "references": [item["reference"] for item in history],
                "entries": record.history_read,
                "limit": MAX_POLICY_HISTORY,
            },
        )
        context = PolicyContext(
            window_index=window.index,
            report=record.report,
            budget=self.budget.snapshot(),
            requested_labels=0,
            can_train=can_train,
            experiment_history=history,
        )
        decision = self.policy.decide(context)
        # 策略响应已经产生，即使实际Token记账接下来熔断，也要让窗口最终假设链保留这次调用证据。
        record.decision = decision
        if decision.llm_attempted:
            runtime.summary.llm_attempts += 1
        if decision.llm_called:
            runtime.summary.llm_calls += 1
            self.tracker.log_event(
                "llm_decision",
                {
                    "window": window.index,
                    "source": decision.source,
                    "model": decision.model,
                    "confidence": decision.confidence,
                    "usage": asdict(decision.usage),
                    "latency_seconds": decision.latency_seconds,
                    "action": decision.action.value,
                },
            )
        if decision.fallback_reason:
            runtime.summary.llm_fallbacks += 1
            self.tracker.log_event(
                "llm_fallback",
                {
                    "window": window.index,
                    "model": decision.model,
                    "reason": decision.fallback_reason,
                },
            )
        if decision.usage.total_tokens:
            self.budget.record_tokens(decision.usage.total_tokens)

        record.action = decision.action
        record.reason = decision.reason
        record.status = "no_update"
        if not can_train and record.report.is_actionable:
            record.reason = f"{decision.reason}；{budget_reason}"
            record.status = "budget_exhausted"

        self.tracker.log_event(
            "update_decision",
            {
                "window": window.index,
                "action": record.action.value,
                "reason": record.reason,
                "source": decision.source,
                "model": decision.model,
                "confidence": decision.confidence,
                "usage": asdict(decision.usage),
                "latency_seconds": decision.latency_seconds,
                "fallback_reason": decision.fallback_reason,
                "llm_attempted": decision.llm_attempted,
                "llm_called": decision.llm_called,
                "budget": self.budget.snapshot().to_dict(),
            },
        )
        if record.action is Action.RETRAIN_WITH_REPLAY:
            self._train_and_evaluate(window, runtime, record, train_size)

    def _training_sample_count(self, window_size: int) -> int:
        holdout = self.config.training.holdout_fraction
        return max(1, int(window_size * (1.0 - holdout)))

    def _train_and_evaluate(
        self,
        window: DataWindow,
        runtime: _RuntimeContext,
        record: _WindowRecord,
        train_size: int,
    ) -> None:
        runtime.summary.updates_attempted += 1
        self._transition(RunState.TRAIN, window.index)
        recent_X = window.X.iloc[:train_size]
        recent_y = window.y[:train_size]
        holdout_X = window.X.iloc[train_size:]
        holdout_y = window.y[train_size:]
        assert window.sample_ids is not None
        recent_ids = window.sample_ids[:train_size]
        guard_id_set = set(runtime.guard_ids.tolist())

        started = perf_counter()
        try:
            outcome = self.trainer.train(
                runtime.champion,
                recent_X,
                recent_y,
                self.replay,
                recent_ids,
                exclude_ids=guard_id_set,
            )
        except Exception:
            elapsed = perf_counter() - started
            self.budget.record_update(elapsed)
            raise
        self.budget.record_update(outcome.seconds)
        self.policy.mark_update(window.index)

        if set(outcome.training_sample_ids.tolist()) & guard_id_set:
            raise RuntimeError("候选训练数据与历史守护集重叠")
        self._transition(RunState.EVALUATE, window.index)
        evaluation = self.evaluator.evaluate(
            runtime.champion,
            outcome.model,
            holdout_X,
            holdout_y,
            runtime.guard_X,
            runtime.guard_y,
        )
        record.evaluation = asdict(evaluation)
        self.tracker.log_event(
            "candidate_evaluated",
            {
                "window": window.index,
                "training_samples": outcome.training_samples,
                "training_seconds": outcome.seconds,
                "training_guard_overlap": 0,
                "evaluation": record.evaluation,
            },
        )

        if evaluation.accepted:
            self._deploy_candidate(window, runtime, record, outcome, recent_y)
        else:
            self._transition(RunState.REJECT, window.index)
            runtime.summary.updates_rejected += 1
            record.status = "rejected"

    def _deploy_candidate(
        self,
        window: DataWindow,
        runtime: _RuntimeContext,
        record: _WindowRecord,
        outcome: TrainingOutcome,
        recent_y: np.ndarray,
    ) -> None:
        self._transition(RunState.DEPLOY, window.index)
        runtime.champion = outcome.model
        candidate_score = record.evaluation["challenger_recent_score"]
        version = self.registry.register(
            runtime.champion,
            {
                "source": f"window_{window.index}",
                "score": candidate_score,
                "training_seconds": round(outcome.seconds, 6),
            },
        )
        runtime.feature_detector.update_reference(outcome.reference_X)
        runtime.diagnoser.update_reference_label_distribution(self._label_distribution(recent_y))
        runtime.performance_detector.set_baseline(candidate_score)
        runtime.last_deployment_score = candidate_score
        runtime.regression_streak = 0
        runtime.summary.updates_deployed += 1
        record.deployed_model_version = version
        record.status = "deployed"
        runtime.deployment_states[version] = self._snapshot_runtime(runtime)
        runtime.deployment_evaluations[version] = {
            "window": window.index,
            "evaluation": dict(record.evaluation),
        }
        self.tracker.log_event(
            "model_deployed",
            {
                "window": window.index,
                "scored_model_version": record.scored_model_version,
                "version": version,
                "evaluation": record.evaluation,
            },
        )

    def _handle_window_failure(
        self,
        window: DataWindow,
        runtime: _RuntimeContext,
        record: _WindowRecord,
        error: Exception,
    ) -> None:
        self._transition(RunState.SAFE_FALLBACK, window.index)
        runtime.summary.failures += 1
        record.action = Action.ERROR
        record.deployed_model_version = self.registry.current_version()
        if isinstance(error, DataQualityError):
            record.status = "data_quality_failed"
        elif isinstance(error, BudgetExceededError):
            record.status = "budget_exhausted"
        else:
            record.status = "failed"
        record.reason = f"窗口处理失败，继续使用最后稳定模型：{type(error).__name__}: {error}"
        self.tracker.log_event(
            "window_failure",
            {
                "window": window.index,
                "error_type": type(error).__name__,
                "error": str(error),
                "fallback_version": self.registry.current_version(),
            },
        )

    def _record_window(
        self,
        window: DataWindow,
        runtime: _RuntimeContext,
        record: _WindowRecord,
    ) -> None:
        runtime.summary.windows_processed += 1
        action = record.action.value
        runtime.summary.action_counts[action] = runtime.summary.action_counts.get(action, 0) + 1
        budget = self.budget.snapshot()
        report = record.report
        self.tracker.log_metrics(
            {
                "window": window.index,
                "phase": window.phase,
                "state": self.state.value,
                "model_version": record.scored_model_version,
                "scored_model_version": record.scored_model_version,
                "deployed_model_version": record.deployed_model_version,
                "score": round(record.score, 6) if np.isfinite(record.score) else None,
                "drift_kind": report.kind.value if report else "unknown",
                "feature_drift_score": report.feature.overall_score if report else 0.0,
                "performance_drop": report.performance.score_drop if report else 0.0,
                "label_shift": report.label_shift if report else 0.0,
                "action": action,
                "status": record.status,
                "reason": record.reason,
                "llm_attempted": record.decision.llm_attempted if record.decision else False,
                "llm_called": record.decision.llm_called if record.decision else False,
                "decision_source": record.decision.source if record.decision else "system",
                "llm_model": record.decision.model if record.decision else None,
                "llm_confidence": record.decision.confidence if record.decision else None,
                "llm_input_tokens": (record.decision.usage.input_tokens if record.decision else 0),
                "llm_output_tokens": (
                    record.decision.usage.output_tokens if record.decision else 0
                ),
                "llm_reasoning_tokens": (
                    record.decision.usage.reasoning_tokens if record.decision else 0
                ),
                "llm_total_tokens": (record.decision.usage.total_tokens if record.decision else 0),
                "llm_latency_seconds": (
                    record.decision.latency_seconds if record.decision else 0.0
                ),
                "llm_fallback_reason": (
                    record.decision.fallback_reason if record.decision else None
                ),
                "candidate_improvement": record.evaluation.get("recent_improvement", 0.0),
                "updates_used": budget.updates_used,
                "training_seconds_used": budget.training_seconds_used,
                "labels_used": budget.labels_used,
                "tokens_used": budget.tokens_used,
                "budget_violations": budget.budget_violations,
            }
        )
        hypothesis = self._build_hypothesis(window, runtime, record)
        self.tracker.log_hypothesis(hypothesis)
        report = record.report
        evaluation = record.evaluation or {}
        runtime.experiment_history.append(
            {
                "reference": f"{runtime.summary.run_id}:window:{window.index}",
                "window": window.index,
                "drift_kind": report.kind.value if report else "unknown",
                "top_features": list(report.feature.top_features) if report else [],
                "action": record.action.value,
                "status": record.status,
                "score_drop": report.performance.score_drop if report else None,
                "candidate_improvement": evaluation.get("recent_improvement"),
                "historical_drop": evaluation.get("historical_drop"),
                "verdict": hypothesis["verdict"],
            }
        )
        del runtime.experiment_history[:-MAX_POLICY_HISTORY]
        self._emit_batch_decision(window, runtime, record)

    def _emit_batch_decision(
        self,
        window: DataWindow,
        runtime: _RuntimeContext,
        record: _WindowRecord,
    ) -> None:
        """逐批次输出「上线 / 不上线」判断。

        赛题要求先给出当前批次模型或不上线的判断，才能获取下一批数据。这里把判断
        固定落盘到 ``decisions.jsonl``（每窗口一行），并可选同步打印到 stdout，作为
        后续在线握手协议的对接点。``decision`` 严格二值，细分原因保留在 ``status``。
        """
        deployed = record.status == "deployed"
        row = {
            "batch": window.index,
            "decision": "deploy" if deployed else "no_update",
            "model_version": record.deployed_model_version or self.registry.current_version(),
            "score": round(record.score, 6) if np.isfinite(record.score) else None,
            "status": record.status,
            "action": record.action.value,
            "reason": record.reason,
            "history_read": [item["reference"] for item in record.history_read],
            "candidate_improvement": record.evaluation.get("recent_improvement"),
            "updates_attempted": runtime.summary.updates_attempted,
        }
        with self._decisions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.config.tracking.emit_batch_decisions:
            print(json.dumps(row, ensure_ascii=False), flush=True)

    def _build_hypothesis(
        self,
        window: DataWindow,
        runtime: _RuntimeContext,
        record: _WindowRecord,
    ) -> dict[str, Any]:
        """把单个窗口整理成一条可复核的假设链。

        评审关心的是"你为什么这么改、改完真的好了吗"，标量曲线回答不了。这里固定
        观察 → 诊断 → 决策 → 预期 → 结果 → 判定 六段，让每一轮自适应都有可被反驳的
        主张和对应的证据；``verdict`` 为 ``refuted`` 时说明"重训挽回漂移"的判断这次
        没成立，是负结果，同样保留。
        """
        report = record.report
        baseline = record.baseline_score
        if baseline is None:
            baseline = runtime.summary.initial_score
        delta = record.score - baseline if np.isfinite(record.score) else None
        evaluation = record.evaluation or {}
        verdict, verdict_reason = self._judge_hypothesis(record, evaluation)
        decision = record.decision

        return {
            "window": window.index,
            "phase": window.phase,
            "history_read": record.history_read,
            "observation": {
                "score": record.score if np.isfinite(record.score) else None,
                "baseline": baseline,
                "delta": delta,
                "drift_kind": report.kind.value if report else "unknown",
                "feature_drift": report.feature.overall_score if report else None,
                "performance_drop": report.performance.score_drop if report else None,
                "label_shift": report.label_shift if report else None,
                "is_actionable": report.is_actionable if report else None,
            },
            "diagnosis": (
                "标签预算不足，未执行评分与漂移诊断"
                if record.status == "budget_skipped"
                else report.explanation
                if report
                else "窗口未产出诊断结果"
            ),
            "decision": {
                # action 是窗口最终动作（超预算异常时为 error）；proposed_action 是策略返回值。
                "action": record.action.value,
                "proposed_action": decision.action.value if decision else None,
                "source": decision.source if decision else "system",
                "model": decision.model if decision else None,
                "confidence": decision.confidence if decision else None,
                "reason": record.reason,
                "policy_reason": decision.reason if decision else None,
                "fallback_reason": decision.fallback_reason if decision else None,
                "usage": asdict(decision.usage) if decision else asdict(LLMUsage()),
                "llm_attempted": decision.llm_attempted if decision else False,
                "llm_called": decision.llm_called if decision else False,
            },
            "expectation": self._expectation_text(record.action),
            "outcome": {
                "status": record.status,
                "candidate_improvement": evaluation.get("recent_improvement"),
                "historical_drop": evaluation.get("historical_drop"),
                "deployed_version": (
                    record.deployed_model_version if record.status == "deployed" else None
                ),
            },
            "verdict": verdict,
            "verdict_reason": verdict_reason,
        }

    def _expectation_text(self, action: Action) -> str:
        validation = self.config.validation
        if action is Action.RETRAIN_WITH_REPLAY:
            return (
                f"候选在本窗 holdout 上提升 ≥ {validation.min_improvement:.4f}，"
                f"且历史守护集回退 ≤ {validation.max_historical_drop:.4f}，则上线"
            )
        if action is Action.ROLLBACK:
            return "回滚到最后稳定版本后，分数应回到部署阈值之上"
        return "维持当前模型，等待漂移信号连续确认后再动作"

    @staticmethod
    def _judge_hypothesis(
        record: _WindowRecord,
        evaluation: dict[str, Any],
    ) -> tuple[str, str]:
        if record.status == "budget_skipped":
            return "inconclusive", "标签预算不足，未评分或诊断，不构成模型策略的证据"
        if record.action is Action.ROLLBACK or record.status == "rolled_back":
            return "supported", "连续回退触发回滚，已恢复到上一稳定版本并同步监测状态"
        if record.action is not Action.RETRAIN_WITH_REPLAY:
            return "not_applicable", "本窗口未触发重训，决策本身不需要验证"
        if record.status == "deployed":
            improvement = evaluation.get("recent_improvement")
            played = f"{improvement:+.4f}" if isinstance(improvement, (int, float)) else "—"
            return (
                "supported",
                f"候选在 holdout 上提升 {played}，历史回退在阈值内，已上线并作为新基线",
            )
        if record.status == "rejected":
            improvement = evaluation.get("recent_improvement") or 0.0
            if improvement <= 0:
                return "refuted", "候选在 holdout 上无提升，'重训可挽回本次漂移'不成立，保留旧模型"
            drop = evaluation.get("historical_drop")
            played = f"{drop:.4f}" if isinstance(drop, (int, float)) else "—"
            return "refuted", f"holdout 有提升但历史回退 {played} 超阈值，整体收益为负，否决"
        if record.status in {"failed", "data_quality_failed", "budget_exhausted"}:
            return "inconclusive", f"窗口停在 {record.status}，验证环节未走完，不构成证据"
        return "inconclusive", "无可用的候选评估结果"

    def _finalize(self, runtime: _RuntimeContext) -> dict[str, Any]:
        self._transition(RunState.COMPLETE)
        fallback_score = runtime.summary.initial_score
        runtime.summary.final_score = (
            round(runtime.scores[-1], 6) if runtime.scores else fallback_score
        )
        runtime.summary.mean_score = (
            round(float(np.mean(runtime.scores)), 6) if runtime.scores else fallback_score
        )
        if runtime.summary.successful_windows == 0:
            runtime.summary.run_status = "failed"
        elif runtime.summary.failures or runtime.summary.skipped_windows:
            runtime.summary.run_status = "partial"
        else:
            runtime.summary.run_status = "complete"
        runtime.summary.budget = self.budget.snapshot().to_dict()

        result = runtime.summary.to_dict()
        result["final_model_version"] = runtime.last_scored_model_version
        result["final_score_model_version"] = runtime.last_scored_model_version
        result["final_deployed_model_version"] = self.registry.current_version()
        result.update(self._final_deployment_report(runtime))
        result["metric_name"] = runtime.champion.metric_name
        result["label_budget_mode"] = self.config.budget.label_budget_mode
        result["min_updates_required"] = self.config.budget.min_updates
        result["min_updates_met"] = (
            runtime.summary.updates_attempted >= self.config.budget.min_updates
        )
        if not result["min_updates_met"]:
            self.tracker.log_event(
                "min_updates_unmet",
                {
                    "required": self.config.budget.min_updates,
                    "attempted": runtime.summary.updates_attempted,
                    "note": "赛题要求至少三次尝试迭代；本次记录不足，不作为可提交证据",
                },
            )
        result.update(self._official_baseline_report(runtime))
        result["reproducibility"] = _run_provenance(self.config)
        self.tracker.close(result)
        return result

    def _official_baseline_report(self, runtime: _RuntimeContext) -> dict[str, Any]:
        """把官方静态基线写进摘要；口径不一致时显式拒绝比较，不制造虚假增益。"""
        baseline = self.config.validation.official_baseline_score
        if baseline is None:
            return {"vs_official_baseline": None}
        metric = runtime.champion.metric_name
        official_metric = self.config.validation.official_metric_name
        comparable = official_metric is not None and official_metric == metric
        final_score = runtime.summary.final_score
        return {
            "vs_official_baseline": {
                "official_metric": official_metric,
                "project_metric": metric,
                "official_baseline_score": baseline,
                "final_score": final_score,
                "delta": round(final_score - baseline, 6) if comparable else None,
                "meets_baseline": bool(comparable and final_score >= baseline),
                "comparable": comparable,
                "note": (
                    "同口径，可直接比较"
                    if comparable
                    else "指标口径不同，不计算 delta；先用 validation.score_metric 对齐官方主指标"
                ),
            }
        }

    def _final_deployment_report(self, runtime: _RuntimeContext) -> dict[str, Any]:
        """补充「最终部署模型」自己的评分，避免用旧冠军的分数代表最终模型。

        部署发生在窗口打分之后，所以 ``final_score`` 记的始终是当时冠军（例如 v003）的分数；
        最后一个窗口之后上线的版本（v004）不会再被任何窗口评分。这里改用该版本上线时的
        候选评估得分：候选只在窗口前 70%（``holdout_fraction``）上训练，尾部 holdout 与
        历史守护集都未参与训练，是现成的无泄漏证据。缺失时显式给 ``None``，不拿旧分数顶替。
        """
        version = self.registry.current_version()
        record = runtime.deployment_evaluations.get(version or "")
        if record is None:
            self.tracker.log_event(
                "final_deployment_score_missing",
                {
                    "final_deployed_model_version": version,
                    "final_score_model_version": runtime.last_scored_model_version,
                    "reason": "该版本不是通过候选评估上线的，没有未见数据评分",
                },
            )
            return {
                "final_deployed_score": None,
                "final_deployed_guard_score": None,
                "final_deployed_score_source": f"none:{version}",
                "final_deployed_window": None,
            }

        evaluation = record["evaluation"]
        self.tracker.log_event(
            "final_deployment_summary",
            {
                "final_deployed_model_version": version,
                "final_score_model_version": runtime.last_scored_model_version,
                "window": record["window"],
                "final_deployed_score": evaluation.get("challenger_recent_score"),
                "final_deployed_guard_score": evaluation.get("challenger_historical_score"),
                "note": (
                    "final_score 为最后窗口全量得分；"
                    "final_deployed_score 为候选上线时的 holdout 得分"
                ),
            },
        )
        return {
            "final_deployed_score": evaluation.get("challenger_recent_score"),
            "final_deployed_guard_score": evaluation.get("challenger_historical_score"),
            "final_deployed_score_source": f"candidate_holdout_window_{record['window']}",
            "final_deployed_window": record["window"],
        }

    def _snapshot_runtime(self, runtime: _RuntimeContext) -> _DeploymentSnapshot:
        return _DeploymentSnapshot(
            feature_state=deepcopy(runtime.feature_detector.snapshot_state()),
            performance_state=deepcopy(runtime.performance_detector.snapshot_state()),
            diagnosis_state=deepcopy(runtime.diagnoser.snapshot_state()),
            policy_state=deepcopy(self._policy_snapshot()),
            deployment_score=runtime.last_deployment_score,
        )

    def _restore_runtime(
        self,
        runtime: _RuntimeContext,
        snapshot: _DeploymentSnapshot,
    ) -> None:
        runtime.feature_detector.restore_state(deepcopy(snapshot.feature_state))
        runtime.performance_detector.restore_state(deepcopy(snapshot.performance_state))
        runtime.diagnoser.restore_state(deepcopy(snapshot.diagnosis_state))
        self._restore_policy(deepcopy(snapshot.policy_state))
        runtime.last_deployment_score = snapshot.deployment_score

    def _policy_snapshot(self) -> dict[str, Any]:
        snapshot = getattr(self.policy, "snapshot_state", None)
        return snapshot() if callable(snapshot) else {}

    def _restore_policy(self, state: dict[str, Any]) -> None:
        restore = getattr(self.policy, "restore_state", None)
        if callable(restore):
            restore(state)

    def _transition(self, state: RunState, window: int | None = None) -> None:
        previous = self.state
        self.state = state
        self.tracker.log_event(
            "state_transition",
            {"from": previous.value, "to": state.value, "window": window},
        )

    def _label_distribution(self, y: np.ndarray) -> dict[str, float]:
        if self.config.task_type != "classification" or len(y) == 0:
            return {}
        labels, counts = np.unique(np.asarray(y), return_counts=True)
        total = float(np.sum(counts))
        return {
            f"{type(label).__name__}:{label!r}": float(count / total)
            for label, count in zip(labels, counts, strict=True)
        }
