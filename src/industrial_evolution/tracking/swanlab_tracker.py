"""SwanLab 镜像记录器。

职责边界：SwanLab 只做**镜像**，本地 ``runs/<run_id>/`` 才是主记录。任何上传动作
失败均记为事件，进化闭环继续运行；本地记录保证全程可追溯，看板状态不影响自适应流程。

记录分三层，对应评审关心的三件事：

1. **曲线层**（``log_metrics``）：每个窗口的得分、漂移量、预算消耗，``step`` 用窗口序号，
   这样多轮运行可以直接在同一坐标系里对比。
2. **文本层**（``log_event`` / ``log_hypothesis``）：决策依据、模型上线/回滚、窗口失败，
   以及"观察→诊断→决策→预期→结果→判定"的假设链，人可以直接读。
3. **产物层**（``close``）：``events.jsonl`` / ``metrics.csv`` / ``summary.json`` /
   模型注册表 上传为 run 文件，任何人拿到一次 run 就能复算，不依赖本地目录。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from ..interfaces import ExperimentTracker, TrackerMirror

#: 字符串标签不能直接 log——SwanLab 0.10 会报
#: ``TypeError: Unsupported scalar string value`` 并在内部吞掉，列静默变空。
#: 所以状态类字段统一转成数值码，名字仍保留在判断链文本里，看板上靠下面的码表读。
_CODEBOOK: dict[str, dict[str, int]] = {
    "action": {"no_update": 0, "retrain_with_replay": 1, "rollback": 2, "error": 3},
    "status": {
        "no_update": 0,
        "deployed": 1,
        "rejected": 2,
        "rolled_back": 3,
        "budget_exhausted": 4,
        "data_quality_failed": 5,
        "failed": 6,
    },
    "drift_kind": {
        "none": 0,
        "covariate": 1,
        "label": 2,
        "concept_suspected": 3,
        "mixed": 4,
        "unknown": -1,
    },
    "decision_source": {"system": 0, "rule": 1, "llm": 2, "hybrid": 3},
}

#: 需要单独留一条文本事件的关键事件（每窗口都会发生的状态迁移不放进来，太吵）。
_TEXT_EVENTS = frozenset(
    {
        "initial_model_ready",
        "candidate_evaluated",
        "model_deployed",
        "model_rolled_back",
        "window_failure",
        "llm_decision",
        "llm_fallback",
        "final_deployment_summary",
        "final_deployment_score_missing",
        "swanlab_disabled",
    }
)

_VERDICT_MARK = {
    "supported": "✅ 假设成立",
    "refuted": "❌ 假设被否定",
    "inconclusive": "➖ 证据不足",
    "not_applicable": "⏭️ 未触发更新",
}


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


class SwanLabTracker:
    """可选镜像记录器；init 失败时由调用方降级到纯本地记录。"""

    def __init__(
        self,
        project: str,
        config: dict[str, Any],
        *,
        run_name: str | None = None,
        mode: str = "online",
        workspace: str | None = None,
        log_dir: str | Path | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        artifact_dir: str | Path | None = None,
        upload_artifacts: bool = True,
        artifact_max_mb: float = 64.0,
    ) -> None:
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError("未安装 SwanLab，可使用 pip install -e .[swanlab] 安装") from exc

        self._swanlab = swanlab
        self._artifact_dir = Path(artifact_dir) if artifact_dir else None
        self._upload_artifacts = upload_artifacts
        self._artifact_max_bytes = int(artifact_max_mb * 1024 * 1024)
        self._auto_step = 0
        self._event_counts: dict[str, int] = {}
        self._failure_count = 0
        self._uploaded: set[str] = set()
        self.mode, self.mode_note = self._resolve_mode(mode)

        self._run = self._init_run(
            swanlab,
            project=project,
            name=run_name,
            config=config,
            mode=self.mode,
            workspace=workspace,
            log_dir=log_dir,
            tags=tags,
            group=group,
        )
        # 老版本 SwanLab 没有 log_text / save，探测后降级，不让接口差异直接炸掉运行。
        self._supports_text = callable(getattr(self._run, "log_text", None))
        self._supports_save = callable(getattr(self._run, "save", None))
        self.run_id = str(getattr(self._run, "id", None) or run_name or "swanlab")
        self.url = self._resolve_url()

    @staticmethod
    def _init_run(swanlab: Any, **kwargs: Any) -> Any:
        common = {
            "project": kwargs["project"],
            "name": kwargs["name"],
            "config": kwargs["config"],
            "mode": kwargs["mode"],
            "workspace": kwargs["workspace"],
            "log_dir": str(kwargs["log_dir"]) if kwargs["log_dir"] else None,
            "tags": list(kwargs["tags"]) if kwargs["tags"] else None,
            "group": kwargs["group"],
            "description": (
                "工业模型自进化闭环：每窗口 monitor→diagnose→decide→train→evaluate→"
                "deploy/reject/rollback，记录可复算。"
            ),
        }
        try:
            # 联网失败时退到 offline 落盘，而不是抛异常中断进化闭环；事后可 swanlab sync 补传。
            return swanlab.init(**common, on_online_error="offline")
        except TypeError:
            return swanlab.init(**common)

    @staticmethod
    def _has_saved_credential() -> bool:
        """判断本机是否已有 ``swanlab login`` 留下的凭证。

        优先用 SwanLab 自己的凭证读取接口；内部 API 变动时退回 ``~/.swanlab/.netrc``
        存在性检查，宁可多认一次登录，也不要凭一次探测失败就把在线记录关掉。
        """
        try:
            from swanlab.sdk.internal.pkg import nrc
            from swanlab.sdk.internal.settings import Settings

            netrc_path = nrc.path(Settings.get_user_config_dir())
            if nrc.read(netrc_path) is not None:
                return True
        except Exception:
            pass
        return (Path.home() / ".swanlab" / ".netrc").exists()

    @classmethod
    def _resolve_mode(cls, mode: str) -> tuple[str, str | None]:
        """没凭证时把 online 降级成 offline，避免被交互登录提示卡住。

        实测：``init(mode="online")`` 在没有凭证时会打印"创建账号/输入 API Key"的菜单
        并等待标准输入；即使 stdin 是管道也不会跳过（Git Bash / ConPTY 下 ``isatty()``
        还会返回真）。无人值守运行必须靠凭证判断来提前分流，否则整轮自适应会挂住。
        """
        if mode != "online":
            return mode, None
        if os.getenv("SWANLAB_API_KEY"):
            return "online", None
        if cls._has_saved_credential():
            return "online", None
        return (
            "offline",
            "未检测到 SwanLab 凭证（SWANLAB_API_KEY 或 ~/.swanlab/.netrc），"
            "已降级为 offline 落盘；需要云端看板请先执行 swanlab login",
        )

    # ------------------------------------------------------------------ 基础设施

    def _resolve_url(self) -> str | None:
        try:
            return str(self._run.url)
        except Exception:
            return None

    def _note_failure(self, action: str, exc: Exception) -> None:
        self._failure_count += 1
        if self._failure_count <= 3 or self._failure_count % 20 == 0:
            print(
                f"[swanlab] {action} 失败（第 {self._failure_count} 次）："
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _next_step(self, step: Any) -> int:
        if _is_number(step):
            self._auto_step = max(self._auto_step, int(step) + 1)
            return int(step)
        self._auto_step += 1
        return self._auto_step

    def _log(self, data: dict[str, Any], step: int) -> None:
        if not data:
            return
        try:
            self._run.log(data, step=step)
        except Exception as exc:
            self._note_failure("log", exc)

    def _log_text(self, key: str, content: str, step: int, caption: str | None = None) -> None:
        if not self._supports_text:
            # 老版本没有文本通道，退化成普通字段，至少不丢内容。
            self._log({key: content}, step)
            return
        try:
            self._run.log_text(key=key, data=content, caption=caption, step=step)
        except Exception as exc:
            self._note_failure("log_text", exc)

    # ------------------------------------------------------------------ 曲线层

    def log_metrics(self, row: dict[str, Any]) -> None:
        step = self._next_step(row.get("window"))
        scalars: dict[str, Any] = {}
        for key, value in row.items():
            if key == "window":
                continue
            if _is_number(value):
                scalars[key] = float(value)
            elif isinstance(value, bool):
                scalars[key] = int(value)
            elif isinstance(value, str) and key in _CODEBOOK:
                code = _CODEBOOK[key].get(value)
                if code is not None:
                    scalars[f"{key}_code"] = code
        self._log(scalars, step)

    # ------------------------------------------------------------------ 文本层

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        count = self._event_counts.get(event_type, 0) + 1
        self._event_counts[event_type] = count
        step = self._next_step(payload.get("window"))
        self._log({f"event_count/{event_type}": count}, step)
        if event_type in _TEXT_EVENTS:
            self._log_text(
                key=f"事件/{event_type}",
                content=self._render_event(event_type, payload),
                step=step,
                caption=f"窗口 {payload.get('window')}",
            )

    def log_hypothesis(self, record: dict[str, Any]) -> None:
        step = self._next_step(record.get("window"))
        verdict = str(record.get("verdict", "inconclusive"))
        window = record.get("window")
        # 判定结果同时存成数值，便于统计"假设被否定了多少次"这条曲线。
        self._log(
            {
                "hypothesis/count": 1,
                "hypothesis/supported": int(verdict == "supported"),
                "hypothesis/refuted": int(verdict == "refuted"),
                "hypothesis/inconclusive": int(verdict == "inconclusive"),
            },
            step,
        )
        self._log_text(
            key="判断链/窗口结论",
            content=self._render_hypothesis(record),
            step=step,
            caption=f"W{window} · {record.get('phase', '')}",
        )

    # ------------------------------------------------------------------ 产物层

    def close(self, summary: dict[str, Any]) -> None:
        self._log(
            {f"summary/{key}": value for key, value in summary.items() if _is_number(value)},
            self._next_step(None),
        )
        if self._upload_artifacts:
            self._upload_run_files(summary)
        try:
            self._run.finish()
        except Exception as exc:
            self._note_failure("finish", exc)

    def _upload_run_files(self, summary: dict[str, Any]) -> None:
        """把一次运行的可复算证据挂到 run 上，而不是只留曲线。"""
        if self._artifact_dir is None or not self._artifact_dir.exists():
            return
        if not self._supports_save:
            self._log({"artifact/uploaded_files": 0}, self._next_step(None))
            return
        candidates = [
            "summary.json",
            "metrics.csv",
            "events.jsonl",
            "models/registry.json",
        ]
        uploaded = 0
        for relative in candidates:
            path = self._artifact_dir / relative
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > self._artifact_max_bytes:
                self._note_failure(
                    f"skip {relative}",
                    RuntimeError(f"{size / 1024 / 1024:.1f}MB 超过单文件上限"),
                )
                continue
            try:
                saved = self._run.save(str(path), base_path=str(self._artifact_dir), policy="now")
            except Exception as exc:
                self._note_failure(f"save {relative}", exc)
                continue
            if saved:
                self._uploaded.update(str(item) for item in saved)
                uploaded += 1
        self._log({"artifact/uploaded_files": uploaded}, self._next_step(None))

    # ------------------------------------------------------------------ 渲染

    @staticmethod
    def _render_event(event_type: str, payload: dict[str, Any]) -> str:
        rows = "".join(
            f"- `{key}`：{_fmt(value)}\n" for key, value in payload.items() if key != "window"
        )
        return f"**{event_type}**\n\n{rows}"

    @staticmethod
    def _render_hypothesis(record: dict[str, Any]) -> str:
        observation = record.get("observation", {})
        decision = record.get("decision", {})
        outcome = record.get("outcome", {})
        verdict = str(record.get("verdict", "inconclusive"))

        lines = [
            f"### 窗口 {record.get('window')} · {record.get('phase', '—')}",
            "",
            (
                f"**观察** 得分 {_fmt(observation.get('score'))}"
                f"（基线 {_fmt(observation.get('baseline'))}，"
                f"Δ {_fmt(observation.get('delta'))}）"
                f"｜漂移类型 {_fmt(observation.get('drift_kind'))}"
                f"｜特征漂移 {_fmt(observation.get('feature_drift'))}"
                f"｜性能下降 {_fmt(observation.get('performance_drop'))}"
                f"｜标签漂移 {_fmt(observation.get('label_shift'))}"
                f"｜可执行信号 {_fmt(observation.get('is_actionable'))}"
            ),
            "",
            f"**诊断** {record.get('diagnosis') or '—'}",
            "",
            (
                f"**决策** `{_fmt(decision.get('action'))}`"
                f"（来源 {_fmt(decision.get('source'))}"
                f"，模型 {_fmt(decision.get('model'))}"
                f"，置信 {_fmt(decision.get('confidence'), 3)}）"
            ),
            "",
            f"> {decision.get('reason') or '—'}",
            "",
            f"**预期** {record.get('expectation') or '—'}",
            "",
            (
                f"**结果** {_fmt(outcome.get('status'))}"
                f"｜holdout 提升 {_fmt(outcome.get('candidate_improvement'))}"
                f"｜历史回退 {_fmt(outcome.get('historical_drop'))}"
                f"｜上线版本 {_fmt(outcome.get('deployed_version'))}"
            ),
            "",
            (
                f"**判定** {_VERDICT_MARK.get(verdict, verdict)}"
                f" —— {record.get('verdict_reason') or ''}"
            ),
        ]
        return "\n".join(lines)


class CompositeTracker:
    """主记录器 + 若干镜像；镜像连续失败达到阈值后静默退出，不影响主流程。"""

    #: 连续失败多少次后停用该镜像（网络抖动不该永久踢掉它）。
    MIRROR_FAILURE_LIMIT = 5

    def __init__(
        self,
        primary: ExperimentTracker,
        mirrors: list[TrackerMirror] | None = None,
    ) -> None:
        self.primary = primary
        self._mirrors = mirrors or []
        self._failures: dict[int, int] = {}
        self._disabled: set[int] = set()
        self.run_id = primary.run_id
        self.run_dir = primary.run_dir

    def _notify_mirrors(self, method: str, *args: Any) -> None:
        for mirror in self._mirrors:
            key = id(mirror)
            if key in self._disabled:
                continue
            try:
                getattr(mirror, method)(*args)
            except Exception as exc:
                streak = self._failures.get(key, 0) + 1
                self._failures[key] = streak
                self.primary.log_event(
                    "tracker_mirror_failure",
                    {
                        "mirror": type(mirror).__name__,
                        "method": method,
                        "error": f"{type(exc).__name__}: {exc}",
                        "consecutive": streak,
                    },
                )
                if streak >= self.MIRROR_FAILURE_LIMIT:
                    self._disabled.add(key)
                    self.primary.log_event(
                        "tracker_mirror_disabled",
                        {"mirror": type(mirror).__name__, "consecutive_failures": streak},
                    )
            else:
                self._failures[key] = 0

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.primary.log_event(event_type, payload)
        self._notify_mirrors("log_event", event_type, payload)

    def log_metrics(self, row: dict[str, Any]) -> None:
        self.primary.log_metrics(row)
        self._notify_mirrors("log_metrics", row)

    def log_hypothesis(self, record: dict[str, Any]) -> None:
        self.primary.log_hypothesis(record)
        self._notify_mirrors("log_hypothesis", record)

    def close(self, summary: dict[str, Any]) -> None:
        # 先落本地，再上传：镜像挂了也一定留下可复算的本地证据。
        self.primary.close(summary)
        self._notify_mirrors("close", summary)
