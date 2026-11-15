from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib

from .sklearn_adapter import SklearnModelAdapter


class ModelRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self._index: dict[str, Any] = {"current": None, "versions": []}
        if self.index_path.exists():
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))

    @property
    def _versions(self) -> list[dict[str, Any]]:
        return self._index["versions"]

    def _version_names(self) -> list[str]:
        return [entry["version"] for entry in self._versions]

    def _entry_for(self, version: str) -> dict[str, Any]:
        try:
            return next(entry for entry in self._versions if entry["version"] == version)
        except StopIteration as exc:
            raise RuntimeError(f"注册表中不存在模型版本：{version}") from exc

    def _previous_version(self, version: str) -> str | None:
        entry = self._entry_for(version)
        if "previous" in entry:
            return entry["previous"]
        names = self._version_names()
        position = names.index(version)
        return names[position - 1] if position > 0 else None

    def _save_index(self) -> None:
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.index_path)

    def register(self, model: SklearnModelAdapter, metadata: dict[str, Any]) -> str:
        version = f"v{len(self._versions):03d}"
        previous = self.current_version()
        path = self.root / f"{version}.joblib"
        temporary = path.with_suffix(".tmp")
        joblib.dump(model, temporary)
        os.replace(temporary, path)

        protected = {"version", "path", "previous"}
        safe_metadata = {key: value for key, value in metadata.items() if key not in protected}
        self._versions.append(
            {"version": version, "path": path.name, "previous": previous, **safe_metadata}
        )
        self._index["current"] = version
        self._save_index()
        return version

    def current_version(self) -> str | None:
        return self._index["current"]

    def load(self, version: str) -> SklearnModelAdapter:
        return joblib.load(self.root / self._entry_for(version)["path"])

    def load_current(self) -> SklearnModelAdapter:
        version = self.current_version()
        if version is None:
            raise RuntimeError("注册表中没有可用模型")
        return self.load(version)

    def rollback_target(self) -> str:
        current = self.current_version()
        if current is None:
            raise RuntimeError("没有可回滚的历史模型")
        target = self._previous_version(current)
        if target is None:
            raise RuntimeError("没有可回滚的历史模型")
        return target

    def activate(self, version: str) -> None:
        self._entry_for(version)
        self._index["current"] = version
        self._save_index()

    def can_rollback(self) -> bool:
        current = self.current_version()
        return current is not None and self._previous_version(current) is not None

    def rollback(self) -> tuple[str, SklearnModelAdapter]:
        target = self.rollback_target()
        model = self.load(target)
        self.activate(target)
        return target, model
