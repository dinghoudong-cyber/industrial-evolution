"""把官方 CPU 基线包（contest.cpu-mlp.v1）包装成本项目可驱动的模型适配器。

官方 A/B 包与自建 C 包共用同一套 NumPy MLP 实现，差别只在
``baselines/configs/{A,B,C}.json`` 与 ``checkpoint.npz``。因此这里只有一个适配器，
用 ``profile`` 区分赛道。

设计要点：

1. **权重而非对象**：适配器只保存裸数组（w1/b1/w2/b2、Adam 一阶二阶矩、mean/scale）
   和序列化后的 metadata，不持有 ``baselines.Model`` 实例。官方模块是运行时按包路径
   动态加载的，直接 pickle 实例会在反序列化时找不到模块；存数组则 joblib 完全可移植。
2. **new_unfitted 是 warm start**：官方协议要求「从随包检查点继续训练」，所以
   ``new_unfitted()`` 返回的是当前权重的**副本**，而不是重新初始化的模型。名字沿用
   项目内既有约定，但语义与 sklearn 版相反，已在方法 docstring 中写明。
3. **指标一律「越大越好」**：项目内部用统一方向做 gating、冷却和阈值判断。分类用
   macro_f1（官方 A 的主指标，本身越大越好）；回归返回 ``-rmse``，对外上报时再取反。
   注意官方 C 还额外看 NASA 非对称分（高估寿命罚更重），该分数不参与内部排序，
   只作为旁证记录——否则会把「保守」当成「准确」。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

PARAMETERS = ("w1", "b1", "w2", "b2")
SUPPORTED_PROFILES = ("A", "B", "C")
PROFILE_CONTRACTS = {
    "A": {"task": "classification", "feature_count": 128},
    "B": {"task": "regression", "feature_count": 34},
    "C": {"task": "regression", "feature_count": 24},
}
A_CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def _load_baseline_module(pack_root: Path):
    """按包路径加载官方 baselines.model，避免多个包同名模块互相覆盖。"""
    model_path = pack_root / "baselines" / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"包内缺少 baselines/model.py：{model_path}")
    key = f"_official_baseline_{abs(hash(str(pack_root.resolve())))}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载官方模型实现：{model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def _macro_f1(expected: np.ndarray, predicted: np.ndarray) -> float:
    """与官方 train.py 的 metrics() 完全一致的 macro-F1。

    官方在「真实标签集合 ∪ 预测标签集合」上取平均，不假设所有类别都出现；
    这里照抄，避免与平台口径出现偏差。
    """
    observed = sorted(set(expected.tolist()) | set(predicted.tolist()))
    scores = []
    for label in observed:
        denominator = np.sum(predicted == label) + np.sum(expected == label)
        scores.append(
            0.0
            if denominator == 0
            else 2 * np.sum((predicted == label) & (expected == label)) / denominator
        )
    return float(np.mean(scores)) if scores else 0.0


class OfficialBaselineAdapter:
    """官方 CPU 基线（NumPy MLP + Adam）的可续训适配器。

    参数
    ----
    pack_root:
        官方包根目录，需包含 ``baselines/model.py`` 与该 profile 的 checkpoint。
    profile:
        ``"A"`` / ``"B"`` / ``"C"``。
    checkpoint:
        起始检查点路径。传 ``None`` 表示必须随后用 ``from_fresh()`` 初始化。
    epochs:
        每次 ``fit()`` 训练多少个 epoch（官方 ``--epochs``）。
    learning_rate / batch_size:
        覆盖检查点内的学习率与批大小，留空则沿用检查点记录的值。
    """

    def __init__(
        self,
        pack_root: str | Path,
        profile: str,
        checkpoint: str | Path | None = None,
        *,
        epochs: int = 5,
        learning_rate: float | None = None,
        batch_size: int | None = None,
    ) -> None:
        if profile not in SUPPORTED_PROFILES:
            raise ValueError(f"不支持的赛题 profile：{profile}")
        self.pack_root = str(Path(pack_root).resolve())
        self.profile = profile
        self.epochs = int(epochs)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weights: dict[str, np.ndarray] = {}
        self.meta_json: str = ""
        if checkpoint is not None:
            self._load_checkpoint(Path(checkpoint))

    # ---------------------------------------------------------------- 检查点读写

    def _module(self):
        return _load_baseline_module(Path(self.pack_root))

    def _load_checkpoint(self, path: Path) -> None:
        module = self._module()
        model = module.Model.load(path)
        self._store(model)
        self.validate_profile()

    def validate_profile(self) -> None:
        """Fail closed when the selected task does not match its official profile."""
        contract = PROFILE_CONTRACTS[self.profile]
        metadata = json.loads(self.meta_json)
        if metadata.get("task") != contract["task"]:
            raise ValueError(
                f"profile {self.profile} 期望任务 {contract['task']}，"
                f"检查点任务为 {metadata.get('task')}"
            )
        if len(metadata.get("paths", [])) != contract["feature_count"]:
            raise ValueError(
                f"profile {self.profile} 期望 {contract['feature_count']} 个特征，"
                f"检查点声明 {len(metadata.get('paths', []))} 个"
            )
        checkpoint_profile = metadata.get("profile")
        if checkpoint_profile is not None and checkpoint_profile != self.profile:
            raise ValueError(f"profile {self.profile} 与检查点 profile {checkpoint_profile} 不匹配")
        if self.profile == "A" and metadata.get("classes") != A_CLASSES:
            raise ValueError("A 检查点类别名称或顺序与官方任务定义不匹配")

    def _store(self, model) -> None:
        weights: dict[str, np.ndarray] = {}
        for name in PARAMETERS:
            weights[name] = np.array(model.params[name], dtype=np.float64, copy=True)
            weights["adam_first_" + name] = np.array(model.first[name], dtype=np.float64, copy=True)
            weights["adam_second_" + name] = np.array(
                model.second[name], dtype=np.float64, copy=True
            )
        weights["mean"] = np.array(model.mean, dtype=np.float64, copy=True)
        weights["scale"] = np.array(model.scale, dtype=np.float64, copy=True)
        self.weights = weights
        self.meta_json = json.dumps(model.meta, sort_keys=True, allow_nan=False)

    def to_model(self):
        """用当前权重重建官方 Model 实例（调用方负责不要长期持有它）。"""
        if not self.weights:
            raise RuntimeError("适配器尚未加载任何权重")
        module = self._module()
        model = module.Model.__new__(module.Model)
        model.meta = json.loads(self.meta_json)
        model.params = {name: self.weights[name].copy() for name in PARAMETERS}
        model.first = {name: self.weights["adam_first_" + name].copy() for name in PARAMETERS}
        model.second = {name: self.weights["adam_second_" + name].copy() for name in PARAMETERS}
        model.mean = self.weights["mean"].copy()
        model.scale = self.weights["scale"].copy()
        model.validate()
        return model

    def save_checkpoint(self, path: str | Path) -> str:
        """按官方格式落盘，返回 sha256 摘要（可直接写进训练报告）。"""
        target = Path(path)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"拒绝覆盖已有 checkpoint：{target}")
        model = self.to_model()
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(target)
        return self._module().checkpoint_digest(target)

    # ---------------------------------------------------------------- 元信息

    @property
    def task(self) -> str:
        return json.loads(self.meta_json)["task"] if self.meta_json else "unknown"

    @property
    def classes(self) -> list[str]:
        return list(json.loads(self.meta_json)["classes"]) if self.meta_json else []

    @property
    def feature_count(self) -> int:
        return len(json.loads(self.meta_json)["paths"]) if self.meta_json else 0

    @property
    def parameter_count(self) -> int:
        return int(json.loads(self.meta_json)["parameter_count"]) if self.meta_json else 0

    def metric_name(self) -> str:
        """项目内部约定：数值越大越好。"""
        return "macro_f1" if self.task == "classification" else "negative_rmse"

    def official_metric(self, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        """官方口径指标，用于上报与留痕（回归的 rmse 越小越好）。"""
        predicted = self.predict(X)
        expected = np.asarray(y)
        if self.task == "classification":
            return {
                "macro_f1": _macro_f1(expected.astype(int), predicted.astype(int)),
                "accuracy": float(np.mean(predicted.astype(int) == expected.astype(int))),
            }
        error = predicted - expected.astype(float)
        return {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
        }

    # ---------------------------------------------------------------- 适配器协议

    def _as_matrix(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            expected_paths = json.loads(self.meta_json).get("paths", [])
            if list(X.columns) != expected_paths:
                raise ValueError("特征路径或顺序与检查点不一致")
            matrix = X.to_numpy(dtype=np.float64, copy=False)
        else:
            matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"期望二维特征矩阵，实际 {matrix.ndim} 维")
        if self.feature_count and matrix.shape[1] != self.feature_count:
            raise ValueError(
                f"特征维度与检查点不一致：数据 {matrix.shape[1]} 维，检查点 {self.feature_count} 维"
            )
        return matrix

    def new_unfitted(self) -> OfficialBaselineAdapter:
        """返回**保留当前权重**的副本，用于 warm start 续训。

        与 sklearn 版同名方法语义相反：官方协议要求从随包检查点继续训练，
        若在这里重新初始化，等于把「续训」偷换成「重训」，起点不再可比。
        """
        clone = OfficialBaselineAdapter(
            self.pack_root,
            self.profile,
            None,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
        )
        clone.weights = {name: value.copy() for name, value in self.weights.items()}
        clone.meta_json = self.meta_json
        return clone

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> OfficialBaselineAdapter:
        model = self.to_model()
        if self.learning_rate is not None:
            model.meta["learning_rate"] = float(self.learning_rate)
        if self.batch_size is not None:
            model.meta["batch_size"] = int(self.batch_size)
        model.validate()
        target = np.asarray(y)
        if model.meta["task"] == "classification":
            target = target.astype(np.int64)
        else:
            target = target.astype(np.float64)
        model.fit(self._as_matrix(X), target, self.epochs)
        self._store(model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        model = self.to_model()
        output = model.predict(self._as_matrix(X))
        if model.meta["task"] == "classification":
            return output.argmax(axis=1).astype(np.int64)
        return np.asarray(output, dtype=np.float64)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """分类返回 softmax 概率；回归没有概率语义，返回 None。"""
        model = self.to_model()
        if model.meta["task"] != "classification":
            return None
        return np.asarray(model.predict(self._as_matrix(X)), dtype=np.float64)

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """项目内部排序用分数，统一「越大越好」。"""
        predicted = self.predict(X)
        expected = np.asarray(y)
        if self.task == "classification":
            return _macro_f1(expected.astype(int), predicted.astype(int))
        error = predicted - expected.astype(float)
        return -float(np.sqrt(np.mean(error**2)))

    # ---------------------------------------------------------------- 便利构造

    @classmethod
    def from_pack(cls, pack_root: str | Path, profile: str, **kwargs: Any) -> Self:
        """按约定从 ``<pack_root>/checkpoint.npz`` 加载起点。"""
        checkpoint = Path(pack_root) / "checkpoint.npz"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"包内缺少起始检查点：{checkpoint}")
        return cls(pack_root, profile, checkpoint, **kwargs)

    def describe(self) -> dict[str, Any]:
        meta = json.loads(self.meta_json) if self.meta_json else {}
        return {
            "backend": "official_cpu_mlp",
            "profile": self.profile,
            "pack_root": self.pack_root,
            "task": meta.get("task"),
            "classes": meta.get("classes"),
            "hidden": meta.get("hidden"),
            "parameter_count": meta.get("parameter_count"),
            "input_features": len(meta.get("paths", [])),
            "epochs_in_checkpoint": meta.get("epochs"),
            "steps_in_checkpoint": meta.get("steps"),
            "learning_rate": meta.get("learning_rate"),
            "batch_size": meta.get("batch_size"),
            "fit_epochs_per_call": self.epochs,
            "metric": self.metric_name(),
        }
