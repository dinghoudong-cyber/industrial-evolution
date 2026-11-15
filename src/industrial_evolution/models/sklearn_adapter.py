from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_squared_error


@dataclass
class SklearnModelAdapter:
    """工业模型适配器。

    ``score_metric`` 可配置，用于对齐官方主指标：A 包分类用 ``macro_f1``，
    B/C 包回归用 ``negative_rmse``。默认 ``auto`` 保持历史行为
    （分类 = balanced_accuracy，回归 = negative_rmse）。所有指标统一「越大越好」，
    以便 Champion-Challenger 门控不改动即可复用。
    """

    task_type: str = "classification"
    seed: int = 42
    model: object | None = None
    score_metric: str = "auto"

    def _resolved_metric(self) -> str:
        if self.score_metric != "auto":
            if self.task_type == "regression" and self.score_metric != "negative_rmse":
                raise ValueError("回归任务只支持 negative_rmse")
            if self.task_type == "classification" and self.score_metric == "negative_rmse":
                raise ValueError("分类任务不支持 negative_rmse")
            return self.score_metric
        return "balanced_accuracy" if self.task_type == "classification" else "negative_rmse"

    def _build_model(self) -> object:
        common = {
            "max_iter": 140,
            "learning_rate": 0.08,
            "max_leaf_nodes": 15,
            "l2_regularization": 0.1,
            "random_state": self.seed,
        }
        if self.task_type == "classification":
            return HistGradientBoostingClassifier(**common)
        if self.task_type == "regression":
            return HistGradientBoostingRegressor(**common)
        raise ValueError(f"不支持的任务类型：{self.task_type}")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SklearnModelAdapter:
        self.model = self._build_model()
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("模型尚未训练")
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        if self.model is None:
            raise RuntimeError("模型尚未训练")
        if self.task_type == "classification" and hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))
        return None

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        predictions = self.predict(X)
        metric = self._resolved_metric()
        if metric == "macro_f1":
            return float(f1_score(y, predictions, average="macro", zero_division=0))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y, predictions))
        return -float(np.sqrt(mean_squared_error(y, predictions)))

    def new_unfitted(self) -> SklearnModelAdapter:
        return SklearnModelAdapter(
            task_type=self.task_type,
            seed=self.seed,
            score_metric=self.score_metric,
        )

    @property
    def metric_name(self) -> str:
        return self._resolved_metric()
