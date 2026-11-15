from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ValidationConfig
from ..domain import CandidateEvaluation
from ..models.sklearn_adapter import SklearnModelAdapter


class ChampionChallengerEvaluator:
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def evaluate(
        self,
        champion: SklearnModelAdapter,
        challenger: SklearnModelAdapter,
        recent_X: pd.DataFrame,
        recent_y: np.ndarray,
        historical_X: pd.DataFrame,
        historical_y: np.ndarray,
    ) -> CandidateEvaluation:
        champion_recent = champion.score(recent_X, recent_y)
        challenger_recent = challenger.score(recent_X, recent_y)
        improvement = challenger_recent - champion_recent

        champion_historical: float | None = None
        challenger_historical: float | None = None
        historical_drop = 0.0
        if len(historical_y) > 0:
            champion_historical = champion.score(historical_X, historical_y)
            challenger_historical = challenger.score(historical_X, historical_y)
            historical_drop = max(0.0, champion_historical - challenger_historical)

        if improvement < self.config.min_improvement:
            accepted = False
            reason = f"近期提升 {improvement:.4f} 未达到门槛 {self.config.min_improvement:.4f}"
        elif historical_drop > self.config.max_historical_drop:
            accepted = False
            reason = (
                f"历史守护集退化 {historical_drop:.4f} 超过上限 "
                f"{self.config.max_historical_drop:.4f}"
            )
        else:
            accepted = True
            reason = f"近期提升 {improvement:.4f}，历史退化 {historical_drop:.4f}，允许上线"

        return CandidateEvaluation(
            accepted=accepted,
            champion_recent_score=round(champion_recent, 6),
            challenger_recent_score=round(challenger_recent, 6),
            recent_improvement=round(improvement, 6),
            champion_historical_score=None
            if champion_historical is None
            else round(champion_historical, 6),
            challenger_historical_score=None
            if challenger_historical is None
            else round(challenger_historical, 6),
            historical_drop=round(historical_drop, 6),
            reason=reason,
        )
