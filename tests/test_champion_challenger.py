import numpy as np
import pandas as pd

from industrial_evolution.config import ValidationConfig
from industrial_evolution.evaluation.champion_challenger import ChampionChallengerEvaluator


class ScoreStub:
    def __init__(self, recent: float, historical: float):
        self.recent = recent
        self.historical = historical

    def score(self, X, y):
        return self.recent if X.iloc[0, 0] == 1 else self.historical


def test_candidate_accepts_only_when_recent_gain_and_history_are_safe():
    evaluator = ChampionChallengerEvaluator(
        ValidationConfig(min_improvement=0.02, max_historical_drop=0.04, guard_samples=100)
    )
    recent_X = pd.DataFrame({"marker": [1, 1]})
    guard_X = pd.DataFrame({"marker": [0, 0]})
    y = np.array([0, 1])

    accepted = evaluator.evaluate(
        ScoreStub(0.70, 0.80), ScoreStub(0.75, 0.78), recent_X, y, guard_X, y
    )
    rejected = evaluator.evaluate(
        ScoreStub(0.70, 0.80), ScoreStub(0.76, 0.70), recent_X, y, guard_X, y
    )

    assert accepted.accepted
    assert not rejected.accepted
    assert rejected.historical_drop > 0.04
