import numpy as np
import pandas as pd

from industrial_evolution.models.registry import ModelRegistry
from industrial_evolution.models.sklearn_adapter import SklearnModelAdapter


def test_registry_can_restore_previous_model(tmp_path):
    X = pd.DataFrame({"a": [-2, -1, 1, 2], "b": [0, 1, 0, 1]})
    y = np.array([0, 0, 1, 1])
    first = SklearnModelAdapter(seed=1).fit(X, y)
    second = SklearnModelAdapter(seed=2).fit(X, 1 - y)
    registry = ModelRegistry(tmp_path)

    assert registry.register(first, {"score": 0.8}) == "v000"
    assert registry.register(second, {"score": 0.7}) == "v001"
    version, restored = registry.rollback()

    assert version == "v000"
    np.testing.assert_array_equal(restored.predict(X), first.predict(X))

    third = SklearnModelAdapter(seed=3).fit(X, y)
    assert registry.register(third, {"score": 0.9}) == "v002"
    version, restored = registry.rollback()

    assert version == "v000"
    np.testing.assert_array_equal(restored.predict(X), first.predict(X))
