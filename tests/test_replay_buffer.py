import numpy as np
import pandas as pd

from industrial_evolution.memory.replay_buffer import ReplayBuffer


def test_replay_buffer_is_bounded_stratified_and_reproducible():
    X = pd.DataFrame({"x": np.arange(500), "z": np.arange(500) * 2})
    y = np.array([0] * 430 + [1] * 70)
    first = ReplayBuffer(capacity=120, seed=11, task_type="classification")
    second = ReplayBuffer(capacity=120, seed=11, task_type="classification")

    first.update(X, y, np.arange(500))
    second.update(X, y, np.arange(500))

    assert len(first) == 120
    assert set(first.y) == {0, 1}
    pd.testing.assert_frame_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    np.testing.assert_array_equal(first.sample_ids, second.sample_ids)

    excluded = set(first.sample_ids[:20].tolist())
    _, _, sampled_ids = first.sample_with_ids(80, exclude_ids=excluded)
    assert not (set(sampled_ids.tolist()) & excluded)
