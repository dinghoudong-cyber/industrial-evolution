from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from industrial_evolution.models.official_adapter import A_CLASSES, OfficialBaselineAdapter


@pytest.mark.parametrize(
    ("profile", "task", "feature_count", "classes"),
    [
        ("A", "classification", 128, A_CLASSES),
        ("B", "regression", 34, []),
        ("C", "regression", 24, []),
    ],
)
def test_profile_contract_accepts_matching_checkpoint_metadata(
    tmp_path, profile, task, feature_count, classes
):
    adapter = OfficialBaselineAdapter(tmp_path, profile)
    adapter.meta_json = json.dumps(
        {
            "profile": profile,
            "task": task,
            "paths": [f"feature-{index}" for index in range(feature_count)],
            "classes": classes,
        }
    )

    adapter.validate_profile()
    assert adapter.feature_count == feature_count


def test_profile_contract_rejects_checkpoint_with_wrong_task(tmp_path):
    adapter = OfficialBaselineAdapter(tmp_path, "B")
    adapter.meta_json = json.dumps(
        {"task": "classification", "paths": ["x"] * 34, "classes": ["a", "b"]}
    )
    with pytest.raises(ValueError, match="期望任务"):
        adapter.validate_profile()


def test_dataframe_columns_must_match_checkpoint_paths_and_order(tmp_path):
    adapter = OfficialBaselineAdapter(tmp_path, "C")
    paths = [f"path-{index}" for index in range(24)]
    adapter.meta_json = json.dumps({"paths": paths})
    X = pd.DataFrame(np.zeros((2, 24)), columns=paths)
    assert adapter._as_matrix(X).shape == (2, 24)

    with pytest.raises(ValueError, match="路径或顺序"):
        adapter._as_matrix(X[paths[::-1]])


def test_save_checkpoint_refuses_to_overwrite_before_loading_model(tmp_path):
    adapter = OfficialBaselineAdapter(tmp_path, "A")
    existing = tmp_path / "checkpoint.npz"
    existing.write_bytes(b"preserve me")

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        adapter.save_checkpoint(existing)
    assert existing.read_bytes() == b"preserve me"
