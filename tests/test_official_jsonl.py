from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrial_evolution.data.official_jsonl import (
    flatten_inputs,
    read_official_jsonl,
    validate_official_pair,
)


def _row(profile: str, identity: str, group: int) -> dict:
    if profile == "A":
        inputs = {"features": [float(index) for index in range(128)]}
        answer = "Ethanol"
        meta = {"source_batch": group, "source_index": 1}
    elif profile == "B":
        names = [f"feature_{index:02d}" for index in range(17)]
        inputs = {
            "features": {
                "horizontal": {name: float(index) for index, name in enumerate(names)},
                "vertical": {name: float(index + 17) for index, name in enumerate(names)},
            }
        }
        answer = 120.0
        meta = {"bearing": f"Bearing{group}_1", "snapshot": 1}
    else:
        inputs = {
            "settings": [float(index) for index in range(3)],
            "sensors": [float(index) for index in range(21)],
        }
        answer = 50.0
        meta = {"source_subset": "FD001", "unit": group, "cycle": 1}
    return {"id": identity, "inputs": inputs, "answer": answer, "meta": meta}


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("profile", ["A", "B", "C"])
def test_reader_maps_official_profile_schema_and_ignores_meta(tmp_path: Path, profile: str):
    classes = (
        ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
        if profile == "A"
        else []
    )
    data = read_official_jsonl(
        _write(tmp_path / f"{profile}.jsonl", [_row(profile, "sample-1", 1)]),
        profile,
        classes=classes,
    )

    assert len(data.ids) == 1
    assert data.X.shape == (1, {"A": 128, "B": 34, "C": 24}[profile])
    assert "meta" not in " ".join(data.feature_paths)
    assert "answer" not in " ".join(data.feature_paths)
    if profile == "A":
        assert data.y.tolist() == [0]
    else:
        assert data.y.tolist() == [120.0 if profile == "B" else 50.0]


def test_flatten_is_key_order_independent_and_rejects_target_derived_features():
    paths_a, values_a = flatten_inputs({"z": 2.0, "a": [1.0]})
    paths_b, values_b = flatten_inputs({"a": [1.0], "z": 2.0})
    assert paths_a == paths_b
    assert values_a == values_b
    with pytest.raises(ValueError, match="目标或实体派生"):
        flatten_inputs({"rul": 20.0})


def test_reader_rejects_duplicate_ids_and_nonfinite_json(tmp_path: Path):
    row = _row("A", "sample-1", 1)
    duplicate = _write(tmp_path / "duplicate.jsonl", [row, row])
    classes = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
    with pytest.raises(ValueError, match="唯一"):
        read_official_jsonl(duplicate, "A", classes=classes)

    bad = tmp_path / "nan.jsonl"
    bad.write_text(
        '{"id":"x","inputs":{"features":[NaN]},"answer":"Ethanol","meta":{"source_batch":1}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="NaN"):
        read_official_jsonl(bad, "A", classes=classes)


def test_reader_rejects_checkpoint_path_mismatch(tmp_path: Path):
    row = _row("C", "sample-1", 1)
    with pytest.raises(ValueError, match="特征路径或顺序"):
        read_official_jsonl(
            _write(tmp_path / "data.jsonl", [row]),
            "C",
            expected_paths=["wrong-path"] * 24,
        )


def test_public_split_requires_disjoint_entities(tmp_path: Path):
    train = read_official_jsonl(_write(tmp_path / "train.jsonl", [_row("B", "train-1", 1)]), "B")
    validation = read_official_jsonl(
        _write(tmp_path / "validation.jsonl", [_row("B", "validation-1", 1)]), "B"
    )
    with pytest.raises(ValueError, match="实体/批次"):
        validate_official_pair(train, validation, "B")
