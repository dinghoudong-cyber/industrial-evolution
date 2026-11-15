from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from industrial_evolution.config import load_config
from industrial_evolution.data.official import (
    OfficialPackError,
    OfficialPackStream,
    load_pack_spec,
)
from industrial_evolution.orchestrator import EvolutionOrchestrator
from industrial_evolution.tracking.local_tracker import LocalTracker
from industrial_evolution.tracking.swanlab_tracker import CompositeTracker

CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _frame(
    rows: int,
    *,
    seed: int = 0,
    label_column: str = "label",
    start_id: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "row_id": np.arange(start_id, start_id + rows),
        "ts": np.arange(rows) + 1_700_000_000,
        "x": rng.normal(size=rows),
        "z": rng.integers(0, 5, size=rows),
    }
    # 漂移检测要求足够多的特征列，这里按真实数据包规模铺开。
    for index in range(6):
        data[f"f{index}"] = rng.normal(size=rows)
    frame = pd.DataFrame(data)
    frame[label_column] = np.where(frame["x"] > 0, "normal", "warning")
    return frame


def write_pack(
    root: Path,
    *,
    initial_rows: int = 120,
    stream_rows: int = 240,
    manifest: dict | None = None,
    initial_name: str = "initial.csv",
    stream_name: str = "stream.csv",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _frame(initial_rows, seed=1).to_csv(root / initial_name, index=False)
    # 真实数据包里初始集与流的样本ID是全局唯一的，这里按同样约定生成。
    _frame(stream_rows, seed=2, start_id=initial_rows).to_csv(root / stream_name, index=False)
    if manifest is not None:
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return root


MANIFEST = {
    "target_column": "label",
    "id_column": "row_id",
    "timestamp_column": "ts",
    "initial_file": "initial.csv",
    "stream_file": "stream.csv",
    "initial_size": 120,
    "window_size": 60,
    "task_type": "classification",
}


def test_manifest_drives_layout_and_excludes_id_and_timestamp(tmp_path: Path):
    pack = write_pack(tmp_path / "pack", manifest=MANIFEST)

    spec = load_pack_spec(pack)
    stream = OfficialPackStream(spec)

    assert spec.source == "manifest"
    assert spec.target_column == "label"
    assert spec.window_size == 60
    declared = list(pd.read_csv(pack / "stream.csv", nrows=0).columns)
    assert stream.feature_columns == [
        column for column in declared if column not in {"row_id", "ts", "label"}
    ]
    assert "row_id" not in stream.feature_columns
    assert "ts" not in stream.feature_columns
    assert stream.initial_rows == 120
    assert stream.total_stream_rows == 240
    assert stream.full_windows == 4
    assert stream.dropped_tail_rows == 0

    X_initial, y_initial = stream.initial_data()
    assert len(X_initial) == 120
    assert set(np.unique(y_initial)) == {"normal", "warning"}


def test_windows_keep_row_order_and_continue_ids_from_initial_set(tmp_path: Path):
    pack = write_pack(tmp_path / "pack", manifest=MANIFEST)
    stream = OfficialPackStream(load_pack_spec(pack))

    windows = list(stream)

    assert [window.index for window in windows] == [0, 1, 2, 3]
    assert all(len(window.y) == 60 for window in windows)
    assert all(window.phase == "official" for window in windows)
    # ID 列存在时直接使用数据包里的样本ID，便于跨窗口去重与守护集隔离。
    np.testing.assert_array_equal(windows[0].sample_ids, np.arange(120, 180))
    np.testing.assert_array_equal(windows[-1].sample_ids, np.arange(300, 360))


def test_layout_is_inferred_without_manifest_and_cli_can_override(tmp_path: Path):
    pack = write_pack(
        tmp_path / "pack",
        initial_name="train.csv",
        stream_name="test.csv",
    )

    with pytest.raises(OfficialPackError, match="未声明目标列"):
        load_pack_spec(pack)

    spec = load_pack_spec(
        pack,
        target_column="label",
        initial_size=100,
        window_size=50,
        drop_columns=["row_id", "ts"],
    )
    stream = OfficialPackStream(spec)

    assert spec.source == "inferred"
    assert spec.initial_file.name == "train.csv"
    assert spec.stream_file.name == "test.csv"
    assert stream.full_windows == 4
    # 未声明ID列时退化为连续整数ID，从初始集行数继续编号。
    windows = list(stream)
    np.testing.assert_array_equal(windows[0].sample_ids, np.arange(100, 150))


def test_manifest_can_use_aliases_and_custom_filenames(tmp_path: Path):
    pack = write_pack(
        tmp_path / "pack",
        initial_name="cold_start.csv",
        stream_name="hidden.csv",
        manifest={
            "label": "label",
            "id": "row_id",
            "initial_file": "cold_start.csv",
            "stream_file": "hidden.csv",
            "initial_size": 120,
            "window_size": 40,
        },
    )

    spec = load_pack_spec(pack)

    assert spec.target_column == "label"
    assert spec.id_column == "row_id"
    assert OfficialPackStream(spec).full_windows == 6


def test_missing_target_column_lists_actual_columns(tmp_path: Path):
    pack = write_pack(tmp_path / "pack", manifest={**MANIFEST, "target_column": "y_true"})

    with pytest.raises(OfficialPackError, match="缺少目标列 y_true"):
        OfficialPackStream(load_pack_spec(pack))


def test_non_numeric_feature_is_rejected_with_actionable_message(tmp_path: Path):
    pack = write_pack(tmp_path / "pack", manifest=MANIFEST)
    for name in ("initial.csv", "stream.csv"):
        frame = pd.read_csv(pack / name)
        frame["device"] = "sensor-a"
        frame.to_csv(pack / name, index=False)

    with pytest.raises(OfficialPackError, match="drop_columns"):
        OfficialPackStream(load_pack_spec(pack))


def test_missing_stream_file_reports_candidates(tmp_path: Path):
    pack = write_pack(tmp_path / "pack", manifest={**MANIFEST, "stream_file": "absent.csv"})

    with pytest.raises(OfficialPackError, match="评测流文件不存在"):
        load_pack_spec(pack)


def test_short_stream_is_rejected_before_running(tmp_path: Path):
    pack = write_pack(tmp_path / "pack", stream_rows=40, manifest=MANIFEST)

    with pytest.raises(OfficialPackError, match="不足一个"):
        OfficialPackStream(load_pack_spec(pack))


def test_pack_runs_through_the_orchestrator(tmp_path: Path):
    pack = write_pack(tmp_path / "pack", stream_rows=240, manifest=MANIFEST)
    config = load_config(CONFIG_DIR / "default.yaml")
    config.tracking.run_root = tmp_path / "runs"
    local = LocalTracker(config.tracking.run_root)

    result = EvolutionOrchestrator(
        config,
        OfficialPackStream(load_pack_spec(pack)),
        CompositeTracker(local),
    ).run()

    assert result["windows_processed"] == 4
    assert result["run_status"] == "complete"
    assert result["metric_name"] == "balanced_accuracy"
    metrics = (local.metrics_path).read_text(encoding="utf-8-sig")
    assert "official" in metrics


def test_duplicate_ids_between_initial_and_stream_are_rejected(tmp_path: Path):
    pack = write_pack(tmp_path / "pack", manifest=MANIFEST)
    # 覆盖成两份都从 0 开始编号的数据，模拟数据包ID未全局唯一的情况。
    _frame(120, seed=1, start_id=0).to_csv(pack / "initial.csv", index=False)
    _frame(240, seed=2, start_id=0).to_csv(pack / "stream.csv", index=False)

    with pytest.raises(OfficialPackError, match="重复样本ID"):
        OfficialPackStream(load_pack_spec(pack))
