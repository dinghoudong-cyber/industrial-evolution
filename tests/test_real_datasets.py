from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from industrial_evolution.data.real_datasets import (
    CmapssStream,
    GasStream,
    XjtuBearingStream,
    _read_libsvm,
    load_cmapss,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CMAPSS_DIR = _PROJECT_ROOT / "datasets" / "cmapss"
_GAS_DIR = _PROJECT_ROOT / "datasets" / "gas"
_XJTU_DIR = Path(r"D:\桌面\XJTU-SY轴承数据集\Data\XJTU-SY_Bearing_Datasets")


def _cmapss_line(unit: int, cycle: int, sensors: list[float]) -> str:
    parts = [
        str(unit),
        str(cycle),
        str(cycle / 10),
        str(cycle / 20),
        str(cycle / 30),
        *[str(value) for value in sensors],
    ]
    return " ".join(parts) + "\n"


def _write_cmapss(path: Path, n_engines: int, cycles: int) -> None:
    lines = []
    for unit in range(1, n_engines + 1):
        for cycle in range(1, cycles + 1):
            sensors = [unit * 1000.0 + cycle * 10.0 + i for i in range(21)]
            lines.append(_cmapss_line(unit, cycle, sensors))
    path.write_text("".join(lines), encoding="ascii")


def test_read_libsvm_parses_sparse_lines(tmp_path: Path):
    data = tmp_path / "mini.dat"
    data.write_text(
        "1 1:0.5 3:1.2\n2 1:0.1 2:0.9 3:0.4\n",
        encoding="ascii",
    )
    X, y = _read_libsvm(data, n_features=3)
    assert X.shape == (2, 3)
    np.testing.assert_array_equal(y, np.array([1, 2]))
    np.testing.assert_allclose(X[0], [0.5, 0.0, 1.2])
    np.testing.assert_allclose(X[1], [0.1, 0.9, 0.4])


def test_load_cmapss_constructs_clipped_rul(tmp_path: Path):
    _write_cmapss(tmp_path / "train_FD001.txt", n_engines=2, cycles=3)
    loaded = load_cmapss(tmp_path, fd="FD001", use_rolling=False)
    # engine1 max=3 -> RUL [2,1,0]；engine2 max=3 -> RUL [2,1,0]（均小于 clip 上限）。
    np.testing.assert_allclose(loaded.rul, [2.0, 1.0, 0.0, 2.0, 1.0, 0.0])


def test_load_cmapss_clips_rul_to_upper_bound(tmp_path: Path):
    path = tmp_path / "train_FD001.txt"
    lines = []
    for cycle in range(1, 5):
        sensors = [float(cycle) + i for i in range(21)]
        lines.append(_cmapss_line(1, cycle, sensors))
    path.write_text("".join(lines), encoding="ascii")
    loaded = load_cmapss(tmp_path, fd="FD001", use_rolling=False, rul_clip=2)
    # 单个 engine，max=4 -> RUL [3,2,1,0]，clip 到 2 -> [2,2,1,0]。
    np.testing.assert_allclose(loaded.rul, [2.0, 2.0, 1.0, 0.0])


def test_load_cmapss_drops_constant_columns(tmp_path: Path):
    path = tmp_path / "train_FD001.txt"
    # 21 个传感器里让第 1 个恒为 0，其余随 cycle 变化；应被剔除。
    lines = []
    for cycle in range(1, 5):
        sensors = [0.0] + [float(cycle) + i for i in range(1, 21)]
        lines.append(_cmapss_line(1, cycle, sensors))
    path.write_text("".join(lines), encoding="ascii")
    loaded = load_cmapss(tmp_path, fd="FD001", use_rolling=False)
    assert "sensor_1" not in loaded.feature_columns
    assert "sensor_2" in loaded.feature_columns


def test_load_cmapss_rolling_features_expand_columns(tmp_path: Path):
    path = tmp_path / "train_FD001.txt"
    lines = []
    for cycle in range(1, 6):
        sensors = [float(cycle) + i for i in range(21)]
        lines.append(_cmapss_line(1, cycle, sensors))
    path.write_text("".join(lines), encoding="ascii")
    loaded = load_cmapss(tmp_path, fd="FD001", use_rolling=True, rolling_window=3)
    # 3 个 setting + 21 个传感器各扩成 原值/均值/标准差/趋势 四类 -> (3+21)*4 = 96。
    assert len(loaded.feature_columns) == (3 + 21) * 4


def test_cmapss_stream_slices_windows_in_order(tmp_path: Path):
    _write_cmapss(tmp_path / "train_FD001.txt", n_engines=8, cycles=10)
    stream = CmapssStream(tmp_path, fd="FD001", initial_size=20, window_size=10, use_rolling=False)

    initial_X, initial_y = stream.initial_data()
    windows = list(stream)

    assert len(initial_X) == 20
    assert len(windows) == 6
    assert [len(window.y) for window in windows] == [10] * 6
    np.testing.assert_array_equal(windows[0].sample_ids, np.arange(20, 30))
    np.testing.assert_array_equal(windows[-1].sample_ids, np.arange(70, 80))
    assert stream.dropped_tail_rows == 0


def test_gas_stream_concatenates_batches_in_order(tmp_path: Path):
    dataset = tmp_path / "Dataset"
    dataset.mkdir()
    (dataset / "batch1.dat").write_text(
        "".join(f"{i % 3 + 1} 1:{i}.0 2:{i + 1}.0\n" for i in range(6)),
        encoding="ascii",
    )
    (dataset / "batch2.dat").write_text(
        "".join(f"{i % 3 + 1} 1:{i + 6}.0 2:{i + 7}.0\n" for i in range(6)),
        encoding="ascii",
    )
    stream = GasStream(tmp_path, initial_size=2, window_size=10, n_features=2)

    initial_X, initial_y = stream.initial_data()
    windows = list(stream)

    assert len(initial_X) == 2
    np.testing.assert_array_equal(initial_y, np.array([1, 2]))
    assert len(windows) == 1
    # batch 顺序拼接：batch1 的 4 个剩余行先于 batch2 的 6 行进入窗口。
    np.testing.assert_array_equal(windows[0].y, np.array([3, 1, 2, 3, 1, 2, 3, 1, 2, 3]))


@pytest.mark.skipif(not _CMAPSS_DIR.is_dir(), reason="C-MAPSS 数据未解压")
def test_cmapss_stream_loads_real_fd001():
    stream = CmapssStream(_CMAPSS_DIR, fd="FD001")
    initial_X, initial_y = stream.initial_data()
    windows = list(stream)
    assert len(initial_X) == 3000
    assert len(initial_y) == 3000
    assert len(windows) == stream.full_windows
    assert windows  # 冷启动后至少有一个完整窗口
    # RUL 是连续回归标签，且全部落在 [0, RUL_CLIP]。
    assert np.all(initial_y >= 0) and np.all(initial_y <= 130)


@pytest.mark.skipif(not _GAS_DIR.is_dir(), reason="Gas 数据未解压")
def test_gas_stream_loads_real_batches():
    stream = GasStream(_GAS_DIR)
    initial_X, initial_y = stream.initial_data()
    windows = list(stream)
    assert len(initial_X) == 3000
    assert len(windows) == stream.full_windows
    assert windows
    assert set(np.unique(initial_y)).issubset({1, 2, 3, 4, 5, 6})


def _write_xjtu_csv(path: Path, offset: float) -> None:
    import pandas as pd

    signal = np.arange(16, dtype=float) + offset
    pd.DataFrame(
        {
            "Horizontal_vibration_signals": signal,
            "Vertical_vibration_signals": signal * -2,
        }
    ).to_csv(path, index=False)


def test_xjtu_stream_aggregates_sorted_files_and_constructs_rul(tmp_path: Path):
    bearing_dir = tmp_path / "XJTU-SY_Bearing_Datasets" / "35Hz12kN" / "Bearing1_1"
    bearing_dir.mkdir(parents=True)
    for number in reversed(range(1, 24)):
        _write_xjtu_csv(bearing_dir / f"{number}.csv", float(number))
    stream = XjtuBearingStream(
        tmp_path,
        "35Hz12kN",
        "Bearing1_1",
        initial_size=3,
        window_size=10,
    )

    initial_X, initial_y = stream.initial_data()
    windows = list(stream)

    assert initial_X.shape == (3, 12)
    assert initial_X.iloc[0]["Horizontal_vibration_signals_mean"] == pytest.approx(8.5)
    np.testing.assert_array_equal(initial_y, [22.0, 21.0, 20.0])
    assert len(windows) == 2
    np.testing.assert_array_equal(windows[0].y, np.arange(19.0, 9.0, -1.0))
    np.testing.assert_array_equal(windows[1].y, np.arange(9.0, -1.0, -1.0))
    assert windows[0].sample_ids[0] == "35Hz12kN/Bearing1_1/4"
    assert windows[1].sample_ids[-1] == "35Hz12kN/Bearing1_1/23"
    assert len(set(windows[0].sample_ids) | set(windows[1].sample_ids)) == 20
    assert stream.dropped_tail_rows == 0


def test_xjtu_stream_rejects_missing_sequence_number(tmp_path: Path):
    bearing_dir = tmp_path / "XJTU-SY_Bearing_Datasets" / "35Hz12kN" / "Bearing1_1"
    bearing_dir.mkdir(parents=True)
    for number in [1, *range(3, 15)]:
        _write_xjtu_csv(bearing_dir / f"{number}.csv", float(number))
    with pytest.raises(ValueError, match="缺失序号：\\[2\\]"):
        XjtuBearingStream(
            tmp_path,
            "35Hz12kN",
            "Bearing1_1",
            initial_size=2,
            window_size=10,
        )


@pytest.mark.skipif(not _XJTU_DIR.is_dir(), reason="XJTU-SY 数据未解压")
def test_xjtu_stream_loads_real_bearing():
    stream = XjtuBearingStream(
        _XJTU_DIR,
        "35Hz12kN",
        "Bearing1_5",
        initial_size=12,
        window_size=10,
    )
    initial_X, initial_y = stream.initial_data()
    windows = list(stream)
    assert initial_X.shape == (12, 12)
    assert len(windows) == stream.full_windows == 4
    assert np.isfinite(initial_X.to_numpy()).all()
    assert initial_y[0] == 51 and initial_y[-1] == 40
    assert windows[-1].y[-1] == 0
    assert windows[0].sample_ids[0] == "35Hz12kN/Bearing1_5/13"
