"""真实工业退化数据集接入层。

与 ``CsvWindowStream`` / ``OfficialPackStream`` 一样，本模块的每个流都实现 ``StreamSource``
协议（``initial_data()`` 返回冷启动 (X, y)，``__iter__`` 按时间顺序产出 ``DataWindow``），
可直接被 ``EvolutionOrchestrator`` 消费。

目前内置两个赛题常见数据集：

- **C-MAPSS**（涡扇发动机退化，NASA）：RUL 回归，4 个子集 FD001~FD004；
- **Gas Sensor Array Drift**（气体传感器阵列漂移，UCI）：6 类气体分类，10 个月批次；
- **XJTU-SY**（滚动轴承加速寿命试验）：单轴承分钟级振动特征与离线 RUL 回归。

共同点：都是**多元时间序列 + 天然数据漂移**，正是赛题「工业漂移数据流」的形态。因此这里的
关键不只是"能读文件"，而是把原始时序压成表格特征、构造可回归/可分类的标签、并按真实时间
顺序切窗，让现有的漂移检测与持续学习闭环能在真实数据上运转。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain import DataWindow

RUL_CLIP = 130  # C-MAPSS 常用 piecewise linear RUL 上限：把"还很健康"截断为 130

# C-MAPSS 26 列的含义（0-based）：unit / cycle / 3 个工况 / 21 个传感器
_CMAPSS_UNIT_COL = 0
_CMAPSS_CYCLE_COL = 1
_CMAPSS_SETTING_COLS = (2, 3, 4)
_CMAPSS_SENSOR_COLS = tuple(range(5, 26))


def _read_libsvm(path: Path, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """解析 libsvm 文本（``label idx:val idx:val ...``）成稠密 (X, y)。

    特征索引是 1-based；Gas 数据集为 1..128 连续，这里按声明维度对齐，越界的索引报错。
    """
    labels: list[int] = []
    rows: list[np.ndarray] = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            labels.append(int(float(parts[0])))
            row = np.zeros(n_features, dtype=float)
            for token in parts[1:]:
                idx_str, _, val_str = token.partition(":")
                idx = int(idx_str) - 1
                if not 0 <= idx < n_features:
                    raise ValueError(f"{path.name} 的特征索引 {idx + 1} 超出声明维度 {n_features}")
                row[idx] = float(val_str)
            rows.append(row)
    return np.vstack(rows), np.array(labels)


@dataclass(frozen=True, slots=True)
class _CmapssFrame:
    """加载并整理好的 C-MAPSS 训练子集：特征表 + RUL 标签 + 有效特征列名。"""

    frame: pd.DataFrame
    rul: np.ndarray
    feature_columns: list[str]


def load_cmapss(
    data_dir: str | Path,
    fd: str = "FD001",
    *,
    rul_clip: int = RUL_CLIP,
    use_rolling: bool = True,
    rolling_window: int = 20,
) -> _CmapssFrame:
    """加载 C-MAPSS 训练集，构造 piecewise linear RUL 标签与表格特征。

    - 按 engine 顺序把每个 engine 的 cycle 序列在时间上展开，保证「时间顺序」即防泄漏边界；
    - RUL = 该 engine 的最大 cycle − 当前 cycle，再 clip 到 ``[0, rul_clip]``；
    - 剔除常量列（FD001/FD003 有 6 个传感器 + 1 个工况在单工况下恒值）；
    - 可选地在每个 engine 内部对每个传感器加滑动窗均值/标准差/趋势，捕捉退化走向。
    """
    root = Path(data_dir).expanduser().resolve()
    train_file = root / f"train_{fd}.txt"
    if not train_file.is_file():
        raise FileNotFoundError(f"C-MAPSS 训练文件不存在：{train_file}")

    raw = pd.read_csv(train_file, sep=r"\s+", header=None)
    n_cols = raw.shape[1]
    if n_cols != 26:
        raise ValueError(f"{fd} 期望 26 列，实际 {n_cols} 列")

    raw.columns = (
        ["unit", "cycle"]
        + [f"setting_{i}" for i in range(3)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    # RUL：按 engine 算最大 cycle，再整体 clip。
    max_cycle = raw.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - raw["cycle"]).clip(upper=rul_clip).to_numpy(dtype=float)

    # 剔除常量列（不同 engine 间取值完全不变的列不携带信息，还会让漂移检测失真）。
    value_cols = [c for c in raw.columns if c not in ("unit", "cycle")]
    nonconstant = [c for c in value_cols if raw[c].nunique() > 1]
    frame = raw[["unit", "cycle", *nonconstant]].copy()

    feature_cols = list(nonconstant)
    if use_rolling:
        feature_cols = _add_rolling_features(frame, feature_cols, rolling_window)

    return _CmapssFrame(
        frame=frame,
        rul=rul,
        feature_columns=feature_cols,
    )


def _add_rolling_features(
    frame: pd.DataFrame,
    base_cols: list[str],
    window: int,
) -> list[str]:
    """在每个 engine 内部对每个基础列追加滑动窗统计量，返回完整特征列名。

    追加三个量，分别回答：当前值附近"平均水平 / 波动程度 / 变化趋势"。这三者组合起来
    能显著改善 RUL 预测（退化期的均值抬升、方差放大、单调趋势正是故障的早期信号）。
    """
    grouped = frame.groupby("unit", sort=False)
    new_cols = list(base_cols)
    for col in base_cols:
        series = grouped[col]
        roll = series.rolling(window, min_periods=1)
        mean_col = f"{col}_roll_mean"
        std_col = f"{col}_roll_std"
        trend_col = f"{col}_roll_trend"
        frame[mean_col] = roll.mean().reset_index(level=0, drop=True)
        frame[std_col] = roll.std(ddof=0).reset_index(level=0, drop=True)
        # 趋势：当前值 − (window-1) 步前的值，即窗口跨度内的净变化；engine 开头不足
        # window 步时退化为当前值 − 该 engine 首个值。
        lag = series.shift(window - 1)
        first = series.transform("first")
        frame[trend_col] = frame[col] - lag.fillna(first)
        new_cols.extend([mean_col, std_col, trend_col])
    return new_cols


class CmapssStream:
    """把 C-MAPSS 训练集包装成 ``StreamSource``（RUL 回归流）。

    冷启动取前 ``initial_size`` 行（按 engine 顺序展开后的时间序），其余按 ``window_size``
    行切窗，绝不打乱顺序。engine 边界处 RUL 会从 0 跳回 130，这本身就是一个真实的"新样本
    分布"信号，交由漂移检测器去捕获，而不是在这里抹平。
    """

    def __init__(
        self,
        data_dir: str | Path,
        fd: str = "FD001",
        *,
        initial_size: int = 3000,
        window_size: int = 500,
        rul_clip: int = RUL_CLIP,
        use_rolling: bool = True,
        rolling_window: int = 20,
    ) -> None:
        if initial_size < 2:
            raise ValueError("初始样本数至少为 2")
        if window_size < 10:
            raise ValueError("窗口大小至少为 10")
        loaded = load_cmapss(
            data_dir,
            fd,
            rul_clip=rul_clip,
            use_rolling=use_rolling,
            rolling_window=rolling_window,
        )
        self._frame = loaded.frame
        self._rul = loaded.rul
        self.feature_columns = loaded.feature_columns
        self.initial_size = initial_size
        self.window_size = window_size
        self.total_rows = len(self._frame)
        if self.total_rows <= initial_size:
            raise ValueError(
                f"C-MAPSS {fd} 共 {self.total_rows} 行，不足以构造 {initial_size} 行冷启动集"
            )
        self.full_windows = (self.total_rows - initial_size) // window_size
        self.dropped_tail_rows = (self.total_rows - initial_size) % window_size
        if self.full_windows < 1:
            raise ValueError(
                f"C-MAPSS {fd} 冷启动后只剩 {self.total_rows - initial_size} 行，"
                f"不足一个 {window_size} 行的窗口"
            )

    def _xy(self, start: int, stop: int) -> tuple[pd.DataFrame, np.ndarray]:
        X = self._frame.iloc[start:stop].loc[:, self.feature_columns].reset_index(drop=True)
        X = X.apply(pd.to_numeric, errors="raise")
        y = self._rul[start:stop]
        return X, y

    def initial_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        return self._xy(0, self.initial_size)

    def __iter__(self) -> Iterator[DataWindow]:
        window_index = 0
        pos = self.initial_size
        while pos + self.window_size <= self.total_rows:
            X, y = self._xy(pos, pos + self.window_size)
            yield DataWindow(
                index=window_index,
                X=X,
                y=y,
                phase="cmapss",
                sample_ids=np.arange(pos, pos + self.window_size, dtype=np.int64),
            )
            pos += self.window_size
            window_index += 1


class GasStream:
    """把 Gas Sensor Array Drift 数据集包装成 ``StreamSource``（6 类气体分类流）。

    按 batch1→batch10 顺序拼接，模拟传感器随 10 个月逐渐漂移。冷启动取前 ``initial_size``
    行，其余按 ``window_size`` 行切窗。标签为 1~6 的类别，直接作为分类目标。
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        initial_size: int = 3000,
        window_size: int = 400,
        n_features: int = 128,
    ) -> None:
        if initial_size < 2:
            raise ValueError("初始样本数至少为 2")
        if window_size < 10:
            raise ValueError("窗口大小至少为 10")
        root = Path(data_dir).expanduser().resolve()
        dataset_dir = root / "Dataset"
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Gas 数据集目录不存在：{dataset_dir}")

        batches = sorted(dataset_dir.glob("batch*.dat"))
        if not batches:
            raise FileNotFoundError(f"{dataset_dir} 下没有 batch*.dat 文件")

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for batch in batches:
            X, y = _read_libsvm(batch, n_features)
            xs.append(X)
            ys.append(y)
        feature_names = [f"sensor_{i}" for i in range(1, n_features + 1)]
        self._X = pd.DataFrame(np.vstack(xs), columns=feature_names)
        self._y = np.concatenate(ys)
        self.feature_columns = list(self._X.columns)
        self.initial_size = initial_size
        self.window_size = window_size
        self.total_rows = len(self._X)
        if self.total_rows <= initial_size:
            raise ValueError(
                f"Gas 数据集共 {self.total_rows} 行，不足以构造 {initial_size} 行冷启动集"
            )
        self.full_windows = (self.total_rows - initial_size) // window_size
        self.dropped_tail_rows = (self.total_rows - initial_size) % window_size
        if self.full_windows < 1:
            raise ValueError(
                f"Gas 冷启动后只剩 {self.total_rows - initial_size} 行，"
                f"不足一个 {window_size} 行的窗口"
            )

    def _xy(self, start: int, stop: int) -> tuple[pd.DataFrame, np.ndarray]:
        X = self._X.iloc[start:stop].reset_index(drop=True)
        y = self._y[start:stop]
        return X, y

    def initial_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        return self._xy(0, self.initial_size)

    def __iter__(self) -> Iterator[DataWindow]:
        window_index = 0
        pos = self.initial_size
        while pos + self.window_size <= self.total_rows:
            X, y = self._xy(pos, pos + self.window_size)
            yield DataWindow(
                index=window_index,
                X=X,
                y=y,
                phase="gas",
                sample_ids=np.arange(pos, pos + self.window_size, dtype=np.int64),
            )
            pos += self.window_size
            window_index += 1


class XjtuBearingStream:
    """把一个 XJTU-SY 轴承包装成按分钟前推的离线 RUL 回归流。

    标签根据完整寿命数据的最后文件序号推导，仅适用于全寿命离线实验；
    现场数据尚未到达寿命终点时不能提前知道这些标签。
    """

    CHANNELS = ("Horizontal_vibration_signals", "Vertical_vibration_signals")
    STATISTICS = ("mean", "std", "rms", "peak_to_peak", "skewness", "kurtosis")
    CONDITIONS = ("35Hz12kN", "37.5Hz11kN", "40Hz10kN")

    def __init__(
        self,
        data_dir: str | Path,
        condition: str,
        bearing: str,
        *,
        initial_size: int = 100,
        window_size: int = 50,
    ) -> None:
        if condition not in self.CONDITIONS:
            raise ValueError(
                f"不支持的 XJTU-SY 工况 {condition!r}；可选值：{', '.join(self.CONDITIONS)}"
            )
        if not bearing.startswith("Bearing") or not bearing[7:].replace("_", "").isdigit():
            raise ValueError("bearing 格式应为 Bearing1_1 等官方轴承名称")
        if initial_size < 2:
            raise ValueError("初始样本数至少为 2")
        if window_size < 10:
            raise ValueError("窗口大小至少为 10")

        root = Path(data_dir).expanduser().resolve()
        if root.name != "XJTU-SY_Bearing_Datasets":
            root = root / "XJTU-SY_Bearing_Datasets"
        bearing_dir = root / condition / bearing
        if not bearing_dir.is_dir():
            raise FileNotFoundError(f"XJTU-SY 轴承目录不存在：{bearing_dir}")

        files = list(bearing_dir.glob("*.csv"))
        numbered: dict[int, Path] = {}
        for path in files:
            if not path.stem.isdigit():
                raise ValueError(f"XJTU-SY 文件名必须是正整数序号：{path.name}")
            number = int(path.stem)
            if number in numbered:
                raise ValueError(f"XJTU-SY 文件序号重复：{number}")
            numbered[number] = path
        if not numbered:
            raise FileNotFoundError(f"{bearing_dir} 下没有数字序号 CSV 文件")
        ordered_numbers = sorted(numbered)
        expected_numbers = list(range(1, len(ordered_numbers) + 1))
        if ordered_numbers != expected_numbers:
            missing = sorted(set(expected_numbers) - set(ordered_numbers))
            raise ValueError(f"XJTU-SY 文件序号必须从 1 连续排列；缺失序号：{missing[:20]}")
        if len(ordered_numbers) <= initial_size:
            raise ValueError(
                f"{condition}/{bearing} 共 {len(ordered_numbers)} 个文件，不足以构造"
                f" {initial_size} 行冷启动集"
            )

        self.condition = condition
        self.bearing = bearing
        self.files = [numbered[number] for number in ordered_numbers]
        self.total_rows = len(self.files)
        self.initial_size = initial_size
        self.window_size = window_size
        self.full_windows = (self.total_rows - initial_size) // window_size
        self.dropped_tail_rows = (self.total_rows - initial_size) % window_size
        if self.full_windows < 1:
            raise ValueError(
                f"{condition}/{bearing} 冷启动后只剩 {self.total_rows - initial_size} 行，"
                f"不足一个 {window_size} 行的窗口"
            )
        self.feature_columns = [
            f"{channel}_{stat}" for channel in self.CHANNELS for stat in self.STATISTICS
        ]

    @classmethod
    def _read_features(cls, path: Path) -> list[float]:
        frame = pd.read_csv(path)
        if list(frame.columns) != list(cls.CHANNELS):
            raise ValueError(
                f"{path} 列名错误；期望 {list(cls.CHANNELS)}，实际 {list(frame.columns)}"
            )
        if frame.empty:
            raise ValueError(f"{path} 没有振动采样点")
        values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path} 含缺失值或非有限数值")
        result: list[float] = []
        for signal in values.T:
            mean = float(np.mean(signal))
            centered = signal - mean
            variance = float(np.mean(centered**2))
            std = float(np.sqrt(variance))
            if std > 0:
                standardized = centered / std
                skewness = float(np.mean(standardized**3))
                kurtosis = float(np.mean(standardized**4) - 3.0)
            else:
                skewness = 0.0
                kurtosis = 0.0
            result.extend(
                [
                    mean,
                    std,
                    float(np.sqrt(np.mean(signal**2))),
                    float(np.ptp(signal)),
                    skewness,
                    kurtosis,
                ]
            )
        return result

    def _xy(self, start: int, stop: int) -> tuple[pd.DataFrame, np.ndarray]:
        rows = [self._read_features(path) for path in self.files[start:stop]]
        X = pd.DataFrame(rows, columns=self.feature_columns)
        y = np.arange(self.total_rows - start - 1, self.total_rows - stop - 1, -1, dtype=float)
        return X, y

    def initial_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        return self._xy(0, self.initial_size)

    def __iter__(self) -> Iterator[DataWindow]:
        pos = self.initial_size
        window_index = 0
        while pos + self.window_size <= self.total_rows:
            X, y = self._xy(pos, pos + self.window_size)
            sample_ids = np.asarray(
                [
                    f"{self.condition}/{self.bearing}/{index + 1}"
                    for index in range(pos, pos + self.window_size)
                ],
                dtype=object,
            )
            yield DataWindow(
                index=window_index,
                X=X,
                y=y,
                phase=f"xjtu:{self.condition}:{self.bearing}",
                sample_ids=sample_ids,
            )
            pos += self.window_size
            window_index += 1
