from pathlib import Path

import numpy as np

from industrial_evolution.data.stream import CsvWindowStream


def test_csv_stream_preserves_rows_across_chunk_boundaries(tmp_path: Path):
    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "x,z,label\n"
        + "".join(
            f"{index},{index * 2},{'normal' if index % 2 else 'warning'}\n" for index in range(190)
        ),
        encoding="utf-8",
    )
    stream = CsvWindowStream(
        csv_path,
        "label",
        initial_size=100,
        window_size=30,
        task_type="classification",
        chunk_rows=37,
    )

    initial_X, initial_y = stream.initial_data()
    windows = list(stream)

    assert len(initial_X) == 100
    assert initial_y[0] == "warning"
    assert len(windows) == 3
    assert [len(window.y) for window in windows] == [30, 30, 30]
    assert windows[0].X.iloc[0]["x"] == 100
    assert windows[-1].X.iloc[-1]["x"] == 189
    np.testing.assert_array_equal(windows[0].sample_ids, np.arange(100, 130))
    assert stream.dropped_tail_rows == 0


def test_csv_stream_reports_and_drops_only_short_tail(tmp_path: Path):
    csv_path = tmp_path / "tail.csv"
    csv_path.write_text(
        "x,label\n" + "".join(f"{index},{index % 2}\n" for index in range(195)),
        encoding="utf-8",
    )
    stream = CsvWindowStream(
        csv_path,
        "label",
        initial_size=100,
        window_size=30,
        chunk_rows=41,
    )

    assert len(list(stream)) == 3
    assert stream.dropped_tail_rows == 5
