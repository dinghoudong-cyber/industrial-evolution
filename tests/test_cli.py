from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "industrial_evolution.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env=env,
    )


@pytest.mark.parametrize(
    ("filename", "content", "target", "initial_size"),
    [
        ("missing-target.csv", "a,b\n1,2\n3,4\n5,6\n", "label", 2),
        (
            "nonnumeric.csv",
            "sensor,status,label\n1,normal,0\n2,warning,1\n3,normal,0\n",
            "label",
            2,
        ),
        ("too-small.csv", "sensor,label\n1,0\n2,1\n", "label", 100),
    ],
)
def test_known_csv_input_errors_are_concise(
    tmp_path: Path,
    filename: str,
    content: str,
    target: str,
    initial_size: int,
) -> None:
    csv_path = tmp_path / filename
    csv_path.write_text(content, encoding="utf-8")
    output = tmp_path / "output"

    completed = run_cli(
        "run-csv",
        "--csv",
        str(csv_path),
        "--target",
        target,
        "--initial-size",
        str(initial_size),
        "--output",
        str(output),
    )

    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "error:" in completed.stderr
    assert not output.exists()


def test_missing_csv_is_reported_without_traceback(tmp_path: Path) -> None:
    output = tmp_path / "output"
    completed = run_cli(
        "run-csv",
        "--csv",
        str(tmp_path / "missing.csv"),
        "--target",
        "label",
        "--initial-size",
        "100",
        "--output",
        str(output),
    )

    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "error:" in completed.stderr
    assert not output.exists()


def test_bad_yaml_and_zero_window_are_input_errors(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("seed: [broken", encoding="utf-8")
    bad = run_cli("demo", "--config", str(bad_yaml))
    assert bad.returncode == 2
    assert "Traceback" not in bad.stderr

    csv_path = tmp_path / "short.csv"
    csv_path.write_text(
        "x,label\n" + "".join(f"{index},{index % 2}\n" for index in range(105)),
        encoding="utf-8",
    )
    output = tmp_path / "zero-window-output"
    zero = run_cli(
        "run-csv",
        "--csv",
        str(csv_path),
        "--target",
        "label",
        "--initial-size",
        "100",
        "--window-size",
        "30",
        "--output",
        str(output),
    )
    assert zero.returncode == 2
    assert "完整流式窗口" in zero.stderr
    assert not output.exists()


def test_string_classification_labels_are_supported(tmp_path: Path) -> None:
    csv_path = tmp_path / "string-label.csv"
    csv_path.write_text(
        "x,label\n"
        + "".join(f"{index},{'normal' if index % 2 else 'warning'}\n" for index in range(160)),
        encoding="utf-8",
    )
    completed = run_cli(
        "run-csv",
        "--csv",
        str(csv_path),
        "--target",
        "label",
        "--initial-size",
        "100",
        "--window-size",
        "30",
        "--output",
        str(tmp_path / "string-output"),
    )
    assert completed.returncode == 0
    assert '"run_status": "complete"' in completed.stdout
