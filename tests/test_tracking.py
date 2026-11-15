import csv

from industrial_evolution.tracking.local_tracker import LocalTracker
from industrial_evolution.tracking.swanlab_tracker import CompositeTracker


class FailingMirror:
    def log_event(self, event_type, payload):
        raise RuntimeError("mirror event failed")

    def log_metrics(self, row):
        raise RuntimeError("mirror metrics failed")

    def close(self, summary):
        raise RuntimeError("mirror close failed")


def test_local_tracker_appends_metric_rows(tmp_path):
    tracker = LocalTracker(tmp_path, run_id="append-test")
    tracker.log_metrics({"window": 0, "score": 0.5})
    tracker.log_metrics({"window": 1, "score": 0.6})

    with tracker.metrics_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"window": "0", "score": "0.5"},
        {"window": "1", "score": "0.6"},
    ]


def test_optional_mirror_failure_does_not_interrupt_primary_tracker(tmp_path):
    primary = LocalTracker(tmp_path, run_id="mirror-test")
    tracker = CompositeTracker(primary, [FailingMirror()])

    tracker.log_event("started", {"ok": True})
    tracker.log_metrics({"window": 0, "score": 0.5})
    tracker.close({"status": "complete"})

    events = primary.events_path.read_text(encoding="utf-8")
    assert '"event_type": "started"' in events
    assert '"event_type": "tracker_mirror_failure"' in events
    assert (primary.run_dir / "summary.json").exists()
