from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import spatial_predictor_selection as selection


def test_parallel_executor_uses_extended_idle_timeout(monkeypatch):
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_parallel(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(selection, "Parallel", fake_parallel)

    executor = selection._parallel_executor(n_jobs=-2)

    assert executor is sentinel
    assert captured == {
        "n_jobs": -2,
        "backend": "loky",
        "idle_worker_timeout": 1800,
    }
