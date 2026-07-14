from __future__ import annotations

import threading
import time

import numpy as np
import pytest

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

import mppi.pointworld_ext.tracker_interface as tracker_interface_mod
from mppi.pointworld_ext.tracker_interface import (
    CoTrackerOnlinePointTracker,
    MultiDeviceOnlinePointTracker,
    TrackWindowOutput,
    TrackWindowRequest,
    expand_device_list,
)


class _FakeOnlinePredictor:
    interp_shape = (4, 4)

    def __init__(self) -> None:
        self.queries = None
        self.model_iters = None
        self.model = self._model

    def __call__(self, video, *, is_first_step: bool, queries=None):
        if is_first_step:
            self.queries = queries
            return None
        raise AssertionError("expected model fast path")

    def _model(self, *, video, queries, iters: int, is_online: bool):
        self.model_iters = int(iters)
        b, t, _c, _h, _w = video.shape
        n = int(queries.shape[1])
        tracks = torch.zeros((b, t, n, 2), device=video.device, dtype=video.dtype)
        visibility = torch.ones((b, t, n), device=video.device, dtype=video.dtype)
        confidence = torch.ones((b, t, n), device=video.device, dtype=video.dtype)
        return tracks, visibility, confidence


def test_track_window_uses_configured_cotracker_iters() -> None:
    if torch is None or not hasattr(torch, "from_numpy"):
        pytest.skip("torch is not installed")

    tracker = CoTrackerOnlinePointTracker.__new__(CoTrackerOnlinePointTracker)
    tracker._torch = torch
    tracker._predictor = _FakeOnlinePredictor()
    tracker._device = "cpu"
    tracker._iters = 3
    tracker._vis_thr = 0.6

    frames = np.zeros((11, 4, 4, 3), dtype=np.uint8)
    query_points = np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)

    tracker.track_window(frames, query_points)

    assert tracker._predictor.model_iters == 3


class _FakeTracker:
    def __init__(self, *, device: str) -> None:
        self.device = str(device)
        self.calls = 0

    def track_window(self, frames, query_points):
        self.calls += 1
        fr = np.asarray(frames)
        q = np.asarray(query_points, dtype=np.float32)
        tracks = np.zeros((fr.shape[0], q.shape[0], 2), dtype=np.float32)
        visibility = np.ones((fr.shape[0], q.shape[0]), dtype=bool)
        confidence = np.ones((fr.shape[0], q.shape[0]), dtype=np.float32)
        return TrackWindowOutput(uv_tracks=tracks, visibility=visibility, confidence=confidence)


def test_expand_device_list_accepts_comma_separated_devices() -> None:
    assert expand_device_list("cuda:0, cuda:1", fallback="cuda") == ("cuda:0", "cuda:1")
    assert expand_device_list("", fallback="cuda") == ("cuda",)
    assert expand_device_list(None, fallback="cpu") == ("cpu",)


def test_cotracker_factory_preserves_none_device_for_auto_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeCoTracker:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(tracker_interface_mod, "CoTrackerOnlinePointTracker", _FakeCoTracker)

    tracker_interface_mod.build_cotracker_online_point_tracker(checkpoint="fake.pt", device=None)

    assert captured["device"] is None


def test_multi_device_tracker_assigns_camera_keys_stably() -> None:
    made: list[_FakeTracker] = []

    def factory(device: str):
        tracker = _FakeTracker(device=device)
        made.append(tracker)
        return tracker

    tracker = MultiDeviceOnlinePointTracker(
        devices=("cuda:0", "cuda:1"),
        tracker_factory=factory,
    )

    frames = np.zeros((11, 4, 4, 3), dtype=np.uint8)
    q = np.asarray([[1.0, 1.0]], dtype=np.float32)
    requests = (
        TrackWindowRequest(key="back", frames=frames, query_points=q),
        TrackWindowRequest(key="side", frames=frames, query_points=q),
    )

    out = tracker.track_windows(requests)

    assert set(out.keys()) == {"back", "side"}
    assert tracker.device_assignments == {"back": "cuda:0", "side": "cuda:1"}
    assert [t.device for t in made] == ["cuda:0", "cuda:1"]

    tracker.track_windows((TrackWindowRequest(key="back", frames=frames, query_points=q),))

    assert tracker.device_assignments["back"] == "cuda:0"
    assert made[0].calls == 2
    assert made[1].calls == 1


class _SlowStatefulTracker(_FakeTracker):
    def __init__(self, *, device: str) -> None:
        super().__init__(device=device)
        self.active = 0
        self.max_active = 0
        self.active_lock = threading.Lock()

    def track_window(self, frames, query_points):
        with self.active_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return super().track_window(frames, query_points)
        finally:
            with self.active_lock:
                self.active -= 1


def test_multi_device_tracker_serializes_calls_to_same_underlying_tracker() -> None:
    made: list[_SlowStatefulTracker] = []

    def factory(device: str):
        tracker = _SlowStatefulTracker(device=device)
        made.append(tracker)
        return tracker

    tracker = MultiDeviceOnlinePointTracker(devices=("cuda:0",), tracker_factory=factory)
    frames = np.zeros((11, 4, 4, 3), dtype=np.uint8)
    q = np.asarray([[1.0, 1.0]], dtype=np.float32)
    start = threading.Barrier(3)

    def worker(key: str) -> None:
        start.wait(timeout=2.0)
        tracker.track_windows((TrackWindowRequest(key=key, frames=frames, query_points=q),))

    t0 = threading.Thread(target=worker, args=("back",))
    t1 = threading.Thread(target=worker, args=("side",))
    t0.start()
    t1.start()
    start.wait(timeout=2.0)
    t0.join(timeout=2.0)
    t1.join(timeout=2.0)

    assert not t0.is_alive()
    assert not t1.is_alive()
    assert made[0].max_active == 1
