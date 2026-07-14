from __future__ import annotations

import numpy as np
import pytest

from mppi.pointworld_ext.geometry import PinholeIntrinsics
from mppi.pointworld_ext.input_config import PointWorldInputConfig, TrackingConfig, WorkspaceFilterConfig
import mppi.pointworld_ext.scene_flow_builder as scene_flow_builder_mod
from mppi.pointworld_ext.scene_flow_builder import OnlineSceneFlowBuilder
from mppi.pointworld_ext.tracker_interface import TrackWindowOutput
from mppi.pointworld_ext.window_buffer import CameraFrame, PointWorldWindowBuffer


class _FakeQueryManager:
    def __init__(self) -> None:
        self.advanced: list[str] = []

    def get_or_create(self, cam_name: str, **_kwargs):
        if str(cam_name) == "back":
            return np.asarray([[1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
        return np.asarray([[1.0, 2.0], [2.0, 2.0]], dtype=np.float32)

    def advance_window(self, cam_name: str, **_kwargs) -> None:
        self.advanced.append(str(cam_name))


class _BatchOnlyTracker:
    def __init__(self) -> None:
        self.request_keys: tuple[str, ...] = ()

    def track_window(self, _frames, _query_points):
        raise AssertionError("builder should submit cameras through track_windows()")

    def track_windows(self, requests):
        self.request_keys = tuple(req.key for req in requests)
        out = {}
        for req in requests:
            q = np.asarray(req.query_points, dtype=np.float32)
            tracks = np.repeat(q[None, :, :], 11, axis=0).astype(np.float32)
            visibility = np.ones((11, q.shape[0]), dtype=bool)
            confidence = np.ones((11, q.shape[0]), dtype=np.float32)
            out[req.key] = TrackWindowOutput(
                uv_tracks=tracks,
                visibility=visibility,
                confidence=confidence,
            )
        return out


def _make_window() -> PointWorldWindowBuffer:
    window = PointWorldWindowBuffer(window_size=11)
    intr = PinholeIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    extr = np.eye(4, dtype=np.float32)
    depth = np.ones((4, 4), dtype=np.float32)
    rgb_back = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_side = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_back[..., 0] = 128
    rgb_side[..., 1] = 128

    for t in range(11):
        window.push_frame(
            cameras={
                "back": CameraFrame(rgb=rgb_back, depth=depth, intrinsics=intr, extrinsics=extr),
                "side": CameraFrame(rgb=rgb_side, depth=depth, intrinsics=intr, extrinsics=extr),
            },
            joint_positions=np.zeros((7,), dtype=np.float32),
            gripper_positions=np.zeros((1,), dtype=np.float32),
            timestamp=float(t),
        )
    return window


def test_scene_flow_builder_submits_selected_cameras_as_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scene_flow_builder_mod,
        "compute_workspace_mask_2d",
        lambda *, height, width, **_kwargs: np.ones((int(height), int(width)), dtype=bool),
    )

    cfg = PointWorldInputConfig(
        window_size=11,
        tracking=TrackingConfig(max_query_points_per_camera=2, depth_min_m=0.05, depth_max_m=4.0),
        workspace_filter=WorkspaceFilterConfig(
            workspace_min=(-1.0, -1.0, 0.0),
            workspace_max=(4.0, 4.0, 2.0),
            strict_all_time_enabled=False,
        ),
        seed_robot_mask_enabled=False,
        camera_names=("back", "side"),
        camera_selection="subset",
        min_cameras=2,
    )
    tracker = _BatchOnlyTracker()
    query_manager = _FakeQueryManager()
    builder = OnlineSceneFlowBuilder(
        cfg=cfg,
        window_buffer=_make_window(),
        tracker=tracker,
        query_manager=query_manager,
    )

    out = builder.build(window_shift=1)

    assert tracker.request_keys == ("back", "side")
    assert query_manager.advanced == ["back", "side"]
    assert out.scene_flows.shape == (11, 4, 3)
    assert out.cameras_used == ("back", "side")
