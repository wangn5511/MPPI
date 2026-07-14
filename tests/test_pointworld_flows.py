from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

from mppi.pointworld_ext.flows import (
    build_robot_inputs,
    build_scene_features,
    build_scene_features_torch,
    compute_flow_derivatives,
    normalize_dist2robot_mode,
    prepare_scene_inputs,
)


def test_compute_flow_derivatives_shapes() -> None:
    flows = np.arange(2 * 4 * 3 * 3, dtype=np.float32).reshape(2, 4, 3, 3)
    velocity, acceleration = compute_flow_derivatives(flows)
    assert velocity.shape == flows.shape
    assert acceleration.shape == flows.shape
    assert np.allclose(velocity[:, 0], flows[:, 1] - flows[:, 0])


def test_prepare_scene_and_robot_feature_shapes() -> None:
    scene = prepare_scene_inputs(
        scene_flows=np.zeros((4, 5, 3), dtype=np.float32),
        scene_colors=np.zeros((4, 5, 3), dtype=np.uint8),
        scene_exists=np.ones((4, 5), dtype=bool),
        scene_track_confidence=None,
        batch_size=3,
        max_scene_points=6,
    )
    assert scene["scene_flows"].shape == (3, 4, 6, 3)
    assert scene["scene_exists"].shape == (3, 4, 6)

    robot = build_robot_inputs(
        robot_flows=np.zeros((3, 4, 2, 3), dtype=np.float32),
        robot_colors=np.zeros((3, 4, 2, 3), dtype=np.float32),
        robot_normals=np.zeros((3, 4, 2, 3), dtype=np.float32),
        gripper_positions=np.zeros((3, 4), dtype=np.float32),
        max_robot_points=4,
    )
    assert robot["robot_features"].shape == (3, 4, 4, 16)
    assert robot["robot_exists"].shape == (3, 4, 4)

    robot_flows = np.ones((3, 4, 2, 3), dtype=np.float32)
    scene_features = build_scene_features(
        scene_flows=scene["scene_flows"],
        scene_colors=scene["scene_colors"],
        gripper_positions=np.zeros((3, 4), dtype=np.float32),
        robot_flows=robot_flows,
    )
    assert scene_features.shape == (3, 1, 6, 17)


def test_build_scene_features_dist2robot_fast_modes() -> None:
    rng = np.random.default_rng(7)
    scene_flows = rng.normal(size=(2, 5, 4, 3)).astype(np.float32)
    scene_colors = rng.normal(size=(2, 5, 4, 3)).astype(np.float32)
    gripper_positions = rng.normal(size=(2, 5)).astype(np.float32)
    robot_flows = rng.normal(size=(2, 5, 3, 3)).astype(np.float32)

    full = build_scene_features(
        scene_flows=scene_flows,
        scene_colors=scene_colors,
        gripper_positions=gripper_positions,
        robot_flows=robot_flows,
        dist2robot_mode="full",
    )
    t0_repeat = build_scene_features(
        scene_flows=scene_flows,
        scene_colors=scene_colors,
        gripper_positions=gripper_positions,
        robot_flows=robot_flows,
        dist2robot_mode="t0_repeat",
    )
    none = build_scene_features(
        scene_flows=scene_flows,
        scene_colors=scene_colors,
        gripper_positions=gripper_positions,
        robot_flows=robot_flows,
        dist2robot_mode="none",
    )

    T = scene_flows.shape[1]
    full_dist = full[..., -T:]
    t0_dist = t0_repeat[..., -T:]
    none_dist = none[..., -T:]

    assert np.allclose(t0_dist[..., 0], full_dist[..., 0])
    assert np.allclose(t0_dist, np.repeat(full_dist[..., :1], T, axis=-1))
    assert np.allclose(none_dist, 0.0)
    assert normalize_dist2robot_mode("zero") == "none"


def test_build_scene_features_torch_matches_numpy() -> None:
    if torch is None:
        return

    rng = np.random.default_rng(5)
    scene_flows = rng.normal(size=(3, 4, 6, 3)).astype(np.float32)
    scene_colors = rng.normal(size=(3, 4, 6, 3)).astype(np.float32)
    gripper_positions = rng.normal(size=(3, 4)).astype(np.float32)
    robot_flows = rng.normal(size=(3, 4, 5, 3)).astype(np.float32)

    expected = None
    for mode in ("full", "t0_repeat", "none"):
        expected = build_scene_features(
            scene_flows=scene_flows,
            scene_colors=scene_colors,
            gripper_positions=gripper_positions,
            robot_flows=robot_flows,
            dist2robot_mode=mode,
        )
        actual = build_scene_features_torch(
            scene_flows=torch.as_tensor(scene_flows),
            scene_colors=torch.as_tensor(scene_colors),
            gripper_positions=torch.as_tensor(gripper_positions),
            robot_flows=torch.as_tensor(robot_flows),
            dist2robot_mode=mode,
        )
        assert np.allclose(actual.cpu().numpy(), expected, atol=1e-6)

    assert expected is not None
    expected = build_scene_features(
        scene_flows=scene_flows,
        scene_colors=scene_colors,
        gripper_positions=gripper_positions,
        robot_flows=robot_flows,
        dist2robot_mode="full",
    )

    dist2robot = expected[..., -scene_flows.shape[1]:]
    assert not np.allclose(dist2robot, 0.0)
    assert float(dist2robot.max() - dist2robot.min()) > 1e-4
