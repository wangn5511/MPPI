from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from mppi.pointworld_ext.scene_flow_builder import SceneFlowBuildOutput


def _pad_or_trim_TN3(x: np.ndarray, *, T: int, N: int, dtype: np.dtype) -> np.ndarray:
    a = np.asarray(x, dtype=dtype)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"Expected (T,N,3), got {a.shape}")
    out = np.zeros((T, N, 3), dtype=dtype)
    t0 = min(T, a.shape[0])
    n0 = min(N, a.shape[1])
    out[:t0, :n0] = a[:t0, :n0]
    return out


def _pad_or_trim_TN(x: np.ndarray, *, T: int, N: int, dtype: np.dtype) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"Expected (T,N), got {a.shape}")
    out = np.zeros((T, N), dtype=dtype)
    t0 = min(T, a.shape[0])
    n0 = min(N, a.shape[1])
    out[:t0, :n0] = a[:t0, :n0].astype(dtype, copy=False)
    return out


def _pad_or_trim_TN3_u8(x: np.ndarray, *, T: int, N: int) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"Expected (T,N,3), got {a.shape}")
    out = np.zeros((T, N, 3), dtype=np.uint8)
    t0 = min(T, a.shape[0])
    n0 = min(N, a.shape[1])
    out[:t0, :n0] = np.asarray(a[:t0, :n0], dtype=np.uint8)
    return out


def build_pointworld_batch(
    *,
    scene: SceneFlowBuildOutput,
    robot_flows: Optional[np.ndarray],
    robot_positions: np.ndarray,
    gripper_positions: np.ndarray,
    max_scene_points: Optional[int] = None,
    max_robot_points: Optional[int] = None,
) -> Dict[str, Any]:
    scene_flows = np.asarray(scene.scene_flows, dtype=np.float32)
    scene_colors = np.asarray(scene.scene_colors, dtype=np.uint8)
    scene_exists = np.asarray(scene.scene_exists, dtype=bool)
    scene_conf = np.asarray(scene.scene_track_confidence, dtype=np.float32)

    if scene_flows.ndim != 3 or scene_flows.shape[2] != 3:
        raise ValueError(f"scene_flows must be (T,N,3), got {scene_flows.shape}")
    if scene_colors.shape != scene_flows.shape:
        raise ValueError(f"scene_colors shape {scene_colors.shape} must match scene_flows shape {scene_flows.shape}")
    if scene_exists.shape != scene_flows.shape[:2]:
        raise ValueError(f"scene_exists shape {scene_exists.shape} must match (T,N)={scene_flows.shape[:2]}")
    if scene_conf.shape != scene_flows.shape[:2]:
        raise ValueError(f"scene_track_confidence shape {scene_conf.shape} must match (T,N)={scene_flows.shape[:2]}")

    T, N_scene = int(scene_flows.shape[0]), int(scene_flows.shape[1])

    rp = np.asarray(robot_positions, dtype=np.float32)
    if rp.ndim != 2 or rp.shape[1] != 7:
        raise ValueError(f"robot_positions must be (T,7), got {rp.shape}")
    if rp.shape[0] != T:
        raise ValueError(f"robot_positions length mismatch: {rp.shape[0]} != T={T}")

    gp = np.asarray(gripper_positions, dtype=np.float32).reshape(-1)
    if gp.shape[0] != T:
        raise ValueError(f"gripper_positions length mismatch: {gp.shape[0]} != T={T}")

    if robot_flows is None:
        rf = np.zeros((T, 0, 3), dtype=np.float32)
    else:
        rf = np.asarray(robot_flows, dtype=np.float32)
        if rf.ndim != 3 or rf.shape[0] != T or rf.shape[2] != 3:
            raise ValueError(f"robot_flows must be (T,M,3) with T={T}, got {rf.shape}")

    if max_scene_points is not None:
        Ns = int(max_scene_points)
        if Ns < 0:
            raise ValueError("max_scene_points must be >= 0")
        scene_flows = _pad_or_trim_TN3(scene_flows, T=T, N=Ns, dtype=np.float32)
        scene_colors = _pad_or_trim_TN3_u8(scene_colors, T=T, N=Ns)
        scene_exists = _pad_or_trim_TN(scene_exists, T=T, N=Ns, dtype=bool)
        scene_conf = _pad_or_trim_TN(scene_conf, T=T, N=Ns, dtype=np.float32)
        N_scene = Ns

    if max_robot_points is not None:
        Mr = int(max_robot_points)
        if Mr < 0:
            raise ValueError("max_robot_points must be >= 0")
        rf = _pad_or_trim_TN3(rf, T=T, N=Mr, dtype=np.float32)

    batch: Dict[str, Any] = {
        "scene_flows": scene_flows,
        "scene_colors": scene_colors,
        "scene_exists": scene_exists,
        "robot_flows": rf,
        "robot_positions": rp,
        "gripper_positions": gp.astype(np.float32, copy=False),
        "scene_track_confidence": scene_conf,
        "cameras_used": np.asarray(scene.cameras_used, dtype=np.str_),
        "camera_track_slices": np.asarray(scene.camera_track_slices, dtype=np.int32),
        "camera_track_ids": np.asarray(scene.camera_track_ids, dtype=np.int32),
    }
    return batch