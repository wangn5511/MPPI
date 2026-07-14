from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import time
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from mppi.mpc.solver import JointMPPIConfig, JointMPPISolver, PointWorldCostFn, _first_device
from mppi.curobo_ext.check_depth_pcl import (
    load_T_row_major_4x4_yaml,
    load_intrinsics_from_cam_info_yaml,
    parse_obs_camera_params,
    rgbd_to_pointcloud_base,
)
from mppi.curobo_ext.collision_checker import CuRoboCollisionConfig, fk_T_base_link, get_curobo_collision_checker
from mppi.curobo_ext.scene_builder import SceneBuildConfig, build_scene_points_base_and_colors_from_pcd_back_cam
from mppi.costs.pointworld_cost import PointWorldCostConfig
from mppi.protocol.msgpack_codec import decode_message, encode_message
from mppi.protocol.types_pcl import (
    ActionChunkPCL,
    ErrorPCL,
    InferResponsePCL,
    ObsPCL,
    SCHEMA_VERSION_PCL,
    ServerTimingPCL,
)
from mppi.pointworld_ext.geometry import PinholeIntrinsics as PWPinholeIntrinsics
from mppi.pointworld_ext.input_config import (
    PointWorldInputConfig,
    RobotFilterConfig,
    TrackingConfig,
    WorkspaceFilterConfig,
    parse_spheres_spec,
)
from mppi.pointworld_ext.query_manager import QueryPointManager, QueryPointManagerConfig
from mppi.pointworld_ext.scene_flow_builder import OnlineSceneFlowBuilder
from mppi.pointworld_ext.pointworld_batch_adapter import build_pointworld_batch
from mppi.pointworld_ext.tracker_interface import build_cotracker_online_point_tracker
from mppi.pointworld_ext.wrapper import PointWorldCostModel, PointWorldModelConfig
from mppi.utils.paths import default_urdf_path, repo_path
from mppi.pointworld_ext.window_buffer import CameraFrame as PWCameraFrame
from mppi.pointworld_ext.window_buffer import PointWorldWindowBuffer


def _require_websockets():
    try:
        import websockets  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: websockets. Install it in the container env.") from e
    return websockets


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 9011
    open_loop_horizon: int = 8
    policy: str = "dummy_hold"
    request_timeout_s: float = 2.0
    cam_id: str = "back"


REQUIRED_CAM_IDS: Tuple[str, ...] = ("back", "side")


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def _env_f(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_i(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _env_vec3(name: str, default: str) -> Tuple[float, float, float]:
    s = os.getenv(name, default).strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must be 'x,y,z'")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


_STARTUP_NOTICES: set[str] = set()


def _print_once(key: str, msg: str) -> None:
    if str(key) in _STARTUP_NOTICES:
        return
    _STARTUP_NOTICES.add(str(key))
    print(msg, flush=True)


@lru_cache(maxsize=1)
def _cuda_runtime_available() -> bool:
    if _env_bool("MPPI_FORCE_NO_CUDA", "0"):
        return False

    visible = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if visible is not None and visible.strip().lower() in ("", "-1", "none", "null", "void"):
        return False

    try:
        import torch  # type: ignore

        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return False
        is_available = getattr(cuda, "is_available", None)
        if callable(is_available) and not bool(is_available()):
            return False
        device_count = getattr(cuda, "device_count", None)
        if callable(device_count):
            return int(device_count()) > 0
        return bool(callable(is_available))
    except Exception:
        return False


def _allow_cpu_curobo() -> bool:
    return _env_bool("MPPI_ALLOW_CPU_CUROBO", "0")


def _allow_cpu_pointworld() -> bool:
    return _env_bool("MPPI_ALLOW_CPU_POINTWORLD", "0")


def _effective_use_curobo_collision() -> bool:
    enabled = _env_bool("MPPI_USE_CUROBO_COLLISION", "0")
    if not enabled:
        return False
    if _cuda_runtime_available() or _allow_cpu_curobo():
        return True
    _print_once(
        "disable_curobo_no_cuda",
        "[pcl_server] CUDA is not available; disabling MPPI_USE_CUROBO_COLLISION for CPU-only inference.",
    )
    return False


def _effective_pointworld_enabled() -> bool:
    enabled = _env_bool("MPPI_PW_ENABLE", "0")
    if not enabled:
        return False
    if _cuda_runtime_available() or _allow_cpu_pointworld():
        return True
    _print_once(
        "disable_pointworld_no_cuda",
        "[pcl_server] CUDA is not available; disabling MPPI_PW_ENABLE for CPU-only inference.",
    )
    return False


def _effective_use_pointworld_cost() -> bool:
    enabled = _env_bool("MPPI_USE_POINTWORLD_COST", "0")
    if not enabled:
        return False
    if _cuda_runtime_available() or _allow_cpu_pointworld():
        return True
    _print_once(
        "disable_pointworld_cost_no_cuda",
        "[pcl_server] CUDA is not available; disabling MPPI_USE_POINTWORLD_COST for CPU-only inference.",
    )
    return False


def _default_num_samples() -> str:
    if _cuda_runtime_available():
        return "256"
    return os.getenv("MPPI_CPU_NUM_SAMPLES", "32")


def _effective_scene_from_pcd_enabled(policy: str) -> bool:
    if str(policy) != "mppi_joint":
        return False
    return (
        _effective_use_curobo_collision()
        and _env_bool("MPPI_SCENE_FROM_PCD_BACK_CAM", "0")
        and float(os.getenv("MPPI_W_SCENE_COLLISION", "1.0")) > 0.0
    )


def _request_needs_pcd(policy: str) -> bool:
    return _env_bool("MPPI_PCL_SAVE_PCD", "0") or _effective_scene_from_pcd_enabled(policy)


def _request_needs_camera_decode(policy: str, pw: Optional[object]) -> bool:
    pw_enabled = bool(pw is not None and bool(getattr(pw, "enabled", False)))
    return pw_enabled or _request_needs_pcd(policy)


def _tracker_device_summary(tracker: object) -> str:
    devices = getattr(tracker, "devices", None)
    if devices:
        return ",".join(str(d) for d in tuple(devices))
    device = getattr(tracker, "_device", "")
    return str(device) if str(device) else "unknown"


_JOINT_SOLVERS: dict[int, JointMPPISolver] = {}


def _get_joint_solver(open_loop_horizon: int) -> JointMPPISolver:
    h = int(open_loop_horizon)
    if h in _JOINT_SOLVERS:
        return _JOINT_SOLVERS[h]

    urdf_path = os.getenv("MPPI_URDF_PATH", "").strip() or default_urdf_path()

    cfg = JointMPPIConfig(
        horizon=h,
        num_samples=int(os.getenv("MPPI_NUM_SAMPLES", _default_num_samples())),
        infer_budget_ms=float(os.getenv("MPPI_INFER_BUDGET_MS", "0.0")),
        budget_max_dynamic_cuboids=int(os.getenv("MPPI_BUDGET_MAX_DYNAMIC_CUBOIDS", "0")),
        debug_cost_stats=_env_bool("MPPI_DEBUG_COST_STATS", "0"),
        debug_cost_stats_q=float(os.getenv("MPPI_DEBUG_COST_STATS_Q", "0.5")),
        temperature=float(os.getenv("MPPI_TEMPERATURE", "1.0")),
        noise_std=float(os.getenv("MPPI_NOISE_STD", "0.05")),
        dt=float(os.getenv("MPPI_DT", "1.0")),
        w_smooth=float(os.getenv("MPPI_W_SMOOTH", "1.0")),
        w_action=float(os.getenv("MPPI_W_ACTION", "0.01")),
        w_joint_limit=float(os.getenv("MPPI_W_JOINT_LIMIT", "50.0")),
        use_curobo_collision=_effective_use_curobo_collision(),
        w_scene_collision=float(os.getenv("MPPI_W_SCENE_COLLISION", "1.0")),
        w_self_collision=float(os.getenv("MPPI_W_SELF_COLLISION", os.getenv("MPPI_W_SCENE_COLLISION", "1.0"))),
        curobo_device=str(os.getenv("MPPI_CUROBO_DEVICE", "cuda:0")),
        curobo_robot_yaml=str(os.getenv("MPPI_CUROBO_ROBOT_YAML", "franka.yml")),
        curobo_tool_frame=str(os.getenv("MPPI_CUROBO_TOOL_FRAME", "robotiq_85_base_link")),
        curobo_with_world=_env_bool("MPPI_CUROBO_WITH_WORLD", "1"),
        curobo_collision_activation_distance=float(os.getenv("MPPI_CUROBO_COLLISION_ACTIVATION_DISTANCE", "0.2")),
        curobo_self_collision_activation_distance=float(os.getenv("MPPI_CUROBO_SELF_COLLISION_ACTIVATION_DISTANCE", "0.0")),
        curobo_max_collision_distance=float(os.getenv("MPPI_CUROBO_MAX_COLLISION_DISTANCE", "1.0")),
        scene_from_pcd_back_cam=_env_bool("MPPI_SCENE_FROM_PCD_BACK_CAM", "0"),
        scene_pcd_scale=float(os.getenv("MPPI_SCENE_PCD_SCALE", "1.0")),
        scene_pcd_in_base=_env_bool("MPPI_SCENE_PCD_IN_BASE", "1"),
        scene_add_table=_env_bool("MPPI_SCENE_ADD_TABLE", "1"),
        scene_table_dims=_env_vec3("MPPI_SCENE_TABLE_DIMS", "2.0,2.0,0.2"),
        scene_table_center=_env_vec3("MPPI_SCENE_TABLE_CENTER", "0.4,0.0,-0.1"),
        scene_remove_table_points=_env_bool("MPPI_SCENE_REMOVE_TABLE_POINTS", "1"),
        scene_table_eps_m=float(os.getenv("MPPI_SCENE_TABLE_EPS_M", "0.01")),
        scene_remove_wall_points=_env_bool("MPPI_SCENE_REMOVE_WALL_POINTS", "0"),
        scene_wall_dims=_env_vec3("MPPI_SCENE_WALL_DIMS", "0.0,0.0,0.0"),
        scene_wall_center=_env_vec3("MPPI_SCENE_WALL_CENTER", "0.0,0.0,0.0"),
        scene_wall_margin_m=float(os.getenv("MPPI_SCENE_WALL_MARGIN_M", "0.02")),
        t_base_cam_back_path=str(os.getenv("MPPI_T_BASE_CAM_BACK_PATH", repo_path("configs", "T_base_cam.yaml"))),
        scene_roi_min=_env_vec3("MPPI_SCENE_ROI_MIN", "-0.1,-0.7,-0.05"),
        scene_roi_max=_env_vec3("MPPI_SCENE_ROI_MAX", "1.2,0.7,1.2"),
        scene_voxel_size_m=float(os.getenv("MPPI_SCENE_VOXEL_SIZE_M", "0.01")),
        scene_padding_m=float(os.getenv("MPPI_SCENE_PADDING_M", "0.02")),
        scene_max_cuboids=int(os.getenv("MPPI_SCENE_MAX_CUBOIDS", "20")),
        scene_robot_mask_margin_m=float(os.getenv("MPPI_SCENE_ROBOT_MASK_MARGIN_M", "0.02")),
        scene_min_cluster_voxels=int(os.getenv("MPPI_SCENE_MIN_CLUSTER_VOXELS", "10")),
        scene_track_alpha=float(os.getenv("MPPI_SCENE_TRACK_ALPHA", "0.6")),
        scene_track_remove_after_misses=int(os.getenv("MPPI_SCENE_TRACK_REMOVE_AFTER_MISSES", "5")),
        scene_track_max_tracks=int(os.getenv("MPPI_SCENE_TRACK_MAX_TRACKS", "20")),
        scene_track_match_center_dist_m=float(os.getenv("MPPI_SCENE_TRACK_MATCH_CENTER_DIST_M", "0.10")),
        scene_track_match_iou_min=float(os.getenv("MPPI_SCENE_TRACK_MATCH_IOU_MIN", "0.05")),
        min_effective_samples_ratio=float(os.getenv("MPPI_MIN_EFFECTIVE_SAMPLES_RATIO", "0.01")),
        use_pointworld_cost=_effective_use_pointworld_cost(),
        w_pointworld=float(os.getenv("MPPI_W_POINTWORLD", "0.0")),
        pointworld_cost_timeout_ms=float(os.getenv("MPPI_PW_COST_TIMEOUT_MS", "0.0")),
        urdf_path=str(urdf_path),
    )

    solver = JointMPPISolver(cfg)
    _JOINT_SOLVERS[h] = solver
    return solver


DEFAULT_PW_AABB_CONFIG_PATH = str(repo_path("configs/pointworld_static_aabbs.json"))
DEFAULT_PW_TASK_ABLATION = "obs_infl"
DEFAULT_PW_TASK_W_OBS = 1.0
DEFAULT_PW_TASK_W_INFL = 0.5
DEFAULT_PW_TASK_MIN_POINTS = 1
DEFAULT_PW_SEED_ROBOT_MASK_ENABLED = True
DEFAULT_PW_GRIPPER_POSE_ENABLED = True
DEFAULT_PW_TASK_USE_VISIBILITY = True
DEFAULT_PW_TASK_USE_DEPTH_VALID = True


def _load_aabb_config_json(path: str) -> dict[str, Any]:
    p = str(Path(path))
    obj = json.load(open(p, "r", encoding="utf-8"))
    obstacles = obj.get("obstacles", obj.get("aabbs", []))
    if not isinstance(obstacles, list):
        raise ValueError("AABB config must contain a list 'obstacles'")

    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    infls: list[float] = []

    for it in obstacles:
        if not isinstance(it, dict):
            continue
        if it.get("enabled", True) is False:
            continue

        aabb_min = it.get("aabb_min")
        aabb_max = it.get("aabb_max")
        if aabb_min is None or aabb_max is None:
            c = it.get("center")
            d = it.get("dims")
            if c is None or d is None:
                continue
            c = np.asarray(c, dtype=np.float32).reshape(3)
            d = np.asarray(d, dtype=np.float32).reshape(3)
            aabb_min = (c - 0.5 * d).tolist()
            aabb_max = (c + 0.5 * d).tolist()

        mn = np.asarray(aabb_min, dtype=np.float32).reshape(3)
        mx = np.asarray(aabb_max, dtype=np.float32).reshape(3)
        mins.append(mn)
        maxs.append(mx)
        infls.append(float(it.get("inflation_m", obj.get("inflation_default_m", 0.0))))

    if not mins:
        raise ValueError(f"No enabled obstacles found in {p}")

    mn_all = np.stack(mins, axis=0)
    mx_all = np.stack(maxs, axis=0)
    infl = np.asarray(infls, dtype=np.float32).reshape(-1)

    mn_infl = mn_all - infl[:, None]
    mx_infl = mx_all + infl[:, None]

    return {
        "aabb_min": mn_all,
        "aabb_max": mx_all,
        "aabb_min_infl": mn_infl,
        "aabb_max_infl": mx_infl,
    }


def _decode_rgb(*, codec: Optional[str], data: Any) -> np.ndarray:
    if codec is None:
        raise ValueError("rgb_codec is required for rgb_bytes")
    c = str(codec).strip().lower()

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"rgb_bytes must be bytes-like, got {type(data)}")
    b = bytes(data)

    if c != "jpeg":
        raise ValueError(f"Unsupported rgb_codec: {c}")

    try:
        import cv2  # type: ignore

        buf = np.frombuffer(b, dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode returned None")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return np.asarray(img_rgb)
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(b)).convert("RGB")
        return np.asarray(img)
    except Exception as e:
        raise RuntimeError(f"Failed to decode JPEG RGB. Install opencv-python or pillow. Error: {e}") from e


def _decode_depth(*, codec: Optional[str], data: Any) -> np.ndarray:
    if codec is None:
        raise ValueError("depth_codec is required for depth_bytes")
    c = str(codec).strip().lower()

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"depth_bytes must be bytes-like, got {type(data)}")
    b = bytes(data)

    if c == "npy_zlib":
        raw = zlib.decompress(b)
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        d = np.asarray(arr)
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        if d.ndim != 2:
            raise ValueError(f"Expected depth shape (H,W), got {d.shape}")
        return np.asarray(d, dtype=np.float32)

    raise ValueError(f"Unsupported depth_codec: {c}")


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: cv2 (opencv-python) is required for contract resize") from e
    return cv2


def _resize_rgbd_to_contract(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    intr: Any,
    target_hw: Tuple[int, int] = (180, 320),
) -> Tuple[np.ndarray, np.ndarray, Any]:
    d = np.asarray(depth)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"Expected depth shape (H,W), got {d.shape}")

    rgb_arr = np.asarray(rgb)
    if rgb_arr.ndim != 3 or rgb_arr.shape[-1] != 3:
        raise ValueError(f"Expected rgb shape (H,W,3), got {rgb_arr.shape}")

    src_h, src_w = int(d.shape[0]), int(d.shape[1])
    dst_h, dst_w = int(target_hw[0]), int(target_hw[1])

    if (src_h, src_w) == (dst_h, dst_w):
        return rgb_arr, np.asarray(d, dtype=np.float32), intr

    cv2 = _require_cv2()

    rgb_u8 = rgb_arr
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = np.clip(rgb_u8, 0, 255).astype(np.uint8)

    depth_f32 = np.asarray(d, dtype=np.float32)

    rgb_rs = cv2.resize(rgb_u8, (int(dst_w), int(dst_h)), interpolation=cv2.INTER_AREA)
    depth_rs = cv2.resize(depth_f32, (int(dst_w), int(dst_h)), interpolation=cv2.INTER_NEAREST)

    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)

    intr_out = type(intr)(
        fx=float(getattr(intr, "fx")) * sx,
        fy=float(getattr(intr, "fy")) * sy,
        cx=float(getattr(intr, "cx")) * sx,
        cy=float(getattr(intr, "cy")) * sy,
    )

    return np.asarray(rgb_rs), np.asarray(depth_rs, dtype=np.float32), intr_out


def _sphere_surface_points(center: np.ndarray, radius: float, n: int) -> np.ndarray:
    nn = int(n)
    if nn <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    i = (np.arange(nn, dtype=np.float32) + 0.5).astype(np.float32)
    phi = np.arccos(1.0 - 2.0 * i / float(nn)).astype(np.float32)
    theta = (np.pi * (1.0 + 5.0**0.5) * i).astype(np.float32)

    x = (np.cos(theta) * np.sin(phi)).astype(np.float32)
    y = (np.sin(theta) * np.sin(phi)).astype(np.float32)
    z = (np.cos(phi)).astype(np.float32)

    p = np.stack([x, y, z], axis=1).astype(np.float32)
    return p * float(radius) + np.asarray(center, dtype=np.float32).reshape(1, 3)


def _robot_mask_vis_points(robot_spheres_base: np.ndarray) -> np.ndarray:
    spheres = np.asarray(robot_spheres_base, dtype=np.float32)
    if spheres.ndim != 2 or spheres.shape[1] != 4 or spheres.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    per = _env_i("MPPI_PCL_ROBOT_MASK_VIS_POINTS_PER_SPHERE", "250")
    max_total = _env_i("MPPI_PCL_ROBOT_MASK_VIS_MAX_POINTS", "60000")

    pts_all: list[np.ndarray] = []
    total = 0
    for s in spheres:
        if total >= int(max_total):
            break
        c = s[:3]
        r = float(s[3])
        if not np.all(np.isfinite(c)) or (not np.isfinite(r)) or r <= 0.0:
            continue
        n = int(per)
        if total + n > int(max_total):
            n = max(0, int(max_total) - total)
        if n <= 0:
            break
        pts_all.append(_sphere_surface_points(c, r, n))
        total += n

    if not pts_all:
        return np.zeros((0, 3), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(pts_all, axis=0).astype(np.float32))


def _cuboid_wire_points(center_xyz: tuple[float, float, float], dims_xyz: tuple[float, float, float]) -> np.ndarray:
    cx, cy, cz = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
    dx, dy, dz = float(dims_xyz[0]), float(dims_xyz[1]), float(dims_xyz[2])
    if not np.isfinite([cx, cy, cz, dx, dy, dz]).all() or dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        return np.zeros((0, 3), dtype=np.float32)

    hx, hy, hz = 0.5 * dx, 0.5 * dy, 0.5 * dz
    corners = np.asarray(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    n_per = max(2, int(_env_i("MPPI_PCL_CUBOID_WIRE_POINTS_PER_EDGE", "40")))
    pts: list[np.ndarray] = []
    for a, b in edges:
        p0 = corners[a]
        p1 = corners[b]
        t = np.linspace(0.0, 1.0, num=n_per, dtype=np.float32)[:, None]
        pts.append((1.0 - t) * p0[None, :] + t * p1[None, :])

    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(pts, axis=0).astype(np.float32))


def _resolve_save_path(base: str, *, step_id: int) -> str:
    s = str(base).strip()
    if not s:
        s = repo_path("data", "test", "pcl_scene_filtered.npz")
    if s.lower().endswith(".npz"):
        return s
    out_dir = Path(s)
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{int(step_id):06d}.npz")


def _maybe_dump_pointworld_acceptance(
    *,
    pointworld_obs: Optional[Dict[str, Any]],
    step_id: int,
    pw_cost_debug: Optional[Dict[str, Any]] = None,
    timing_breakdown: Optional[Dict[str, Any]] = None,
) -> None:
    out_dir = os.getenv("MPPI_PW_ACCEPTANCE_DUMP_DIR", "").strip()
    if not out_dir or not isinstance(pointworld_obs, dict):
        return

    def _shape(name: str) -> Optional[list[int]]:
        if name not in pointworld_obs:
            return None
        try:
            return [int(x) for x in np.asarray(pointworld_obs[name]).shape]
        except Exception:
            return None

    summary: Dict[str, Any] = {
        "step_id": int(step_id),
        "has_scene_flows": "scene_flows" in pointworld_obs,
        "has_scene_visibility": "scene_visibility" in pointworld_obs,
        "has_scene_depth_valid_mask": "scene_depth_valid_mask" in pointworld_obs,
        "has_task_n_obs": "task_n_obs" in pointworld_obs,
        "has_task_n_infl": "task_n_infl" in pointworld_obs,
        "has_runtime_policy": "runtime_policy" in pointworld_obs,
        "scene_flows_shape": _shape("scene_flows"),
        "scene_visibility_shape": _shape("scene_visibility"),
        "scene_depth_valid_mask_shape": _shape("scene_depth_valid_mask"),
        "task_n_obs": int(pointworld_obs.get("task_n_obs", 0) or 0),
        "task_n_infl": int(pointworld_obs.get("task_n_infl", 0) or 0),
        "task_selector_reason": str(pointworld_obs.get("task_selector_reason", "") or ""),
        "runtime_policy": (dict(pointworld_obs.get("runtime_policy", {})) if isinstance(pointworld_obs.get("runtime_policy", {}), dict) else {}),
    }

    if isinstance(pw_cost_debug, dict) and pw_cost_debug:
        summary["pw_cost_debug"] = dict(pw_cost_debug)

    if isinstance(timing_breakdown, dict) and timing_breakdown:
        summary["timing_breakdown"] = dict(timing_breakdown)

    out_path = Path(out_dir).expanduser().resolve() / f"{int(step_id):06d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_scene_npz(
    *,
    out_path: str,
    pts_filtered: np.ndarray,
    cols_filtered: np.ndarray,
    scene_cuboids: Optional[list[dict[str, Any]]],
    has_table: bool,
    robot_spheres_base: Optional[np.ndarray],
) -> None:
    pts_f = np.asarray(pts_filtered, dtype=np.float32)

    cols_f = np.asarray(cols_filtered)
    if cols_f.dtype != np.uint8:
        cols_f = np.clip(cols_f, 0, 255).astype(np.uint8)
    if cols_f.ndim != 2 or cols_f.shape[1] != 3 or cols_f.shape[0] != pts_f.shape[0]:
        cols_f = np.tile(np.asarray([180, 180, 180], dtype=np.uint8)[None, :], (int(pts_f.shape[0]), 1))

    spheres = (
        np.asarray(robot_spheres_base, dtype=np.float32)
        if robot_spheres_base is not None
        else np.zeros((0, 4), dtype=np.float32)
    )

    mask_vis = _robot_mask_vis_points(spheres)
    mask_c = np.tile(np.asarray([255, 0, 0], dtype=np.uint8)[None, :], (int(mask_vis.shape[0]), 1))

    cubs = list(scene_cuboids) if isinstance(scene_cuboids, list) else []
    wire_all: list[np.ndarray] = []
    wire_c_all: list[np.ndarray] = []
    centers: list[list[float]] = []
    dims: list[list[float]] = []

    for i, c in enumerate(cubs):
        if not isinstance(c, dict):
            continue
        cc = c.get("center")
        dd = c.get("dims")
        if not isinstance(cc, (list, tuple)) or not isinstance(dd, (list, tuple)):
            continue
        if len(cc) != 3 or len(dd) != 3:
            continue

        center = (float(cc[0]), float(cc[1]), float(cc[2]))
        dim = (float(dd[0]), float(dd[1]), float(dd[2]))
        centers.append([center[0], center[1], center[2]])
        dims.append([dim[0], dim[1], dim[2]])

        w = _cuboid_wire_points(center, dim)
        if int(w.shape[0]) == 0:
            continue

        # Table (if present) is expected as cuboid 0.
        col = np.asarray([0, 0, 255], dtype=np.uint8) if (bool(has_table) and i == 0) else np.asarray([0, 255, 0], dtype=np.uint8)
        wire_all.append(w)
        wire_c_all.append(np.tile(col[None, :], (int(w.shape[0]), 1)))

    wire = np.concatenate(wire_all, axis=0).astype(np.float32) if wire_all else np.zeros((0, 3), dtype=np.float32)
    wire_c = np.concatenate(wire_c_all, axis=0).astype(np.uint8) if wire_c_all else np.zeros((0, 3), dtype=np.uint8)

    pts_vis = pts_f
    cols_vis = cols_f
    if int(wire.shape[0]) > 0:
        pts_vis = np.concatenate([pts_vis, wire], axis=0)
        cols_vis = np.concatenate([cols_vis, wire_c], axis=0)
    if int(mask_vis.shape[0]) > 0:
        pts_vis = np.concatenate([pts_vis, mask_vis], axis=0)
        cols_vis = np.concatenate([cols_vis, mask_c], axis=0)

    out_dir = Path(str(out_path)).expanduser().resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(out_path),
        points=pts_vis,
        colors=cols_vis,
        points_filtered=pts_f,
        colors_filtered=cols_f,
        cuboid_wire_points=wire,
        cuboid_wire_colors=wire_c,
        cuboid_centers=np.asarray(centers, dtype=np.float32).reshape(-1, 3),
        cuboid_dims=np.asarray(dims, dtype=np.float32).reshape(-1, 3),
        cuboid_has_table=(1 if bool(has_table) else 0),
        robot_mask_vis_points=mask_vis,
        robot_mask_vis_colors=mask_c,
        robot_spheres_base=spheres,
    )


def _make_actions_dummy_hold(q: list[float], gripper: float, horizon: int) -> Any:
    row = list(q) + [float(gripper)]
    return np.asarray([row] * int(horizon), dtype=np.float32)


def _load_cam_configs_from_env(default_cam_id: str) -> dict[str, tuple[Any, np.ndarray]]:
    out: dict[str, tuple[Any, np.ndarray]] = {}

    def _load_one(cam_id: str) -> None:
        suf = str(cam_id).upper()
        cam_info = os.getenv(f"MPPI_PCL_CAM_INFO_{suf}_PATH", "").strip()
        T_path = os.getenv(f"MPPI_PCL_T_BASE_CAM_{suf}_PATH", "").strip()
        if not cam_info or not T_path:
            return
        intr, _ = load_intrinsics_from_cam_info_yaml(cam_info)
        T = load_T_row_major_4x4_yaml(T_path).astype(np.float32)
        out[str(cam_id)] = (intr, T)

    seen: set[str] = set()
    for cam_id in (str(default_cam_id),) + REQUIRED_CAM_IDS:
        c = str(cam_id)
        if c in seen:
            continue
        seen.add(c)
        _load_one(c)

    return out


@dataclass
class _PointWorldRuntime:
    cfg: PointWorldInputConfig
    window: PointWorldWindowBuffer
    builder: OnlineSceneFlowBuilder
    query_manager: QueryPointManager
    cost_model: Optional[PointWorldCostModel] = None

    enabled: bool = True
    last_pointworld_obs: Optional[Dict[str, Any]] = None
    last_camera_names: Tuple[str, ...] = ()
    last_hw_by_camera: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    last_step_id: int = -1
    last_ts_s: float = float("nan")

    consecutive_build_failures: int = 0
    disabled_until_ts_s: float = 0.0
    last_build_error: str = ""
    last_build_error_message: str = ""
    last_build_error_traceback: str = ""

    task_aabb_min: Optional[np.ndarray] = None
    task_aabb_max: Optional[np.ndarray] = None
    task_aabb_min_infl: Optional[np.ndarray] = None
    task_aabb_max_infl: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.window.reset()
        self.query_manager.reset()
        self.last_pointworld_obs = None
        self.last_camera_names = ()
        self.last_hw_by_camera.clear()
        self.last_step_id = -1
        self.last_ts_s = float("nan")
        self.consecutive_build_failures = 0
        self.disabled_until_ts_s = 0.0
        self.last_build_error = ""
        self.last_build_error_message = ""
        self.last_build_error_traceback = ""

    def push_and_maybe_build(
        self,
        *,
        cameras: Dict[str, PWCameraFrame],
        step_id: int,
        ts_s: float,
        q: np.ndarray,
        gripper: float,
    ) -> Optional[Dict[str, Any]]:
        if not cameras:
            raise ValueError("cameras must be non-empty")

        names = tuple(sorted(str(k) for k in cameras.keys()))
        hw_by_cam = {n: (int(cameras[n].depth.shape[0]), int(cameras[n].depth.shape[1])) for n in names}

        if self.last_camera_names and names != self.last_camera_names:
            self.reset()
        if self.last_hw_by_camera and hw_by_cam != self.last_hw_by_camera:
            self.reset()
        if self.last_step_id >= 0 and int(step_id) <= int(self.last_step_id):
            self.reset()
        if np.isfinite(self.last_ts_s) and float(ts_s) <= float(self.last_ts_s):
            self.reset()

        self.last_camera_names = names
        self.last_hw_by_camera = hw_by_cam
        self.last_step_id = int(step_id)
        self.last_ts_s = float(ts_s)

        q7 = np.asarray(q, dtype=np.float32).reshape(-1)
        if q7.shape[0] < 7:
            raise ValueError("q must have at least 7 elements")
        q7 = q7[:7]

        self.window.push_frame(
            cameras={str(k): v for k, v in dict(cameras).items()},
            joint_positions=q7,
            gripper_positions=np.asarray([float(gripper)], dtype=np.float32),
            timestamp=float(ts_s),
        )

        if not self.window.is_ready():
            return None

        if float(ts_s) < float(self.disabled_until_ts_s):
            return self.last_pointworld_obs

        try:
            scene = self.builder.build(window_shift=1, robot_spheres_base=None)
        except Exception as e:
            import traceback

            self.consecutive_build_failures = int(self.consecutive_build_failures) + 1
            self.last_build_error = type(e).__name__
            self.last_build_error_message = str(e)
            self.last_build_error_traceback = traceback.format_exc()

            fail_thr = int(os.getenv("MPPI_PW_BUILD_FAIL_THRESH", "3"))
            cooldown_s = float(os.getenv("MPPI_PW_BUILD_COOLDOWN_S", "1.0"))
            if fail_thr < 1:
                fail_thr = 1
            if cooldown_s < 0.0:
                cooldown_s = 0.0

            if int(self.consecutive_build_failures) >= int(fail_thr):
                self.disabled_until_ts_s = float(ts_s) + float(cooldown_s)

            if _env_bool("MPPI_PW_LOG_BUILD_ERRORS", "1"):
                msg = str(self.last_build_error_message)
                if len(msg) > 500:
                    msg = msg[:500] + "..."
                print(
                    "[pcl_server] pointworld_build_error "
                    f"type={self.last_build_error} failures={self.consecutive_build_failures} "
                    f"message={msg}",
                    flush=True,
                )
            return self.last_pointworld_obs

        self.consecutive_build_failures = 0
        self.disabled_until_ts_s = 0.0
        self.last_build_error = ""
        self.last_build_error_message = ""
        self.last_build_error_traceback = ""

        steps = self.window.get_window()
        ts = np.asarray([float(s.timestamp) for s in steps], dtype=np.float64)
        q_win = np.stack([np.asarray(s.joint_positions, dtype=np.float32).reshape(-1)[:7] for s in steps], axis=0)
        g_win = np.asarray([float(np.asarray(s.gripper_positions).reshape(-1)[0]) for s in steps], dtype=np.float32)

        batch = build_pointworld_batch(
            scene=scene,
            robot_flows=None,
            robot_positions=q_win,
            gripper_positions=g_win,
        )

        T = int(q_win.shape[0])
        gripper_open = (g_win < 0.1).astype(bool).reshape(int(T), 1)

        obs: Dict[str, Any] = dict(batch)
        obs["timestamps"] = ts
        obs["joint_positions_window"] = q_win
        obs["gripper_positions_window"] = g_win

        # droid contract proprio fields
        obs["joint_positions"] = q_win
        obs["gripper_positions"] = g_win.reshape(int(T), 1)
        obs["gripper_open"] = gripper_open

        if _env_bool("MPPI_PW_GRIPPER_POSE_ENABLED", "1" if DEFAULT_PW_GRIPPER_POSE_ENABLED else "0"):
            urdf_path = self.cfg.urdf_path
            if not urdf_path:
                raise ValueError("cfg.urdf_path is required to compute gripper_pose. Set MPPI_PW_URDF_PATH.")
            link_name = os.getenv("MPPI_PW_GRIPPER_LINK", "robotiq_85_base_link").strip() or "robotiq_85_base_link"
            poses = [fk_T_base_link(urdf_path=str(urdf_path), q7=q_win[i], link_name=link_name) for i in range(int(T))]
            obs["gripper_pose"] = np.stack(poses, axis=0).astype(np.float32, copy=False)

        ablation = os.getenv("MPPI_PW_TASK_ABLATION", DEFAULT_PW_TASK_ABLATION).strip().lower()
        if ablation not in {"no_pw", "obs_only", "obs_infl"}:
            ablation = DEFAULT_PW_TASK_ABLATION
        obs["task_ablation_mode"] = ablation

        if ablation != "no_pw" and (self.task_aabb_min is not None) and (self.task_aabb_max is not None):
            try:
                p0 = np.asarray(obs["scene_flows"], dtype=np.float32).reshape(int(T), -1, 3)[0]
                exists0 = np.asarray(obs["scene_exists"], dtype=bool).reshape(int(T), -1)[0]
                vis0 = np.asarray(obs["scene_visibility"], dtype=bool).reshape(int(T), -1)[0] if "scene_visibility" in obs else np.ones_like(exists0)
                dv0 = np.asarray(obs["scene_depth_valid_mask"], dtype=bool).reshape(int(T), -1)[0] if "scene_depth_valid_mask" in obs else np.ones_like(exists0)
                use_vis = _env_bool("MPPI_PW_TASK_USE_VISIBILITY", "1" if DEFAULT_PW_TASK_USE_VISIBILITY else "0")
                use_dv = _env_bool("MPPI_PW_TASK_USE_DEPTH_VALID", "1" if DEFAULT_PW_TASK_USE_DEPTH_VALID else "0")
                valid = exists0.copy()
                if use_vis:
                    valid &= vis0
                if use_dv:
                    valid &= dv0

                mn = np.asarray(self.task_aabb_min, dtype=np.float32).reshape(-1, 3)
                mx = np.asarray(self.task_aabb_max, dtype=np.float32).reshape(-1, 3)
                hit_obs = np.zeros((p0.shape[0],), dtype=bool)
                for j in range(int(mn.shape[0])):
                    hit_obs |= (
                        (p0[:, 0] >= mn[j, 0]) & (p0[:, 0] <= mx[j, 0])
                        & (p0[:, 1] >= mn[j, 1]) & (p0[:, 1] <= mx[j, 1])
                        & (p0[:, 2] >= mn[j, 2]) & (p0[:, 2] <= mx[j, 2])
                    )
                obs["task_n_obs_raw"] = int(hit_obs.sum())
                obs["task_n_obs_after_exists"] = int((hit_obs & exists0).sum())
                obs["task_n_obs_after_visibility"] = int((hit_obs & exists0 & (vis0 if use_vis else np.ones_like(vis0))).sum())
                obs["task_n_obs_after_depth_valid"] = int((hit_obs & exists0 & (vis0 if use_vis else np.ones_like(vis0)) & (dv0 if use_dv else np.ones_like(dv0))).sum())
                idx_obs = np.flatnonzero(valid & hit_obs).astype(np.int64, copy=False)
                if int(idx_obs.size) > 0:
                    idx_obs = np.unique(idx_obs)
                min_points = int(os.getenv("MPPI_PW_TASK_MIN_POINTS_OBS", os.getenv("MPPI_PW_TASK_MIN_POINTS", str(DEFAULT_PW_TASK_MIN_POINTS))))
                if min_points < 1:
                    min_points = DEFAULT_PW_TASK_MIN_POINTS
                if int(idx_obs.size) >= int(min_points):
                    obs["task_point_indices_obs"] = idx_obs
                    obs["task_goal_positions_obs"] = p0[idx_obs].astype(np.float32, copy=False)
                    obs["task_weight_obs"] = float(os.getenv("MPPI_PW_TASK_W_OBS", str(DEFAULT_PW_TASK_W_OBS)))
                    obs["task_n_obs"] = int(idx_obs.size)
                else:
                    obs["task_n_obs"] = int(idx_obs.size)
                    if int(obs["task_n_obs_raw"]) == 0:
                        obs["task_selector_reason"] = "obs_empty_aabb"
                    elif int(obs["task_n_obs_after_exists"]) == 0:
                        obs["task_selector_reason"] = "obs_filtered_by_exists"
                    elif use_vis and int(obs["task_n_obs_after_visibility"]) == 0:
                        obs["task_selector_reason"] = "obs_filtered_by_visibility"
                    elif use_dv and int(obs["task_n_obs_after_depth_valid"]) == 0:
                        obs["task_selector_reason"] = "obs_filtered_by_depth_valid"
                    else:
                        obs["task_selector_reason"] = "obs_too_few_points"

                if ablation == "obs_infl" and (self.task_aabb_min_infl is not None) and (self.task_aabb_max_infl is not None):
                    mn2 = np.asarray(self.task_aabb_min_infl, dtype=np.float32).reshape(-1, 3)
                    mx2 = np.asarray(self.task_aabb_max_infl, dtype=np.float32).reshape(-1, 3)
                    hit_infl = np.zeros((p0.shape[0],), dtype=bool)
                    for j in range(int(mn2.shape[0])):
                        hit_infl |= (
                            (p0[:, 0] >= mn2[j, 0]) & (p0[:, 0] <= mx2[j, 0])
                            & (p0[:, 1] >= mn2[j, 1]) & (p0[:, 1] <= mx2[j, 1])
                            & (p0[:, 2] >= mn2[j, 2]) & (p0[:, 2] <= mx2[j, 2])
                        )
                    obs["task_n_infl_raw"] = int(hit_infl.sum())
                    obs["task_n_infl_after_exists"] = int((hit_infl & exists0).sum())
                    obs["task_n_infl_after_visibility"] = int((hit_infl & exists0 & (vis0 if use_vis else np.ones_like(vis0))).sum())
                    obs["task_n_infl_after_depth_valid"] = int((hit_infl & exists0 & (vis0 if use_vis else np.ones_like(vis0)) & (dv0 if use_dv else np.ones_like(dv0))).sum())
                    idx_infl = np.flatnonzero(valid & hit_infl).astype(np.int64, copy=False)
                    if int(idx_infl.size) > 0:
                        idx_infl = np.unique(idx_infl)
                    min_points_infl = int(os.getenv("MPPI_PW_TASK_MIN_POINTS_INFL", os.getenv("MPPI_PW_TASK_MIN_POINTS", str(DEFAULT_PW_TASK_MIN_POINTS))))
                    if min_points_infl < 1:
                        min_points_infl = DEFAULT_PW_TASK_MIN_POINTS
                    if int(idx_infl.size) >= int(min_points_infl):
                        obs["task_point_indices_infl"] = idx_infl
                        obs["task_goal_positions_infl"] = p0[idx_infl].astype(np.float32, copy=False)
                        obs["task_weight_infl"] = float(os.getenv("MPPI_PW_TASK_W_INFL", str(DEFAULT_PW_TASK_W_INFL)))
                        obs["task_n_infl"] = int(idx_infl.size)
                    else:
                        obs["task_n_infl"] = int(idx_infl.size)
                        if int(obs["task_n_infl_raw"]) == 0:
                            obs["task_selector_reason"] = obs.get("task_selector_reason", "") or "infl_empty_aabb"
                        elif int(obs["task_n_infl_after_exists"]) == 0:
                            obs["task_selector_reason"] = obs.get("task_selector_reason", "") or "infl_filtered_by_exists"
                        elif use_vis and int(obs["task_n_infl_after_visibility"]) == 0:
                            obs["task_selector_reason"] = obs.get("task_selector_reason", "") or "infl_filtered_by_visibility"
                        elif use_dv and int(obs["task_n_infl_after_depth_valid"]) == 0:
                            obs["task_selector_reason"] = obs.get("task_selector_reason", "") or "infl_filtered_by_depth_valid"
                        else:
                            obs["task_selector_reason"] = obs.get("task_selector_reason", "") or "infl_too_few_points"

                if ablation == "obs_only":
                    obs.pop("task_point_indices_infl", None)
                    obs.pop("task_goal_positions_infl", None)
                    obs.pop("task_weight_infl", None)
                    obs.pop("task_n_infl", None)
            except Exception as e:
                obs["task_selector_reason"] = type(e).__name__

        if ablation == "no_pw":
            obs.pop("task_point_indices_obs", None)
            obs.pop("task_goal_positions_obs", None)
            obs.pop("task_weight_obs", None)
            obs.pop("task_n_obs", None)
            obs.pop("task_point_indices_infl", None)
            obs.pop("task_goal_positions_infl", None)
            obs.pop("task_weight_infl", None)
            obs.pop("task_n_infl", None)

        raw_idx = os.getenv("MPPI_PW_TASK_POINT_INDICES", "").strip()
        raw_goal = os.getenv("MPPI_PW_TASK_GOAL_XYZ", "").strip()
        obs["runtime_policy"] = {
            "seed_robot_mask_enabled": bool(self.cfg.seed_robot_mask_enabled),
            "gripper_pose_enabled": _env_bool("MPPI_PW_GRIPPER_POSE_ENABLED", "1" if DEFAULT_PW_GRIPPER_POSE_ENABLED else "0"),
            "task_use_visibility": _env_bool("MPPI_PW_TASK_USE_VISIBILITY", "1" if DEFAULT_PW_TASK_USE_VISIBILITY else "0"),
            "task_use_depth_valid": _env_bool("MPPI_PW_TASK_USE_DEPTH_VALID", "1" if DEFAULT_PW_TASK_USE_DEPTH_VALID else "0"),
            "task_ablation_mode": ablation,
            "cooldown_active": bool(float(ts_s) < float(self.disabled_until_ts_s)),
        }

        if raw_idx and raw_goal and ("task_point_indices_obs" not in obs) and ("task_point_indices" not in obs):
            try:
                idx = np.asarray([int(p.strip()) for p in raw_idx.split(",") if p.strip()], dtype=np.int32)
                goal = np.asarray([float(p.strip()) for p in raw_goal.split(",") if p.strip()], dtype=np.float32)
                if int(idx.size) > 0 and int(goal.size) == 3:
                    obs["task_point_indices"] = idx
                    obs["task_goal_positions"] = goal.reshape(3)
            except Exception:
                pass

        self.last_pointworld_obs = obs
        return obs


def _build_pointworld_runtime() -> Optional[_PointWorldRuntime]:
    if not _effective_pointworld_enabled():
        return None

    ckpt = os.getenv("MPPI_PW_COTRACKER_CKPT", "").strip()
    if not ckpt:
        raise ValueError("MPPI_PW_COTRACKER_CKPT is required when MPPI_PW_ENABLE=1")

    device = os.getenv("MPPI_PW_COTRACKER_DEVICE", "cuda").strip() or None

    max_q = _env_i("MPPI_PW_MAX_QUERY_POINTS_PER_CAMERA", "2048")
    min_conf = float(_env_f("MPPI_PW_MIN_TRACK_CONFIDENCE", "0.0"))
    rng_seed = _env_i("MPPI_PW_QUERY_RNG_SEED", "0")

    pw_depth_min_m = float(_env_f("MPPI_PW_DEPTH_MIN_M", "0.0"))
    pw_depth_max_m = float(_env_f("MPPI_PW_DEPTH_MAX_M", "4.0"))

    ws_min = _env_vec3("MPPI_PW_WORKSPACE_MIN", "0.00,-0.38,-0.30")
    ws_max = _env_vec3("MPPI_PW_WORKSPACE_MAX", "0.80,0.30,1.20")

    seed_robot_mask_enabled = _env_bool("MPPI_PW_SEED_ROBOT_MASK_ENABLED", "1" if DEFAULT_PW_SEED_ROBOT_MASK_ENABLED else "0")
    robot_mask_seed = int(os.getenv("MPPI_PW_ROBOT_MASK_SEED", "0"))

    ee_filter_enabled = _env_bool("MPPI_PW_EE_FILTER_ENABLED", "0")
    ee_filter_link = os.getenv("MPPI_PW_EE_FILTER_LINK", "panda_link7")
    ee_filter_spheres = parse_spheres_spec(os.getenv("MPPI_PW_EE_FILTER_SPHERES", ""))

    urdf_path = os.getenv("MPPI_PW_URDF_PATH", os.getenv("MPPI_URDF_PATH", "")).strip() or None
    if seed_robot_mask_enabled and not urdf_path:
        raise ValueError("MPPI_PW_SEED_ROBOT_MASK_ENABLED=1 requires MPPI_PW_URDF_PATH (or MPPI_URDF_PATH)")
    if _env_bool("MPPI_PW_GRIPPER_POSE_ENABLED", "1" if DEFAULT_PW_GRIPPER_POSE_ENABLED else "0") and not urdf_path:
        raise ValueError("MPPI_PW_GRIPPER_POSE_ENABLED=1 requires MPPI_PW_URDF_PATH (or MPPI_URDF_PATH)")

    tracking = TrackingConfig(
        max_query_points_per_camera=int(max_q),
        min_track_confidence=float(min_conf),
        depth_min_m=float(pw_depth_min_m),
        depth_max_m=float(pw_depth_max_m),
    )
    workspace = WorkspaceFilterConfig(workspace_min=ws_min, workspace_max=ws_max)
    robot = RobotFilterConfig(
        robot_mask_margin_m=float(os.getenv("MPPI_PW_ROBOT_MASK_MARGIN_M", "0.02")),
        ee_filter_enabled=bool(ee_filter_enabled),
        ee_filter_link=str(ee_filter_link),
        ee_filter_spheres=tuple(ee_filter_spheres),
        ee_filter_margin_m=float(os.getenv("MPPI_PW_EE_FILTER_MARGIN_M", "0.0")),
    )

    cfg = PointWorldInputConfig(
        window_size=11,
        tracking=tracking,
        workspace_filter=workspace,
        robot_filter=robot,
        urdf_path=urdf_path,
        seed_robot_mask_enabled=bool(seed_robot_mask_enabled),
        robot_mask_seed=int(robot_mask_seed),
        camera_names=REQUIRED_CAM_IDS,
        camera_selection="subset",
        min_cameras=len(REQUIRED_CAM_IDS),
    )

    window = PointWorldWindowBuffer(window_size=11)
    query_manager = QueryPointManager(cfg=QueryPointManagerConfig(max_query_points_per_camera=int(max_q), min_track_confidence=float(min_conf), rng_seed=int(rng_seed)))
    tracker = build_cotracker_online_point_tracker(
        checkpoint=str(ckpt),
        window_len=11,
        device=device,
        iters=int(os.getenv("MPPI_PW_COTRACKER_ITERS", "6")),
    )
    builder = OnlineSceneFlowBuilder(cfg=cfg, window_buffer=window, tracker=tracker, query_manager=query_manager)
    print(f"[pcl_server] pointworld_tracker_devices={_tracker_device_summary(tracker)}")

    cost_model: Optional[PointWorldCostModel] = None
    if _effective_use_pointworld_cost():
        model_path = os.getenv("MPPI_PW_MODEL_PATH", "").strip()
        if not model_path:
            raise ValueError("MPPI_PW_MODEL_PATH is required when MPPI_USE_POINTWORLD_COST=1")
        urdf_path = os.getenv("MPPI_PW_URDF_PATH", os.getenv("MPPI_URDF_PATH", "")).strip() or default_urdf_path()

        cost_cfg = PointWorldCostConfig(
            mode=str(os.getenv("MPPI_PW_COST_MODE", "task_point_goal_l2")),
            use_model_confidence=_env_bool("MPPI_PW_USE_MODEL_CONFIDENCE", "1"),
            use_track_confidence=_env_bool("MPPI_PW_USE_TRACK_CONFIDENCE", "1"),
            min_confidence=float(os.getenv("MPPI_PW_MIN_COST_CONFIDENCE", "0.0")),
            ignore_t0=_env_bool("MPPI_PW_COST_IGNORE_T0", "1"),
        )
        cost_model = PointWorldCostModel(
            PointWorldModelConfig(
                checkpoint_path=str(model_path),
                device=str(os.getenv("MPPI_PW_MODEL_DEVICE", os.getenv("MPPI_PW_COTRACKER_DEVICE", "cuda"))),
                domain=(os.getenv("MPPI_PW_MODEL_DOMAIN", "").strip() or None),
                urdf_path=str(urdf_path),
                max_scene_points=(
                    int(os.getenv("MPPI_PW_MAX_SCENE_POINTS"))
                    if os.getenv("MPPI_PW_MAX_SCENE_POINTS", "").strip()
                    else None
                ),
                max_robot_points=(
                    int(os.getenv("MPPI_PW_MAX_ROBOT_POINTS"))
                    if os.getenv("MPPI_PW_MAX_ROBOT_POINTS", "").strip()
                    else None
                ),
                robot_sampler_device=(os.getenv("MPPI_PW_ROBOT_SAMPLER_DEVICE", "").strip() or None),
                robot_gripper_only=_env_bool("MPPI_PW_ROBOT_GRIPPER_ONLY", "1"),
                seed=int(os.getenv("MPPI_PW_SEED", "1")),
                disable_compile=_env_bool("MPPI_PW_DISABLE_COMPILE", "1"),
                eval_batch_size=int(os.getenv("MPPI_PW_EVAL_BATCH_SIZE", "32")),
                dist2robot_mode=str(os.getenv("MPPI_PW_DIST2ROBOT_MODE", "full")),
                cost=cost_cfg,
            )
        )
        print(
            "[pcl_server] pointworld_cost_devices="
            f"{','.join(cost_model._devices)} robot_devices={','.join(cost_model._robot_devices)} "
            f"replicas={len(cost_model._replicas)} eval_batch_size={cost_model.cfg.eval_batch_size} "
            f"dist2robot_mode={cost_model._dist2robot_mode}"
        )

    rt = _PointWorldRuntime(
        cfg=cfg,
        window=window,
        builder=builder,
        query_manager=query_manager,
        cost_model=cost_model,
    )

    default_aabb_path = DEFAULT_PW_AABB_CONFIG_PATH if Path(DEFAULT_PW_AABB_CONFIG_PATH).is_file() else ""
    aabb_path = os.getenv("MPPI_PW_AABB_CONFIG_PATH", default_aabb_path).strip()
    if aabb_path:
        try:
            a = _load_aabb_config_json(aabb_path)
            rt.task_aabb_min = np.asarray(a["aabb_min"], dtype=np.float32)
            rt.task_aabb_max = np.asarray(a["aabb_max"], dtype=np.float32)
            rt.task_aabb_min_infl = np.asarray(a["aabb_min_infl"], dtype=np.float32)
            rt.task_aabb_max_infl = np.asarray(a["aabb_max_infl"], dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load MPPI_PW_AABB_CONFIG_PATH={aabb_path}: {type(e).__name__}") from e

    return rt


async def _handle_connection(
    ws: Any,
    cfg: ServerConfig,
    cam_configs: dict[str, tuple[Any, np.ndarray]],
    pw: Optional[_PointWorldRuntime],
) -> None:
    verbose = _env_bool("MPPI_PCL_VERBOSE", "1")
    print_every = max(1, int(os.getenv("MPPI_PCL_PRINT_EVERY", "1")))
    heartbeat_s = float(os.getenv("MPPI_PCL_HEARTBEAT_S", "5.0"))

    conn_id = hex(id(ws))
    remote = getattr(ws, "remote_address", None)
    if verbose:
        print(f"[pcl_server] connect id={conn_id} remote={remote}")

    last_hb_s = time.time()
    n_req = 0

    while True:
        try:
            data = await asyncio.wait_for(ws.recv(), timeout=cfg.request_timeout_s)
        except asyncio.TimeoutError:
            now_s = time.time()
            if verbose and heartbeat_s > 0.0 and (now_s - last_hb_s) >= heartbeat_s:
                print(f"[pcl_server] heartbeat id={conn_id} policy={cfg.policy} waiting")
                last_hb_s = now_s
            continue
        except Exception:
            break

        t_server_recv_ns = time.time_ns()
        t0 = time.perf_counter()
        tb: Dict[str, Any] = {}

        try:
            if isinstance(data, str):
                raise ValueError("Expected binary msgpack payload, got text frame.")
            envelope = decode_message(data)

            if int(envelope.get("schema_version", -1)) != SCHEMA_VERSION_PCL:
                raise ValueError(f"Unsupported schema_version: {envelope.get('schema_version')}")
            if envelope.get("type") != "infer_request_pcl":
                raise ValueError(f"Unexpected message type: {envelope.get('type')}")

            request_id = str(envelope["request_id"])
            obs = ObsPCL.from_payload(dict(envelope["payload"]))
            n_req += 1

            tb["request_id"] = request_id
            tb["step_id"] = int(getattr(obs, "step_id", -1))
            tb["t_decode_ms"] = (time.perf_counter() - t0) * 1000.0

            pw_cameras: Dict[str, PWCameraFrame] = {}
            intr_by_cam: Dict[str, Any] = {}
            T_by_cam: Dict[str, Any] = {}
            depth_unit_scale_by_cam: Dict[str, float] = {}
            pcd_base: Optional[Dict[str, Any]] = None

            depth_unit_scale_primary = float(obs.depth_unit_scale) if obs.depth_unit_scale is not None else _env_f("MPPI_PCL_DEPTH_UNIT_SCALE", "1.0")
            need_camera_decode = _request_needs_camera_decode(cfg.policy, pw)
            need_pcd = _request_needs_pcd(cfg.policy)

            cams_payload = getattr(obs, "cameras", None)
            if need_camera_decode:
                if not (isinstance(cams_payload, dict) and cams_payload):
                    raise ValueError("This server requires ObsPCL.cameras with back+side when perception costs or PCD save are enabled.")

                for need in REQUIRED_CAM_IDS:
                    if str(need) not in cams_payload:
                        raise ValueError(f"Missing required camera in payload: {need}")

                t_cam0 = time.perf_counter()
                for cam_name in REQUIRED_CAM_IDS:
                    cam_pl = cams_payload[str(cam_name)]
                    if not isinstance(cam_pl, dict):
                        raise ValueError(f"cameras[{cam_name}] must be a dict")

                    intr_i, T_i = parse_obs_camera_params(
                        cam_id=cam_name,
                        intrinsics=(dict(cam_pl["intrinsics"]) if "intrinsics" in cam_pl and cam_pl["intrinsics"] is not None else None),
                        T_base_cam=(cam_pl.get("T_base_cam", None)),
                        cam_configs=cam_configs,
                    )

                    if cam_pl.get("rgb_bytes", None) is not None:
                        rgb_i = _decode_rgb(codec=(cam_pl.get("rgb_codec", None)), data=cam_pl.get("rgb_bytes", None))
                    elif cam_pl.get("rgb_back", None) is not None:
                        rgb_i = np.asarray(cam_pl.get("rgb_back", None))
                    else:
                        raise ValueError(f"Missing rgb for camera {cam_name}")

                    if cam_pl.get("depth_bytes", None) is not None:
                        depth_i = _decode_depth(codec=(cam_pl.get("depth_codec", None)), data=cam_pl.get("depth_bytes", None))
                    elif cam_pl.get("depth_back", None) is not None:
                        depth_i = np.asarray(cam_pl.get("depth_back", None))
                    else:
                        raise ValueError(f"Missing depth for camera {cam_name}")

                    rgb_i, depth_i, intr_i = _resize_rgbd_to_contract(rgb=np.asarray(rgb_i), depth=np.asarray(depth_i), intr=intr_i)

                    intr_pw = PWPinholeIntrinsics(fx=float(intr_i.fx), fy=float(intr_i.fy), cx=float(intr_i.cx), cy=float(intr_i.cy))
                    pw_cameras[cam_name] = PWCameraFrame(
                        rgb=np.asarray(rgb_i),
                        depth=np.asarray(depth_i),
                        intrinsics=intr_pw,
                        extrinsics=np.asarray(T_i, dtype=np.float32).reshape(4, 4),
                    )

                    intr_by_cam[cam_name] = intr_i
                    T_by_cam[cam_name] = T_i
                    depth_unit_scale_by_cam[cam_name] = (
                        float(cam_pl["depth_unit_scale"])
                        if "depth_unit_scale" in cam_pl and cam_pl["depth_unit_scale"] is not None
                        else depth_unit_scale_primary
                    )

                    if cam_name == "back":
                        depth_unit_scale_primary = float(depth_unit_scale_by_cam[cam_name])

                tb["t_cameras_ms"] = (time.perf_counter() - t_cam0) * 1000.0
            else:
                tb["t_cameras_ms"] = 0.0
                tb["perception_fast_path"] = "no_camera_decode"

            if need_camera_decode and pw is not None and bool(pw.enabled):
                ts_s = float(obs.t_client_send_ns) * 1e-9
                t_pw0 = time.perf_counter()
                try:
                    pw.push_and_maybe_build(
                        cameras=pw_cameras,
                        step_id=int(obs.step_id),
                        ts_s=float(ts_s),
                        q=np.asarray(obs.q, dtype=np.float32),
                        gripper=float(obs.gripper),
                    )
                    tb["t_pw_build_ms"] = (time.perf_counter() - t_pw0) * 1000.0
                    if _env_bool("MPPI_PW_LOG_WINDOW", "0"):
                        print(
                            "[pcl_server] pointworld_window "
                            f"step={int(obs.step_id)} len={len(pw.window)} ready={pw.window.is_ready()} "
                            f"has_obs={pw.last_pointworld_obs is not None} "
                            f"failures={pw.consecutive_build_failures} error={pw.last_build_error}",
                            flush=True,
                        )
                    if str(getattr(pw, "last_build_error", "") or ""):
                        tb["pw_build_error_type"] = str(getattr(pw, "last_build_error", "") or "")
                        tb["pw_build_error_message"] = str(getattr(pw, "last_build_error_message", "") or "")
                    _maybe_dump_pointworld_acceptance(pointworld_obs=pw.last_pointworld_obs, step_id=int(obs.step_id))
                except Exception as e:  # noqa: BLE001
                    tb["t_pw_build_ms"] = (time.perf_counter() - t_pw0) * 1000.0
                    tb["pw_build_error_type"] = type(e).__name__
                    tb["pw_build_error_message"] = str(e)
                    pw.reset()

            if need_pcd:
                depth_min_m = _env_f("MPPI_PCL_DEPTH_MIN_M", "0.0")
                depth_max_m = _env_f("MPPI_PCL_DEPTH_MAX_M", "4.0")
                stride = _env_i("MPPI_PCL_STRIDE", "1")

                t_pcd0 = time.perf_counter()
                pcd_list: list[Dict[str, Any]] = []
                for cam_name in REQUIRED_CAM_IDS:
                    pcd_i = rgbd_to_pointcloud_base(
                        depth=np.asarray(pw_cameras[cam_name].depth),
                        rgb=np.asarray(pw_cameras[cam_name].rgb),
                        intr=intr_by_cam[cam_name],
                        T_base_cam=T_by_cam[cam_name],
                        depth_unit_scale=float(depth_unit_scale_by_cam.get(cam_name, depth_unit_scale_primary)),
                        depth_min_m=float(depth_min_m),
                        depth_max_m=float(depth_max_m),
                        stride=int(stride),
                        roi_min=None,
                        roi_max=None,
                        voxel_size_m=0.0,
                    )
                    pcd_list.append(dict(pcd_i))

                pts_list = [np.asarray(p["points"], dtype=np.float32) for p in pcd_list if np.asarray(p["points"]).size > 0]
                pts = np.concatenate(pts_list, axis=0) if pts_list else np.zeros((0, 3), dtype=np.float32)
                out_pcd: Dict[str, Any] = {"points": np.ascontiguousarray(pts)}

                if all("colors" in p for p in pcd_list):
                    col_list = [np.asarray(p["colors"], dtype=np.uint8) for p in pcd_list if np.asarray(p["points"]).size > 0]
                    cols = np.concatenate(col_list, axis=0) if col_list else np.zeros((0, 3), dtype=np.uint8)
                    out_pcd["colors"] = np.ascontiguousarray(cols)

                pcd_base = out_pcd

                try:
                    tb["pcd_points"] = int(np.asarray(pcd_base.get("points")).shape[0]) if isinstance(pcd_base, dict) and "points" in pcd_base else 0
                except Exception:
                    tb["pcd_points"] = 0
                tb["t_pcd_ms"] = (time.perf_counter() - t_pcd0) * 1000.0
            else:
                tb["pcd_points"] = 0
                tb["t_pcd_ms"] = 0.0

            timing_policy = cfg.policy
            t_solve0 = time.perf_counter()
            if cfg.policy == "dummy_hold":
                actions = _make_actions_dummy_hold(obs.q, float(obs.gripper), cfg.open_loop_horizon)
            elif cfg.policy == "mppi_joint":
                solver = _get_joint_solver(cfg.open_loop_horizon)
                pw_obs = pw.last_pointworld_obs if (pw is not None and bool(pw.enabled)) else None

                pw_cost_fn0: Optional[PointWorldCostFn] = (
                    pw.cost_model if (pw is not None and bool(pw.enabled) and pw.cost_model is not None) else None
                )

                pw_cost_error: Dict[str, Any] = {}

                if pw_cost_fn0 is not None:

                    def _wrapped_pw_cost_fn(*, q_traj: np.ndarray, u_traj: np.ndarray, pointworld_obs: dict[str, Any], gripper: Optional[float] = None):
                        try:
                            return pw_cost_fn0(
                                q_traj=q_traj,
                                u_traj=u_traj,
                                pointworld_obs=pointworld_obs,
                                gripper=gripper,
                            )
                        except Exception as e:  # noqa: BLE001
                            import traceback

                            pw_cost_error["type"] = type(e).__name__
                            pw_cost_error["message"] = str(e)
                            pw_cost_error["traceback"] = traceback.format_exc()
                            raise

                    pw_cost_fn: Optional[PointWorldCostFn] = _wrapped_pw_cost_fn
                else:
                    pw_cost_fn = None

                actions = solver.infer_actions(
                    q0=obs.q,
                    gripper=float(obs.gripper),
                    pcd_back_cam=pcd_base,
                    pointworld_obs=pw_obs,
                    pointworld_cost_fn=pw_cost_fn,
                )

                tb["t_solver_ms"] = (time.perf_counter() - t_solve0) * 1000.0
                curobo_eval_ranges = getattr(solver, "last_curobo_eval_ranges", ())
                if curobo_eval_ranges:
                    tb["curobo_eval_ranges"] = [dict(x) for x in tuple(curobo_eval_ranges)]

                pw_cost_debug: Dict[str, Any] = {
                    "enabled": bool(getattr(solver, "last_pw_enabled", False)),
                    "reason": str(getattr(solver, "last_pw_reason", "") or ""),
                    "ms": float(getattr(solver, "last_pw_ms", 0.0) or 0.0),
                }
                if pw_cost_fn0 is not None:
                    eval_ranges = getattr(pw_cost_fn0, "last_eval_ranges", ())
                    if eval_ranges:
                        pw_cost_debug["eval_ranges"] = [dict(x) for x in tuple(eval_ranges)]

                if pw_cost_error:
                    tb = str(pw_cost_error.get("traceback", "") or "")
                    if len(tb) > 4000:
                        tb = tb[-4000:]
                    pw_cost_debug["error_type"] = str(pw_cost_error.get("type", "") or "")
                    pw_cost_debug["error_message"] = str(pw_cost_error.get("message", "") or "")
                    pw_cost_debug["error_traceback"] = tb

                tb["pw_ms"] = float(pw_cost_debug.get("ms", 0.0) or 0.0)
                _maybe_dump_pointworld_acceptance(
                    pointworld_obs=pw_obs,
                    step_id=int(obs.step_id),
                    pw_cost_debug=pw_cost_debug,
                    timing_breakdown=tb,
                )

                if (
                    _env_bool("MPPI_PCL_SAVE_PCD", "0")
                    and getattr(solver, "cfg", None) is not None
                    and bool(getattr(solver.cfg, "use_curobo_collision", False))
                    and isinstance(pcd_base, dict)
                ):
                    scfg = solver.cfg

                    checker = get_curobo_collision_checker(
                        CuRoboCollisionConfig(
                            device=_first_device(scfg.curobo_device),
                            robot_yaml=str(scfg.curobo_robot_yaml),
                            urdf_path=str(scfg.urdf_path),
                            tool_frame=str(scfg.curobo_tool_frame),
                            with_world=True,
                            collision_activation_distance=float(scfg.curobo_collision_activation_distance),
                            self_collision_activation_distance=float(scfg.curobo_self_collision_activation_distance),
                            max_collision_distance=float(scfg.curobo_max_collision_distance),
                        )
                    )
                    spheres = checker.get_robot_spheres_base(
                        np.asarray(obs.q, dtype=np.float32),
                        margin_m=float(scfg.scene_robot_mask_margin_m),
                    )

                    table_top_z = float(scfg.scene_table_center[2]) + 0.5 * float(scfg.scene_table_dims[2])

                    sb_cfg = SceneBuildConfig(
                        t_base_cam_back_path=str(scfg.t_base_cam_back_path),
                        roi_min=scfg.scene_roi_min,
                        roi_max=scfg.scene_roi_max,
                        voxel_size_m=float(scfg.scene_voxel_size_m),
                        padding_m=float(scfg.scene_padding_m),
                        max_cuboids=int(scfg.scene_max_cuboids),
                        robot_mask_margin_m=0.0,
                        min_cluster_voxels=int(scfg.scene_min_cluster_voxels),
                        remove_table_points=bool(scfg.scene_remove_table_points),
                        table_top_z_m=float(table_top_z),
                        table_eps_m=float(scfg.scene_table_eps_m),
                        remove_wall_points=bool(scfg.scene_remove_wall_points),
                        wall_center=scfg.scene_wall_center,
                        wall_dims=scfg.scene_wall_dims,
                        wall_margin_m=float(scfg.scene_wall_margin_m),
                    )

                    pts_in = pcd_base.get("points") if isinstance(pcd_base, dict) else None
                    cols_in = pcd_base.get("colors") if isinstance(pcd_base, dict) else None

                    pts_arr = np.asarray(pts_in, dtype=np.float32) if pts_in is not None else np.zeros((0, 3), dtype=np.float32)

                    if cols_in is None:
                        cols_arr = np.tile(np.asarray([180, 180, 180], dtype=np.uint8)[None, :], (int(pts_arr.shape[0]), 1))
                    else:
                        cols_arr = np.asarray(cols_in)
                        if cols_arr.dtype != np.uint8:
                            cols_arr = np.clip(cols_arr, 0, 255).astype(np.uint8)
                        if cols_arr.ndim != 2 or cols_arr.shape[1] != 3 or cols_arr.shape[0] != pts_arr.shape[0]:
                            cols_arr = np.tile(np.asarray([180, 180, 180], dtype=np.uint8)[None, :], (int(pts_arr.shape[0]), 1))

                    pts_filtered, cols_filtered = build_scene_points_base_and_colors_from_pcd_back_cam(
                        pts_arr,
                        cols_arr,
                        cfg=sb_cfg,
                        pcd_scale=float(scfg.scene_pcd_scale),
                        pcd_in_base=bool(scfg.scene_pcd_in_base),
                        robot_spheres=spheres,
                    )

                    scene_cuboids = getattr(solver, "last_scene_cuboids", None)
                    has_table = bool(getattr(solver, "last_scene_has_table", False))

                    out_base = os.getenv("MPPI_PCL_SAVE_PCD_OUT", repo_path("data", "test", "pcl_scene_filtered.npz"))
                    out_path = _resolve_save_path(out_base, step_id=int(obs.step_id))
                    _save_scene_npz(
                        out_path=str(out_path),
                        pts_filtered=pts_filtered,
                        cols_filtered=cols_filtered,
                        scene_cuboids=(list(scene_cuboids) if isinstance(scene_cuboids, list) else None),
                        has_table=bool(has_table),
                        robot_spheres_base=spheres,
                    )

                use_curobo = bool(getattr(solver, "cfg", None) and getattr(solver.cfg, "use_curobo_collision", False))
                ess_ratio = float(getattr(solver, "last_effective_samples_ratio", 0.0) or 0.0)
                n_cub = int(getattr(solver, "last_scene_num_cuboids", 0) or 0)
                n_sph = int(getattr(solver, "last_scene_num_robot_spheres", 0) or 0)
                has_table = bool(getattr(solver, "last_scene_has_table", False))

                stable_trk = int(getattr(solver, "last_scene_num_dynamic_tracks", 0) or 0)
                scene_key_short = str(getattr(solver, "last_scene_key_short", "") or "")

                pw_enabled = bool(getattr(solver, "last_pw_enabled", False))
                pw_reason = str(getattr(solver, "last_pw_reason", "") or "")
                pw_ms = float(getattr(solver, "last_pw_ms", 0.0) or 0.0)
                pw_tag = f"pw{1 if pw_enabled else 0}:{pw_reason}:{pw_ms:.1f}ms"

                timing_policy = f"mppi_joint+{'curobo' if use_curobo else 'nocurobo'}+ess{ess_ratio:.3f}+tab{1 if has_table else 0}+cub{n_cub}+trk{int(stable_trk)}+key{scene_key_short}+sph{n_sph}+{pw_tag}"

                suffix = str(getattr(solver, "last_timing_policy_suffix", "") or "")
                if suffix:
                    timing_policy = f"{timing_policy}{suffix}"

                if getattr(solver, "last_fallback", False):
                    reason = str(getattr(solver, "last_fallback_reason", ""))
                    actions = _make_actions_dummy_hold(obs.q, float(obs.gripper), cfg.open_loop_horizon)
                    timing_policy = f"{timing_policy}:fallback_hold:{reason}"
            else:
                raise ValueError(f"Unknown policy: {cfg.policy}")

            t1 = time.perf_counter()
            t_server_send_ns = time.time_ns()

            timing = ServerTimingPCL(infer_ms=(t1 - t0) * 1000.0, queue_ms=0.0, policy=timing_policy)
            chunk = ActionChunkPCL(
                t_server_recv_ns=t_server_recv_ns,
                t_server_send_ns=t_server_send_ns,
                t_client_send_ns_echo=obs.t_client_send_ns,
                open_loop_horizon=cfg.open_loop_horizon,
                actions=actions,
                server_timing=timing,
            )
            resp = InferResponsePCL(request_id=request_id, action_chunk=chunk).to_envelope()
            t_enc0 = time.perf_counter()
            payload_out = encode_message(resp)
            tb["t_encode_ms"] = (time.perf_counter() - t_enc0) * 1000.0
            await ws.send(payload_out)

            if verbose and (n_req % print_every) == 0:
                cams_payload = getattr(obs, "cameras", None)
                cam_names = sorted(str(k) for k in cams_payload.keys()) if isinstance(cams_payload, dict) else []
                n_pts = 0
                try:
                    n_pts = int(np.asarray(pcd_base.get("points")).shape[0]) if isinstance(pcd_base, dict) and "points" in pcd_base else 0
                except Exception:
                    n_pts = 0
                print(f"[pcl_server] id={conn_id} step={int(obs.step_id)} req={request_id} cams={cam_names} points={n_pts} infer_ms={(t1 - t0) * 1000.0:.3f} policy={timing_policy}")
        except Exception as e:  # noqa: BLE001
            try:
                request_id = str(envelope.get("request_id", "")) if "envelope" in locals() else ""
                err = ErrorPCL(
                    request_id=request_id,
                    code="bad_request",
                    message=str(e),
                    t_server_send_ns=time.time_ns(),
                ).to_envelope()
                await ws.send(encode_message(err))
            except Exception:
                break


async def serve(cfg: ServerConfig) -> None:
    websockets = _require_websockets()
    cam_configs = _load_cam_configs_from_env(cfg.cam_id)
    pw = _build_pointworld_runtime()

    needs_camera_configs = _request_needs_camera_decode(cfg.policy, pw)
    missing = [c for c in REQUIRED_CAM_IDS if str(c) not in cam_configs]
    if needs_camera_configs and missing:
        name_list = ",".join(str(x) for x in missing)
        raise RuntimeError(
            f"Missing cam config(s): {name_list}. "
            f"Set MPPI_PCL_CAM_INFO_<CAM>_PATH and MPPI_PCL_T_BASE_CAM_<CAM>_PATH for each, e.g. BACK/SIDE."
        )

    verbose = _env_bool("MPPI_PCL_VERBOSE", "1")
    if verbose:
        pw_enabled = bool(pw is not None and bool(getattr(pw, "enabled", False)))
        dump_dir = os.getenv("MPPI_PW_ACCEPTANCE_DUMP_DIR", "").strip()
        print(f"[pcl_server] boot host={cfg.host} port={cfg.port} policy={cfg.policy} horizon={cfg.open_loop_horizon} cam_id={cfg.cam_id}")
        print(f"[pcl_server] required_cams={list(REQUIRED_CAM_IDS)} loaded_cam_configs={sorted(cam_configs.keys())}")
        print(f"[pcl_server] needs_camera_decode={needs_camera_configs} cuda_available={_cuda_runtime_available()} default_num_samples={_default_num_samples()}")
        print(f"[pcl_server] curobo_device={os.getenv('MPPI_CUROBO_DEVICE', 'cuda:0')}")
        print(f"[pcl_server] pointworld_enabled={pw_enabled} acceptance_dump_dir={dump_dir if dump_dir else 'disabled'}")

    async def handler(ws: Any) -> None:
        await _handle_connection(ws, cfg, cam_configs, pw)

    async with websockets.serve(handler, cfg.host, cfg.port, max_size=None):
        if verbose:
            print(f"[pcl_server] listening ws://{cfg.host}:{cfg.port}")
        await asyncio.Future()


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="ws_server_async_pcl")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9011)
    ap.add_argument("--open-loop-horizon", type=int, default=8)
    ap.add_argument("--policy", type=str, default="dummy_hold")
    ap.add_argument("--request-timeout-s", type=float, default=2.0)
    ap.add_argument("--cam-id", type=str, default="back")
    args = ap.parse_args(argv)

    cfg = ServerConfig(
        host=str(args.host),
        port=int(args.port),
        open_loop_horizon=int(args.open_loop_horizon),
        policy=str(args.policy),
        request_timeout_s=float(args.request_timeout_s),
        cam_id=str(args.cam_id),
    )
    asyncio.run(serve(cfg))


if __name__ == "__main__":
    main()
