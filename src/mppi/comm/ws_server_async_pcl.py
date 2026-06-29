from __future__ import annotations

import argparse
import asyncio
import io
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from mppi.comm.ws_server_async import _get_joint_solver
from mppi.mpc.solver import PointWorldCostFn
from mppi.curobo_ext.check_depth_pcl import (
    load_T_row_major_4x4_yaml,
    load_intrinsics_from_cam_info_yaml,
    parse_obs_camera_params,
    rgbd_to_pointcloud_base,
)
from mppi.curobo_ext.collision_checker import CuRoboCollisionConfig, get_curobo_collision_checker
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
from mppi.pointworld_ext.tracker_interface import CoTrackerOnlinePointTracker
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

    _load_one(default_cam_id)
    if str(default_cam_id) != "back":
        _load_one("back")

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
    last_cam_id: str = ""
    last_hw: Tuple[int, int] = (0, 0)
    last_step_id: int = -1
    last_ts_s: float = float("nan")

    def reset(self) -> None:
        self.window.reset()
        self.query_manager.reset()
        self.last_pointworld_obs = None

    def push_and_maybe_build(
        self,
        *,
        cam_id: str,
        step_id: int,
        ts_s: float,
        rgb: np.ndarray,
        depth: np.ndarray,
        intr: Any,
        T_base_cam: np.ndarray,
        q: np.ndarray,
        gripper: float,
    ) -> Optional[Dict[str, Any]]:
        cid = str(cam_id)
        H, W = int(depth.shape[0]), int(depth.shape[1])

        if self.last_cam_id and cid != self.last_cam_id:
            self.reset()
        if self.last_hw != (0, 0) and (H, W) != self.last_hw:
            self.reset()
        if self.last_step_id >= 0 and int(step_id) <= int(self.last_step_id):
            self.reset()
        if np.isfinite(self.last_ts_s) and float(ts_s) <= float(self.last_ts_s):
            self.reset()

        self.last_cam_id = cid
        self.last_hw = (H, W)
        self.last_step_id = int(step_id)
        self.last_ts_s = float(ts_s)

        intr_pw = PWPinholeIntrinsics(fx=float(intr.fx), fy=float(intr.fy), cx=float(intr.cx), cy=float(intr.cy))
        frame = PWCameraFrame(
            rgb=np.asarray(rgb),
            depth=np.asarray(depth),
            intrinsics=intr_pw,
            extrinsics=np.asarray(T_base_cam, dtype=np.float32).reshape(4, 4),
        )

        q7 = np.asarray(q, dtype=np.float32).reshape(-1)
        if q7.shape[0] < 7:
            raise ValueError("q must have at least 7 elements")
        q7 = q7[:7]

        self.window.push_frame(
            cameras={cid: frame},
            joint_positions=q7,
            gripper_positions=np.asarray([float(gripper)], dtype=np.float32),
            timestamp=float(ts_s),
        )

        if not self.window.is_ready():
            return None

        scene = self.builder.build(window_shift=1, robot_spheres_base=None)
        steps = self.window.get_window()
        ts = np.asarray([float(s.timestamp) for s in steps], dtype=np.float64)
        q_win = np.stack([np.asarray(s.joint_positions, dtype=np.float32).reshape(-1)[:7] for s in steps], axis=0)
        g_win = np.asarray([float(np.asarray(s.gripper_positions).reshape(-1)[0]) for s in steps], dtype=np.float32)

        obs: Dict[str, Any] = {
            "scene_flows": np.asarray(scene.scene_flows, dtype=np.float32),
            "scene_exists": np.asarray(scene.scene_exists, dtype=bool),
            "scene_track_confidence": np.asarray(scene.scene_track_confidence, dtype=np.float32),
            "scene_colors": np.asarray(scene.scene_colors, dtype=np.uint8),
            "cameras_used": np.asarray(list(scene.cameras_used)),
            "camera_track_slices": np.asarray(scene.camera_track_slices, dtype=np.int32),
            "camera_track_ids": np.asarray(scene.camera_track_ids, dtype=np.int32),
            "timestamps": ts,
            "joint_positions_window": q_win,
            "gripper_positions_window": g_win,
        }
        self.last_pointworld_obs = obs
        return obs


def _build_pointworld_runtime() -> Optional[_PointWorldRuntime]:
    if not _env_bool("MPPI_PW_ENABLE", "0"):
        return None

    ckpt = os.getenv("MPPI_PW_COTRACKER_CKPT", "").strip()
    if not ckpt:
        raise ValueError("MPPI_PW_COTRACKER_CKPT is required when MPPI_PW_ENABLE=1")

    device = os.getenv("MPPI_PW_COTRACKER_DEVICE", "cuda").strip() or None

    max_q = _env_i("MPPI_PW_MAX_QUERY_POINTS_PER_CAMERA", "2048")
    min_conf = float(_env_f("MPPI_PW_MIN_TRACK_CONFIDENCE", "0.0"))
    rng_seed = _env_i("MPPI_PW_QUERY_RNG_SEED", "0")

    ws_min = _env_vec3("MPPI_PW_WORKSPACE_MIN", "-0.1,-0.7,-0.05")
    ws_max = _env_vec3("MPPI_PW_WORKSPACE_MAX", "1.2,0.7,1.2")

    seed_robot_mask_enabled = _env_bool("MPPI_PW_SEED_ROBOT_MASK_ENABLED", "0")
    robot_mask_seed = int(os.getenv("MPPI_PW_ROBOT_MASK_SEED", "0"))

    ee_filter_enabled = _env_bool("MPPI_PW_EE_FILTER_ENABLED", "0")
    ee_filter_link = os.getenv("MPPI_PW_EE_FILTER_LINK", "panda_link7")
    ee_filter_spheres = parse_spheres_spec(os.getenv("MPPI_PW_EE_FILTER_SPHERES", ""))

    urdf_path = os.getenv("MPPI_PW_URDF_PATH", os.getenv("MPPI_URDF_PATH", "")).strip() or None

    tracking = TrackingConfig(max_query_points_per_camera=int(max_q), min_track_confidence=float(min_conf))
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
        camera_selection="all_available",
        min_cameras=1,
    )

    window = PointWorldWindowBuffer(window_size=11)
    query_manager = QueryPointManager(cfg=QueryPointManagerConfig(max_query_points_per_camera=int(max_q), min_track_confidence=float(min_conf), rng_seed=int(rng_seed)))
    tracker = CoTrackerOnlinePointTracker(checkpoint=str(ckpt), window_len=11, device=device)
    builder = OnlineSceneFlowBuilder(cfg=cfg, window_buffer=window, tracker=tracker, query_manager=query_manager)

    cost_model: Optional[PointWorldCostModel] = None
    if _env_bool("MPPI_USE_POINTWORLD_COST", "0"):
        model_path = os.getenv("MPPI_PW_MODEL_PATH", "").strip()
        if not model_path:
            raise ValueError("MPPI_PW_MODEL_PATH is required when MPPI_USE_POINTWORLD_COST=1")
        urdf_path = os.getenv("MPPI_PW_URDF_PATH", os.getenv("MPPI_URDF_PATH", "")).strip() or default_urdf_path()

        cost_cfg = PointWorldCostConfig(
            mode=str(os.getenv("MPPI_PW_COST_MODE", "flow_l2")),
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
                cost=cost_cfg,
            )
        )

    return _PointWorldRuntime(
        cfg=cfg,
        window=window,
        builder=builder,
        query_manager=query_manager,
        cost_model=cost_model,
    )


async def _handle_connection(
    ws: Any,
    cfg: ServerConfig,
    cam_configs: dict[str, tuple[Any, np.ndarray]],
    pw: Optional[_PointWorldRuntime],
) -> None:
    while True:
        try:
            data = await asyncio.wait_for(ws.recv(), timeout=cfg.request_timeout_s)
        except asyncio.TimeoutError:
            continue
        except Exception:
            break

        t_server_recv_ns = time.time_ns()
        t0 = time.perf_counter()

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

            intr, T_base_cam = parse_obs_camera_params(
                cam_id=obs.cam_id,
                intrinsics=obs.intrinsics,
                T_base_cam=obs.T_base_cam,
                cam_configs=cam_configs,
            )

            depth_unit_scale = float(obs.depth_unit_scale) if obs.depth_unit_scale is not None else _env_f("MPPI_PCL_DEPTH_UNIT_SCALE", "1.0")
            depth_min_m = _env_f("MPPI_PCL_DEPTH_MIN_M", "0.05")
            depth_max_m = _env_f("MPPI_PCL_DEPTH_MAX_M", "2.0")
            stride = _env_i("MPPI_PCL_STRIDE", "1")

            if obs.rgb_bytes is not None:
                rgb = _decode_rgb(codec=obs.rgb_codec, data=obs.rgb_bytes)
            elif obs.rgb_back is not None:
                rgb = np.asarray(obs.rgb_back)
            else:
                raise ValueError("Missing rgb payload: provide rgb_bytes or rgb_back")

            if obs.depth_bytes is not None:
                depth = _decode_depth(codec=obs.depth_codec, data=obs.depth_bytes)
            elif obs.depth_back is not None:
                depth = np.asarray(obs.depth_back)
            else:
                raise ValueError("Missing depth payload: provide depth_bytes or depth_back")

            if pw is not None and bool(pw.enabled):
                ts_s = float(obs.t_client_send_ns) * 1e-9
                try:
                    pw.push_and_maybe_build(
                        cam_id=str(obs.cam_id) if obs.cam_id is not None else str(cfg.cam_id),
                        step_id=int(obs.step_id),
                        ts_s=float(ts_s),
                        rgb=np.asarray(rgb),
                        depth=np.asarray(depth),
                        intr=intr,
                        T_base_cam=T_base_cam,
                        q=np.asarray(obs.q, dtype=np.float32),
                        gripper=float(obs.gripper),
                    )
                except Exception:
                    pw.reset()

            pcd_base = rgbd_to_pointcloud_base(
                depth=np.asarray(depth),
                rgb=np.asarray(rgb),
                intr=intr,
                T_base_cam=T_base_cam,
                depth_unit_scale=float(depth_unit_scale),
                depth_min_m=float(depth_min_m),
                depth_max_m=float(depth_max_m),
                stride=int(stride),
                roi_min=None,
                roi_max=None,
                voxel_size_m=0.0,
            )

            timing_policy = cfg.policy
            if cfg.policy == "dummy_hold":
                actions = _make_actions_dummy_hold(obs.q, float(obs.gripper), cfg.open_loop_horizon)
            elif cfg.policy == "mppi_joint":
                solver = _get_joint_solver(cfg.open_loop_horizon)
                pw_obs = pw.last_pointworld_obs if (pw is not None and bool(pw.enabled)) else None
                pw_cost_fn: Optional[PointWorldCostFn] = (
                    pw.cost_model if (pw is not None and bool(pw.enabled) and pw.cost_model is not None) else None
                )
                actions = solver.infer_actions(
                    q0=obs.q,
                    gripper=float(obs.gripper),
                    pcd_back_cam=pcd_base,
                    pointworld_obs=pw_obs,
                    pointworld_cost_fn=pw_cost_fn,
                )

                if _env_bool("MPPI_PCL_SAVE_PCD", "0") and getattr(solver, "cfg", None) is not None:
                    scfg = solver.cfg

                    checker = get_curobo_collision_checker(
                        CuRoboCollisionConfig(
                            device=str(scfg.curobo_device),
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
            await ws.send(encode_message(resp))
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

    if cfg.cam_id not in cam_configs:
        raise RuntimeError(
            f"Missing cam config for cam_id={cfg.cam_id}. "
            f"Set MPPI_PCL_CAM_INFO_{cfg.cam_id.upper()}_PATH and MPPI_PCL_T_BASE_CAM_{cfg.cam_id.upper()}_PATH."
        )

    async def handler(ws: Any) -> None:
        await _handle_connection(ws, cfg, cam_configs, pw)

    async with websockets.serve(handler, cfg.host, cfg.port, max_size=None):
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
