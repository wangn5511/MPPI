from __future__ import annotations

import argparse
import asyncio
import io
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from mppi.comm.ws_server_async import _get_joint_solver
from mppi.curobo_ext.check_depth_pcl import (
    load_T_row_major_4x4_yaml,
    load_intrinsics_from_cam_info_yaml,
    parse_obs_camera_params,
    rgbd_to_pointcloud_base,
)
from mppi.curobo_ext.collision_checker import CuRoboCollisionConfig, get_curobo_collision_checker
from mppi.curobo_ext.scene_builder import SceneBuildConfig, build_scene_points_base_and_colors_from_pcd_back_cam
from mppi.protocol.msgpack_codec import decode_message, encode_message
from mppi.protocol.types_pcl import (
    ActionChunkPCL,
    ErrorPCL,
    InferResponsePCL,
    ObsPCL,
    SCHEMA_VERSION_PCL,
    ServerTimingPCL,
)


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
        s = "/home/wangyuhan/MPPI/data/test/pcl_scene_filtered.npz"
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


async def _handle_connection(ws: Any, cfg: ServerConfig, cam_configs: dict[str, tuple[Any, np.ndarray]]) -> None:
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
                actions = solver.infer_actions(
                    q0=obs.q,
                    gripper=float(obs.gripper),
                    pcd_back_cam=pcd_base,
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

                    out_base = os.getenv("MPPI_PCL_SAVE_PCD_OUT", "/home/wangyuhan/MPPI/data/test/pcl_scene_filtered.npz")
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
                timing_policy = f"mppi_joint+{'curobo' if use_curobo else 'nocurobo'}+ess{ess_ratio:.3f}+tab{1 if has_table else 0}+cub{n_cub}+trk{int(stable_trk)}+key{scene_key_short}+sph{n_sph}"

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

    if cfg.cam_id not in cam_configs:
        raise RuntimeError(
            f"Missing cam config for cam_id={cfg.cam_id}. "
            f"Set MPPI_PCL_CAM_INFO_{cfg.cam_id.upper()}_PATH and MPPI_PCL_T_BASE_CAM_{cfg.cam_id.upper()}_PATH."
        )

    async def handler(ws: Any) -> None:
        await _handle_connection(ws, cfg, cam_configs)

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