from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional

from mppi.mpc.solver import JointMPPIConfig, JointMPPISolver, PointWorldCostFn
from mppi.protocol.msgpack_codec import decode_message, encode_message
from mppi.protocol.types import (
    ActionChunkV1,
    ErrorV1,
    InferResponseV1,
    ObsV1,
    SCHEMA_VERSION_V1,
    ServerTimingV1,
)
from mppi.utils.paths import default_urdf_path, repo_path

_JOINT_SOLVER: JointMPPISolver | None = None


def _require_websockets():
    try:
        import websockets  # type: ignore
        from websockets.server import WebSocketServerProtocol  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: websockets. Install it in the container env.") from e
    return websockets, WebSocketServerProtocol


def _optional_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    return np


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 9010
    path: Optional[str] = None
    open_loop_horizon: int = 8
    policy: str = "dummy_hold"
    request_timeout_s: float = 2.0


def _get_joint_solver(horizon: int) -> JointMPPISolver:
    import os

    def _env_bool(name: str, default: str = "0") -> bool:
        return os.getenv(name, default).strip().lower() in ("1", "true", "yes")

    def _env_f(name: str, default: str) -> float:
        return float(os.getenv(name, default))

    def _env_i(name: str, default: str) -> int:
        return int(os.getenv(name, default))

    def _env_vec3(name: str, default: str) -> tuple[float, float, float]:
        s = os.getenv(name, default).strip()
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"{name} must be 'x,y,z'")
        return (float(parts[0]), float(parts[1]), float(parts[2]))

    global _JOINT_SOLVER

    ee_goal_env = os.getenv("MPPI_EE_POS_GOAL", "").strip()
    ee_goal = None
    if ee_goal_env:
        parts = [p.strip() for p in ee_goal_env.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("MPPI_EE_POS_GOAL must be 'x,y,z'")
        ee_goal = (float(parts[0]), float(parts[1]), float(parts[2]))

    link7_goal_env = os.getenv("MPPI_LINK7_POS_GOAL", "").strip()
    link7_goal = None
    if link7_goal_env:
        parts = [p.strip() for p in link7_goal_env.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("MPPI_LINK7_POS_GOAL must be 'x,y,z'")
        link7_goal = (float(parts[0]), float(parts[1]), float(parts[2]))

    w_ee_pos = float(os.getenv("MPPI_W_EE_POS", "0"))
    w_link7_pos = float(os.getenv("MPPI_W_LINK7_POS", "0"))

    use_pointworld_cost = _env_bool("MPPI_USE_POINTWORLD_COST", "0")
    w_pointworld = float(os.getenv("MPPI_W_POINTWORLD", "0"))
    pw_timeout_ms = float(os.getenv("MPPI_POINTWORLD_COST_TIMEOUT_MS", "0"))
    pw_require_h11 = _env_bool("MPPI_POINTWORLD_REQUIRE_HORIZON_11", "1")

    urdf_path = os.getenv(
        "MPPI_URDF_PATH",
        default_urdf_path(),
    )
    ee_link = os.getenv("MPPI_EE_LINK", "robotiq_85_base_link")
    link7_link = os.getenv("MPPI_LINK7_LINK", "panda_link7")

    use_curobo = _env_bool("MPPI_USE_CUROBO_COLLISION", "0")
    w_self = float(os.getenv("MPPI_W_SELF_COLLISION", "1.0"))
    w_scene = float(os.getenv("MPPI_W_SCENE_COLLISION", "1.0"))
    curobo_device = os.getenv("MPPI_CUROBO_DEVICE", "cuda:0")
    curobo_robot_yaml = os.getenv("MPPI_CUROBO_ROBOT_YAML", "franka.yml")
    curobo_tool_frame = os.getenv("MPPI_CUROBO_TOOL_FRAME", str(ee_link))
    curobo_with_world = _env_bool("MPPI_CUROBO_WITH_WORLD", "1")

    scene_from_pcd = _env_bool("MPPI_SCENE_FROM_PCD_BACK_CAM", "0")
    scene_pcd_scale = _env_f("MPPI_SCENE_PCD_SCALE", "1.0")
    scene_pcd_in_base = _env_bool("MPPI_SCENE_PCD_IN_BASE", "1")

    scene_add_table = _env_bool("MPPI_SCENE_ADD_TABLE", "1")
    scene_table_dims = _env_vec3("MPPI_SCENE_TABLE_DIMS", "2.0,2.0,0.2")
    scene_table_center = _env_vec3("MPPI_SCENE_TABLE_CENTER", "0.4,0.0,-0.1")

    scene_remove_table_points = _env_bool("MPPI_SCENE_REMOVE_TABLE_POINTS", "1")
    scene_table_eps_m = _env_f("MPPI_SCENE_TABLE_EPS_M", "0.01")

    scene_remove_wall_points = _env_bool("MPPI_SCENE_REMOVE_WALL_POINTS", "1")
    scene_wall_dims = _env_vec3("MPPI_SCENE_WALL_DIMS", "2.5,0.5,2.0")
    scene_wall_center = _env_vec3("MPPI_SCENE_WALL_CENTER", "0.5,0.5,-0.5")
    scene_wall_margin_m = _env_f("MPPI_SCENE_WALL_MARGIN_M", "0.05")

    t_base_cam_back_path = os.getenv("MPPI_T_BASE_CAM_BACK_PATH", repo_path("configs", "T_base_cam.yaml"))
    scene_roi_min = _env_vec3("MPPI_SCENE_ROI_MIN", "-0.1,-0.7,-0.05")
    scene_roi_max = _env_vec3("MPPI_SCENE_ROI_MAX", "1.2,0.7,1.2")
    scene_voxel_size_m = _env_f("MPPI_SCENE_VOXEL_SIZE_M", "0.01")
    scene_padding_m = _env_f("MPPI_SCENE_PADDING_M", "0.02")
    scene_max_cuboids = _env_i("MPPI_SCENE_MAX_CUBOIDS", "20")
    scene_robot_mask_margin_m = _env_f("MPPI_SCENE_ROBOT_MASK_MARGIN_M", "0.02")
    scene_min_cluster_voxels = _env_i("MPPI_SCENE_MIN_CLUSTER_VOXELS", "10")

    scene_track_alpha = _env_f("MPPI_SCENE_TRACK_ALPHA", "0.6")
    scene_track_remove_after_misses = _env_i("MPPI_SCENE_TRACK_REMOVE_AFTER_MISSES", "5")
    scene_track_max_tracks = _env_i("MPPI_SCENE_TRACK_MAX_TRACKS", "20")
    scene_track_match_center_dist_m = _env_f("MPPI_SCENE_TRACK_MATCH_CENTER_DIST_M", "0.10")
    scene_track_match_iou_min = _env_f("MPPI_SCENE_TRACK_MATCH_IOU_MIN", "0.05")

    infer_budget_ms = _env_f("MPPI_INFER_BUDGET_MS", "0")
    budget_max_dynamic_cuboids = _env_i("MPPI_BUDGET_MAX_DYNAMIC_CUBOIDS", "0")
    num_samples = _env_i("MPPI_NUM_SAMPLES", "256")

    debug_cost_stats = _env_bool("MPPI_DEBUG_COST_STATS", "0")
    debug_cost_stats_q = _env_f("MPPI_DEBUG_COST_STATS_Q", "0.5")

    if bool(use_pointworld_cost) and bool(pw_require_h11) and int(horizon) != 11:
        raise ValueError(f"PointWorld requires horizon=11, got {int(horizon)}. Set open_loop_horizon=11.")

    if _JOINT_SOLVER is None or _JOINT_SOLVER.cfg.horizon != int(horizon):
        cfg = JointMPPIConfig(
            horizon=int(horizon),
            num_samples=int(num_samples),
            w_ee_pos=w_ee_pos,
            ee_pos_goal=ee_goal,
            w_link7_pos=w_link7_pos,
            link7_pos_goal=link7_goal,
            use_pointworld_cost=bool(use_pointworld_cost),
            w_pointworld=float(w_pointworld),
            pointworld_require_horizon_11=bool(pw_require_h11),
            pointworld_cost_timeout_ms=float(pw_timeout_ms),
            urdf_path=str(urdf_path),
            ee_link=str(ee_link),
            link7_link=str(link7_link),
            use_curobo_collision=bool(use_curobo),
            w_self_collision=float(w_self),
            w_scene_collision=float(w_scene),
            curobo_device=str(curobo_device),
            curobo_robot_yaml=str(curobo_robot_yaml),
            curobo_tool_frame=str(curobo_tool_frame),
            curobo_with_world=bool(curobo_with_world),
            scene_from_pcd_back_cam=bool(scene_from_pcd),
            scene_pcd_scale=float(scene_pcd_scale),
            scene_pcd_in_base=bool(scene_pcd_in_base),
            scene_add_table=bool(scene_add_table),
            scene_table_dims=scene_table_dims,
            scene_table_center=scene_table_center,
            scene_remove_table_points=bool(scene_remove_table_points),
            scene_table_eps_m=float(scene_table_eps_m),
            scene_remove_wall_points=bool(scene_remove_wall_points),
            scene_wall_dims=scene_wall_dims,
            scene_wall_center=scene_wall_center,
            scene_wall_margin_m=float(scene_wall_margin_m),
            t_base_cam_back_path=str(t_base_cam_back_path),
            scene_roi_min=scene_roi_min,
            scene_roi_max=scene_roi_max,
            scene_voxel_size_m=float(scene_voxel_size_m),
            scene_padding_m=float(scene_padding_m),
            scene_max_cuboids=int(scene_max_cuboids),
            scene_robot_mask_margin_m=float(scene_robot_mask_margin_m),
            scene_min_cluster_voxels=int(scene_min_cluster_voxels),
            scene_track_alpha=float(scene_track_alpha),
            scene_track_remove_after_misses=int(scene_track_remove_after_misses),
            scene_track_max_tracks=int(scene_track_max_tracks),
            scene_track_match_center_dist_m=float(scene_track_match_center_dist_m),
            scene_track_match_iou_min=float(scene_track_match_iou_min),
            infer_budget_ms=float(infer_budget_ms),
            budget_max_dynamic_cuboids=int(budget_max_dynamic_cuboids),
            debug_cost_stats=bool(debug_cost_stats),
            debug_cost_stats_q=float(debug_cost_stats_q),
        )
        _JOINT_SOLVER = JointMPPISolver(cfg)
    return _JOINT_SOLVER


def _make_actions_dummy_hold(obs: ObsV1, horizon: int) -> Any:
    np = _optional_numpy()
    row = list(obs.q) + [float(obs.gripper)]
    if np is None:
        return [row[:] for _ in range(horizon)]
    return np.asarray([row] * horizon, dtype=np.float32)


async def _handle_connection(ws: Any, cfg: ServerConfig) -> None:
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
            if int(envelope.get("schema_version", -1)) != SCHEMA_VERSION_V1:
                raise ValueError(f"Unsupported schema_version: {envelope.get('schema_version')}")
            if envelope.get("type") != "infer_request":
                raise ValueError(f"Unexpected message type: {envelope.get('type')}")
            request_id = str(envelope["request_id"])
            obs = ObsV1.from_payload(dict(envelope["payload"]))

            timing_policy = cfg.policy
            if cfg.policy == "dummy_hold":
                actions = _make_actions_dummy_hold(obs, cfg.open_loop_horizon)
            elif cfg.policy == "mppi_joint":
                solver = _get_joint_solver(cfg.open_loop_horizon)
                actions = solver.infer_actions(
                    q0=obs.q,
                    gripper=float(obs.gripper),
                    pcd_back_cam=obs.pcd_back_cam,
                )

                use_curobo = bool(getattr(solver, "cfg", None) and getattr(solver.cfg, "use_curobo_collision", False))
                ess_ratio = float(getattr(solver, "last_effective_samples_ratio", 0.0) or 0.0)
                n_cub = int(getattr(solver, "last_scene_num_cuboids", 0) or 0)
                n_sph = int(getattr(solver, "last_scene_num_robot_spheres", 0) or 0)
                has_table = bool(getattr(solver, "last_scene_has_table", False))

                stable_trk = None
                scene_key_short = None
                try:
                    stable_trk = int(getattr(solver, "last_scene_num_dynamic_tracks", 0) or 0)
                    scene_key_short = str(getattr(solver, "last_scene_key_short", "") or "")
                except Exception:
                    stable_trk = 0
                    scene_key_short = ""

                timing_policy = f"mppi_joint+{'curobo' if use_curobo else 'nocurobo'}+ess{ess_ratio:.3f}+tab{1 if has_table else 0}+cub{n_cub}+trk{int(stable_trk)}+key{scene_key_short}+sph{n_sph}"

                suffix = str(getattr(solver, "last_timing_policy_suffix", "") or "")
                if suffix:
                    timing_policy = f"{timing_policy}{suffix}"

                if bool(getattr(getattr(solver, "cfg", None), "debug_cost_stats", False)):
                    stats = getattr(solver, "last_cost_stats", None)
                    dsc = getattr(solver, "last_min_distance_scene", None)
                    dsl = getattr(solver, "last_min_distance_self", None)

                    tokens: list[str] = []
                    if isinstance(stats, dict):
                        for k in (
                            "cost_smooth_q50",
                            "cost_action_q50",
                            "cost_joint_limit_q50",
                            "cost_ee_pos_q50",
                            "cost_link7_pos_q50",
                            "cost_scene_q50",
                            "cost_self_q50",
                            "cost_pointworld_q50",
                        ):
                            if k in stats:
                                try:
                                    tokens.append(f"{k}={float(stats[k]):.3g}")
                                except Exception:
                                    pass
                    try:
                        if dsc is not None and float(dsc) == float(dsc):
                            tokens.append(f"dmin_scene={float(dsc):.3g}")
                    except Exception:
                        pass
                    try:
                        if dsl is not None and float(dsl) == float(dsl):
                            tokens.append(f"dmin_self={float(dsl):.3g}")
                    except Exception:
                        pass

                    if tokens:
                        timing_policy = timing_policy + ":dbg{" + ",".join(tokens) + "}"

                if getattr(solver, "last_fallback", False):
                    reason = str(getattr(solver, "last_fallback_reason", ""))
                    actions = _make_actions_dummy_hold(obs, cfg.open_loop_horizon)
                    timing_policy = f"{timing_policy}:fallback_hold:{reason}"
            else:
                raise ValueError(f"Unknown policy: {cfg.policy}")

            t1 = time.perf_counter()
            t_server_send_ns = time.time_ns()
            timing = ServerTimingV1(infer_ms=(t1 - t0) * 1000.0, queue_ms=0.0, policy=timing_policy)
            chunk = ActionChunkV1(
                t_server_recv_ns=t_server_recv_ns,
                t_server_send_ns=t_server_send_ns,
                t_client_send_ns_echo=obs.t_client_send_ns,
                open_loop_horizon=cfg.open_loop_horizon,
                actions=actions,
                server_timing=timing,
            )
            resp = InferResponseV1(request_id=request_id, action_chunk=chunk).to_envelope()
            await ws.send(encode_message(resp))
        except Exception as e:  # noqa: BLE001
            try:
                request_id = str(envelope.get("request_id", "")) if "envelope" in locals() else ""
                err = ErrorV1(
                    request_id=request_id,
                    code="bad_request",
                    message=str(e),
                    t_server_send_ns=time.time_ns(),
                ).to_envelope()
                await ws.send(encode_message(err))
            except Exception:
                break


async def serve(cfg: ServerConfig) -> None:
    websockets, _ = _require_websockets()

    async def handler(ws: Any) -> None:
        await _handle_connection(ws, cfg)

    if cfg.path:
        async with websockets.serve(handler, cfg.host, cfg.port, process_request=None):
            await asyncio.Future()
    else:
        async with websockets.serve(handler, cfg.host, cfg.port):
            await asyncio.Future()


def main(host: str = "0.0.0.0", port: int = 9010, open_loop_horizon: int = 8, policy: str = "dummy_hold") -> None:
    cfg = ServerConfig(host=host, port=port, open_loop_horizon=open_loop_horizon, policy=policy)
    asyncio.run(serve(cfg))


if __name__ == "__main__":
    main()
