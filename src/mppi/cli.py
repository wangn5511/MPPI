from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from mppi.curobo_ext.collision_checker import _load_robot_cfg_dict, _require_curobo_v2, _sanitize_kinematics_cfg
from mppi.utils.paths import default_urdf_path

def _load_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json",):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Config is YAML but PyYAML is not installed.") from e
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("YAML root must be a mapping.")
        return obj
    raise ValueError(f"Unsupported config extension: {ext}")


def _get(d: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _run_curobo_smoke(
    *,
    device: str,
    robot_yaml: str,
    urdf_path: str,
    tool_frame: str,
    batch: int,
    horizon: int,
    with_world: bool,
) -> None:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: torch (required by cuRobo)") from e

    (
        RobotCollisionChecker,
        RobotCollisionCheckerCfg,
        DeviceCfg,
        get_robot_configs_path,
        join_path,
        load_yaml,
    ) = _require_curobo_v2()

    if os.path.isfile(robot_yaml):
        robot_yaml_path = robot_yaml
    else:
        robot_yaml_path = join_path(get_robot_configs_path(), robot_yaml)

    robot_cfg_dict = _load_robot_cfg_dict(str(robot_yaml_path))
    robot_cfg_dict = _sanitize_kinematics_cfg(
        robot_cfg_dict=robot_cfg_dict,
        urdf_path=str(urdf_path),
        tool_frame=str(tool_frame),
    )

    device_cfg = DeviceCfg(device=torch.device(device), dtype=torch.float32)

    scene_cfg = None
    if with_world:
        scene_cfg = {
            "cuboid": {
                "table": {"dims": [2.0, 2.0, 0.2], "pose": [0.4, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0]},
                "cube_1": {"dims": [0.1, 0.1, 0.2], "pose": [0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]},
            }
        }

    cfg = RobotCollisionCheckerCfg.load_from_config(
        robot_config=robot_cfg_dict,
        scene_model=scene_cfg,
        device_cfg=device_cfg,
        collision_activation_distance=0.2,
        self_collision_activation_distance=0.0,
        max_collision_distance=1.0,
    )
    checker = RobotCollisionChecker(cfg)

    with torch.no_grad():
        q_traj = checker.sample_trajectory(batch=int(batch), horizon=int(horizon), mask_valid=False)
        world_d, self_d = checker.get_scene_self_collision_distance_from_joint_trajectory(q_traj)
        kin_state = checker.get_kinematics(q_traj)

    if device_cfg.device.type == "cuda":
        torch.cuda.synchronize(device_cfg.device)

    urdf_used = robot_cfg_dict.get("robot_cfg", {}).get("kinematics", {}).get("urdf_path")

    print("[curobo-smoke] ok")
    print(f"[curobo-smoke] device={device}")
    print(f"[curobo-smoke] robot_yaml={robot_yaml_path}")
    print(f"[curobo-smoke] urdf_path_input={urdf_path}")
    print(f"[curobo-smoke] urdf_path_cfg={urdf_used}")
    print(f"[curobo-smoke] tool_frames={robot_cfg_dict.get('robot_cfg', {}).get('kinematics', {}).get('tool_frames')}")
    print(f"[curobo-smoke] q_traj shape={tuple(q_traj.shape)}")
    print(
        f"[curobo-smoke] scene_dist shape={tuple(world_d.shape)} min={float(torch.min(world_d))} max={float(torch.max(world_d))}"
    )
    print(
        f"[curobo-smoke] self_dist shape={tuple(self_d.shape)} min={float(torch.min(self_d))} max={float(torch.max(self_d))}"
    )

    spheres = getattr(kin_state, "robot_spheres", None)
    if spheres is not None:
        print(f"[curobo-smoke] robot_spheres shape={tuple(spheres.shape)}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="mppi")
    parser.add_argument("--config", type=str, default=None)

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_server = sub.add_parser("server")
    p_server.add_argument("--host", type=str, default=None)
    p_server.add_argument("--port", type=int, default=None)
    p_server.add_argument("--open-loop-horizon", type=int, default=None)
    p_server.add_argument("--policy", type=str, default=None)
    p_server.add_argument("--cam-id", type=str, default=None)

    p_client = sub.add_parser("client")
    p_client.add_argument("--url", type=str, default=None)
    p_client.add_argument("--rgb", type=str, default=None)
    p_client.add_argument("--depth", type=str, default=None)
    p_client.add_argument("--request-timeout-s", type=float, default=None)
    p_client.add_argument("--cam-id", type=str, default=None)
    p_client.add_argument("--depth-unit-scale", type=float, default=None)
    p_client.add_argument("--gripper", type=float, default=0.0)
    p_client.add_argument("--step-id", type=int, default=0)
    p_client.add_argument("--initial-q", type=str, default="")
    p_client.add_argument("--print-actions", action="store_true")

    p_smoke = sub.add_parser("curobo-smoke")
    p_smoke.add_argument("--device", type=str, default="cuda:0")
    p_smoke.add_argument("--robot-yaml", type=str, default="franka.yml")
    p_smoke.add_argument("--urdf", type=str, default=None)
    p_smoke.add_argument("--tool-frame", type=str, default=None)
    p_smoke.add_argument("--batch", type=int, default=32)
    p_smoke.add_argument("--horizon", type=int, default=8)
    p_smoke.add_argument("--no-world", action="store_true")

    args = parser.parse_args(argv)

    cfg: Dict[str, Any] = {}
    if args.config is not None:
        cfg = _load_config(args.config)

    if args.cmd == "server":
        from mppi.comm.ws_server_async_pcl import main as server_main

        host = args.host if args.host is not None else _get(cfg, "server.host", "0.0.0.0")
        port = args.port if args.port is not None else int(_get(cfg, "server.port", 9011))
        horizon = (
            args.open_loop_horizon
            if args.open_loop_horizon is not None
            else int(_get(cfg, "control.open_loop_horizon", 11))
        )
        policy = args.policy if args.policy is not None else str(_get(cfg, "policy.name", "mppi_joint"))
        cam_id = args.cam_id if args.cam_id is not None else str(_get(cfg, "camera.cam_id", "back"))
        server_main(["--host", str(host), "--port", str(int(port)), "--open-loop-horizon", str(int(horizon)), "--policy", str(policy), "--cam-id", str(cam_id)])
        return

    if args.cmd == "client":
        from mppi.comm.ws_client_sync_pcl import main as client_main

        def _parse_q_csv(s: str) -> list[float]:
            parts = [p.strip() for p in str(s).split(",") if p.strip()]
            if len(parts) != 7:
                raise ValueError("--initial-q must be 7 comma-separated floats")
            return [float(x) for x in parts]

        url = args.url if args.url is not None else str(_get(cfg, "client.url", "ws://127.0.0.1:9011"))
        rgb = args.rgb if args.rgb is not None else str(_get(cfg, "client.rgb", ""))
        depth = args.depth if args.depth is not None else str(_get(cfg, "client.depth", ""))
        if not str(rgb).strip() or not str(depth).strip():
            raise ValueError("PCL client requires --rgb and --depth")

        request_timeout_s = (
            float(args.request_timeout_s)
            if args.request_timeout_s is not None
            else float(_get(cfg, "client.request_timeout_s", 2.0))
        )
        cam_id = args.cam_id if args.cam_id is not None else str(_get(cfg, "camera.cam_id", "back"))
        q0 = _parse_q_csv(str(args.initial_q)) if str(args.initial_q).strip() else None
        q_csv = "" if q0 is None else ",".join(str(float(x)) for x in q0)

        argv2 = [
            "--url",
            str(url),
            "--rgb",
            str(rgb),
            "--depth",
            str(depth),
            "--request-timeout-s",
            str(float(request_timeout_s)),
            "--cam-id",
            str(cam_id),
            "--gripper",
            str(float(args.gripper)),
            "--step-id",
            str(int(args.step_id)),
        ]
        if args.depth_unit_scale is not None:
            argv2.extend(["--depth-unit-scale", str(float(args.depth_unit_scale))])
        if str(q_csv).strip():
            argv2.extend(["--initial-q", str(q_csv)])
        if bool(args.print_actions):
            argv2.append("--print-actions")

        client_main(argv2)
        return

    if args.cmd == "curobo-smoke":
        urdf_default = os.getenv(
            "MPPI_URDF_PATH",
            default_urdf_path(),
        )
        urdf_path = str(args.urdf) if args.urdf is not None else str(urdf_default)
        tool_frame = (
            str(args.tool_frame)
            if args.tool_frame is not None
            else str(os.getenv("MPPI_EE_LINK", "robotiq_85_base_link"))
        )
        _run_curobo_smoke(
            device=str(args.device),
            robot_yaml=str(args.robot_yaml),
            urdf_path=urdf_path,
            tool_frame=tool_frame,
            batch=int(args.batch),
            horizon=int(args.horizon),
            with_world=not bool(args.no_world),
        )
        return

    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    main()
