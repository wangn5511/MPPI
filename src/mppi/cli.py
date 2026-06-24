from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional


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

    try:
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: curobo (GPU env only)") from e

    if os.path.isfile(robot_yaml):
        robot_yaml_path = robot_yaml
    else:
        robot_yaml_path = join_path(get_robot_configs_path(), robot_yaml)

    loaded = load_yaml(robot_yaml_path)
    if not isinstance(loaded, dict):
        raise ValueError("Invalid robot yaml: expected mapping")

    if "robot_cfg" in loaded:
        robot_cfg_dict: Dict[str, Any] = loaded
    else:
        robot_cfg_dict = {"robot_cfg": loaded}

    kin = robot_cfg_dict.get("robot_cfg", {}).get("kinematics")
    if not isinstance(kin, dict):
        raise ValueError("Invalid robot yaml: missing robot_cfg.kinematics")

    kin["urdf_path"] = str(urdf_path)

    link_names: set[str] | None = None
    joint_names: set[str] | None = None
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        ln: set[str] = set()
        jn: set[str] = set()
        for el in root.iter():
            tag = str(el.tag)
            if tag.endswith("link"):
                name = el.attrib.get("name")
                if name:
                    ln.add(str(name))
            elif tag.endswith("joint"):
                name = el.attrib.get("name")
                if name:
                    jn.add(str(name))
        link_names = ln if len(ln) > 0 else None
        joint_names = jn if len(jn) > 0 else None
    except Exception:
        link_names = None
        joint_names = None

    requested_tool = str(tool_frame).strip() or "robotiq_85_base_link"
    if link_names is not None:
        fallback_order = [
            requested_tool,
            "robotiq_85_base_link",
            "camera_mount_link",
            "panda_link8",
            "panda_link7",
            str(kin.get("base_link", "")).strip(),
        ]
        chosen = next((x for x in fallback_order if x and x in link_names), None)
        if chosen is None:
            chosen = next(iter(sorted(link_names)))
        kin["tool_frames"] = [chosen]
    else:
        kin["tool_frames"] = [requested_tool]

    if "extra_links" in kin:
        kin.pop("extra_links", None)
    if "extra_collision_spheres" in kin:
        kin.pop("extra_collision_spheres", None)

    if "lock_joints" in kin and joint_names is not None:
        lock = kin.get("lock_joints")
        if isinstance(lock, dict):
            kin["lock_joints"] = {k: v for k, v in lock.items() if str(k) in joint_names}

    if link_names is not None:
        for k_list in ("collision_link_names", "mesh_link_names", "grasp_contact_link_names"):
            v = kin.get(k_list)
            if isinstance(v, list):
                kin[k_list] = [x for x in v if str(x) in link_names]

        v = kin.get("collision_spheres")
        if isinstance(v, dict):
            kin["collision_spheres"] = {k: vv for k, vv in v.items() if str(k) in link_names}

        v = kin.get("self_collision_buffer")
        if isinstance(v, dict):
            kin["self_collision_buffer"] = {k: vv for k, vv in v.items() if str(k) in link_names}

        v = kin.get("self_collision_ignore")
        if isinstance(v, dict):
            cleaned: Dict[str, Any] = {}
            for kk, vv in v.items():
                if str(kk) not in link_names or not isinstance(vv, list):
                    continue
                kept = [x for x in vv if str(x) in link_names]
                if kept:
                    cleaned[str(kk)] = kept
            kin["self_collision_ignore"] = cleaned

    tf = kin.get("tool_frames")
    if not isinstance(tf, list) or len(tf) == 0:
        kin["tool_frames"] = [str(tool_frame).strip() or "robotiq_85_base_link"]

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

    p_client = sub.add_parser("client")
    p_client.add_argument("--url", type=str, default=None)
    p_client.add_argument("--run-seconds", type=float, default=None)
    p_client.add_argument("--control-hz", type=float, default=None)
    p_client.add_argument("--open-loop-horizon", type=int, default=None)
    p_client.add_argument("--request-timeout-s", type=float, default=None)
    p_client.add_argument("--gripper", type=float, default=0.0)
    p_client.add_argument("--initial-q", type=str, default="")
    p_client.add_argument("--pcd-npz", type=str, default="")
    p_client.add_argument("--pcd-a", type=str, default="")
    p_client.add_argument("--pcd-b", type=str, default="")
    p_client.add_argument("--ab-alternate", action="store_true")

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
        from mppi.comm.ws_server_async import main as server_main

        host = args.host if args.host is not None else _get(cfg, "server.host", "0.0.0.0")
        port = args.port if args.port is not None else int(_get(cfg, "server.port", 9010))
        horizon = (
            args.open_loop_horizon
            if args.open_loop_horizon is not None
            else int(_get(cfg, "control.open_loop_horizon", 8))
        )
        policy = args.policy if args.policy is not None else str(_get(cfg, "policy.name", "dummy_hold"))
        server_main(host=host, port=port, open_loop_horizon=horizon, policy=policy)
        return

    if args.cmd == "client":
        from mppi.comm.ws_client_sync import main as client_main

        def _parse_q_csv(s: str) -> list[float]:
            parts = [p.strip() for p in str(s).split(",") if p.strip()]
            if len(parts) != 7:
                raise ValueError("--initial-q must be 7 comma-separated floats")
            return [float(x) for x in parts]

        url = args.url if args.url is not None else str(_get(cfg, "client.url", "ws://127.0.0.1:9010"))
        run_seconds = (
            float(args.run_seconds)
            if args.run_seconds is not None
            else float(_get(cfg, "client.run_seconds", 60.0))
        )
        control_hz = (
            float(args.control_hz)
            if args.control_hz is not None
            else float(_get(cfg, "control.frequency", 20.0))
        )
        horizon = (
            int(args.open_loop_horizon)
            if args.open_loop_horizon is not None
            else int(_get(cfg, "control.open_loop_horizon", 8))
        )
        request_timeout_s = (
            float(args.request_timeout_s)
            if args.request_timeout_s is not None
            else float(_get(cfg, "client.request_timeout_s", 2.0))
        )

        q0 = _parse_q_csv(str(args.initial_q)) if str(args.initial_q).strip() else None

        client_main(
            url=str(url),
            run_seconds=float(run_seconds),
            control_hz=float(control_hz),
            open_loop_horizon=int(horizon),
            request_timeout_s=float(request_timeout_s),
            gripper=float(args.gripper),
            initial_q=q0,
            pcd_npz=(str(args.pcd_npz) if str(args.pcd_npz).strip() else None),
            pcd_a_npz=(str(args.pcd_a) if str(args.pcd_a).strip() else None),
            pcd_b_npz=(str(args.pcd_b) if str(args.pcd_b).strip() else None),
            ab_alternate=bool(args.ab_alternate),
        )
        return

    if args.cmd == "curobo-smoke":
        urdf_default = os.getenv(
            "MPPI_URDF_PATH",
            "/home/wangyuhan/PointWorld/assets/franka_description/franka_panda_robotiq_2f85.urdf",
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