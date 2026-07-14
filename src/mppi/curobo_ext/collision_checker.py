from __future__ import annotations

import os
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from mppi.utils.paths import default_urdf_path, ensure_sys_path_for_runtime

def _require_torch():
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: torch") from e
    return torch


def _require_torch_pk():
    torch = _require_torch()
    try:
        import pytorch_kinematics as pk
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: pytorch_kinematics") from e
    return torch, pk


_PK_FK_LOCK = threading.Lock()
_PK_FK_CACHE: dict[tuple[str, str], tuple[Any, Any, Any, tuple[str, ...]]] = {}


def _get_fk_chain_for_link(urdf_path: str, link_name: str):
    key = (str(urdf_path), str(link_name))
    with _PK_FK_LOCK:
        cached = _PK_FK_CACHE.get(key)
        if cached is not None:
            return cached

        torch, pk = _require_torch_pk()
        with open(str(urdf_path), "rb") as f:
            urdf_bytes = f.read()

        chain = pk.build_chain_from_urdf(bytes(urdf_bytes))
        chain = chain.to(device=torch.device("cpu"), dtype=torch.float32)
        frame_indices = chain.get_frame_indices(str(link_name))

        joint_names = None
        if hasattr(chain, "get_joint_parameter_names"):
            joint_names = list(chain.get_joint_parameter_names())
        elif hasattr(chain, "get_joint_names"):
            joint_names = list(chain.get_joint_names())

        if not joint_names:
            raise RuntimeError("Failed to query joint names from pytorch_kinematics chain")

        joint_names = tuple(str(n) for n in joint_names)
        cached = (torch, chain, frame_indices, joint_names)
        _PK_FK_CACHE[key] = cached
        return cached


def _fk_T_base_link(urdf_path: str, q7: np.ndarray, link_name: str) -> np.ndarray:
    torch, chain, frame_indices, joint_names = _get_fk_chain_for_link(str(urdf_path), str(link_name))

    q = np.asarray(q7, dtype=np.float32).reshape(1, 7)
    q_t = torch.from_numpy(q).to(device=torch.device("cpu"), dtype=torch.float32)
    zeros = torch.zeros((q_t.shape[0],), device=q_t.device, dtype=q_t.dtype)
    joint_dict = {name: zeros for name in joint_names}
    joint_dict.update(
        {
            "panda_joint1": q_t[:, 0],
            "panda_joint2": q_t[:, 1],
            "panda_joint3": q_t[:, 2],
            "panda_joint4": q_t[:, 3],
            "panda_joint5": q_t[:, 4],
            "panda_joint6": q_t[:, 5],
            "panda_joint7": q_t[:, 6],
        }
    )

    with torch.no_grad():
        tf = chain.forward_kinematics(joint_dict, frame_indices=frame_indices)
        T = tf[str(link_name)].get_matrix()

    return T[0].detach().cpu().numpy().astype(np.float32)


def fk_T_base_link(*, urdf_path: str, q7: np.ndarray, link_name: str) -> np.ndarray:
    return _fk_T_base_link(str(urdf_path), np.asarray(q7, dtype=np.float32).reshape(7), str(link_name))


def _require_curobo_v2():
    ensure_sys_path_for_runtime()
    try:
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: curobo (v2)") from e
    return RobotCollisionChecker, RobotCollisionCheckerCfg, DeviceCfg, get_robot_configs_path, join_path, load_yaml


def _parse_urdf_names(urdf_path: str) -> Tuple[Optional[set[str]], Optional[set[str]]]:
    try:
        root = ET.parse(str(urdf_path)).getroot()
        links: set[str] = set()
        joints: set[str] = set()
        for el in root.iter():
            tag = str(el.tag)
            if tag.endswith("link"):
                name = el.attrib.get("name")
                if name:
                    links.add(str(name))
            elif tag.endswith("joint"):
                name = el.attrib.get("name")
                if name:
                    joints.add(str(name))
        return (links if links else None, joints if joints else None)
    except Exception:
        return (None, None)


def _resolve_robot_yaml_path(robot_yaml: str) -> str:
    RobotCollisionChecker, RobotCollisionCheckerCfg, DeviceCfg, get_robot_configs_path, join_path, load_yaml = _require_curobo_v2()
    if os.path.isfile(robot_yaml):
        return str(robot_yaml)
    return str(join_path(get_robot_configs_path(), str(robot_yaml)))


def _load_robot_cfg_dict(robot_yaml_path: str) -> Dict[str, Any]:
    RobotCollisionChecker, RobotCollisionCheckerCfg, DeviceCfg, get_robot_configs_path, join_path, load_yaml = _require_curobo_v2()
    loaded = load_yaml(str(robot_yaml_path))
    if not isinstance(loaded, dict):
        raise ValueError("Invalid robot yaml: expected mapping")
    if "robot_cfg" in loaded:
        return dict(loaded)
    return {"robot_cfg": dict(loaded)}


def _sanitize_kinematics_cfg(
    *,
    robot_cfg_dict: Dict[str, Any],
    urdf_path: str,
    tool_frame: str,
) -> Dict[str, Any]:
    kin = robot_cfg_dict.get("robot_cfg", {}).get("kinematics")
    if not isinstance(kin, dict):
        raise ValueError("Invalid robot yaml: missing robot_cfg.kinematics")

    kin["urdf_path"] = str(urdf_path)

    links, joints = _parse_urdf_names(str(urdf_path))

    requested_tool = str(tool_frame).strip() or "robotiq_85_base_link"
    if links is not None:
        fallback = [
            requested_tool,
            "robotiq_85_base_link",
            "camera_mount_link",
            "panda_link8",
            "panda_link7",
            str(kin.get("base_link", "")).strip(),
        ]
        chosen = next((x for x in fallback if x and x in links), None)
        if chosen is None:
            chosen = next(iter(sorted(links)))
        kin["tool_frames"] = [chosen]
    else:
        kin["tool_frames"] = [requested_tool]

    kin.pop("extra_links", None)
    kin.pop("extra_collision_spheres", None)

    if joints is not None:
        cspace = kin.get("cspace")
        if isinstance(cspace, dict):
            names = cspace.get("joint_names")
            if isinstance(names, list):
                keep_idx = [i for i, name in enumerate(names) if str(name) in joints]
                cspace["joint_names"] = [names[i] for i in keep_idx]
                for list_key in ("default_joint_position", "retract_config", "null_space_weight", "cspace_distance_weight"):
                    values = cspace.get(list_key)
                    if isinstance(values, list) and len(values) == len(names):
                        cspace[list_key] = [values[i] for i in keep_idx]

        lock = kin.get("lock_joints")
        if isinstance(lock, dict):
            kin["lock_joints"] = {k: v for k, v in lock.items() if str(k) in joints}

    if links is not None:
        for k_list in ("collision_link_names", "mesh_link_names", "grasp_contact_link_names"):
            v = kin.get(k_list)
            if isinstance(v, list):
                kin[k_list] = [x for x in v if str(x) in links]

        v = kin.get("collision_spheres")
        if isinstance(v, dict):
            kin["collision_spheres"] = {k: vv for k, vv in v.items() if str(k) in links}

        v = kin.get("self_collision_buffer")
        if isinstance(v, dict):
            kin["self_collision_buffer"] = {k: vv for k, vv in v.items() if str(k) in links}

        v = kin.get("self_collision_ignore")
        if isinstance(v, dict):
            cleaned: Dict[str, Any] = {}
            for kk, vv in v.items():
                if str(kk) not in links or not isinstance(vv, list):
                    continue
                kept = [x for x in vv if str(x) in links]
                if kept:
                    cleaned[str(kk)] = kept
            kin["self_collision_ignore"] = cleaned

    tf = kin.get("tool_frames")
    if not isinstance(tf, list) or len(tf) == 0:
        kin["tool_frames"] = [requested_tool]

    return robot_cfg_dict


@dataclass(frozen=True)
class CuRoboCollisionConfig:
    device: str = "cuda:0"
    robot_yaml: str = "franka.yml"
    urdf_path: str = default_urdf_path()
    tool_frame: str = "robotiq_85_base_link"

    with_world: bool = True
    collision_activation_distance: float = 0.2
    self_collision_activation_distance: float = 0.0
    max_collision_distance: float = 1.0


class CuRoboCollisionChecker:
    def __init__(self, cfg: CuRoboCollisionConfig):
        self.cfg = cfg
        self._checker_cache: dict[str, Any] = {}
        self._torch = None

    def _set_device_context(self) -> None:
        torch = _require_torch()
        device = torch.device(str(self.cfg.device))
        if device.type == "cuda":
            torch.cuda.set_device(device.index if device.index is not None else 0)

    def _make_scene_cfg(self, scene_cuboids: Optional[list[dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        if not bool(self.cfg.with_world):
            return None
        if not scene_cuboids:
            return None
        cuboid_dict: Dict[str, Any] = {}
        for i, c in enumerate(scene_cuboids):
            name = f"obs_{i}"
            dims = c.get("dims")
            center = c.get("center")
            if not isinstance(dims, (list, tuple)) or not isinstance(center, (list, tuple)):
                continue
            if len(dims) != 3 or len(center) != 3:
                continue
            cuboid_dict[name] = {
                "dims": [float(dims[0]), float(dims[1]), float(dims[2])],
                "pose": [float(center[0]), float(center[1]), float(center[2]), 1.0, 0.0, 0.0, 0.0],
            }
        return {"cuboid": cuboid_dict} if cuboid_dict else None

    def _scene_key(self, scene_cuboids: Optional[list[dict[str, Any]]]) -> str:
        if not scene_cuboids:
            return "__default__"
        parts: list[str] = []
        for c in scene_cuboids:
            center = c.get("center")
            dims = c.get("dims")
            if not isinstance(center, (list, tuple)) or not isinstance(dims, (list, tuple)):
                continue
            if len(center) != 3 or len(dims) != 3:
                continue
            parts.append(
                f"{center[0]:.4f},{center[1]:.4f},{center[2]:.4f}|{dims[0]:.4f},{dims[1]:.4f},{dims[2]:.4f}"
            )
        parts.sort()
        return "__".join(parts) if parts else "__default__"

    def _get_checker(self, scene_cuboids: Optional[list[dict[str, Any]]] = None):
        self._set_device_context()
        key = self._scene_key(scene_cuboids)
        cached = self._checker_cache.get(key)
        if cached is not None:
            return cached

        torch = _require_torch()
        (
            RobotCollisionChecker,
            RobotCollisionCheckerCfg,
            DeviceCfg,
            get_robot_configs_path,
            join_path,
            load_yaml,
        ) = _require_curobo_v2()

        device_cfg = DeviceCfg(device=torch.device(str(self.cfg.device)), dtype=torch.float32)

        robot_yaml_path = _resolve_robot_yaml_path(str(self.cfg.robot_yaml))
        robot_cfg_dict = _load_robot_cfg_dict(robot_yaml_path)
        robot_cfg_dict = _sanitize_kinematics_cfg(
            robot_cfg_dict=robot_cfg_dict,
            urdf_path=str(self.cfg.urdf_path),
            tool_frame=str(self.cfg.tool_frame),
        )

        scene_cfg = self._make_scene_cfg(scene_cuboids)

        c_cfg = RobotCollisionCheckerCfg.load_from_config(
            robot_config=robot_cfg_dict,
            scene_model=scene_cfg,
            device_cfg=device_cfg,
            collision_activation_distance=float(self.cfg.collision_activation_distance),
            self_collision_activation_distance=float(self.cfg.self_collision_activation_distance),
            max_collision_distance=float(self.cfg.max_collision_distance),
        )
        checker = RobotCollisionChecker(c_cfg)

        self._torch = torch
        self._checker_cache[key] = checker
        return checker

    def batch_distance(
        self,
        q_traj: np.ndarray,
        *,
        scene_cuboids: Optional[list[dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._set_device_context()
        checker = self._get_checker(scene_cuboids)
        torch = self._torch

        q = np.asarray(q_traj, dtype=np.float32)
        if q.ndim != 3 or q.shape[-1] != 7:
            raise ValueError(f"Expected q_traj shape (B,H,7), got {q.shape}")

        q_t = torch.from_numpy(np.ascontiguousarray(q)).to(device=torch.device(str(self.cfg.device)), dtype=torch.float32)

        with torch.no_grad():
            d_scene, d_self = checker.get_scene_self_collision_distance_from_joint_trajectory(q_t)

        if torch.device(str(self.cfg.device)).type == "cuda":
            torch.cuda.synchronize(torch.device(str(self.cfg.device)))

        return (
            d_scene.detach().cpu().numpy().astype(np.float32),
            d_self.detach().cpu().numpy().astype(np.float32),
        )

    def collision_penalty(
        self,
        q_traj: np.ndarray,
        *,
        w_scene: float,
        w_self: float,
        scene_cuboids: Optional[list[dict[str, Any]]] = None,
    ) -> np.ndarray:
        d_scene, d_self = self.batch_distance(q_traj, scene_cuboids=scene_cuboids)
        c = float(w_scene) * np.sum(d_scene, axis=(1, 2)) + float(w_self) * np.sum(d_self, axis=(1, 2))
        return c.astype(np.float32)

    def robot_spheres(self, q: np.ndarray) -> np.ndarray:
        self._set_device_context()
        checker = self._get_checker(None)
        torch = self._torch

        q_np = np.asarray(q, dtype=np.float32)
        if q_np.ndim != 3 or q_np.shape[-1] != 7:
            raise ValueError(f"Expected q shape (B,H,7), got {q_np.shape}")

        q_t = torch.from_numpy(np.ascontiguousarray(q_np)).to(device=torch.device(str(self.cfg.device)), dtype=torch.float32)
        with torch.no_grad():
            kin_state = checker.get_kinematics(q_t)
            spheres = kin_state.robot_spheres

        if spheres is None:
            raise RuntimeError("curobo returned no robot_spheres")

        if torch.device(str(self.cfg.device)).type == "cuda":
            torch.cuda.synchronize(torch.device(str(self.cfg.device)))

        return spheres.detach().cpu().numpy().astype(np.float32)

    def get_robot_spheres_base(self, q_base: np.ndarray, *, margin_m: float = 0.02) -> np.ndarray:
        q0 = np.asarray(q_base, dtype=np.float32).reshape(7)
        q_np = q0.reshape(1, 1, 7)
        spheres = self.robot_spheres(q_np)[0, 0]
        if spheres.ndim != 2 or spheres.shape[1] != 4:
            raise ValueError(f"Expected robot_spheres[0,0] shape (M,4), got {spheres.shape}")
        spheres = np.asarray(spheres, dtype=np.float32).copy()
        if float(margin_m) != 0.0:
            spheres[:, 3] = spheres[:, 3] + float(margin_m)

        T_base_l7 = _fk_T_base_link(str(self.cfg.urdf_path), q0, "panda_link7")

        def _make_extra(offset_z_m: float, radius_m: float) -> np.ndarray:
            p_l7 = np.asarray([0.0, 0.0, float(offset_z_m), 1.0], dtype=np.float32)
            p_b = (T_base_l7 @ p_l7)[:3]
            return np.asarray([p_b[0], p_b[1], p_b[2], float(radius_m)], dtype=np.float32)

        extra = np.stack([
            _make_extra(0.05, 0.17),
            _make_extra(0.20, 0.10),
        ], axis=0)

        return np.ascontiguousarray(np.concatenate([spheres, extra], axis=0).astype(np.float32))


_LOCK = threading.Lock()
_CACHE: dict[tuple[Any, ...], CuRoboCollisionChecker] = {}


def get_curobo_collision_checker(cfg: CuRoboCollisionConfig) -> CuRoboCollisionChecker:
    key = (
        str(cfg.device),
        str(cfg.robot_yaml),
        str(cfg.urdf_path),
        str(cfg.tool_frame),
        bool(cfg.with_world),
        float(cfg.collision_activation_distance),
        float(cfg.self_collision_activation_distance),
        float(cfg.max_collision_distance),
    )
    with _LOCK:
        inst = _CACHE.get(key)
        if inst is None:
            inst = CuRoboCollisionChecker(cfg)
            _CACHE[key] = inst
        return inst
