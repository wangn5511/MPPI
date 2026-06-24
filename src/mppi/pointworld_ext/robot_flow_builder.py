from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def _ensure_urdfpy_numpy_compat() -> None:
    if "float" not in np.__dict__:
        np.float = float  # type: ignore[attr-defined]
    if "int" not in np.__dict__:
        np.int = int  # type: ignore[attr-defined]
    if "bool" not in np.__dict__:
        np.bool = bool  # type: ignore[attr-defined]


def _require_urdfpy_trimesh():
    _ensure_urdfpy_numpy_compat()
    try:
        import trimesh
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: trimesh") from e

    try:
        import urdfpy
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: urdfpy") from e

    return urdfpy, trimesh


def get_mesh_stable_id(mesh, idx: int | None = None) -> str:
    try:
        src = getattr(mesh, "source", None)
        if src is not None:
            fname = getattr(src, "file_name", None)
            if isinstance(fname, (str, bytes)) and fname:
                base = str(fname).lower()
                return f"{base}_{idx}" if idx is not None else base
    except Exception:
        pass
    try:
        metadata = mesh.metadata or {}
        for key in ("name", "file_name"):
            val = metadata.get(key)
            if isinstance(val, (str, bytes)) and val:
                base = str(val).lower()
                return f"{base}_{idx}" if idx is not None else base
    except Exception:
        pass
    try:
        bounds = np.asarray(mesh.bounds).reshape(-1)
        bounds = np.round(bounds, 6)
        vcount = len(getattr(mesh, "vertices", []))
        fcount = len(getattr(mesh, "faces", []))
        base = f"b{','.join(map(str, bounds))}_v{vcount}_f{fcount}"
        return f"{base}_{idx}" if idx is not None else base
    except Exception:
        base = f"unknown_{id(mesh)}"
        return f"{base}_{idx}" if idx is not None else base


def _sample_surface_deterministic(mesh, count: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if count <= 0 or float(getattr(mesh, "area", 0.0)) <= 0.0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    faces = mesh.faces
    verts = mesh.vertices
    areas = mesh.area_faces.astype(np.float64)
    prob = areas / max(float(areas.sum()), 1e-12)
    idx = rng.choice(len(faces), size=int(count), p=prob)
    tri = faces[idx]
    v0 = verts[tri[:, 0]]
    v1 = verts[tri[:, 1]]
    v2 = verts[tri[:, 2]]
    u = rng.random(size=(count,)).astype(np.float64)
    v = rng.random(size=(count,)).astype(np.float64)
    su = np.sqrt(u)
    w0 = 1.0 - su
    w1 = su * (1.0 - v)
    w2 = su * v
    pts = (v0 * w0[:, None] + v1 * w1[:, None] + v2 * w2[:, None]).astype(np.float32)
    nrm = mesh.face_normals[idx].astype(np.float32)
    return pts, nrm


@dataclass(frozen=True)
class RobotFlowBuilderConfig:
    urdf_path: str
    total_samples: int = 4000
    min_samples_per_mesh: int = 50
    gripper_only: bool = True
    seed: int = 1


class _URDFMeshSampler:
    def __init__(self, *, cfg: RobotFlowBuilderConfig) -> None:
        urdfpy, _trimesh = _require_urdfpy_trimesh()
        urdf_path = Path(str(cfg.urdf_path))
        if not urdf_path.is_file():
            raise FileNotFoundError(str(urdf_path))

        self._cfg = cfg
        self._urdf = urdfpy.URDF.load(str(urdf_path))
        self._presampled_pts: Dict[str, np.ndarray] = {}
        self._presampled_nrm: Dict[str, np.ndarray] = {}
        self._target_count: Optional[int] = None
        self._rng: Optional[np.random.Generator] = None

    def _neutral_cfg(self) -> Dict[str, float]:
        cfg = {j.name: 0.0 for j in self._urdf.actuated_joints}
        cfg.setdefault("finger_joint", 0.0)
        for i in range(1, 8):
            cfg.setdefault(f"panda_joint{i}", 0.0)
        return cfg

    def presample(self) -> None:
        num_points = int(self._cfg.total_samples)
        if num_points <= 0:
            self._presampled_pts = {}
            self._presampled_nrm = {}
            return

        rng = np.random.default_rng(self._cfg.seed)
        self._rng = rng
        self._target_count = int(num_points)
        fk_ref = self._urdf.visual_trimesh_fk(cfg=self._neutral_cfg())

        names, meshes, areas = [], [], []
        for i, (mesh, _T) in enumerate(fk_ref.items()):
            mid = get_mesh_stable_id(mesh, i)
            if bool(self._cfg.gripper_only):
                ml = mid.lower()
                if not any(k in ml for k in ("finger", "knuckle", "robotiq", "gripper")):
                    continue
            eff_area = float(mesh.area)
            if "hand_camera_part" in mid.lower():
                eff_area *= 1e-6
            names.append(mid)
            meshes.append(mesh)
            areas.append(max(eff_area, 1e-9))

        if not names:
            for i, (mesh, _T) in enumerate(fk_ref.items()):
                mid = get_mesh_stable_id(mesh, i)
                names.append(mid)
                meshes.append(mesh)
                areas.append(max(float(mesh.area), 1e-9))

        total_area = float(sum(areas))
        counts = []
        allocated = 0
        for i, a in enumerate(areas):
            if i == len(areas) - 1:
                c = max(int(self._cfg.min_samples_per_mesh), num_points - allocated)
            else:
                w = a / total_area if total_area > 0 else 0.0
                c = max(int(self._cfg.min_samples_per_mesh), int(round(w * num_points)))
                allocated += c
            counts.append(int(c))

        pts_dict: Dict[str, np.ndarray] = {}
        nrm_dict: Dict[str, np.ndarray] = {}
        for mid, mesh, c in zip(names, meshes, counts):
            pts, nrm = _sample_surface_deterministic(mesh, int(c), rng)
            pts_dict[mid] = pts
            nrm_dict[mid] = nrm

        total = sum(arr.shape[0] for arr in pts_dict.values())
        if total > num_points:
            mesh_keys = list(pts_dict.keys())
            offsets = np.cumsum([0] + [pts_dict[k].shape[0] for k in mesh_keys])
            sel = rng.choice(total, size=num_points, replace=False)
            sel.sort()
            new_pts: Dict[str, np.ndarray] = {}
            new_nrm: Dict[str, np.ndarray] = {}
            for i, key in enumerate(mesh_keys):
                start, end = offsets[i], offsets[i + 1]
                mask = (sel >= start) & (sel < end)
                if not np.any(mask):
                    continue
                local_idx = sel[mask] - start
                new_pts[key] = pts_dict[key][local_idx]
                new_nrm[key] = nrm_dict[key][local_idx]
            pts_dict = new_pts
            nrm_dict = new_nrm

        self._presampled_pts = pts_dict
        self._presampled_nrm = nrm_dict

    def compute_world_trajectories(
        self,
        joint_positions: np.ndarray,
        gripper_positions: np.ndarray,
        *,
        T_world_base: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self._presampled_pts:
            jp = np.asarray(joint_positions)
            T = int(jp.shape[0]) if jp.ndim >= 1 else 0
            return np.zeros((T, 0, 3), dtype=np.float32)

        jp = np.asarray(joint_positions, dtype=np.float32)
        gp = np.asarray(gripper_positions, dtype=np.float32).reshape(-1)
        if jp.ndim != 2 or jp.shape[1] != 7:
            raise ValueError(f"joint_positions must be (T,7), got {jp.shape}")
        if gp.shape[0] != jp.shape[0]:
            raise ValueError(f"gripper_positions length mismatch: {gp.shape[0]} != {jp.shape[0]}")

        Tn = int(jp.shape[0])

        if T_world_base is not None:
            Tb = np.asarray(T_world_base, dtype=np.float32)
            if Tb.shape == (4, 4):
                Tb = np.repeat(Tb[None, :, :], Tn, axis=0)
            if Tb.shape != (Tn, 4, 4):
                raise ValueError(f"T_world_base must be (4,4) or (T,4,4), got {Tb.shape}")
        else:
            Tb = None

        transforms: Dict[str, np.ndarray] = {mid: np.zeros((Tn, 4, 4), dtype=np.float32) for mid in self._presampled_pts.keys()}

        for t in range(Tn):
            cfg = {f"panda_joint{i + 1}": float(jp[t, i]) for i in range(7)}
            cfg["finger_joint"] = float(gp[t])
            fk = self._urdf.visual_trimesh_fk(cfg=cfg)
            for i, (mesh, T_wm) in enumerate(fk.items()):
                mid = get_mesh_stable_id(mesh, i)
                if mid in transforms:
                    Tm = np.asarray(T_wm, dtype=np.float32)
                    if Tb is not None:
                        Tm = Tb[t] @ Tm
                    transforms[mid][t] = Tm

        traj_list = []
        for mid, pts in self._presampled_pts.items():
            Tm = transforms.get(mid)
            if Tm is None:
                continue
            n = int(pts.shape[0])
            if n == 0:
                continue
            homo = np.concatenate([pts.astype(np.float32), np.ones((n, 1), dtype=np.float32)], axis=1)
            world = (Tm @ homo.T).transpose(0, 2, 1)[..., :3]
            traj_list.append(world.astype(np.float32))

        if not traj_list:
            return np.zeros((Tn, 0, 3), dtype=np.float32)

        traj = np.concatenate(traj_list, axis=1)
        if self._target_count is not None and traj.shape[1] > int(self._target_count):
            rng = self._rng or np.random.default_rng(0)
            idx = rng.choice(traj.shape[1], size=int(self._target_count), replace=False)
            traj = traj[:, idx, :]
        return traj.astype(np.float32)


class RobotFlowBuilder:
    def __init__(self, *, cfg: RobotFlowBuilderConfig) -> None:
        self.cfg = cfg
        self._sampler = _URDFMeshSampler(cfg=cfg)
        self._sampler.presample()

    def build(
        self,
        *,
        joint_positions: np.ndarray,
        gripper_positions: np.ndarray,
        T_world_base: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._sampler.compute_world_trajectories(
            joint_positions=joint_positions,
            gripper_positions=gripper_positions,
            T_world_base=T_world_base,
        )

    def build_from_actions(
        self,
        *,
        actions: np.ndarray,
        T_world_base: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        a = np.asarray(actions, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] < 8:
            raise ValueError(f"actions must be (T,8+) with joints in [0:7] and gripper in [7], got {a.shape}")
        return self.build(
            joint_positions=a[:, 0:7],
            gripper_positions=a[:, 7],
            T_world_base=T_world_base,
        )