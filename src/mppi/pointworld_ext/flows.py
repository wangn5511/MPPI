from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mppi.pointworld_ext.robot_flow_builder import RobotFlowBuilder, RobotFlowBuilderConfig
from mppi.utils.paths import ensure_sys_path_for_runtime


def compute_flow_derivatives(flows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(flows, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"flows must be (B,T,N,3), got {arr.shape}")

    B, T, N, _ = arr.shape
    velocity = np.zeros((B, T, N, 3), dtype=np.float32)
    acceleration = np.zeros((B, T, N, 3), dtype=np.float32)

    if T >= 2:
        velocity[:, 0] = arr[:, 1] - arr[:, 0]
        if T >= 3:
            velocity[:, 1:-1] = 0.5 * (arr[:, 2:] - arr[:, :-2])
        velocity[:, -1] = arr[:, -1] - arr[:, -2]

    if T >= 3:
        acceleration[:, 0] = velocity[:, 1] - velocity[:, 0]
        if T >= 4:
            acceleration[:, 1:-1] = 0.5 * (velocity[:, 2:] - velocity[:, :-2])
        acceleration[:, -1] = velocity[:, -1] - velocity[:, -2]

    return velocity, acceleration


def _pad_or_trim_btn3(x: np.ndarray, *, B: int, T: int, N: int, dtype: np.dtype) -> np.ndarray:
    a = np.asarray(x, dtype=dtype)
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"Expected (B,T,N,3), got {a.shape}")
    out = np.zeros((B, T, N, 3), dtype=dtype)
    b0 = min(B, a.shape[0])
    t0 = min(T, a.shape[1])
    n0 = min(N, a.shape[2])
    out[:b0, :t0, :n0] = a[:b0, :t0, :n0]
    return out


def _pad_or_trim_btn(x: np.ndarray, *, B: int, T: int, N: int, dtype: np.dtype) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 3:
        raise ValueError(f"Expected (B,T,N), got {a.shape}")
    out = np.zeros((B, T, N), dtype=dtype)
    b0 = min(B, a.shape[0])
    t0 = min(T, a.shape[1])
    n0 = min(N, a.shape[2])
    out[:b0, :t0, :n0] = a[:b0, :t0, :n0].astype(dtype, copy=False)
    return out


@dataclass(frozen=True)
class PointWorldFeatureConfig:
    max_scene_points: int
    max_robot_points: int
    gripper_open_scale: float = 1.0


def normalize_dist2robot_mode(raw: str | None) -> str:
    mode = str(raw or "full").strip().lower().replace("-", "_")
    aliases = {
        "exact": "full",
        "all": "full",
        "t0": "t0_repeat",
        "repeat_t0": "t0_repeat",
        "zero": "none",
        "zeros": "none",
        "off": "none",
        "disabled": "none",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"full", "t0_repeat", "none"}:
        raise ValueError(f"dist2robot_mode must be one of full, t0_repeat, none; got {raw!r}")
    return mode


class RobotFlowAdapter:
    def __init__(
        self,
        *,
        urdf_path: str,
        max_robot_points: int,
        device: str = "cuda",
        seed: int = 1,
        gripper_only: bool = True,
    ) -> None:
        self.urdf_path = str(urdf_path)
        self.max_robot_points = int(max_robot_points)
        self.device = str(device)
        self.seed = int(seed)
        self.gripper_only = bool(gripper_only)

        self._pointworld_sampler = None
        self._fallback_builder = None
        self._build_backend()

    def _build_backend(self) -> None:
        ensure_sys_path_for_runtime()
        try:
            from robot_sampler import RobotSampler, build_robotiq_joint_dict

            self._pw_build_robotiq_joint_dict = build_robotiq_joint_dict
            sampler = RobotSampler(
                urdf_path=self.urdf_path,
                gripper_only=self.gripper_only,
                device=self.device,
            )
            sampler.presample(self.max_robot_points, seed=self.seed)
            self._pointworld_sampler = sampler
            return
        except Exception:
            self._pointworld_sampler = None

        self._fallback_builder = RobotFlowBuilder(
            cfg=RobotFlowBuilderConfig(
                urdf_path=self.urdf_path,
                total_samples=self.max_robot_points,
                gripper_only=self.gripper_only,
                seed=self.seed,
            )
        )

    def build(self, *, q_traj: np.ndarray, gripper_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.asarray(q_traj, dtype=np.float32)
        g = np.asarray(gripper_positions, dtype=np.float32)
        if q.ndim != 3 or q.shape[-1] != 7:
            raise ValueError(f"q_traj must be (B,T,7), got {q.shape}")
        if g.shape != q.shape[:2]:
            raise ValueError(f"gripper_positions shape {g.shape} must match q_traj[:2]={q.shape[:2]}")

        if self._pointworld_sampler is not None:
            return self._build_with_pointworld_sampler(q_traj=q, gripper_positions=g)
        return self._build_with_fallback(q_traj=q, gripper_positions=g)

    def _build_with_pointworld_sampler(
        self,
        *,
        q_traj: np.ndarray,
        gripper_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        sampler = self._pointworld_sampler
        assert sampler is not None

        B, T, _ = q_traj.shape
        q_flat = q_traj.reshape(B * T, 7)
        g_flat = gripper_positions.reshape(B * T)

        device = torch.device(self.device)
        dtype = torch.float32

        joint_dict = {}
        q_t = torch.as_tensor(q_flat, device=device, dtype=dtype)
        g_t = torch.as_tensor(g_flat, device=device, dtype=dtype)
        for idx in range(7):
            joint_dict[f"panda_joint{idx + 1}"] = q_t[:, idx]
        joint_dict.update(self._pw_build_robotiq_joint_dict(g_t, sampler.joint_names))

        flows_t, colors_t, normals_t = sampler.compute_points(joint_dict)
        Nr = int(flows_t.shape[1])
        flows = flows_t.detach().cpu().numpy().astype(np.float32).reshape(B, T, Nr, 3)
        colors = colors_t.detach().cpu().numpy().astype(np.float32).reshape(B, T, Nr, 3)
        normals = normals_t.detach().cpu().numpy().astype(np.float32).reshape(B, T, Nr, 3)
        return flows, colors, normals

    def _build_with_fallback(
        self,
        *,
        q_traj: np.ndarray,
        gripper_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        builder = self._fallback_builder
        assert builder is not None

        B, T, _ = q_traj.shape
        flows_list: list[np.ndarray] = []
        for bi in range(B):
            flows_list.append(
                builder.build(
                    joint_positions=q_traj[bi],
                    gripper_positions=gripper_positions[bi],
                )
            )
        flows = np.stack(flows_list, axis=0).astype(np.float32)
        Nr = int(flows.shape[2])

        colors = np.zeros((B, T, Nr, 3), dtype=np.float32)
        colors[..., 0] = 1.0
        colors[..., 2] = 1.0
        normals = np.zeros((B, T, Nr, 3), dtype=np.float32)
        return flows, colors, normals


def prepare_scene_inputs(
    *,
    scene_flows: np.ndarray,
    scene_colors: np.ndarray,
    scene_exists: np.ndarray,
    scene_track_confidence: Optional[np.ndarray],
    batch_size: int,
    max_scene_points: int,
) -> dict[str, np.ndarray]:
    flows = np.asarray(scene_flows, dtype=np.float32)
    colors = np.asarray(scene_colors)
    exists = np.asarray(scene_exists, dtype=bool)
    conf = (
        np.asarray(scene_track_confidence, dtype=np.float32)
        if scene_track_confidence is not None
        else np.ones(exists.shape, dtype=np.float32)
    )

    if flows.ndim != 3 or flows.shape[-1] != 3:
        raise ValueError(f"scene_flows must be (T,N,3), got {flows.shape}")
    if exists.shape != flows.shape[:2]:
        raise ValueError(f"scene_exists shape {exists.shape} must match scene_flows[:2]={flows.shape[:2]}")
    if conf.shape != flows.shape[:2]:
        raise ValueError(f"scene_track_confidence shape {conf.shape} must match scene_flows[:2]={flows.shape[:2]}")

    T = int(flows.shape[0])
    Ns = int(max_scene_points)

    if colors.dtype == np.uint8:
        colors_f = colors.astype(np.float32) / 255.0
    else:
        colors_f = np.asarray(colors, dtype=np.float32)
    if colors_f.shape != flows.shape:
        colors_f = np.zeros_like(flows, dtype=np.float32)

    flows_b = np.repeat(flows[None, ...], batch_size, axis=0)
    colors_b = np.repeat(colors_f[None, ...], batch_size, axis=0)
    exists_b = np.repeat(exists[None, ...], batch_size, axis=0)
    conf_b = np.repeat(conf[None, ...], batch_size, axis=0)

    flows_b = _pad_or_trim_btn3(flows_b, B=batch_size, T=T, N=Ns, dtype=np.float32)
    colors_b = _pad_or_trim_btn3(colors_b, B=batch_size, T=T, N=Ns, dtype=np.float32)
    exists_b = _pad_or_trim_btn(exists_b, B=batch_size, T=T, N=Ns, dtype=bool)
    conf_b = _pad_or_trim_btn(conf_b, B=batch_size, T=T, N=Ns, dtype=np.float32)

    return {
        "scene_flows": flows_b,
        "scene_colors": colors_b,
        "scene_exists": exists_b,
        "scene_track_confidence": conf_b,
    }


def build_robot_inputs(
    *,
    robot_flows: np.ndarray,
    robot_colors: np.ndarray,
    robot_normals: np.ndarray,
    gripper_positions: np.ndarray,
    max_robot_points: int,
) -> dict[str, np.ndarray]:
    flows = np.asarray(robot_flows, dtype=np.float32)
    colors = np.asarray(robot_colors, dtype=np.float32)
    normals = np.asarray(robot_normals, dtype=np.float32)
    gripper = np.asarray(gripper_positions, dtype=np.float32)

    if flows.ndim != 4 or flows.shape[-1] != 3:
        raise ValueError(f"robot_flows must be (B,T,N,3), got {flows.shape}")
    if colors.shape != flows.shape:
        colors = np.zeros_like(flows, dtype=np.float32)
        colors[..., 0] = 1.0
        colors[..., 2] = 1.0
    if normals.shape != flows.shape:
        normals = np.zeros_like(flows, dtype=np.float32)
    if gripper.shape != flows.shape[:2]:
        raise ValueError(f"gripper_positions shape {gripper.shape} must match robot_flows[:2]={flows.shape[:2]}")

    B, T, _Nr, _ = flows.shape
    Nr = int(max_robot_points)

    flows = _pad_or_trim_btn3(flows, B=B, T=T, N=Nr, dtype=np.float32)
    colors = _pad_or_trim_btn3(colors, B=B, T=T, N=Nr, dtype=np.float32)
    normals = _pad_or_trim_btn3(normals, B=B, T=T, N=Nr, dtype=np.float32)

    velocity, acceleration = compute_flow_derivatives(flows)
    gripper_feat = np.repeat(gripper[:, :, None, None], Nr, axis=2).astype(np.float32)
    robot_exists = np.linalg.norm(flows, axis=-1) > 0.0

    robot_features = np.concatenate(
        [flows, colors, normals, gripper_feat, velocity, acceleration],
        axis=-1,
    ).astype(np.float32)
    return {
        "robot_flows": flows,
        "robot_features": robot_features,
        "robot_exists": robot_exists,
    }


def build_scene_features(
    *,
    scene_flows: np.ndarray,
    scene_colors: np.ndarray,
    gripper_positions: np.ndarray,
    robot_flows: np.ndarray,
    robot_exists: np.ndarray | None = None,
    dist2robot_mode: str = "full",
) -> np.ndarray:
    flows = np.asarray(scene_flows, dtype=np.float32)
    colors = np.asarray(scene_colors, dtype=np.float32)
    gripper = np.asarray(gripper_positions, dtype=np.float32)
    robot = np.asarray(robot_flows, dtype=np.float32)
    exists = (np.asarray(robot_exists, dtype=bool) if robot_exists is not None else (np.linalg.norm(robot, axis=-1) > 0.0))
    if flows.ndim != 4 or flows.shape[-1] != 3:
        raise ValueError(f"scene_flows must be (B,T,N,3), got {flows.shape}")
    if colors.shape != flows.shape:
        raise ValueError(f"scene_colors must match scene_flows shape, got {colors.shape} vs {flows.shape}")
    if gripper.shape != flows.shape[:2]:
        raise ValueError(f"gripper_positions shape {gripper.shape} must match scene_flows[:2]={flows.shape[:2]}")
    if robot.shape[:2] != flows.shape[:2] or robot.shape[-1] != 3:
        raise ValueError(f"robot_flows must be (B,T,Nr,3) with matching (B,T), got {robot.shape}")
    B, T, Ns, _ = flows.shape
    p0 = flows[:, 0]

    mode = normalize_dist2robot_mode(dist2robot_mode)

    def _dist_at_t(t: int) -> np.ndarray:
        diff = p0[:, :, None, :] - robot[:, t, None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        d = np.where(exists[:, t, None, :], d, np.inf)
        return np.min(d, axis=-1).astype(np.float32, copy=False)

    if mode == "none":
        dist2robot = np.zeros((B, 1, Ns, T), dtype=np.float32)
    elif mode == "t0_repeat":
        dist0 = _dist_at_t(0)
        dist2robot = np.repeat(dist0[:, None, :, None], T, axis=-1)
    else:
        dist2robot = np.empty((B, 1, Ns, T), dtype=np.float32)
        for t in range(T):
            dist2robot[:, 0, :, t] = _dist_at_t(t)
    normals0 = np.zeros((B, 1, Ns, 3), dtype=np.float32)
    flow0 = flows[:, :1]
    color0 = colors[:, :1]
    gripper_ctx = np.repeat(gripper[:, None, None, :], Ns, axis=2).astype(np.float32)
    return np.concatenate([flow0, color0, normals0, gripper_ctx, dist2robot], axis=-1).astype(np.float32)


def build_scene_features_torch(
    *,
    scene_flows,
    scene_colors,
    gripper_positions,
    robot_flows,
    robot_exists=None,
    dist2robot_mode: str = "full",
    dist2robot_chunk_size: int = 256,
):
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: torch") from e
    flows = torch.as_tensor(scene_flows, dtype=torch.float32)
    colors = torch.as_tensor(scene_colors, device=flows.device, dtype=torch.float32)
    gripper = torch.as_tensor(gripper_positions, device=flows.device, dtype=torch.float32)
    robot = torch.as_tensor(robot_flows, device=flows.device, dtype=torch.float32)
    exists = (torch.as_tensor(robot_exists, device=flows.device, dtype=torch.bool) if robot_exists is not None else (torch.linalg.norm(robot, dim=-1) > 0.0))
    if flows.ndim != 4 or int(flows.shape[-1]) != 3:
        raise ValueError(f"scene_flows must be (B,T,N,3), got {tuple(flows.shape)}")
    if tuple(colors.shape) != tuple(flows.shape):
        raise ValueError(f"scene_colors must match scene_flows shape, got {tuple(colors.shape)} vs {tuple(flows.shape)}")
    if tuple(gripper.shape) != tuple(flows.shape[:2]):
        raise ValueError(f"gripper_positions shape {tuple(gripper.shape)} must match scene_flows[:2]={tuple(flows.shape[:2])}")
    if robot.ndim != 4 or tuple(robot.shape[:2]) != tuple(flows.shape[:2]) or int(robot.shape[-1]) != 3:
        raise ValueError(f"robot_flows must be (B,T,Nr,3) with matching (B,T), got {tuple(robot.shape)}")
    B, T, Ns, _ = flows.shape
    p0 = flows[:, 0]

    mode = normalize_dist2robot_mode(dist2robot_mode)
    chunk_size = max(1, int(dist2robot_chunk_size))

    def _dist_at_t(t: int):
        pts = robot[:, t]
        valid = exists[:, t]
        out_t = torch.empty((B, Ns), device=flows.device, dtype=torch.float32)
        for s in range(0, Ns, chunk_size):
            e = min(s + chunk_size, Ns)
            d = torch.cdist(p0[:, s:e], pts)
            d = d.masked_fill(~valid[:, None, :], float("inf"))
            out_t[:, s:e] = d.min(dim=-1).values
        return out_t

    if mode == "none":
        dist2robot = torch.zeros((B, 1, Ns, T), device=flows.device, dtype=torch.float32)
    elif mode == "t0_repeat":
        dist0 = _dist_at_t(0)
        dist2robot = dist0[:, None, :, None].expand(B, 1, Ns, T)
    else:
        dist2robot = torch.empty((B, 1, Ns, T), device=flows.device, dtype=torch.float32)
        for t in range(T):
            dist2robot[:, 0, :, t] = _dist_at_t(t)
    normals0 = torch.zeros((B, 1, Ns, 3), device=flows.device, dtype=torch.float32)
    flow0 = flows[:, :1]
    color0 = colors[:, :1]
    gripper_ctx = gripper[:, None, None, :].expand(B, 1, Ns, T)
    return torch.cat([flow0, color0, normals0, gripper_ctx, dist2robot], dim=-1).to(dtype=torch.float32)
