from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

from mppi.costs.ee_pose import ee_pos_cost
from mppi.curobo_ext.collision_checker import CuRoboCollisionConfig, get_curobo_collision_checker
from mppi.curobo_ext.scene_builder import SceneBuildConfig, build_scene_cuboids_from_pcd_back_cam
from mppi.robots.franka_kinematics import FrankaFK
from mppi.utils.paths import default_urdf_path, repo_path


class PointWorldCostFn(Protocol):
    def __call__(
        self,
        *,
        q_traj: np.ndarray,
        u_traj: np.ndarray,
        pointworld_obs: dict[str, Any],
        gripper: Optional[float],
    ) -> np.ndarray: ...

def _curobo_collision_cost(
    cfg: "JointMPPIConfig",
    q_traj: np.ndarray,
    *,
    scene_cuboids: Optional[list[dict[str, Any]]] = None,
) -> np.ndarray:
    if not bool(cfg.use_curobo_collision):
        return np.zeros((q_traj.shape[0],), dtype=np.float32)

    ccfg = CuRoboCollisionConfig(
        device=str(cfg.curobo_device),
        robot_yaml=str(cfg.curobo_robot_yaml),
        urdf_path=str(cfg.urdf_path),
        tool_frame=str(cfg.curobo_tool_frame),
        with_world=bool(cfg.curobo_with_world),
        collision_activation_distance=float(cfg.curobo_collision_activation_distance),
        self_collision_activation_distance=float(cfg.curobo_self_collision_activation_distance),
        max_collision_distance=float(cfg.curobo_max_collision_distance),
    )
    checker = get_curobo_collision_checker(ccfg)
    q = np.asarray(q_traj, dtype=np.float32)
    return checker.collision_penalty(
        q,
        w_scene=float(cfg.w_scene_collision),
        w_self=float(cfg.w_self_collision),
        scene_cuboids=scene_cuboids,
    )


def _extract_points_from_pcd(pcd_back_cam: object) -> object:
    if isinstance(pcd_back_cam, dict) and "points" in pcd_back_cam:
        return pcd_back_cam["points"]
    return pcd_back_cam


@dataclass
class _SceneTrack:
    track_id: int
    center: np.ndarray
    dims: np.ndarray
    age: int = 0
    misses: int = 0
    score: float = 0.0


def _aabb_iou_3d(center_a: np.ndarray, dims_a: np.ndarray, center_b: np.ndarray, dims_b: np.ndarray) -> float:
    ca = np.asarray(center_a, dtype=np.float32).reshape(3)
    da = np.asarray(dims_a, dtype=np.float32).reshape(3)
    cb = np.asarray(center_b, dtype=np.float32).reshape(3)
    db = np.asarray(dims_b, dtype=np.float32).reshape(3)

    a_min = ca - 0.5 * da
    a_max = ca + 0.5 * da
    b_min = cb - 0.5 * db
    b_max = cb + 0.5 * db

    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.maximum(inter_max - inter_min, 0.0)
    inter_v = float(inter[0] * inter[1] * inter[2])

    va = float(max(0.0, da[0]) * max(0.0, da[1]) * max(0.0, da[2]))
    vb = float(max(0.0, db[0]) * max(0.0, db[1]) * max(0.0, db[2]))
    union = va + vb - inter_v
    if union <= 0.0:
        return 0.0
    return float(inter_v / union)


class SceneState:
    def __init__(
        self,
        *,
        alpha: float,
        remove_after_misses: int,
        max_tracks: int,
        match_center_dist_m: float,
        match_iou_min: float,
    ) -> None:
        self.alpha = float(alpha)
        self.remove_after_misses = int(remove_after_misses)
        self.max_tracks = int(max_tracks)
        self.match_center_dist_m = float(match_center_dist_m)
        self.match_iou_min = float(match_iou_min)

        self._tracks: list[_SceneTrack] = []
        self._next_id: int = 1

    def update(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cands: list[tuple[np.ndarray, np.ndarray]] = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            center = c.get("center")
            dims = c.get("dims")
            if not isinstance(center, (list, tuple)) or not isinstance(dims, (list, tuple)):
                continue
            if len(center) != 3 or len(dims) != 3:
                continue
            cc = np.asarray([float(center[0]), float(center[1]), float(center[2])], dtype=np.float32)
            dd = np.asarray([float(dims[0]), float(dims[1]), float(dims[2])], dtype=np.float32)
            if (not np.all(np.isfinite(cc))) or (not np.all(np.isfinite(dd))):
                continue
            if np.any(dd <= 0.0):
                continue
            cands.append((cc, dd))

        if len(self._tracks) == 0:
            for cc, dd in cands[: max(0, self.max_tracks)]:
                self._tracks.append(_SceneTrack(track_id=self._next_id, center=cc, dims=dd, age=1, misses=0, score=1.0))
                self._next_id += 1
            self._tracks.sort(key=lambda t: t.track_id)
            return [
                {"center": [float(t.center[0]), float(t.center[1]), float(t.center[2])], "dims": [float(t.dims[0]), float(t.dims[1]), float(t.dims[2])]}
                for t in self._tracks
            ]

        pairs: list[tuple[float, float, int, int]] = []
        for i, tr in enumerate(self._tracks):
            for j, (cc, dd) in enumerate(cands):
                iou = _aabb_iou_3d(tr.center, tr.dims, cc, dd)
                dist = float(np.linalg.norm((cc - tr.center).astype(np.float32)))
                if iou >= float(self.match_iou_min) or dist <= float(self.match_center_dist_m):
                    pairs.append((iou, dist, i, j))

        pairs.sort(key=lambda x: (-x[0], x[1]))

        used_tracks: set[int] = set()
        used_cands: set[int] = set()
        matches: list[tuple[int, int, float, float]] = []
        for iou, dist, i, j in pairs:
            if i in used_tracks or j in used_cands:
                continue
            used_tracks.add(i)
            used_cands.add(j)
            matches.append((i, j, float(iou), float(dist)))

        a = float(self.alpha)
        if not (0.0 <= a <= 1.0):
            a = 1.0

        for i, j, iou, dist in matches:
            tr = self._tracks[i]
            cc, dd = cands[j]
            tr.center = (a * cc + (1.0 - a) * tr.center).astype(np.float32)
            tr.dims = (a * dd + (1.0 - a) * tr.dims).astype(np.float32)
            tr.age = int(tr.age) + 1
            tr.misses = 0
            tr.score = float(max(tr.score, 0.0) * 0.9 + max(iou, 0.0) * 0.1)

        kept: list[_SceneTrack] = []
        for idx, tr in enumerate(self._tracks):
            if idx in used_tracks:
                kept.append(tr)
                continue
            tr.age = int(tr.age) + 1
            tr.misses = int(tr.misses) + 1
            if tr.misses < int(self.remove_after_misses):
                kept.append(tr)

        kept.sort(key=lambda t: (t.track_id,))

        if int(self.max_tracks) > 0 and len(kept) > int(self.max_tracks):
            kept = kept[: int(self.max_tracks)]

        remaining = max(0, int(self.max_tracks) - len(kept)) if int(self.max_tracks) > 0 else 0
        if remaining > 0:
            for j, (cc, dd) in enumerate(cands):
                if j in used_cands:
                    continue
                if remaining <= 0:
                    break
                kept.append(_SceneTrack(track_id=self._next_id, center=cc, dims=dd, age=1, misses=0, score=1.0))
                self._next_id += 1
                remaining -= 1

        kept.sort(key=lambda t: t.track_id)
        self._tracks = kept

        return [
            {"center": [float(t.center[0]), float(t.center[1]), float(t.center[2])], "dims": [float(t.dims[0]), float(t.dims[1]), float(t.dims[2])]}
            for t in self._tracks
        ]


@dataclass(frozen=True)
class JointMPPIConfig:
    horizon: int = 8
    num_samples: int = 256
    infer_budget_ms: float = 0.0
    budget_max_dynamic_cuboids: int = 0
    debug_cost_stats: bool = False
    debug_cost_stats_q: float = 0.5
    temperature: float = 1.0
    noise_std: float = 0.05
    dt: float = 1.0

    w_smooth: float = 1.0
    w_action: float = 0.01
    w_joint_limit: float = 50.0

    use_curobo_collision: bool = False
    w_scene_collision: float = 1.0
    w_self_collision: float = 1.0
    curobo_device: str = "cuda:0"
    curobo_robot_yaml: str = "franka.yml"
    curobo_tool_frame: str = "robotiq_85_base_link"
    curobo_with_world: bool = True
    curobo_collision_activation_distance: float = 0.2
    curobo_self_collision_activation_distance: float = 0.0
    curobo_max_collision_distance: float = 1.0

    scene_from_pcd_back_cam: bool = False
    scene_pcd_scale: float = 1.0
    scene_pcd_in_base: bool = True

    scene_add_table: bool = True
    scene_table_dims: tuple[float, float, float] = (2.0, 2.0, 0.2)
    scene_table_center: tuple[float, float, float] = (0.4, 0.0, -0.1)

    scene_remove_table_points: bool = True
    scene_table_eps_m: float = 0.01

    scene_remove_wall_points: bool = False
    scene_wall_dims: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scene_wall_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scene_wall_margin_m: float = 0.02

    t_base_cam_back_path: str = repo_path("configs", "T_base_cam.yaml")
    scene_roi_min: tuple[float, float, float] = (-0.1, -0.7, -0.05)
    scene_roi_max: tuple[float, float, float] = (1.2, 0.7, 1.2)
    scene_voxel_size_m: float = 0.01
    scene_padding_m: float = 0.02
    scene_max_cuboids: int = 20
    scene_robot_mask_margin_m: float = 0.02
    scene_min_cluster_voxels: int = 10

    scene_track_alpha: float = 0.6
    scene_track_remove_after_misses: int = 5
    scene_track_max_tracks: int = 20
    scene_track_match_center_dist_m: float = 0.10
    scene_track_match_iou_min: float = 0.05

    min_effective_samples_ratio: float = 0.01

    w_ee_pos: float = 0.0
    ee_pos_goal: Optional[tuple[float, float, float]] = None
    w_link7_pos: float = 0.0
    link7_pos_goal: Optional[tuple[float, float, float]] = None

    use_pointworld_cost: bool = False
    w_pointworld: float = 0.0
    pointworld_require_horizon_11: bool = True
    pointworld_cost_timeout_ms: float = 0.0

    urdf_path: str = default_urdf_path()
    ee_link: str = "robotiq_85_base_link"
    link7_link: str = "panda_link7"

    q_min: tuple[float, ...] = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
    q_max: tuple[float, ...] = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)

    max_dq: float = 0.25


class JointMPPISolver:
    def __init__(self, cfg: JointMPPIConfig):
        self.cfg = cfg
        self._u = np.zeros((int(cfg.horizon), 7), dtype=np.float32)

        self.last_infer_ms: float = 0.0
        self.last_timing_policy_suffix: str = ""

        self.last_cost_stats: dict[str, float] = {}
        self.last_min_distance_scene: float = float("nan")
        self.last_min_distance_self: float = float("nan")

        self._degrade_level: int = 0
        self._last_scene_dynamic: list[dict[str, Any]] = []
        self._last_scene_has_table: bool = False
        self._q_min = np.asarray(cfg.q_min, dtype=np.float32)
        self._q_max = np.asarray(cfg.q_max, dtype=np.float32)

        self.last_fallback: bool = False
        self.last_fallback_reason: str = ""
        self.last_effective_samples_ratio: float = 0.0

        self.last_scene_num_cuboids: int = 0
        self.last_scene_num_robot_spheres: int = 0
        self.last_scene_has_table: bool = False

        self.last_fallback: bool = False
        self.last_fallback_reason: str = ""
        self.last_effective_samples_ratio: float = 0.0
        self.last_pw_enabled: bool = False
        self.last_pw_reason: str = ""
        self.last_pw_ms: float = 0.0

        self._fk: FrankaFK | None = None
        self._fk_link7: FrankaFK | None = None

        self._scene_state = SceneState(
            alpha=float(cfg.scene_track_alpha),
            remove_after_misses=int(cfg.scene_track_remove_after_misses),
            max_tracks=int(cfg.scene_track_max_tracks),
            match_center_dist_m=float(cfg.scene_track_match_center_dist_m),
            match_iou_min=float(cfg.scene_track_match_iou_min),
        )

    def reset(self) -> None:
        self._u[...] = 0.0

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self._q_min, self._q_max)

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        return np.clip(u, -float(self.cfg.max_dq), float(self.cfg.max_dq))

    def _get_fk(self) -> FrankaFK:
        if self._fk is None:
            self._fk = FrankaFK(urdf_path=self.cfg.urdf_path, ee_link=self.cfg.ee_link, device="cpu")
        return self._fk

    def _get_fk_link7(self) -> FrankaFK:
        if self._fk_link7 is None:
            self._fk_link7 = FrankaFK(urdf_path=self.cfg.urdf_path, ee_link=self.cfg.link7_link, device="cpu")
        return self._fk_link7

    def _rollout_cost(
        self,
        q_traj: np.ndarray,
        u_traj: np.ndarray,
        *,
        scene_cuboids: Optional[list[dict[str, Any]]] = None,
        pointworld_obs: Optional[dict[str, Any]] = None,
        pointworld_cost_fn: Optional[PointWorldCostFn] = None,
        gripper: Optional[float] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        dq = q_traj[:, 1:, :] - q_traj[:, :-1, :]
        smooth_raw = np.sum(dq * dq, axis=(1, 2)).astype(np.float32)

        act_raw = np.sum(u_traj * u_traj, axis=(1, 2)).astype(np.float32)

        below = np.clip(self._q_min[None, None, :] - q_traj, 0.0, None)
        above = np.clip(q_traj - self._q_max[None, None, :], 0.0, None)
        joint_limit_raw = np.sum(below * below + above * above, axis=(1, 2)).astype(np.float32)

        t_smooth = float(self.cfg.w_smooth) * smooth_raw
        t_action = float(self.cfg.w_action) * act_raw
        t_joint_limit = float(self.cfg.w_joint_limit) * joint_limit_raw

        t_ee_pos = np.zeros((q_traj.shape[0],), dtype=np.float32)
        if float(self.cfg.w_ee_pos) > 0.0 and self.cfg.ee_pos_goal is not None:
            q_T = q_traj[:, -1, :]
            t_ee_pos = float(self.cfg.w_ee_pos) * ee_pos_cost(self._get_fk(), q_T, self.cfg.ee_pos_goal)

        t_link7_pos = np.zeros((q_traj.shape[0],), dtype=np.float32)
        if float(self.cfg.w_link7_pos) > 0.0 and self.cfg.link7_pos_goal is not None:
            q_T = q_traj[:, -1, :]
            t_link7_pos = float(self.cfg.w_link7_pos) * ee_pos_cost(self._get_fk_link7(), q_T, self.cfg.link7_pos_goal)

        t_scene = np.zeros((q_traj.shape[0],), dtype=np.float32)
        t_self = np.zeros((q_traj.shape[0],), dtype=np.float32)
        t_pw = np.zeros((q_traj.shape[0],), dtype=np.float32)
        dmin_scene = float("nan")
        dmin_self = float("nan")
        pw_ms = 0.0
        pw_reason = "disabled"

        need_dist = bool(self.cfg.use_curobo_collision) and (
            float(self.cfg.w_scene_collision) > 0.0
            or float(self.cfg.w_self_collision) > 0.0
            or bool(self.cfg.debug_cost_stats)
        )
        if need_dist:
            ccfg = CuRoboCollisionConfig(
                device=str(self.cfg.curobo_device),
                robot_yaml=str(self.cfg.curobo_robot_yaml),
                urdf_path=str(self.cfg.urdf_path),
                tool_frame=str(self.cfg.curobo_tool_frame),
                with_world=bool(self.cfg.curobo_with_world),
                collision_activation_distance=float(self.cfg.curobo_collision_activation_distance),
                self_collision_activation_distance=float(self.cfg.curobo_self_collision_activation_distance),
                max_collision_distance=float(self.cfg.curobo_max_collision_distance),
            )
            checker = get_curobo_collision_checker(ccfg)
            q_h = np.asarray(q_traj[:, 1:, :], dtype=np.float32)
            d_scene, d_self = checker.batch_distance(q_h, scene_cuboids=scene_cuboids)
            if d_scene.size > 0:
                dmin_scene = float(np.min(d_scene))
                axes = tuple(range(1, int(d_scene.ndim)))
                t_scene = (float(self.cfg.w_scene_collision) * np.sum(d_scene, axis=axes)).astype(np.float32)
            if d_self.size > 0:
                dmin_self = float(np.min(d_self))
                axes = tuple(range(1, int(d_self.ndim)))
                t_self = (float(self.cfg.w_self_collision) * np.sum(d_self, axis=axes)).astype(np.float32)

        if bool(self.cfg.use_pointworld_cost) and float(self.cfg.w_pointworld) > 0.0:
            if pointworld_obs is None or pointworld_cost_fn is None:
                pw_reason = "missing_inputs"
            else:
                t_pw0 = __import__("time").perf_counter()
                try:
                    pw_cost = pointworld_cost_fn(
                        q_traj=np.asarray(q_traj[:, 1:, :], dtype=np.float32),
                        u_traj=np.asarray(u_traj, dtype=np.float32),
                        pointworld_obs=pointworld_obs,
                        gripper=float(gripper) if gripper is not None else None,
                    )
                    pw_ms = (__import__("time").perf_counter() - t_pw0) * 1000.0
                    timeout_ms = float(self.cfg.pointworld_cost_timeout_ms)
                    if timeout_ms > 0.0 and pw_ms > timeout_ms:
                        pw_reason = f"timeout:{pw_ms:.1f}>{timeout_ms:.1f}"
                    else:
                        arr = np.asarray(pw_cost, dtype=np.float32)
                        if arr.ndim != 1 or arr.shape[0] != q_traj.shape[0]:
                            pw_reason = "invalid_shape"
                        elif not bool(np.all(np.isfinite(arr))):
                            pw_reason = "nonfinite"
                        else:
                            t_pw = float(self.cfg.w_pointworld) * arr
                            pw_reason = "ok"
                except Exception:
                    pw_ms = (__import__("time").perf_counter() - t_pw0) * 1000.0
                    pw_reason = "exception"

        total = (t_smooth + t_action + t_joint_limit + t_ee_pos + t_link7_pos + t_scene + t_self + t_pw).astype(np.float32)
        extra: dict[str, Any] = {
            "terms": {
                "cost_smooth": t_smooth,
                "cost_action": t_action,
                "cost_joint_limit": t_joint_limit,
                "cost_ee_pos": t_ee_pos,
                "cost_link7_pos": t_link7_pos,
                "cost_scene": t_scene,
                "cost_self": t_self,
                "cost_pointworld": t_pw,
            },
            "min_distance_scene": dmin_scene,
            "min_distance_self": dmin_self,
            "pointworld_ms": pw_ms,
            "pointworld_reason": pw_reason,
        }
        return total, extra

    def infer_actions(
        self,
        q0: list[float],
        gripper: float,
        seed: Optional[int] = None,
        pcd_back_cam: Optional[object] = None,
        pointworld_obs: Optional[dict[str, Any]] = None,
        pointworld_cost_fn: Optional[PointWorldCostFn] = None,
    ) -> np.ndarray:
        self.last_fallback = False
        self.last_fallback_reason = ""
        self.last_effective_samples_ratio = 0.0

        self.last_scene_num_cuboids = 0
        self.last_scene_num_robot_spheres = 0
        self.last_scene_has_table = False
        self.last_scene_num_dynamic_tracks = 0
        self.last_scene_key_short = ""
        self.last_scene_cuboids: list[dict[str, Any]] = []

        self.last_infer_ms = 0.0
        self.last_timing_policy_suffix = ""
        self.last_cost_stats = {}
        self.last_min_distance_scene = float("nan")
        self.last_min_distance_self = float("nan")
        self.last_pw_enabled = False
        self.last_pw_reason = ""
        self.last_pw_ms = 0.0

        t_infer0 = __import__("time").perf_counter()

        cfg = self.cfg
        T = int(cfg.horizon)
        K_cfg = int(cfg.num_samples)
        if bool(cfg.use_pointworld_cost) and bool(cfg.pointworld_require_horizon_11) and T != 11:
            raise ValueError(f"PointWorld requires horizon=11, got {T}")

        budget_ms = float(cfg.infer_budget_ms)
        budget_enabled = budget_ms > 0.0

        used_freeze_scene = False
        used_limit_cuboids = False
        used_reduce_samples = False

        if self._degrade_level >= 4:
            self.last_fallback = True
            self.last_fallback_reason = "budget_hold"
            self.last_timing_policy_suffix = ":hold"
            actions = np.empty((T, 8), dtype=np.float32)
            q0_np = np.asarray(q0, dtype=np.float32).reshape(7)
            q0_np = self._clip_q(q0_np)
            actions[:, 0:7] = q0_np[None, :]
            actions[:, 7] = float(gripper)
            self.last_infer_ms = (__import__("time").perf_counter() - t_infer0) * 1000.0
            return actions

        K = int(K_cfg)
        if self._degrade_level >= 3:
            K = max(32, int(K_cfg // 2))
            used_reduce_samples = True

        scene_cuboids: Optional[list[dict[str, Any]]] = None
        if (
            bool(cfg.use_curobo_collision)
            and bool(cfg.scene_from_pcd_back_cam)
            and float(cfg.w_scene_collision) > 0.0
        ):
            try:
                stable_dynamic: list[dict[str, Any]]
                if self._degrade_level >= 1 and self._last_scene_dynamic:
                    stable_dynamic = list(self._last_scene_dynamic)
                    used_freeze_scene = True
                else:
                    if pcd_back_cam is None:
                        stable_dynamic = []
                    else:
                        checker = get_curobo_collision_checker(
                            CuRoboCollisionConfig(
                                device=str(cfg.curobo_device),
                                robot_yaml=str(cfg.curobo_robot_yaml),
                                urdf_path=str(cfg.urdf_path),
                                tool_frame=str(cfg.curobo_tool_frame),
                                with_world=True,
                                collision_activation_distance=float(cfg.curobo_collision_activation_distance),
                                self_collision_activation_distance=float(cfg.curobo_self_collision_activation_distance),
                                max_collision_distance=float(cfg.curobo_max_collision_distance),
                            )
                        )
                        spheres = checker.get_robot_spheres_base(
                            np.asarray(q0, dtype=np.float32),
                            margin_m=float(cfg.scene_robot_mask_margin_m),
                        )
                        self.last_scene_num_robot_spheres = int(spheres.shape[0])

                        table_top_z = float(cfg.scene_table_center[2]) + 0.5 * float(cfg.scene_table_dims[2])

                        sb_cfg = SceneBuildConfig(
                            t_base_cam_back_path=str(cfg.t_base_cam_back_path),
                            roi_min=cfg.scene_roi_min,
                            roi_max=cfg.scene_roi_max,
                            voxel_size_m=float(cfg.scene_voxel_size_m),
                            padding_m=float(cfg.scene_padding_m),
                            max_cuboids=int(cfg.scene_max_cuboids),
                            robot_mask_margin_m=0.0,
                            min_cluster_voxels=int(cfg.scene_min_cluster_voxels),
                            remove_table_points=bool(cfg.scene_remove_table_points),
                            table_top_z_m=float(table_top_z),
                            table_eps_m=float(cfg.scene_table_eps_m),
                            remove_wall_points=bool(cfg.scene_remove_wall_points),
                            wall_center=cfg.scene_wall_center,
                            wall_dims=cfg.scene_wall_dims,
                            wall_margin_m=float(cfg.scene_wall_margin_m),
                        )

                        pcd_pts = _extract_points_from_pcd(pcd_back_cam)
                        dynamic_cuboids = build_scene_cuboids_from_pcd_back_cam(
                            pcd_pts,
                            cfg=sb_cfg,
                            pcd_scale=float(cfg.scene_pcd_scale),
                            pcd_in_base=bool(cfg.scene_pcd_in_base),
                            robot_spheres=spheres,
                        )

                        stable_dynamic = self._scene_state.update(list(dynamic_cuboids))

                k_dyn = 0
                if self._degrade_level >= 2:
                    k_dyn = int(cfg.budget_max_dynamic_cuboids)
                    if k_dyn <= 0:
                        k_dyn = 0
                    if k_dyn > 0 and len(stable_dynamic) > k_dyn:
                        def _vol(c: dict[str, Any]) -> float:
                            d = c.get("dims")
                            if not isinstance(d, (list, tuple)) or len(d) != 3:
                                return 0.0
                            try:
                                return float(max(float(d[0]), 0.0) * max(float(d[1]), 0.0) * max(float(d[2]), 0.0))
                            except Exception:
                                return 0.0

                        stable_dynamic = sorted(stable_dynamic, key=_vol, reverse=True)[:k_dyn]
                        used_limit_cuboids = True

                self.last_scene_num_dynamic_tracks = int(len(stable_dynamic))

                try:
                    key_parts = []
                    for c in stable_dynamic:
                        center = c.get("center")
                        dims = c.get("dims")
                        if not isinstance(center, (list, tuple)) or not isinstance(dims, (list, tuple)):
                            continue
                        if len(center) != 3 or len(dims) != 3:
                            continue
                        key_parts.append(
                            f"{center[0]:.4f},{center[1]:.4f},{center[2]:.4f}|{dims[0]:.4f},{dims[1]:.4f},{dims[2]:.4f}"
                        )
                    key_parts.sort()
                    key_full = "__".join(key_parts) if key_parts else "__default__"
                    import hashlib

                    self.last_scene_key_short = hashlib.sha1(key_full.encode("utf-8")).hexdigest()[:8]
                except Exception:
                    self.last_scene_key_short = ""

                scene_cuboids = list(stable_dynamic)
                self.last_scene_has_table = False
                if bool(cfg.scene_add_table):
                    table = {
                        "center": [float(cfg.scene_table_center[0]), float(cfg.scene_table_center[1]), float(cfg.scene_table_center[2])],
                        "dims": [float(cfg.scene_table_dims[0]), float(cfg.scene_table_dims[1]), float(cfg.scene_table_dims[2])],
                    }
                    scene_cuboids = [table] + scene_cuboids
                    self.last_scene_has_table = True

                self.last_scene_num_cuboids = int(len(scene_cuboids))
                self.last_scene_cuboids = list(scene_cuboids)

                self._last_scene_dynamic = list(stable_dynamic)
                self._last_scene_has_table = bool(self.last_scene_has_table)
            except Exception:
                scene_cuboids = None
                self.last_scene_cuboids = []

        q0_np = np.asarray(q0, dtype=np.float32).reshape(7)
        q0_np = self._clip_q(q0_np)

        rng = np.random.default_rng(seed)
        eps = rng.normal(0.0, float(cfg.noise_std), size=(K, T, 7)).astype(np.float32)

        u_nom = self._u[None, :, :]
        u_cand = self._clip_u(u_nom + eps)

        q_traj = np.empty((K, T + 1, 7), dtype=np.float32)
        q_traj[:, 0, :] = q0_np[None, :]
        for t in range(T):
            q_traj[:, t + 1, :] = self._clip_q(q_traj[:, t, :] + u_cand[:, t, :])

        try:
            costs, extra = self._rollout_cost(
                q_traj,
                u_cand,
                scene_cuboids=scene_cuboids,
                pointworld_obs=pointworld_obs,
                pointworld_cost_fn=pointworld_cost_fn,
                gripper=gripper,
            )
            self.last_pw_ms = float(extra.get("pointworld_ms", 0.0))
            self.last_pw_reason = str(extra.get("pointworld_reason", ""))
            self.last_pw_enabled = self.last_pw_reason == "ok"
            if bool(cfg.debug_cost_stats):
                q = float(cfg.debug_cost_stats_q)
                if not (0.0 <= q <= 1.0):
                    q = 0.5
                terms = extra.get("terms", {}) if isinstance(extra, dict) else {}
                stats: dict[str, float] = {}
                if isinstance(terms, dict):
                    for k, v in terms.items():
                        arr = np.asarray(v, dtype=np.float64).reshape(-1)
                        arr = arr[np.isfinite(arr)]
                        if arr.size > 0:
                            stats[f"{k}_mean"] = float(arr.mean())
                            stats[f"{k}_q{int(round(q * 100.0))}"] = float(np.quantile(arr, q))
                self.last_cost_stats = stats
                self.last_min_distance_scene = float(extra.get("min_distance_scene", float("nan")))
                self.last_min_distance_self = float(extra.get("min_distance_self", float("nan")))
        except Exception:
            if bool(cfg.use_curobo_collision):
                self.last_fallback = True
                self.last_fallback_reason = "curobo_collision_error"
                actions = np.empty((T, 8), dtype=np.float32)
                actions[:, 0:7] = q0_np[None, :]
                actions[:, 7] = float(gripper)
                return actions
            raise

        c_min = float(np.min(costs))
        weights_unnorm = np.exp(-(costs - c_min) / max(1e-6, float(cfg.temperature))).astype(np.float32)
        w_sum = float(np.sum(weights_unnorm))

        if (not np.isfinite(w_sum)) or w_sum <= 0.0:
            self.last_fallback = True
            self.last_fallback_reason = "weight_sum_invalid"
            self._u = np.zeros_like(self._u)
        else:
            w_flat = (weights_unnorm / w_sum).astype(np.float32)
            ess = float(1.0 / max(1e-12, float(np.sum(w_flat * w_flat))))
            ess_ratio = ess / max(1.0, float(K))
            self.last_effective_samples_ratio = ess_ratio
            if (not np.isfinite(ess_ratio)) or ess_ratio < float(cfg.min_effective_samples_ratio):
                self.last_fallback = True
                self.last_fallback_reason = "effective_samples_too_low"
                self._u = np.zeros_like(self._u)
            else:
                w = w_flat.reshape(K, 1, 1)
                du = np.sum(w * eps, axis=0)
                self._u = self._clip_u(self._u + du)

        actions = np.empty((T, 8), dtype=np.float32)
        if self.last_fallback:
            actions[:, 0:7] = q0_np[None, :]
            actions[:, 7] = float(gripper)
            self.last_infer_ms = ( __import__("time").perf_counter() - t_infer0) * 1000.0
            return actions

        q_cur = q0_np.copy()
        for t in range(T):
            q_cur = self._clip_q(q_cur + self._u[t])
            actions[t, 0:7] = q_cur
            actions[t, 7] = float(gripper)

        if T > 1:
            self._u[:-1] = self._u[1:]
            self._u[-1] = 0.0

        self.last_infer_ms = (__import__("time").perf_counter() - t_infer0) * 1000.0

        parts: list[str] = []
        if used_freeze_scene:
            parts.append("freeze_scene")
        if used_limit_cuboids:
            parts.append("limit_cuboids")
        if used_reduce_samples:
            parts.append(f"reduce_samples{K}")

        suffix = ""
        if parts:
            suffix = ":" + ":".join(parts)

        if budget_enabled and float(self.last_infer_ms) > float(budget_ms):
            next_action = "hold"
            if self._degrade_level <= 0:
                self._degrade_level = 1
                next_action = "freeze_scene"
            elif self._degrade_level == 1:
                self._degrade_level = 2
                next_action = "limit_cuboids"
            elif self._degrade_level == 2:
                self._degrade_level = 3
                next_action = "reduce_samples"
            elif self._degrade_level == 3:
                self._degrade_level = 4
                next_action = "hold"
            else:
                self._degrade_level = 4
                next_action = "hold"
            suffix = f"{suffix}:over_budget{float(self.last_infer_ms):.1f}>{float(budget_ms):.1f}:escalate_{next_action}"
        else:
            if self._degrade_level > 0 and budget_enabled and float(self.last_infer_ms) < 0.7 * float(budget_ms):
                self._degrade_level = max(0, int(self._degrade_level) - 1)

        self.last_timing_policy_suffix = suffix
        return actions
