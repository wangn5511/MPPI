from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from mppi.pointworld_ext.geometry import (
    PinholeIntrinsics,
    apply_robot_sphere_mask_to_points,
    apply_workspace_mask_to_points,
    compute_workspace_mask_2d,
    invert_T,
    lift_tracked_pixels_to_3d,
    project_points_to_pixels,
    transform_points,
)
from mppi.pointworld_ext.input_config import PointWorldInputConfig
from mppi.pointworld_ext.query_manager import QueryPointManager
from mppi.pointworld_ext.tracker_interface import OnlinePointTracker, TrackWindowRequest
from mppi.pointworld_ext.window_buffer import PointWorldWindowBuffer


@dataclass(frozen=True)
class SceneFlowBuildOutput:
    scene_flows: np.ndarray
    scene_colors: np.ndarray
    scene_exists: np.ndarray
    scene_visibility: np.ndarray
    scene_depth_valid_mask: np.ndarray
    scene_track_confidence: np.ndarray
    camera_track_slices: Tuple[Tuple[int, int], ...]
    camera_track_ids: np.ndarray
    cameras_used: Tuple[str, ...]


def _sample_rgb_at_uv(rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    img = np.asarray(rgb)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected rgb shape (H,W,3), got {img.shape}")

    q = np.asarray(uv, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError(f"Expected uv shape (N,2), got {q.shape}")

    H, W = img.shape[0], img.shape[1]
    u = np.rint(q[:, 0]).astype(np.int32)
    v = np.rint(q[:, 1]).astype(np.int32)
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    cols = img[v, u]
    if cols.dtype != np.uint8:
        cols = np.clip(cols, 0, 255).astype(np.uint8)
    return cols


def _as_spheres_array(spheres: Sequence[Tuple[float, float, float, float]]) -> np.ndarray:
    if not spheres:
        return np.zeros((0, 4), dtype=np.float32)
    s = np.asarray(spheres, dtype=np.float32)
    if s.ndim != 2 or s.shape[1] != 4:
        raise ValueError(f"Expected spheres shape (M,4), got {s.shape}")
    return s


def _require_cv2():
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: cv2 (opencv-python) is required for 2D seed masks") from e
    return cv2


def _ensure_numpy_legacy_aliases() -> None:
    # urdfpy versions commonly bundled with robot stacks still reference
    # NumPy aliases removed in NumPy 1.24.
    for name, value in {
        "bool": bool,
        "complex": complex,
        "float": float,
        "int": int,
        "object": object,
        "str": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _require_trimesh_urdfpy():
    _ensure_numpy_legacy_aliases()
    try:
        import trimesh  # noqa: F401
        import urdfpy  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: trimesh and urdfpy are required for URDF-based masks") from e


def _get_mesh_stable_id(mesh: object, idx: int | None = None) -> str:
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
        metadata = getattr(mesh, "metadata", None) or {}
        for key in ("name", "file_name"):
            val = metadata.get(key) if isinstance(metadata, dict) else None
            if isinstance(val, (str, bytes)) and val:
                base = str(val).lower()
                return f"{base}_{idx}" if idx is not None else base
    except Exception:
        pass
    try:
        bounds = np.asarray(getattr(mesh, "bounds"), dtype=np.float32).reshape(-1)
        bounds = np.round(bounds, 6)
        vcount = len(getattr(mesh, "vertices", []))
        fcount = len(getattr(mesh, "faces", []))
        base = f"b{','.join(map(str, bounds))}_v{vcount}_f{fcount}"
        return f"{base}_{idx}" if idx is not None else base
    except Exception:
        base = f"unknown_{id(mesh)}"
        return f"{base}_{idx}" if idx is not None else base


class _URDFHelper:
    def __init__(self, *, urdf_path: str) -> None:
        _require_trimesh_urdfpy()
        _ensure_numpy_legacy_aliases()
        import urdfpy

        self._urdf = urdfpy.URDF.load(str(urdf_path))
        self._link_cache: Dict[str, object] = {}

    def _cfg_from_state(self, joint_positions: np.ndarray, gripper_positions: np.ndarray) -> Dict[str, float]:
        jp = np.asarray(joint_positions, dtype=np.float32).reshape(-1)
        if jp.shape[0] < 7:
            raise ValueError("joint_positions must have at least 7 elements")
        gp = np.asarray(gripper_positions, dtype=np.float32).reshape(-1)
        g0 = float(gp[0]) if gp.size > 0 else 0.0

        cfg: Dict[str, float] = {"finger_joint": float(g0)}
        for ji in range(7):
            cfg[f"panda_joint{ji + 1}"] = float(jp[ji])
        return cfg

    def _resolve_link(self, link_name: str) -> object:
        name = str(link_name)
        if name in self._link_cache:
            return self._link_cache[name]

        link = None
        if hasattr(self._urdf, "link_map") and isinstance(self._urdf.link_map, dict):
            link = self._urdf.link_map.get(name)
        if link is None and hasattr(self._urdf, "links"):
            for lk in self._urdf.links:
                if getattr(lk, "name", None) == name:
                    link = lk
                    break
        if link is None:
            raise ValueError(f"URDF link not found: {name}")
        self._link_cache[name] = link
        return link

    def visual_trimesh_fk(self, *, joint_positions: np.ndarray, gripper_positions: np.ndarray):
        cfg = self._cfg_from_state(joint_positions, gripper_positions)
        return self._urdf.visual_trimesh_fk(cfg=cfg)

    def transform_spheres_from_link(
        self,
        *,
        joint_positions: np.ndarray,
        gripper_positions: np.ndarray,
        link_name: str,
        spheres_link: np.ndarray,
    ) -> np.ndarray:
        s = np.asarray(spheres_link, dtype=np.float32)
        if s.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        if s.ndim != 2 or s.shape[1] != 4:
            raise ValueError(f"Expected spheres_link shape (M,4), got {s.shape}")

        import numpy as _np

        cfg = self._cfg_from_state(joint_positions, gripper_positions)
        link = self._resolve_link(link_name)
        fk = self._urdf.link_fk(cfg)
        T = _np.asarray(fk[link], dtype=_np.float32).reshape(4, 4)
        R = T[:3, :3]
        t = T[:3, 3]

        centers_l = s[:, :3]
        centers_w = (centers_l @ R.T) + t[None, :]
        out = _np.empty_like(s)
        out[:, :3] = centers_w.astype(_np.float32)
        out[:, 3] = s[:, 3]
        return out.astype(_np.float32)


class _RobotMask2DBuilder:
    def __init__(self, *, urdf_helper: _URDFHelper) -> None:
        self._urdf_helper = urdf_helper
        self._ready = False
        self._mesh_presampled_points: Dict[str, np.ndarray] = {}
        self._last_seed: Optional[int] = None

    def _presample_mesh_points(self, fk_result: Dict[object, np.ndarray], *, seed: Optional[int]) -> None:
        _require_trimesh_urdfpy()
        import trimesh

        self._mesh_presampled_points = {}
        mesh_names, mesh_objs, mesh_areas = [], [], []
        for i, mesh in enumerate(fk_result.keys()):
            name = _get_mesh_stable_id(mesh, i)
            if float(getattr(mesh, "area", 0.0)) <= 0.0:
                continue
            eff_area = float(getattr(mesh, "area"))
            if "hand_camera_part" in name.lower():
                eff_area *= 1e-6
            mesh_names.append(name)
            mesh_objs.append(mesh)
            mesh_areas.append(eff_area)

        if not mesh_names:
            self._ready = True
            self._last_seed = seed
            return

        total_area = float(np.sum(mesh_areas))
        total_samples = 100000
        gripper_multiplier = 2.0
        min_per_mesh = 500

        rng_state = None
        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(int(seed) % (2**32 - 1))

        try:
            for name, mesh, area in zip(mesh_names, mesh_objs, mesh_areas):
                frac = 0.0 if total_area <= 0 else float(area / total_area)
                n = int(total_samples * frac)
                if any(k in name.lower() for k in ["finger", "knuckle", "robotiq"]):
                    n = int(n * gripper_multiplier)
                n = max(min_per_mesh, n)
                try:
                    pts = mesh.sample(int(n))
                except Exception:
                    pts, _ = trimesh.sample.sample_surface_even(mesh, int(n))
                self._mesh_presampled_points[name] = np.asarray(pts, dtype=np.float32)
        finally:
            if rng_state is not None:
                np.random.set_state(rng_state)

        self._ready = True
        self._last_seed = seed

    def build_mask(
        self,
        *,
        intr: PinholeIntrinsics,
        world2cam: np.ndarray,
        height: int,
        width: int,
        joint_positions: np.ndarray,
        gripper_positions: np.ndarray,
        seed: Optional[int],
    ) -> np.ndarray:
        cv2 = _require_cv2()

        fk_result = self._urdf_helper.visual_trimesh_fk(joint_positions=joint_positions, gripper_positions=gripper_positions)
        if (not self._ready) or (self._last_seed != seed):
            self._presample_mesh_points(fk_result, seed=seed)

        H = int(height)
        W = int(width)
        robot_mask = np.zeros((H, W), dtype=np.uint8)

        ref_H, ref_W = 180.0, 320.0
        scale = min(float(H) / ref_H, float(W) / ref_W)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        standard_circle_radius = max(1, int(round(30.0 * scale)))
        gripper_circle_radius = max(1, int(round(20.0 * scale)))

        kernel_size = max(3, int(round(30.0 * scale)))
        if (kernel_size % 2) == 0:
            kernel_size += 1

        gripper_keywords = ["finger", "knuckle", "robotiq"]

        for i, mesh in enumerate(fk_result.keys()):
            mesh_name = _get_mesh_stable_id(mesh, i)
            if float(getattr(mesh, "area", 0.0)) <= 0.0:
                continue
            pts_local = self._mesh_presampled_points.get(mesh_name)
            if pts_local is None or pts_local.size == 0:
                continue

            is_gripper_part = any(k in mesh_name.lower() for k in gripper_keywords)
            circle_radius = gripper_circle_radius if is_gripper_part else standard_circle_radius

            T_wm = np.asarray(fk_result[mesh], dtype=np.float32)
            pts_world = transform_points(T_wm, pts_local)
            uv, okz = project_points_to_pixels(pts_world, world2cam=world2cam, intr=intr)
            if uv.shape[0] == 0:
                continue
            uv = np.round(uv[okz]).astype(np.int32)

            for (x, y) in uv:
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(robot_mask, (int(x), int(y)), int(circle_radius), color=1, thickness=-1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel_size), int(kernel_size)))
        robot_mask = cv2.morphologyEx(robot_mask, cv2.MORPH_CLOSE, kernel)
        return robot_mask.astype(bool)


class OnlineSceneFlowBuilder:
    def __init__(
        self,
        *,
        cfg: PointWorldInputConfig,
        window_buffer: PointWorldWindowBuffer,
        tracker: OnlinePointTracker,
        query_manager: QueryPointManager,
    ) -> None:
        self.cfg = cfg
        self.window_buffer = window_buffer
        self.tracker = tracker
        self.query_manager = query_manager
        self._urdf_helper = None
        self._robot_mask_builder = None

    def _ensure_urdf_helper(self) -> None:
        if self._urdf_helper is not None:
            return
        if not self.cfg.urdf_path:
            raise ValueError("cfg.urdf_path is required for URDF-based robot mask / ee filter")
        self._urdf_helper = _URDFHelper(urdf_path=str(self.cfg.urdf_path))

    def _ensure_robot_mask_builder(self) -> None:
        if self._robot_mask_builder is not None:
            return
        self._ensure_urdf_helper()
        self._robot_mask_builder = _RobotMask2DBuilder(urdf_helper=self._urdf_helper)

    def build(
        self,
        *,
        window_shift: int = 1,
        robot_spheres_base: Optional[Sequence[np.ndarray]] = None,
    ) -> SceneFlowBuildOutput:
        steps = self.window_buffer.get_window()
        T = len(steps)
        if int(T) != 11:
            raise RuntimeError("PointWorld requires window length T == 11")

        available = self.window_buffer.get_available_cameras()
        cameras_used = self.cfg.select_cameras(available)
        if not cameras_used:
            raise RuntimeError("No cameras selected")

        if robot_spheres_base is not None and len(robot_spheres_base) != T:
            raise ValueError("robot_spheres_base must be length T when provided")

        per_cam_xyz: Dict[str, np.ndarray] = {}
        per_cam_exists: Dict[str, np.ndarray] = {}
        per_cam_visibility: Dict[str, np.ndarray] = {}
        per_cam_depth_valid: Dict[str, np.ndarray] = {}
        per_cam_conf: Dict[str, np.ndarray] = {}
        per_cam_cols: Dict[str, np.ndarray] = {}
        per_cam_prepared: Dict[str, dict[str, np.ndarray | list[np.ndarray]]] = {}
        track_requests: list[TrackWindowRequest] = []

        ee_spheres_link = _as_spheres_array(self.cfg.robot_filter.ee_filter_spheres)

        seed_robot_mask_enabled = bool(getattr(self.cfg, "seed_robot_mask_enabled", False))
        if bool(getattr(self.cfg.robot_filter, "ee_filter_enabled", False)):
            self._ensure_urdf_helper()
        if seed_robot_mask_enabled:
            self._ensure_robot_mask_builder()

        for cam_name in cameras_used:
            frames = [np.asarray(s.cameras[cam_name].rgb) for s in steps]

            cam0 = steps[0].cameras[cam_name]
            rgb0 = cam0.rgb
            depth0 = cam0.depth

            H0, W0 = int(depth0.shape[0]), int(depth0.shape[1])
            world2cam0 = invert_T(cam0.extrinsics)

            valid_mask0 = np.isfinite(depth0) & (np.asarray(depth0) > 0)
            workspace_mask0 = compute_workspace_mask_2d(
                height=H0,
                width=W0,
                intr=cam0.intrinsics,
                world2cam=world2cam0,
                workspace_min=self.cfg.workspace_filter.workspace_min,
                workspace_max=self.cfg.workspace_filter.workspace_max,
            )

            if seed_robot_mask_enabled:
                robot_mask0 = self._robot_mask_builder.build_mask(
                    intr=cam0.intrinsics,
                    world2cam=world2cam0,
                    height=H0,
                    width=W0,
                    joint_positions=np.asarray(steps[0].joint_positions, dtype=np.float32),
                    gripper_positions=np.asarray(steps[0].gripper_positions, dtype=np.float32),
                    seed=(int(self.cfg.robot_mask_seed) if getattr(self.cfg, "robot_mask_seed", None) is not None else None),
                )
            else:
                robot_mask0 = np.zeros((H0, W0), dtype=bool)

            try:
                q0 = self.query_manager.get_or_create(
                    cam_name,
                    rgb=rgb0,
                    depth=depth0,
                    valid_mask=valid_mask0,
                    workspace_mask=workspace_mask0,
                    robot_mask=robot_mask0,
                )
            except TypeError:
                # Backwards-compatible fallback if the query manager doesn't support 2D gating.
                q0 = self.query_manager.get_or_create(cam_name, rgb=rgb0, depth=depth0)

            cols0 = _sample_rgb_at_uv(rgb0, q0)

            per_cam_prepared[cam_name] = {
                "q0": q0,
                "cols0": cols0,
            }
            track_requests.append(TrackWindowRequest(key=str(cam_name), frames=frames, query_points=q0))

        track_outputs = self.tracker.track_windows(tuple(track_requests))

        for cam_name in cameras_used:
            prepared = per_cam_prepared[str(cam_name)]
            q0 = np.asarray(prepared["q0"], dtype=np.float32)
            cols0 = np.asarray(prepared["cols0"], dtype=np.uint8)

            if str(cam_name) not in track_outputs:
                raise RuntimeError(f"Tracker did not return output for camera '{cam_name}'")
            track_out = track_outputs[str(cam_name)]

            uv_tracks = np.asarray(track_out.uv_tracks, dtype=np.float32)
            visibility = np.asarray(track_out.visibility).astype(bool)
            confidence = np.asarray(track_out.confidence, dtype=np.float32)

            if uv_tracks.shape != (T, q0.shape[0], 2):
                raise ValueError(f"uv_tracks shape {uv_tracks.shape} must be (T,N,2)={(T, q0.shape[0], 2)}")
            if visibility.shape != (T, q0.shape[0]):
                raise ValueError(f"visibility shape {visibility.shape} must be (T,N)={(T, q0.shape[0])}")
            if confidence.shape != (T, q0.shape[0]):
                raise ValueError(f"confidence shape {confidence.shape} must be (T,N)={(T, q0.shape[0])}")

            xyz_base = np.zeros((T, q0.shape[0], 3), dtype=np.float32)
            exists = np.zeros((T, q0.shape[0]), dtype=bool)
            vis_out = np.zeros((T, q0.shape[0]), dtype=bool)
            depth_valid_out = np.zeros((T, q0.shape[0]), dtype=bool)
            conf_out = np.zeros((T, q0.shape[0]), dtype=np.float32)
            ws_ok = np.zeros((T, q0.shape[0]), dtype=bool)

            for t in range(T):
                cam = steps[t].cameras[cam_name]
                intr: PinholeIntrinsics = cam.intrinsics
                depth_t = cam.depth

                xyz_cam, z_ok = lift_tracked_pixels_to_3d(
                    depth_t,
                    uv_tracks[t],
                    intr=intr,
                    depth_min_m=float(self.cfg.tracking.depth_min_m),
                    depth_max_m=float(self.cfg.tracking.depth_max_m),
                )
                xyzb = transform_points(cam.extrinsics, xyz_cam.reshape(-1, 3)).reshape(q0.shape[0], 3)

                keep_ws = apply_workspace_mask_to_points(
                    xyzb,
                    workspace_min=self.cfg.workspace_filter.workspace_min,
                    workspace_max=self.cfg.workspace_filter.workspace_max,
                )
                ws_ok[t] = keep_ws & z_ok

                if robot_spheres_base is None:
                    keep_robot = np.ones((q0.shape[0],), dtype=bool)
                else:
                    keep_robot = apply_robot_sphere_mask_to_points(
                        xyzb,
                        spheres=np.asarray(robot_spheres_base[t], dtype=np.float32),
                        margin=float(self.cfg.robot_filter.robot_mask_margin_m),
                    )

                if bool(self.cfg.robot_filter.ee_filter_enabled):
                    link_name = str(getattr(self.cfg.robot_filter, "ee_filter_link", ""))
                    if not link_name:
                        raise ValueError("ee_filter_enabled is True but cfg.robot_filter.ee_filter_link is empty")

                    spheres_w = self._urdf_helper.transform_spheres_from_link(
                        joint_positions=np.asarray(steps[t].joint_positions, dtype=np.float32),
                        gripper_positions=np.asarray(steps[t].gripper_positions, dtype=np.float32),
                        link_name=link_name,
                        spheres_link=ee_spheres_link,
                    )
                    keep_ee = apply_robot_sphere_mask_to_points(
                        xyzb,
                        spheres=np.asarray(spheres_w, dtype=np.float32),
                        margin=float(self.cfg.robot_filter.ee_filter_margin_m),
                    )
                else:
                    keep_ee = np.ones((q0.shape[0],), dtype=bool)

                ok = (
                    visibility[t]
                    & z_ok
                    & keep_ws
                    & keep_robot
                    & keep_ee
                    & (confidence[t] >= float(self.cfg.tracking.min_track_confidence))
                )

                vis_out[t] = visibility[t]
                depth_valid_out[t] = z_ok
                xyz_base[t] = np.where(ok[:, None], xyzb, np.zeros_like(xyzb))
                exists[t] = ok
                conf_out[t] = np.where(ok, confidence[t], 0.0).astype(np.float32)

            r_ws = ws_ok.astype(np.float32).mean(axis=0) if ws_ok.size else np.zeros((q0.shape[0],), dtype=np.float32)
            cur = np.zeros((q0.shape[0],), dtype=np.int32)
            mx = np.zeros((q0.shape[0],), dtype=np.int32)
            for t in range(T):
                cur = (cur + 1) * ws_ok[t].astype(np.int32)
                mx = np.maximum(mx, cur)

            thr = float(self.cfg.workspace_filter.stability_ws_ratio_thresh)
            run_thr = int(self.cfg.workspace_filter.stability_ws_run_len_thresh)
            stable = (r_ws >= thr) & (mx >= run_thr)

            strict_enabled = bool(getattr(self.cfg.workspace_filter, "strict_all_time_enabled", False))
            strict_all = np.all(ws_ok, axis=0) if ws_ok.size else np.zeros((q0.shape[0],), dtype=bool)

            if strict_enabled:
                exists2 = exists & strict_all[None, :]
                xyz_base = np.where(exists2[..., None], xyz_base, np.zeros_like(xyz_base))
                conf_out = np.where(exists2, conf_out, 0.0).astype(np.float32)
                exists = exists2

            if bool(self.cfg.workspace_filter.stability_apply_to_confidence):
                conf_out = (conf_out * r_ws[None, :]).astype(np.float32)

            if bool(self.cfg.workspace_filter.stability_apply_to_exists):
                exists2 = exists & stable[None, :]
                xyz_base = np.where(exists2[..., None], xyz_base, np.zeros_like(xyz_base))
                conf_out = np.where(exists2, conf_out, 0.0).astype(np.float32)
                exists = exists2

            stable_mask0 = (stable & strict_all) if strict_enabled else stable

            per_cam_xyz[cam_name] = xyz_base
            per_cam_exists[cam_name] = exists
            per_cam_visibility[cam_name] = vis_out
            per_cam_depth_valid[cam_name] = depth_valid_out
            per_cam_conf[cam_name] = conf_out
            per_cam_cols[cam_name] = np.repeat(cols0[None, :, :], T, axis=0)

            shift = int(window_shift)
            if shift < 0:
                shift = 0
            if shift >= T:
                shift = T - 1

            rgb_shift = steps[shift].cameras[cam_name].rgb
            depth_shift = steps[shift].cameras[cam_name].depth

            cam_s = steps[shift].cameras[cam_name]
            rgb_shift = cam_s.rgb
            depth_shift = cam_s.depth

            Hs, Ws = int(depth_shift.shape[0]), int(depth_shift.shape[1])
            world2cams = invert_T(cam_s.extrinsics)

            valid_masks = np.isfinite(depth_shift) & (np.asarray(depth_shift) > 0)
            workspace_masks = compute_workspace_mask_2d(
                height=Hs,
                width=Ws,
                intr=cam_s.intrinsics,
                world2cam=world2cams,
                workspace_min=self.cfg.workspace_filter.workspace_min,
                workspace_max=self.cfg.workspace_filter.workspace_max,
            )

            if seed_robot_mask_enabled:
                robot_masks = self._robot_mask_builder.build_mask(
                    intr=cam_s.intrinsics,
                    world2cam=world2cams,
                    height=Hs,
                    width=Ws,
                    joint_positions=np.asarray(steps[shift].joint_positions, dtype=np.float32),
                    gripper_positions=np.asarray(steps[shift].gripper_positions, dtype=np.float32),
                    seed=(int(self.cfg.robot_mask_seed) if getattr(self.cfg, "robot_mask_seed", None) is not None else None),
                )
            else:
                robot_masks = np.zeros((Hs, Ws), dtype=bool)

            try:
                self.query_manager.advance_window(
                    cam_name,
                    uv_tracks=uv_tracks,
                    visibility=visibility,
                    confidence=confidence,
                    stable_mask0=stable_mask0,
                    new_query_index=shift,
                    rgb0=rgb_shift,
                    depth0=depth_shift,
                    valid_mask0=valid_masks,
                    workspace_mask0=workspace_masks,
                    robot_mask0=robot_masks,
                )
            except TypeError:
                # Backwards-compatible fallback if the query manager doesn't support 2D gating / stability.
                self.query_manager.advance_window(
                    cam_name,
                    uv_tracks=uv_tracks,
                    visibility=visibility,
                    confidence=confidence,
                    new_query_index=shift,
                    rgb0=rgb_shift,
                    depth0=depth_shift,
                    valid_mask0=valid_masks,
                    workspace_mask0=workspace_masks,
                    robot_mask0=robot_masks,
                )

        xyz_list = [per_cam_xyz[n] for n in cameras_used]
        cols_list = [per_cam_cols[n] for n in cameras_used]
        exists_list = [per_cam_exists[n] for n in cameras_used]
        vis_list = [per_cam_visibility[n] for n in cameras_used]
        depth_valid_list = [per_cam_depth_valid[n] for n in cameras_used]
        conf_list = [per_cam_conf[n] for n in cameras_used]

        scene_flows = np.concatenate(xyz_list, axis=1).astype(np.float32)
        scene_colors = np.concatenate(cols_list, axis=1).astype(np.uint8)
        scene_exists = np.concatenate(exists_list, axis=1).astype(bool)
        scene_visibility = np.concatenate(vis_list, axis=1).astype(bool)
        scene_depth_valid_mask = np.concatenate(depth_valid_list, axis=1).astype(bool)
        scene_track_confidence = np.concatenate(conf_list, axis=1).astype(np.float32)

        camera_track_slices = []
        camera_track_ids = np.empty((scene_flows.shape[1],), dtype=np.int32)
        start = 0
        for i, name in enumerate(cameras_used):
            n = per_cam_xyz[name].shape[1]
            end = start + n
            camera_track_slices.append((start, end))
            camera_track_ids[start:end] = int(i)
            start = end

        return SceneFlowBuildOutput(
            scene_flows=scene_flows,
            scene_colors=scene_colors,
            scene_exists=scene_exists,
            scene_visibility=scene_visibility,
            scene_depth_valid_mask=scene_depth_valid_mask,
            scene_track_confidence=scene_track_confidence,
            camera_track_slices=tuple(camera_track_slices),
            camera_track_ids=camera_track_ids,
            cameras_used=tuple(cameras_used),
        )
