from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from mppi.pointworld_ext.geometry import PinholeIntrinsics, _load_T_row_major_4x4, backproject_depth_to_points, transform_points
from mppi.utils.pointcloud import (
    cluster_points_to_aabbs,
    crop_aabb,
    remove_points_in_spheres,
    voxel_downsample,
)


@dataclass(frozen=True)
class SceneBuildConfig:
    t_base_cam_back_path: str
    roi_min: Tuple[float, float, float]
    roi_max: Tuple[float, float, float]
    voxel_size_m: float
    padding_m: float
    max_cuboids: int
    robot_mask_margin_m: float
    min_cluster_voxels: int

    remove_table_points: bool = True
    table_top_z_m: float = 0.0
    table_eps_m: float = 0.01

    remove_wall_points: bool = False
    wall_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    wall_dims: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    wall_margin_m: float = 0.02


def _remove_points_in_cuboid_aabb(
    pts_base: np.ndarray,
    *,
    center: Tuple[float, float, float],
    dims: Tuple[float, float, float],
    margin_m: float,
) -> np.ndarray:
    p = np.asarray(pts_base, dtype=np.float32)
    c = np.asarray(center, dtype=np.float32).reshape(1, 3)
    d = np.asarray(dims, dtype=np.float32).reshape(1, 3)
    half = 0.5 * d + float(margin_m)
    mn = c - half
    mx = c + half
    inside = np.all((p >= mn) & (p <= mx), axis=1)
    return p[~inside]


def mask_robot_points(
    scene_pts_base: np.ndarray,
    spheres_base: np.ndarray,
    *,
    margin_m: float = 0.02,
) -> np.ndarray:
    pts = np.asarray(scene_pts_base, dtype=np.float32)
    spheres = np.asarray(spheres_base, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected scene_pts_base shape (N,3), got {pts.shape}")
    if spheres.ndim != 2 or spheres.shape[1] != 4:
        raise ValueError(f"Expected spheres_base shape (M,4), got {spheres.shape}")
    return remove_points_in_spheres(pts, spheres, margin=float(margin_m))


def build_scene_points_base_from_pcd_back_cam(
    pcd_back_cam: Any,
    *,
    cfg: SceneBuildConfig,
    pcd_scale: float = 1.0,
    pcd_in_base: bool = False,
    robot_spheres: Optional[np.ndarray] = None,
) -> np.ndarray:
    pts = np.asarray(pcd_back_cam, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected pcd_back_cam shape (N,3), got {pts.shape}")

    pts = pts * float(pcd_scale)

    if bool(pcd_in_base):
        pts_base = pts
    else:
        if not os.path.isfile(cfg.t_base_cam_back_path):
            raise FileNotFoundError(cfg.t_base_cam_back_path)
        T_base_cam = _load_T_row_major_4x4(cfg.t_base_cam_back_path)
        pts_base = transform_points(T_base_cam, pts)

    pts_base = crop_aabb(pts_base, cfg.roi_min, cfg.roi_max)

    if bool(cfg.remove_table_points):
        z_th = float(cfg.table_top_z_m) + float(cfg.table_eps_m)
        pts_base = pts_base[pts_base[:, 2] >= z_th]

    if bool(cfg.remove_wall_points):
        pts_base = _remove_points_in_cuboid_aabb(
            pts_base,
            center=cfg.wall_center,
            dims=cfg.wall_dims,
            margin_m=float(cfg.wall_margin_m),
        )

    pts_base = voxel_downsample(pts_base, cfg.voxel_size_m)

    if robot_spheres is not None:
        pts_base = mask_robot_points(pts_base, robot_spheres, margin_m=float(cfg.robot_mask_margin_m))

    return pts_base


def build_scene_cuboids_from_points_base(pts_base: np.ndarray, *, cfg: SceneBuildConfig) -> List[Dict[str, Any]]:
    pts = np.asarray(pts_base, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected pts_base shape (N,3), got {pts.shape}")

    aabbs = cluster_points_to_aabbs(
        pts,
        voxel_size=cfg.voxel_size_m,
        min_cluster_voxels=cfg.min_cluster_voxels,
    )
    aabbs.sort(key=lambda a: a.volume, reverse=True)
    aabbs = aabbs[: int(cfg.max_cuboids)]

    cuboids = [a.to_cuboid(cfg.padding_m) for a in aabbs]
    return cuboids


def build_scene_points_base_and_colors_from_pcd_back_cam(
    pcd_back_cam: Any,
    colors: Any,
    *,
    cfg: SceneBuildConfig,
    pcd_scale: float = 1.0,
    pcd_in_base: bool = False,
    robot_spheres: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pcd_back_cam, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected pcd_back_cam shape (N,3), got {pts.shape}")

    cols = np.asarray(colors)
    if cols.ndim != 2 or cols.shape[1] != 3:
        raise ValueError(f"Expected colors shape (N,3), got {cols.shape}")
    if cols.shape[0] != pts.shape[0]:
        raise ValueError(f"colors length {cols.shape[0]} must match points length {pts.shape[0]}")
    if cols.dtype != np.uint8:
        cols = np.clip(cols, 0, 255).astype(np.uint8)

    pts = pts * float(pcd_scale)

    if bool(pcd_in_base):
        pts_base = pts
    else:
        if not os.path.isfile(cfg.t_base_cam_back_path):
            raise FileNotFoundError(cfg.t_base_cam_back_path)
        T_base_cam = _load_T_row_major_4x4(cfg.t_base_cam_back_path)
        pts_base = transform_points(T_base_cam, pts)

    mn = np.asarray(cfg.roi_min, dtype=np.float32).reshape(1, 3)
    mx = np.asarray(cfg.roi_max, dtype=np.float32).reshape(1, 3)
    keep = np.all((pts_base >= mn) & (pts_base <= mx), axis=1)
    pts_base = pts_base[keep]
    cols = cols[keep]

    if bool(cfg.remove_table_points):
        z_th = float(cfg.table_top_z_m) + float(cfg.table_eps_m)
        keep = pts_base[:, 2] >= z_th
        pts_base = pts_base[keep]
        cols = cols[keep]

    if bool(cfg.remove_wall_points):
        c = np.asarray(cfg.wall_center, dtype=np.float32).reshape(1, 3)
        d = np.asarray(cfg.wall_dims, dtype=np.float32).reshape(1, 3)
        half = 0.5 * d + float(cfg.wall_margin_m)
        mnw = c - half
        mxw = c + half
        inside = np.all((pts_base >= mnw) & (pts_base <= mxw), axis=1)
        keep = ~inside
        pts_base = pts_base[keep]
        cols = cols[keep]

    vs = float(cfg.voxel_size_m)
    if pts_base.shape[0] > 0 and vs > 0.0:
        g = np.floor(pts_base / vs).astype(np.int32)
        keys = g[:, 0].astype(np.int64) * 73856093 ^ g[:, 1].astype(np.int64) * 19349663 ^ g[:, 2].astype(np.int64) * 83492791
        order = np.argsort(keys, kind="mergesort")
        keys_s = keys[order]
        _, idx = np.unique(keys_s, return_index=True)
        sel = order[idx]
        pts_base = pts_base[sel]
        cols = cols[sel]

    if robot_spheres is not None and pts_base.shape[0] > 0:
        spheres = np.asarray(robot_spheres, dtype=np.float32)
        if spheres.ndim != 2 or spheres.shape[1] != 4:
            raise ValueError(f"Expected robot_spheres shape (M,4), got {spheres.shape}")
        keep = np.ones((pts_base.shape[0],), dtype=bool)
        centers = spheres[:, :3]
        radii = spheres[:, 3] + float(cfg.robot_mask_margin_m)
        for cc, rr in zip(centers, radii):
            d2 = np.sum((pts_base - cc[None, :]) ** 2, axis=1)
            keep &= d2 > float(rr * rr)
        pts_base = pts_base[keep]
        cols = cols[keep]

    return np.ascontiguousarray(pts_base.astype(np.float32)), np.ascontiguousarray(cols.astype(np.uint8))


def build_scene_cuboids_from_pcd_back_cam(
    pcd_back_cam: Any,
    *,
    cfg: SceneBuildConfig,
    pcd_scale: float = 1.0,
    pcd_in_base: bool = False,
    robot_spheres: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    pts_base = build_scene_points_base_from_pcd_back_cam(
        pcd_back_cam,
        cfg=cfg,
        pcd_scale=float(pcd_scale),
        pcd_in_base=bool(pcd_in_base),
        robot_spheres=robot_spheres,
    )
    return build_scene_cuboids_from_points_base(pts_base, cfg=cfg)


