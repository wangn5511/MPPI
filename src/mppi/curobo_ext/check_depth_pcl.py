from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from mppi.curobo_ext.scene_builder import PinholeIntrinsics
from mppi.utils.pointcloud import transform_points

from .check_depth import backproject_depth_to_points_with_uv, load_pinhole_intrinsics_from_cam_info


def _default_depth_scale_for_dtype(depth: np.ndarray) -> float:
    d = np.asarray(depth)
    if np.issubdtype(d.dtype, np.integer):
        return 0.001
    return 1.0


def _voxel_downsample_indices(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float32)
    vs = float(voxel_size)
    if p.shape[0] == 0 or vs <= 0.0:
        return np.arange(p.shape[0], dtype=np.int64)
    g = np.floor(p / vs).astype(np.int32)
    keys = g[:, 0].astype(np.int64) * 73856093 ^ g[:, 1].astype(np.int64) * 19349663 ^ g[:, 2].astype(np.int64) * 83492791
    order = np.argsort(keys, kind="mergesort")
    keys_s = keys[order]
    _, idx = np.unique(keys_s, return_index=True)
    return order[idx]


def _parse_intrinsics_dict(d: Dict[str, Any]) -> Tuple[PinholeIntrinsics, Tuple[int, int]]:
    fx = float(d["fx"])
    fy = float(d["fy"])
    cx = float(d["cx"])
    cy = float(d["cy"])
    w = int(d.get("w", 0))
    h = int(d.get("h", 0))
    return PinholeIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy), (w, h)


def rgbd_to_pointcloud_base(
    *,
    depth: np.ndarray,
    rgb: Optional[np.ndarray],
    intr: PinholeIntrinsics,
    T_base_cam: np.ndarray,
    depth_unit_scale: float = 1.0,
    depth_scale: Optional[float] = None,
    depth_min_m: float = 0.05,
    depth_max_m: float = 2.0,
    stride: int = 1,
    roi_min: Optional[Tuple[float, float, float]] = None,
    roi_max: Optional[Tuple[float, float, float]] = None,
    voxel_size_m: float = 0.0,
) -> Dict[str, Any]:
    d = np.asarray(depth)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"Expected depth shape (H,W), got {d.shape}")

    rgb_arr: Optional[np.ndarray] = None
    if rgb is not None:
        rgb_arr = np.asarray(rgb)
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
            raise ValueError(f"Expected rgb shape (H,W,3), got {rgb_arr.shape}")

    if depth_scale is None:
        depth_scale = _default_depth_scale_for_dtype(d)

    effective_depth_scale = float(depth_scale) * float(depth_unit_scale)

    pts_cam, uv = backproject_depth_to_points_with_uv(
        d,
        intr=intr,
        depth_scale=float(effective_depth_scale),
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        stride=int(stride),
    )

    cols: Optional[np.ndarray] = None
    if rgb_arr is not None and uv.shape[0] > 0:
        H, W = rgb_arr.shape[0], rgb_arr.shape[1]
        u = np.clip(uv[:, 0], 0, W - 1)
        v = np.clip(uv[:, 1], 0, H - 1)
        cols = np.asarray(rgb_arr[v, u], dtype=np.uint8)

    T = np.asarray(T_base_cam, dtype=np.float32).reshape(4, 4)
    pts_base = transform_points(T, pts_cam)

    if roi_min is not None and roi_max is not None and pts_base.shape[0] > 0:
        mn = np.asarray(roi_min, dtype=np.float32).reshape(1, 3)
        mx = np.asarray(roi_max, dtype=np.float32).reshape(1, 3)
        keep = np.all((pts_base >= mn) & (pts_base <= mx), axis=1)
        pts_base = pts_base[keep]
        if cols is not None:
            cols = cols[keep]

    if float(voxel_size_m) > 0.0 and pts_base.shape[0] > 0:
        sel = _voxel_downsample_indices(pts_base, float(voxel_size_m))
        pts_base = pts_base[sel]
        if cols is not None:
            cols = cols[sel]

    out: Dict[str, Any] = {"points": np.ascontiguousarray(pts_base.astype(np.float32, copy=False))}
    if cols is not None:
        out["colors"] = np.ascontiguousarray(cols.astype(np.uint8, copy=False))
    return out


def load_intrinsics_from_cam_info_yaml(path: str) -> Tuple[PinholeIntrinsics, Tuple[int, int]]:
    intr_w, intr_h, fx, fy, cx, cy = load_pinhole_intrinsics_from_cam_info(path)
    return PinholeIntrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy)), (int(intr_w), int(intr_h))


def load_T_row_major_4x4_yaml(path: str) -> np.ndarray:
    import os

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(text)
        if isinstance(obj, dict) and "T" in obj:
            flat = obj["T"]
            if isinstance(flat, list) and len(flat) == 16:
                return np.asarray([float(x) for x in flat], dtype=np.float32).reshape(4, 4)
    except Exception:
        pass

    vals: list[float] = []
    in_T = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("T:"):
            in_T = True
            continue
        if not in_T:
            continue
        if s.startswith("order:") or s.startswith("shape:") or s.startswith("frame_id:") or s.startswith("child_frame_id:"):
            continue
        if s.startswith("-"):
            try:
                vals.append(float(s[1:].strip()))
            except Exception:
                continue
        if len(vals) >= 16:
            break
    if len(vals) != 16:
        raise ValueError(f"Failed to parse 4x4 row-major T from: {path}")
    return np.asarray(vals, dtype=np.float32).reshape(4, 4)


def parse_obs_camera_params(
    *,
    cam_id: Optional[str],
    intrinsics: Optional[Dict[str, Any]],
    T_base_cam: Optional[Any],
    cam_configs: Dict[str, Tuple[PinholeIntrinsics, np.ndarray]],
) -> Tuple[PinholeIntrinsics, np.ndarray]:
    if cam_id is not None:
        key = str(cam_id)
        if key not in cam_configs:
            raise ValueError(f"Unknown cam_id: {key}. Known: {sorted(cam_configs.keys())}")
        intr, T = cam_configs[key]
        return intr, T

    if intrinsics is None or T_base_cam is None:
        raise ValueError("ObsPCL must provide either cam_id or (intrinsics + T_base_cam)")

    intr, _ = _parse_intrinsics_dict(dict(intrinsics))
    T = np.asarray(T_base_cam, dtype=np.float32).reshape(4, 4)
    return intr, T