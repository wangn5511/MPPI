from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def _load_T_row_major_4x4(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(text)
        if isinstance(obj, dict) and "T" in obj:
            flat = obj["T"]
            if isinstance(flat, list) and len(flat) == 16:
                T = np.asarray([float(x) for x in flat], dtype=np.float32).reshape(4, 4)
                return T
    except Exception:
        pass

    vals: List[float] = []
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


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    Tm = np.asarray(T, dtype=np.float32).reshape(4, 4)
    p = np.asarray(pts, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected pts shape (N,3), got {p.shape}")
    ones = np.ones((p.shape[0], 1), dtype=np.float32)
    ph = np.concatenate([p, ones], axis=1)
    out = (ph @ Tm.T)[:, :3]
    return out.astype(np.float32)


@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def backproject_depth_to_points(
    depth: np.ndarray,
    *,
    intr: PinholeIntrinsics,
    depth_scale: float = 1.0,
    depth_min_m: float = 0.05,
    depth_max_m: float = 2.0,
    stride: int = 1,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    d = np.asarray(depth)
    if d.ndim != 2:
        raise ValueError(f"Expected depth shape (H,W), got {d.shape}")

    s = int(stride)
    if s < 1:
        s = 1

    if valid_mask is not None:
        m = np.asarray(valid_mask)
        if m.shape != d.shape:
            raise ValueError(f"valid_mask shape {m.shape} must match depth shape {d.shape}")
        m = m[::s, ::s]
    else:
        m = None

    if np.issubdtype(d.dtype, np.integer):
        z = d.astype(np.float32) * float(depth_scale)
    else:
        z = d.astype(np.float32)
        z = z * float(depth_scale)

    z = z[::s, ::s]

    H, W = z.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    fx = float(intr.fx)
    fy = float(intr.fy)
    cx = float(intr.cx)
    cy = float(intr.cy)

    uu0 = uu * float(s)
    vv0 = vv * float(s)

    x = (uu0 - cx) / fx * z
    y = (vv0 - cy) / fy * z

    ok = np.isfinite(z) & (z > float(depth_min_m)) & (z < float(depth_max_m))
    if m is not None:
        ok &= m.astype(bool)

    pts = np.stack([x[ok], y[ok], z[ok]], axis=1)
    return pts.astype(np.float32)


def lift_tracked_pixels_to_3d(
    depth: np.ndarray,
    uv: np.ndarray,
    *,
    intr: PinholeIntrinsics,
    depth_scale: float = 1.0,
    depth_min_m: float = 0.05,
    depth_max_m: float = 2.0,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(depth)
    if d.ndim != 2:
        raise ValueError(f"Expected depth shape (H,W), got {d.shape}")

    uv_arr = np.asarray(uv, dtype=np.float32)
    if uv_arr.ndim < 2 or uv_arr.shape[-1] != 2:
        raise ValueError(f"Expected uv shape (...,2), got {uv_arr.shape}")

    H, W = d.shape
    u = np.rint(uv_arr[..., 0]).astype(np.int32)
    v = np.rint(uv_arr[..., 1]).astype(np.int32)

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u_clip = np.clip(u, 0, W - 1)
    v_clip = np.clip(v, 0, H - 1)

    z_raw = d[v_clip, u_clip]
    if np.issubdtype(z_raw.dtype, np.integer):
        z = z_raw.astype(np.float32) * float(depth_scale)
    else:
        z = z_raw.astype(np.float32) * float(depth_scale)

    ok = in_bounds & np.isfinite(z) & (z > float(depth_min_m)) & (z < float(depth_max_m))

    if valid_mask is not None:
        m = np.asarray(valid_mask)
        if m.shape != d.shape:
            raise ValueError(f"valid_mask shape {m.shape} must match depth shape {d.shape}")
        ok &= m[v_clip, u_clip].astype(bool)

    fx = float(intr.fx)
    fy = float(intr.fy)
    cx = float(intr.cx)
    cy = float(intr.cy)

    x = (uv_arr[..., 0] - cx) / fx * z
    y = (uv_arr[..., 1] - cy) / fy * z

    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    exists = ok.astype(bool)

    if np.any(~exists):
        xyz = np.where(exists[..., None], xyz, np.zeros_like(xyz))

    return xyz, exists


def apply_workspace_mask_to_points(
    pts: np.ndarray,
    *,
    workspace_min: Tuple[float, float, float],
    workspace_max: Tuple[float, float, float],
) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected pts shape (N,3), got {p.shape}")
    mn = np.asarray(workspace_min, dtype=np.float32).reshape(1, 3)
    mx = np.asarray(workspace_max, dtype=np.float32).reshape(1, 3)
    keep = np.all((p >= mn) & (p <= mx), axis=1)
    return keep.astype(bool)


def apply_robot_sphere_mask_to_points(
    pts: np.ndarray,
    *,
    spheres: np.ndarray,
    margin: float = 0.0,
) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected pts shape (N,3), got {p.shape}")

    s = np.asarray(spheres, dtype=np.float32)
    if s.size == 0:
        return np.ones((p.shape[0],), dtype=bool)
    if s.ndim != 2 or s.shape[1] != 4:
        raise ValueError(f"Expected spheres shape (M,4), got {s.shape}")

    centers = s[:, :3]
    radii = s[:, 3] + float(margin)

    keep = np.ones((p.shape[0],), dtype=bool)
    for c, r in zip(centers, radii):
        d2 = np.sum((p - c[None, :]) ** 2, axis=1)
        keep &= d2 > float(r * r)
    return keep.astype(bool)


def apply_ee_filter_to_points(
    pts: np.ndarray,
    *,
    ee_filter_enabled: bool,
    ee_filter_spheres: np.ndarray,
    ee_filter_margin_m: float = 0.0,
) -> np.ndarray:
    if not bool(ee_filter_enabled):
        p = np.asarray(pts)
        if p.ndim != 2:
            raise ValueError(f"Expected pts shape (N,3), got {p.shape}")
        return np.ones((p.shape[0],), dtype=bool)

    return apply_robot_sphere_mask_to_points(pts, spheres=ee_filter_spheres, margin=float(ee_filter_margin_m))


def invert_T(T: np.ndarray) -> np.ndarray:
    Tm = np.asarray(T, dtype=np.float32).reshape(4, 4)
    R = Tm[:3, :3]
    t = Tm[:3, 3]
    Rt = R.T
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = Rt
    out[:3, 3] = (-Rt @ t).astype(np.float32)
    return out


def project_points_to_pixels(
    pts_world: np.ndarray,
    *,
    world2cam: np.ndarray,
    intr: PinholeIntrinsics,
) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pts_world, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected pts_world shape (N,3), got {p.shape}")

    Tcw = np.asarray(world2cam, dtype=np.float32).reshape(4, 4)
    ones = np.ones((p.shape[0], 1), dtype=np.float32)
    ph = np.concatenate([p, ones], axis=1)
    pc = (ph @ Tcw.T)[:, :3]

    z = pc[:, 2]
    ok = np.isfinite(z) & (z > 1e-6)

    fx = float(intr.fx)
    fy = float(intr.fy)
    cx = float(intr.cx)
    cy = float(intr.cy)

    u = np.zeros((p.shape[0],), dtype=np.float32)
    v = np.zeros((p.shape[0],), dtype=np.float32)
    u[ok] = (pc[ok, 0] / z[ok]) * fx + cx
    v[ok] = (pc[ok, 1] / z[ok]) * fy + cy
    uv = np.stack([u, v], axis=1).astype(np.float32)
    return uv, ok.astype(bool)


def compute_workspace_mask_2d(
    *,
    height: int,
    width: int,
    intr: PinholeIntrinsics,
    world2cam: np.ndarray,
    workspace_min: Tuple[float, float, float],
    workspace_max: Tuple[float, float, float],
    face_density: int = 100,
) -> np.ndarray:
    H = int(height)
    W = int(width)
    if H <= 0 or W <= 0:
        raise ValueError("height/width must be positive")

    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: cv2 (opencv-python) is required for workspace 2D mask") from e

    mn = np.asarray(workspace_min, dtype=np.float32).reshape(3)
    mx = np.asarray(workspace_max, dtype=np.float32).reshape(3)

    d = int(face_density)
    if d < 2:
        d = 2

    xs = np.linspace(mn[0], mx[0], d, dtype=np.float32)
    ys = np.linspace(mn[1], mx[1], d, dtype=np.float32)
    zs = np.linspace(mn[2], mx[2], d, dtype=np.float32)

    pts = []
    yy, zz = np.meshgrid(ys, zs)
    pts.append(np.stack([np.full_like(yy, mn[0]), yy, zz], axis=-1).reshape(-1, 3))
    pts.append(np.stack([np.full_like(yy, mx[0]), yy, zz], axis=-1).reshape(-1, 3))

    xx, zz = np.meshgrid(xs, zs)
    pts.append(np.stack([xx, np.full_like(xx, mn[1]), zz], axis=-1).reshape(-1, 3))
    pts.append(np.stack([xx, np.full_like(xx, mx[1]), zz], axis=-1).reshape(-1, 3))

    xx, yy = np.meshgrid(xs, ys)
    pts.append(np.stack([xx, yy, np.full_like(xx, mn[2])], axis=-1).reshape(-1, 3))
    pts.append(np.stack([xx, yy, np.full_like(xx, mx[2])], axis=-1).reshape(-1, 3))

    corners = np.asarray(
        [
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float32,
    )
    pts.append(corners)

    pts_w = np.concatenate(pts, axis=0).astype(np.float32)

    uv, okz = project_points_to_pixels(pts_w, world2cam=world2cam, intr=intr)
    uv = uv[okz]
    if uv.shape[0] < 3:
        return np.ones((H, W), dtype=bool)

    pts_i = np.round(uv).astype(np.int32)
    pts_i[:, 0] = np.clip(pts_i[:, 0], 0, W - 1)
    pts_i[:, 1] = np.clip(pts_i[:, 1], 0, H - 1)

    hull = cv2.convexHull(pts_i.reshape(-1, 1, 2))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def invert_T(T: np.ndarray) -> np.ndarray:
    Tm = np.asarray(T, dtype=np.float32).reshape(4, 4)
    R = Tm[:3, :3]
    t = Tm[:3, 3]
    Rt = R.T
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = Rt
    out[:3, 3] = (-Rt @ t).astype(np.float32)
    return out


def project_points_to_pixels(
    pts_world: np.ndarray,
    *,
    world2cam: np.ndarray,
    intr: PinholeIntrinsics,
) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pts_world, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected pts_world shape (N,3), got {p.shape}")

    Tcw = np.asarray(world2cam, dtype=np.float32).reshape(4, 4)
    ones = np.ones((p.shape[0], 1), dtype=np.float32)
    ph = np.concatenate([p, ones], axis=1)
    pc = (ph @ Tcw.T)[:, :3]

    z = pc[:, 2]
    ok = np.isfinite(z) & (z > 1e-6)

    fx = float(intr.fx)
    fy = float(intr.fy)
    cx = float(intr.cx)
    cy = float(intr.cy)

    u = np.zeros((p.shape[0],), dtype=np.float32)
    v = np.zeros((p.shape[0],), dtype=np.float32)
    u[ok] = (pc[ok, 0] / z[ok]) * fx + cx
    v[ok] = (pc[ok, 1] / z[ok]) * fy + cy
    uv = np.stack([u, v], axis=1).astype(np.float32)
    return uv, ok.astype(bool)


def compute_workspace_mask_2d(
    *,
    height: int,
    width: int,
    intr: PinholeIntrinsics,
    world2cam: np.ndarray,
    workspace_min: Tuple[float, float, float],
    workspace_max: Tuple[float, float, float],
) -> np.ndarray:
    H = int(height)
    W = int(width)
    if H <= 0 or W <= 0:
        raise ValueError("height/width must be positive")

    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: cv2 (opencv-python) is required for workspace 2D mask") from e

    mn = np.asarray(workspace_min, dtype=np.float32).reshape(3)
    mx = np.asarray(workspace_max, dtype=np.float32).reshape(3)

    corners = np.asarray(
        [
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float32,
    )

    uv, okz = project_points_to_pixels(corners, world2cam=world2cam, intr=intr)
    uv = uv[okz]
    if uv.shape[0] < 3:
        return np.ones((H, W), dtype=bool)

    pts = np.round(uv).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)