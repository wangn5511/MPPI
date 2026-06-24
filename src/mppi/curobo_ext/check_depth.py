import argparse
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from mppi.curobo_ext.scene_builder import PinholeIntrinsics
from mppi.utils.pointcloud import transform_points


def _read_yaml_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _try_load_yaml(path: str) -> Optional[Dict[str, Any]]:
    text = _read_yaml_text(path)
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _parse_cam_info_yaml_fallback(path: str) -> Dict[str, Any]:
    text = _read_yaml_text(path)

    def find_float_list_after_key(key: str) -> Sequence[float]:
        m = re.search(rf"{re.escape(key)}:\s*(?:\r?\n)+([\s\S]*?)(?:\r?\n\S|\Z)", text)
        if not m:
            return []
        block = m.group(1)
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", block)
        return [float(x) for x in nums]

    w = re.search(r"image_width:\s*(\d+)", text)
    h = re.search(r"image_height:\s*(\d+)", text)
    data = find_float_list_after_key("data:")

    if w is None or h is None or len(data) < 9:
        raise ValueError(f"Failed to parse camera info from {path}")

    return {
        "image_width": int(w.group(1)),
        "image_height": int(h.group(1)),
        "camera_matrix": {"data": data[:9]},
    }


def load_pinhole_intrinsics_from_cam_info(path: str) -> Tuple[int, int, float, float, float, float]:
    obj = _try_load_yaml(path)
    if obj is None:
        obj = _parse_cam_info_yaml_fallback(path)

    w = int(obj["image_width"])
    h = int(obj["image_height"])
    data = obj["camera_matrix"]["data"]
    fx = float(data[0])
    fy = float(data[4])
    cx = float(data[2])
    cy = float(data[5])
    return w, h, fx, fy, cx, cy


def load_depth_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext in {".npy"}:
        arr = np.load(path, allow_pickle=False)
        return np.asarray(arr)

    if ext in {".npz"}:
        z = np.load(path, allow_pickle=False)
        if "depth" in z:
            return np.asarray(z["depth"])
        if len(z.files) == 1:
            return np.asarray(z[z.files[0]])
        raise ValueError(f"{path} is .npz but has multiple arrays; store as key 'depth' or only one array")

    if ext in {".exr", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
        try:
            import imageio.v3 as iio  # type: ignore

            arr = iio.imread(path)
            return np.asarray(arr)
        except Exception:
            pass

        try:
            from PIL import Image  # type: ignore

            arr = np.array(Image.open(path))
            return np.asarray(arr)
        except Exception:
            pass

        try:
            import cv2  # type: ignore

            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"cv2 failed to read {path}")
            return np.asarray(arr)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read depth image {path}. "
                f"Install one of: imageio, pillow, opencv-python. Original error: {e}"
            ) from e

    raise ValueError(f"Unsupported depth file extension: {ext}. Use .npy/.npz or an image file.")


def _index_rgb_dir(rgb_dir: str) -> Dict[str, str]:
    root = Path(rgb_dir)
    if not root.is_dir():
        raise FileNotFoundError(rgb_dir)
    out: Dict[str, str] = {}
    for name in os.listdir(str(root)):
        p = root / name
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg"}:
            continue
        out[p.stem] = str(p)
    return out


def load_rgb_any(path: str) -> np.ndarray:
    try:
        import cv2  # type: ignore

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2 failed to read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.asarray(img)
    except Exception as e:
        raise RuntimeError(f"Failed to read RGB image {path}: {e}") from e


def backproject_depth_to_points_with_uv(
    depth: np.ndarray,
    *,
    intr: PinholeIntrinsics,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    stride: int,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(depth)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
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

    z = d.astype(np.float32) * float(depth_scale)
    z = z[::s, ::s]

    H, W = z.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    fx = float(intr.fx)
    fy = float(intr.fy)
    cx = float(intr.cx)
    cy = float(intr.cy)

    uu0 = (uu * float(s)).astype(np.int32)
    vv0 = (vv * float(s)).astype(np.int32)

    x = (uu0.astype(np.float32) - cx) / fx * z
    y = (vv0.astype(np.float32) - cy) / fy * z

    ok = np.isfinite(z) & (z > float(depth_min_m)) & (z < float(depth_max_m))
    if m is not None:
        ok &= m.astype(bool)

    pts = np.stack([x[ok], y[ok], z[ok]], axis=1).astype(np.float32, copy=False)
    uv = np.stack([uu0[ok], vv0[ok]], axis=1).astype(np.int32, copy=False)
    return np.ascontiguousarray(pts), np.ascontiguousarray(uv)


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


def save_pointcloud_npz(path: str, pts: np.ndarray, colors: Optional[np.ndarray]) -> None:
    out = Path(path)
    if str(out.parent):
        os.makedirs(str(out.parent), exist_ok=True)
    p = np.asarray(pts, dtype=np.float32)
    if colors is None:
        np.savez_compressed(str(out), points=p)
        return
    c = np.asarray(colors)
    if c.dtype != np.uint8:
        c = np.clip(c, 0, 255).astype(np.uint8)
    np.savez_compressed(str(out), points=p, colors=c)


def save_pointcloud_ply(path: str, pts: np.ndarray, colors: Optional[np.ndarray]) -> None:
    out = Path(path)
    if str(out.parent):
        os.makedirs(str(out.parent), exist_ok=True)
    p = np.asarray(pts, dtype=np.float32)
    c: Optional[np.ndarray] = None
    if colors is not None:
        c = np.asarray(colors)
        if c.dtype != np.uint8:
            c = np.clip(c, 0, 255).astype(np.uint8)
    has_c = c is not None
    with open(str(out), "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {p.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_c:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if has_c:
            for i in range(p.shape[0]):
                x, y, z = p[i]
                r, g, b = c[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
        else:
            for i in range(p.shape[0]):
                x, y, z = p[i]
                f.write(f"{x} {y} {z}\n")


def depth_stats(depth: np.ndarray) -> Dict[str, Any]:
    d = np.asarray(depth)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"Expected depth as (H,W), got {d.shape}")

    info: Dict[str, Any] = {}
    info["shape"] = tuple(int(x) for x in d.shape)
    info["dtype"] = str(d.dtype)

    if np.issubdtype(d.dtype, np.floating):
        finite = np.isfinite(d)
        info["nan_count"] = int(np.isnan(d).sum())
        info["inf_count"] = int(np.isinf(d).sum())
        df = d[finite]
        if df.size == 0:
            info["finite_min"] = None
            info["finite_median"] = None
            info["finite_max"] = None
        else:
            info["finite_min"] = float(np.min(df))
            info["finite_median"] = float(np.median(df))
            info["finite_max"] = float(np.max(df))
    else:
        di = d.astype(np.int64, copy=False)
        info["min"] = int(np.min(di))
        info["median"] = float(np.median(di))
        info["max"] = int(np.max(di))

    if np.issubdtype(d.dtype, np.floating):
        nonzero = np.count_nonzero((d != 0.0) & np.isfinite(d))
    else:
        nonzero = np.count_nonzero(d != 0)
    info["nonzero_ratio"] = float(nonzero / d.size)
    return info


def sample_depth_values(depth: np.ndarray, pixels_uv: Sequence[Tuple[int, int]]) -> Sequence[Tuple[int, int, float]]:
    d = np.asarray(depth)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    H, W = d.shape
    out = []
    for u, v in pixels_uv:
        uu = int(np.clip(u, 0, W - 1))
        vv = int(np.clip(v, 0, H - 1))
        out.append((uu, vv, float(d[vv, uu])))
    return out


def suggest_intrinsics_scale(
    intr_w: int, intr_h: int, fx: float, fy: float, cx: float, cy: float, depth_w: int, depth_h: int
) -> Dict[str, Any]:
    sx = depth_w / float(intr_w)
    sy = depth_h / float(intr_h)
    return {
        "scale_x": sx,
        "scale_y": sy,
        "fx_scaled": fx * sx,
        "fy_scaled": fy * sy,
        "cx_scaled": cx * sx,
        "cy_scaled": cy * sy,
    }


def normalized_offsets(u: float, v: float, fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float]:
    x = (u - cx) / fx
    y = (v - cy) / fy
    return float(x), float(y)


def _default_depth_scale_for_dtype(depth: np.ndarray) -> float:
    d = np.asarray(depth)
    if np.issubdtype(d.dtype, np.integer):
        return 0.001
    return 1.0


def _load_T_row_major_4x4(path: str) -> np.ndarray:
    text = _read_yaml_text(path)

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


def _parse_vec3_csv(s: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated floats, got: {s}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _print_qstats(name: str, pts: np.ndarray) -> None:
    p = np.asarray(pts, dtype=np.float32)
    if p.size == 0:
        print(f"{name} N 0")
        return
    q = np.quantile(p, [0.01, 0.5, 0.99], axis=0)
    print(f"{name} N {p.shape[0]}")
    print(f"  q01 xyz: {q[0]}")
    print(f"  q50 xyz: {q[1]}")
    print(f"  q99 xyz: {q[2]}")


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(x))
    if n <= 0:
        return x * 0.0
    return x / n


def main() -> None:
    ap = argparse.ArgumentParser()

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--depth", type=str, default="", help="Depth file: .npy/.npz or image (.png/.exr/...)")
    src.add_argument("--depth-dir", type=str, default="", help="Directory containing depth files (.npy)")

    ap.add_argument("--rgb-dir", type=str, default="", help="Directory containing RGB images (optional)")

    ap.add_argument("--out", type=str, default="", help="Output point cloud path (.npz); if empty, do not save")
    ap.add_argument("--save-ply", type=str, default="", help="Optional output .ply path")

    ap.add_argument("--merge", action="store_true", help="Merge all frames into a single scene point cloud")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit number of frames (0 = all)")

    ap.add_argument("--cam-info", required=True, help="CameraInfo YAML, e.g. configs/back_cam_info.yaml")
    ap.add_argument("--depth-scale", type=float, default=None, help="Meters per depth unit before applying --depth-unit-scale")
    ap.add_argument("--depth-unit-scale", type=float, default=1.0, help="Extra scale factor applied to depth values (e.g. 0.01 for cm->m)")
    ap.add_argument("--depth-min-m", type=float, default=0.05)
    ap.add_argument("--depth-max-m", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--voxel-size-m", type=float, default=0.0, help="Voxel downsample size in meters (0 disables)")

    ap.add_argument(
        "--t-base-cam",
        type=str,
        default="",
        help="Extrinsics YAML with 4x4 row-major T (base<-cam), used as p_base = T_base_cam * p_cam",
    )
    ap.add_argument("--roi-min", type=str, default="", help="ROI min in base as x,y,z")
    ap.add_argument("--roi-max", type=str, default="", help="ROI max in base as x,y,z")
    ap.add_argument("--max-points", type=int, default=300000)
    ap.add_argument("--compare-base-b", action="store_true")
    ap.add_argument(
        "--pixels",
        type=str,
        default="",
        help="Extra pixels to sample, format: u,v;u,v;... (e.g. 320,240;100,200)",
    )
    args = ap.parse_args()

    if str(args.depth_dir).strip():
        depth_paths = sorted(Path(str(args.depth_dir)).glob("*.npy"))
    else:
        depth_paths = [Path(str(args.depth))]

    if not depth_paths:
        raise SystemExit("No depth files found")

    if int(args.max_frames) > 0:
        depth_paths = depth_paths[: int(args.max_frames)]

    rgb_index: Dict[str, str] = {}
    if str(args.rgb_dir).strip():
        rgb_index = _index_rgb_dir(str(args.rgb_dir))

    intr_w, intr_h, fx, fy, cx, cy = load_pinhole_intrinsics_from_cam_info(args.cam_info)
    intr = PinholeIntrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))

    T_base_cam: Optional[np.ndarray] = None
    if str(args.t_base_cam).strip():
        T_base_cam = _load_T_row_major_4x4(str(args.t_base_cam)).astype(np.float32)

    pts_all: list[np.ndarray] = []
    col_all: list[np.ndarray] = []

    for i, dp in enumerate(depth_paths):
        depth = load_depth_any(str(dp))
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise SystemExit(f"Depth must be 2D (H,W). Got {depth.shape} from {dp}")

        if i == 0:
            H, W = depth.shape
            print("== A) Depth raw stats ==")
            s = depth_stats(depth)
            for k in sorted(s.keys()):
                print(f"{k}: {s[k]}")

            depth_scale = float(args.depth_scale) if args.depth_scale is not None else _default_depth_scale_for_dtype(depth)
            depth_unit_scale = float(args.depth_unit_scale)
            effective_depth_scale = float(depth_scale) * float(depth_unit_scale)
            print(f"depth_scale_used: {depth_scale}")
            print(f"depth_unit_scale: {depth_unit_scale}")
            print(f"effective_depth_scale_used: {effective_depth_scale}")

            center_u = W // 2
            center_v = H // 2
            px = [(center_u, center_v), (int(round(cx)), int(round(cy)))]
            if args.pixels.strip():
                for item in args.pixels.split(";"):
                    item = item.strip()
                    if not item:
                        continue
                    u_str, v_str = item.split(",")
                    px.append((int(u_str), int(v_str)))

            samples = sample_depth_values(depth, px)
            print("sample_pixels_raw:")
            for u, v, val in samples:
                z_m = float(val) * float(effective_depth_scale)
                ok = (math.isfinite(z_m) and (z_m > args.depth_min_m) and (z_m < args.depth_max_m))
                print(f"  (u={u}, v={v}) raw={val} z_m={z_m:.6f} ok[{args.depth_min_m},{args.depth_max_m}]={ok}")

            print("")
            print("== B) Resolution vs intrinsics ==")
            print(f"depth_shape_hw: ({H}, {W})")
            print(f"intrinsics_image_hw: ({intr_h}, {intr_w})")
            print(f"fx,fy,cx,cy: {fx}, {fy}, {cx}, {cy}")
            if (W != intr_w) or (H != intr_h):
                sug = suggest_intrinsics_scale(intr_w, intr_h, fx, fy, cx, cy, W, H)
                print("WARNING: Depth resolution != intrinsics resolution")
                for k in ["scale_x", "scale_y", "fx_scaled", "fy_scaled", "cx_scaled", "cy_scaled"]:
                    print(f"{k}: {sug[k]}")
            else:
                print("OK: Depth resolution matches intrinsics resolution")

            print("")
            print("== B) Principal point sanity ==")
            img_center_u = (W - 1) * 0.5
            img_center_v = (H - 1) * 0.5
            print(f"image_center_uv: ({img_center_u:.3f}, {img_center_v:.3f})")
            print(f"principal_point_uv: ({cx:.3f}, {cy:.3f})")
            print(f"principal_point_offset_pixels: du={cx - img_center_u:.3f}, dv={cy - img_center_v:.3f}")
            u_test = int(np.clip(int(round(cx)), 0, W - 1))
            v_test = int(np.clip(int(round(cy)), 0, H - 1))
            nx, ny = normalized_offsets(u_test, v_test, fx, fy, cx, cy)
            print(f"normalized_offset_at_round(cx,cy): nx={nx:.6e}, ny={ny:.6e} (should be near 0)")
            nx2, ny2 = normalized_offsets(center_u, center_v, fx, fy, cx, cy)
            print(f"normalized_offset_at_image_center: nx={nx2:.6e}, ny={ny2:.6e} (should be small if cx,cy near center)")

            if T_base_cam is not None:
                print("")
                print("== C) Extrinsics sanity (parent_T_child: base<-cam) ==")
                print("T_base_cam t (cam origin in base):", T_base_cam[:3, 3])

        depth_scale = float(args.depth_scale) if args.depth_scale is not None else _default_depth_scale_for_dtype(depth)
        effective_depth_scale = float(depth_scale) * float(args.depth_unit_scale)

        pts_cam, uv = backproject_depth_to_points_with_uv(
            depth,
            intr=intr,
            depth_scale=float(effective_depth_scale),
            depth_min_m=float(args.depth_min_m),
            depth_max_m=float(args.depth_max_m),
            stride=int(args.stride),
        )

        cols: Optional[np.ndarray] = None
        if rgb_index:
            rp = rgb_index.get(dp.stem)
            if rp is not None:
                rgb = load_rgb_any(rp)
                cols = rgb[uv[:, 1], uv[:, 0]]

        pts = pts_cam
        if T_base_cam is not None:
            pts = transform_points(T_base_cam, pts)

        if args.roi_min and args.roi_max:
            roi_min = _parse_vec3_csv(str(args.roi_min))
            roi_max = _parse_vec3_csv(str(args.roi_max))
            keep = np.all(
                (pts >= np.asarray(roi_min, dtype=np.float32)) & (pts <= np.asarray(roi_max, dtype=np.float32)), axis=1
            )
            pts = pts[keep]
            if cols is not None:
                cols = cols[keep]

        if int(args.max_points) > 0 and pts.shape[0] > int(args.max_points):
            idx = np.random.default_rng(0).choice(pts.shape[0], size=int(args.max_points), replace=False)
            pts = pts[idx]
            if cols is not None:
                cols = cols[idx]

        pts_all.append(np.asarray(pts, dtype=np.float32))
        if cols is not None:
            col_all.append(np.asarray(cols, dtype=np.uint8))

        if (not bool(args.merge)) and str(args.out).strip():
            out_path = str(Path(str(args.out)).with_name(f"{dp.stem}.npz"))
            save_pointcloud_npz(out_path, pts, cols)

    if not bool(args.merge):
        return

    pts_scene = np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3), dtype=np.float32)
    cols_scene: Optional[np.ndarray] = None
    if col_all and sum(x.shape[0] for x in col_all) == pts_scene.shape[0]:
        cols_scene = np.concatenate(col_all, axis=0)

    vs = float(args.voxel_size_m)
    sel = _voxel_downsample_indices(pts_scene, vs)
    pts_scene = pts_scene[sel]
    if cols_scene is not None:
        cols_scene = cols_scene[sel]

    if str(args.out).strip():
        save_pointcloud_npz(str(args.out), pts_scene, cols_scene)

    if str(args.save_ply).strip():
        save_pointcloud_ply(str(args.save_ply), pts_scene, cols_scene)

    print("")
    print("== Scene point cloud ==")
    print("frames:", len(depth_paths))
    print("points:", int(pts_scene.shape[0]))


if __name__ == "__main__":
    main()