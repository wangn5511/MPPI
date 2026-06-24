from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import time
import uuid
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mppi.curobo_ext.check_depth import load_depth_any, load_rgb_any
from mppi.protocol.msgpack_codec import decode_message, encode_message
from mppi.protocol.types_pcl import InferRequestPCL, ObsPCL, SCHEMA_VERSION_PCL


def _require_websockets():
    try:
        import websockets  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: websockets") from e
    return websockets


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("data.json must be a list")
    return obj


def _extract_q(item: Dict[str, Any]) -> List[float]:
    js = item.get("/franka/joint_states", {})
    pos = js.get("position", None) if isinstance(js, dict) else None
    if not isinstance(pos, list) or len(pos) != 7:
        raise ValueError("Missing /franka/joint_states.position (len=7)")
    return [float(x) for x in pos]


def _extract_step_id(item: Dict[str, Any], fallback: int) -> int:
    fid = item.get("frame_id", fallback)
    try:
        return int(fid)
    except Exception:
        return int(fallback)


def _parse_policy_fields(policy: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    tab = None
    cub = None
    sph = None
    for token in str(policy).split("+"):
        if token.startswith("tab"):
            try:
                tab = int(token[3:])
            except Exception:
                pass
        elif token.startswith("cub"):
            try:
                cub = int(token[3:])
            except Exception:
                pass
        elif token.startswith("sph"):
            try:
                sph = int(token[3:])
            except Exception:
                pass
    return tab, cub, sph


def _parse_policy_key(policy: str) -> Optional[str]:
    for token in str(policy).split("+"):
        if token.startswith("key") and len(token) > 3:
            return str(token[3:])
    return None


def _normalize_rel_path(p: str) -> str:
    s = str(p).replace("\\", "/").strip()
    s = re.sub(r"^(\./)+", "", s)
    return s


def _encode_rgb_jpeg(rgb: np.ndarray, *, quality: int = 90) -> bytes:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected rgb shape (H,W,3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    try:
        import cv2  # type: ignore

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return bytes(buf.tobytes())
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        img = Image.fromarray(arr, mode="RGB")
        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=int(quality), optimize=True)
        return bio.getvalue()
    except Exception as e:
        raise RuntimeError(f"Failed to encode RGB as JPEG. Install opencv-python or pillow. Error: {e}") from e


def _encode_depth_npy_zlib(depth_m: np.ndarray, *, level: int = 3) -> bytes:
    d = np.asarray(depth_m)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"Expected depth shape (H,W), got {d.shape}")
    d = np.asarray(d, dtype=np.float32)

    bio = io.BytesIO()
    np.save(bio, d, allow_pickle=False)
    raw = bio.getvalue()
    return zlib.compress(raw, level=int(level))


def _resolve_asset_path(*, json_path: str, data_root: str, rel_or_abs: str) -> str:
    p0 = _normalize_rel_path(rel_or_abs)
    if p0.startswith("/"):
        return p0

    p1 = re.sub(r"^ep_\d+/", "", p0)

    base_dir = os.path.dirname(os.path.abspath(json_path))
    dr = str(data_root).strip() or base_dir

    candidates = [
        os.path.join(dr, p0),
        os.path.join(dr, p1),
        os.path.join(base_dir, p0),
        os.path.join(base_dir, p1),
        os.path.join(os.path.dirname(base_dir), p0),
        os.path.join(os.path.dirname(base_dir), p1),
    ]

    for c in candidates:
        if os.path.isfile(c):
            return c

    return candidates[0]


async def run_playback(
    *,
    url: str,
    json_path: str,
    data_root: str,
    cam_id: str,
    start_idx: int,
    max_steps: int,
    sleep_s: float,
    gripper: float,
    depth_unit_scale: Optional[float],
    print_actions: bool,
) -> None:
    websockets = _require_websockets()
    data = _load_json(json_path)

    if start_idx < 0 or start_idx >= len(data):
        raise ValueError(f"start_idx out of range: {start_idx} / {len(data)}")

    end = len(data) if max_steps <= 0 else min(len(data), start_idx + max_steps)

    infer_ms_list: List[float] = []
    policy_list: List[str] = []
    key_list: List[str] = []

    async with websockets.connect(url, max_size=None) as ws:
        for i in range(start_idx, end):
            item = data[i]
            q = _extract_q(item)
            step_id = _extract_step_id(item, i)

            images = item.get("images", {})
            depths = item.get("depths", {})
            if not isinstance(images, dict) or not isinstance(depths, dict):
                raise ValueError("Bad item format: images/depths must be dict")

            rgb_rel = images.get("back", "")
            depth_rel = depths.get("back_depth", "")
            if not rgb_rel or not depth_rel:
                raise ValueError("Missing images.back or depths.back_depth")

            rgb_path = _resolve_asset_path(json_path=json_path, data_root=data_root, rel_or_abs=str(rgb_rel))
            depth_path = _resolve_asset_path(json_path=json_path, data_root=data_root, rel_or_abs=str(depth_rel))

            rgb = load_rgb_any(rgb_path)
            depth = load_depth_any(depth_path)

            rgb_arr = np.asarray(rgb)
            depth_arr = np.asarray(depth)

            rgb_bytes = _encode_rgb_jpeg(rgb_arr, quality=int(90))
            depth_bytes = _encode_depth_npy_zlib(depth_arr, level=int(3))

            obs = ObsPCL(
                t_client_send_ns=time.time_ns(),
                step_id=int(step_id),
                q=q,
                gripper=float(gripper),
                rgb_codec="jpeg",
                rgb_bytes=rgb_bytes,
                rgb_shape_hw=[int(rgb_arr.shape[0]), int(rgb_arr.shape[1])],
                depth_codec="npy_zlib",
                depth_bytes=depth_bytes,
                depth_shape_hw=[int(depth_arr.shape[0]), int(depth_arr.shape[1])],
                cam_id=str(cam_id),
                depth_unit_scale=(float(depth_unit_scale) if depth_unit_scale is not None else 1.0),
            )
            req = InferRequestPCL(request_id=str(uuid.uuid4()), obs=obs).to_envelope()
            await ws.send(encode_message(req))

            resp_raw = await ws.recv()
            if isinstance(resp_raw, str):
                raise RuntimeError("Expected binary msgpack payload, got text frame.")
            resp = decode_message(resp_raw)

            if int(resp.get("schema_version", -1)) != SCHEMA_VERSION_PCL:
                raise RuntimeError(f"Bad schema_version: {resp.get('schema_version')}")
            if resp.get("type") == "error_pcl":
                raise RuntimeError(f"Server error: {resp}")
            if resp.get("type") != "infer_response_pcl":
                raise RuntimeError(f"Unexpected response type: {resp.get('type')}")

            payload = resp.get("payload", {})
            timing = payload.get("server_timing", {}) if isinstance(payload, dict) else {}
            infer_ms = float(timing.get("infer_ms", float("nan"))) if isinstance(timing, dict) else float("nan")
            policy = str(timing.get("policy", "")) if isinstance(timing, dict) else ""
            tab, cub, sph = _parse_policy_fields(policy)
            key = _parse_policy_key(policy)

            infer_ms_list.append(infer_ms)
            policy_list.append(policy)
            if key is not None:
                key_list.append(str(key))

            line = f"[{i}] frame_id={step_id} infer_ms={infer_ms:.3f} policy={policy}"
            if tab is not None or cub is not None or sph is not None:
                line += f" (tab={tab}, cub={cub}, sph={sph})"
            print(line)

            if print_actions and isinstance(payload, dict):
                actions = payload.get("actions", None)
                if actions is not None:
                    a = np.asarray(actions)
                    if a.ndim >= 2 and a.shape[0] > 0:
                        print("  actions[0]:", a[0].tolist())

            if sleep_s > 0:
                await asyncio.sleep(float(sleep_s))

    arr = np.asarray(infer_ms_list, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    print("")
    print("== Playback Summary (PCL) ==")
    print("steps:", int(arr.shape[0]))
    if arr.shape[0] > 0:
        print("infer_ms mean:", float(arr.mean()))
        print("infer_ms p50 :", float(np.quantile(arr, 0.50)))
        print("infer_ms p95 :", float(np.quantile(arr, 0.95)))
        print("infer_ms max :", float(arr.max()))

    unique_policies = sorted(set(policy_list))
    if unique_policies:
        print("unique policies:", len(unique_policies))
        for p in unique_policies[:10]:
            print("  ", p)

    unique_keys = sorted(set([k for k in key_list if str(k).strip()]))
    print("unique key count:", int(len(unique_keys)))


def main() -> None:
    ap = argparse.ArgumentParser(prog="playback_client_pcl")
    ap.add_argument("--url", type=str, required=True)
    ap.add_argument("--json", type=str, required=True)
    ap.add_argument("--data-root", type=str, default="/home/datasets/FrankaNav/test")
    ap.add_argument("--cam-id", type=str, default="back")
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--sleep-s", type=float, default=0.0)
    ap.add_argument("--gripper", type=float, default=0.0)
    ap.add_argument("--depth-unit-scale", type=float, default=None)
    ap.add_argument("--print-actions", action="store_true")
    args = ap.parse_args()

    asyncio.run(
        run_playback(
            url=str(args.url),
            json_path=str(args.json),
            data_root=str(args.data_root),
            cam_id=str(args.cam_id),
            start_idx=int(args.start_idx),
            max_steps=int(args.max_steps),
            sleep_s=float(args.sleep_s),
            gripper=float(args.gripper),
            depth_unit_scale=(float(args.depth_unit_scale) if args.depth_unit_scale is not None else None),
            print_actions=bool(args.print_actions),
        )
    )


if __name__ == "__main__":
    main()