from __future__ import annotations

import argparse
import asyncio
import io
import time
import uuid
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from mppi.curobo_ext.check_depth import load_depth_any, load_rgb_any
from mppi.protocol.msgpack_codec import decode_message, encode_message
from mppi.protocol.types_pcl import InferRequestPCL, ObsPCL, SCHEMA_VERSION_PCL


def _require_websockets():
    try:
        import websockets  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: websockets. Install it in the container env.") from e
    return websockets


@dataclass(frozen=True)
class ClientConfig:
    url: str
    request_timeout_s: float = 2.0


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


async def infer_once(
    cfg: ClientConfig,
    *,
    q: list[float],
    gripper: float,
    step_id: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    cam_id: str,
    depth_unit_scale: Optional[float],
) -> Dict[str, Any]:
    websockets = _require_websockets()

    rgb_arr = np.asarray(rgb)
    depth_arr = np.asarray(depth)

    rgb_bytes = _encode_rgb_jpeg(rgb_arr, quality=int(90))
    depth_bytes = _encode_depth_npy_zlib(depth_arr, level=int(3))

    obs = ObsPCL(
        t_client_send_ns=time.time_ns(),
        step_id=int(step_id),
        q=list(q),
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
    payload = encode_message(req)

    async with websockets.connect(cfg.url, max_size=None) as ws:
        await ws.send(payload)
        resp_raw = await asyncio.wait_for(ws.recv(), timeout=float(cfg.request_timeout_s))

    if isinstance(resp_raw, str):
        raise RuntimeError("Expected binary msgpack payload, got text frame.")
    resp = decode_message(resp_raw)

    if int(resp.get("schema_version", -1)) != SCHEMA_VERSION_PCL:
        raise RuntimeError(f"Bad schema_version: {resp.get('schema_version')}")
    if resp.get("type") == "error_pcl":
        raise RuntimeError(f"Server error: {resp}")
    if resp.get("type") != "infer_response_pcl":
        raise RuntimeError(f"Unexpected response type: {resp.get('type')}")

    payload_dict = resp.get("payload", {})
    if not isinstance(payload_dict, dict):
        raise RuntimeError("Bad payload type")
    return dict(payload_dict)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="ws_client_sync_pcl")
    ap.add_argument("--url", type=str, required=True)
    ap.add_argument("--rgb", type=str, required=True)
    ap.add_argument("--depth", type=str, required=True)
    ap.add_argument("--request-timeout-s", type=float, default=2.0)
    ap.add_argument("--cam-id", type=str, default="back")
    ap.add_argument("--depth-unit-scale", type=float, default=None)
    ap.add_argument("--gripper", type=float, default=0.0)
    ap.add_argument("--step-id", type=int, default=0)
    ap.add_argument("--initial-q", type=str, default="")
    ap.add_argument("--print-actions", action="store_true")
    args = ap.parse_args(argv)

    q0 = [0.0] * 7
    if str(args.initial_q).strip():
        parts = [p.strip() for p in str(args.initial_q).split(",") if p.strip()]
        if len(parts) != 7:
            raise ValueError("--initial-q must be 7 comma-separated floats")
        q0 = [float(x) for x in parts]

    rgb = load_rgb_any(str(args.rgb))
    depth = load_depth_any(str(args.depth))

    cfg = ClientConfig(url=str(args.url), request_timeout_s=float(args.request_timeout_s))
    payload = asyncio.run(
        infer_once(
            cfg,
            q=q0,
            gripper=float(args.gripper),
            step_id=int(args.step_id),
            rgb=np.asarray(rgb),
            depth=np.asarray(depth),
            cam_id=str(args.cam_id),
            depth_unit_scale=(float(args.depth_unit_scale) if args.depth_unit_scale is not None else None),
        )
    )

    timing = payload.get("server_timing", {})
    policy = str(timing.get("policy", "")) if isinstance(timing, dict) else ""
    infer_ms = float(timing.get("infer_ms", float("nan"))) if isinstance(timing, dict) else float("nan")
    print(f"infer_ms={infer_ms:.3f} policy={policy}")

    if bool(args.print_actions):
        actions = payload.get("actions", None)
        if actions is not None:
            a = np.asarray(actions)
            if a.ndim >= 2 and a.shape[0] > 0:
                print("actions[0]:", a[0].tolist())


if __name__ == "__main__":
    main()