from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mppi.protocol.msgpack_codec import decode_message, encode_message
from mppi.protocol.types import InferRequestV1, ObsV1, SCHEMA_VERSION_V1


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


def _load_pcd_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    if "points" not in z:
        raise ValueError(f"{path} missing 'points'")
    pts = np.asarray(z["points"], dtype=np.float32)
    cols = None
    if "colors" in z:
        cols = np.asarray(z["colors"], dtype=np.uint8)
    return {"points": pts, "colors": cols} if cols is not None else {"points": pts}


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


async def run_playback(
    *,
    url: str,
    json_path: str,
    start_idx: int,
    max_steps: int,
    sleep_s: float,
    gripper: float,
    print_actions: bool,
    pcd_back_cam: Optional[Dict[str, Any]],
    pcd_back_cam_a: Optional[Dict[str, Any]],
    pcd_back_cam_b: Optional[Dict[str, Any]],
    ab_alternate: bool,
) -> None:
    websockets = _require_websockets()
    data = _load_json(json_path)

    if start_idx < 0 or start_idx >= len(data):
        raise ValueError(f"start_idx out of range: {start_idx} / {len(data)}")

    end = len(data) if max_steps <= 0 else min(len(data), start_idx + max_steps)

    infer_ms_list: List[float] = []
    policy_list: List[str] = []
    key_list: List[str] = []
    key_set_a: set[str] = set()
    key_set_b: set[str] = set()

    async with websockets.connect(url, max_size=None) as ws:
        for i in range(start_idx, end):
            item = data[i]
            q = _extract_q(item)
            step_id = _extract_step_id(item, i)

            if bool(ab_alternate) and (pcd_back_cam_a is not None) and (pcd_back_cam_b is not None):
                pcd_this = pcd_back_cam_a if ((i - start_idx) % 2 == 0) else pcd_back_cam_b
            else:
                pcd_this = pcd_back_cam

            obs = ObsV1(
                t_client_send_ns=time.time_ns(),
                step_id=int(step_id),
                q=q,
                gripper=float(gripper),
                pcd_back_cam=pcd_this,
            )
            req = InferRequestV1(request_id=str(uuid.uuid4()), obs=obs).to_envelope()
            payload = encode_message(req)

            await ws.send(payload)
            resp_raw = await ws.recv()
            if isinstance(resp_raw, str):
                raise RuntimeError("Expected binary msgpack payload, got text frame.")
            resp = decode_message(resp_raw)

            if int(resp.get("schema_version", -1)) != SCHEMA_VERSION_V1:
                raise RuntimeError(f"Bad schema_version: {resp.get('schema_version')}")
            if resp.get("type") == "error":
                raise RuntimeError(f"Server error: {resp}")

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

                if bool(ab_alternate) and (pcd_back_cam_a is not None) and (pcd_back_cam_b is not None):
                    if ((i - start_idx) % 2 == 0):
                        key_set_a.add(str(key))
                    else:
                        key_set_b.add(str(key))

            line = f"[{i}] frame_id={step_id} infer_ms={infer_ms:.3f} policy={policy}"
            if tab is not None or cub is not None or sph is not None:
                line += f" (tab={tab}, cub={cub}, sph={sph})"
            print(line)

            if print_actions:
                actions = payload.get("actions", None) if isinstance(payload, dict) else None
                if actions is not None:
                    a = np.asarray(actions)
                    print("  actions[0]:", a[0].tolist())

            if sleep_s > 0:
                await asyncio.sleep(float(sleep_s))

    arr = np.asarray(infer_ms_list, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    print("")
    print("== Playback Summary ==")
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
    if bool(ab_alternate) and (pcd_back_cam_a is not None) and (pcd_back_cam_b is not None):
        print("unique (A/B) key count:", f"A={len(key_set_a)} B={len(key_set_b)}")
    else:
        print("unique (A/B) key count:", "A=- B=-")


def _parse_q_csv(s: str) -> List[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 7:
        raise ValueError("--initial-q must be 7 comma-separated floats")
    return [float(x) for x in parts]


async def _infer_once(
    ws: Any,
    *,
    q: List[float],
    gripper: float,
    step_id: int,
    pcd_back_cam: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int]:
    obs = ObsV1(
        t_client_send_ns=time.time_ns(),
        step_id=int(step_id),
        q=list(q),
        gripper=float(gripper),
        pcd_back_cam=pcd_back_cam,
    )
    req = InferRequestV1(request_id=str(uuid.uuid4()), obs=obs).to_envelope()
    payload = encode_message(req)

    await ws.send(payload)
    resp_raw = await ws.recv()
    t_client_recv_ns = time.time_ns()

    if isinstance(resp_raw, str):
        raise RuntimeError("Expected binary msgpack payload, got text frame.")
    resp = decode_message(resp_raw)

    if int(resp.get("schema_version", -1)) != SCHEMA_VERSION_V1:
        raise RuntimeError(f"Bad schema_version: {resp.get('schema_version')}")
    if resp.get("type") == "error":
        raise RuntimeError(f"Server error: {resp}")

    payload_dict = resp.get("payload", {})
    if not isinstance(payload_dict, dict):
        raise RuntimeError("Bad payload type")

    return dict(payload_dict), int(t_client_recv_ns)


async def run_executor(
    *,
    url: str,
    initial_q: List[float],
    run_seconds: float,
    control_hz: float,
    open_loop_horizon: int,
    gripper: float,
    pcd_back_cam: Optional[Dict[str, Any]],
    pcd_back_cam_a: Optional[Dict[str, Any]],
    pcd_back_cam_b: Optional[Dict[str, Any]],
    ab_alternate: bool,
) -> None:
    websockets = _require_websockets()

    dt = 1.0 / max(1e-6, float(control_hz))
    H = int(open_loop_horizon)
    if H <= 0:
        raise ValueError("open_loop_horizon must be > 0")

    tick_dt_ms: List[float] = []
    jitter_ms: List[float] = []

    chunk_rtt_ms: List[float] = []
    send_to_exec_ms: List[float] = []
    recv_to_exec_ms: List[float] = []

    hold_ticks = 0
    policy_list: List[str] = []

    q_cur = np.asarray(initial_q, dtype=np.float32).reshape(7)
    step_id = 0
    req_count = 0

    current_actions: Optional[np.ndarray] = None
    chunk_step = 0
    chunk_send_ns = 0
    chunk_recv_ns = 0

    pending: Optional[asyncio.Task] = None

    def _pcd_for_request(k: int) -> Optional[Dict[str, Any]]:
        if bool(ab_alternate) and (pcd_back_cam_a is not None) and (pcd_back_cam_b is not None):
            return pcd_back_cam_a if (k % 2 == 0) else pcd_back_cam_b
        return pcd_back_cam

    async with websockets.connect(url, max_size=None) as ws:
        t_start = time.perf_counter()
        t_next = t_start
        t_prev_tick = None

        pending = asyncio.create_task(
            _infer_once(
                ws,
                q=q_cur.tolist(),
                gripper=float(gripper),
                step_id=int(step_id),
                pcd_back_cam=_pcd_for_request(req_count),
            )
        )
        req_count += 1

        while True:
            now = time.perf_counter()
            if now - t_start >= float(run_seconds):
                break

            if pending is not None and pending.done():
                payload_dict, t_client_recv_ns = pending.result()
                pending = None

                actions = payload_dict.get("actions", None)
                if actions is not None:
                    a = np.asarray(actions, dtype=np.float32)
                    if a.ndim == 2 and a.shape[1] >= 7:
                        current_actions = a
                        chunk_step = 0

                timing = payload_dict.get("server_timing", {})
                policy = str(timing.get("policy", "")) if isinstance(timing, dict) else ""
                policy_list.append(policy)

                try:
                    t_send_echo = int(payload_dict.get("t_client_send_ns_echo", 0))
                    if t_send_echo > 0:
                        chunk_send_ns = t_send_echo
                        chunk_recv_ns = int(t_client_recv_ns)
                        chunk_rtt_ms.append((chunk_recv_ns - chunk_send_ns) / 1e6)
                except Exception:
                    chunk_send_ns = 0
                    chunk_recv_ns = 0

            now = time.perf_counter()
            sleep_s = t_next - now
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

            t_tick = time.perf_counter()
            if t_prev_tick is not None:
                dt_ms = (t_tick - float(t_prev_tick)) * 1000.0
                tick_dt_ms.append(dt_ms)
                jitter_ms.append(dt_ms - dt * 1000.0)
            t_prev_tick = t_tick
            t_next = t_next + dt

            q_cmd = q_cur
            used_action = False
            if current_actions is not None and chunk_step < min(H, int(current_actions.shape[0])):
                q_cmd = np.asarray(current_actions[chunk_step, 0:7], dtype=np.float32).reshape(7)
                used_action = True
            else:
                hold_ticks += 1

            q_cur = q_cmd

            if used_action and chunk_send_ns > 0 and chunk_recv_ns > 0:
                t_exec_ns = time.time_ns()
                send_to_exec_ms.append((t_exec_ns - int(chunk_send_ns)) / 1e6)
                recv_to_exec_ms.append((t_exec_ns - int(chunk_recv_ns)) / 1e6)

            chunk_step += 1
            step_id += 1

            if pending is None and chunk_step >= H - 1:
                pending = asyncio.create_task(
                    _infer_once(
                        ws,
                        q=q_cur.tolist(),
                        gripper=float(gripper),
                        step_id=int(step_id),
                        pcd_back_cam=_pcd_for_request(req_count),
                    )
                )
                req_count += 1

    def _summary_stats(name: str, vals: List[float]) -> None:
        a = np.asarray(vals, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.shape[0] == 0:
            print(name + ": n=0")
            return
        print(name + ": n=" + str(int(a.shape[0])) + " mean=" + str(float(a.mean())) + " p50=" + str(float(np.quantile(a, 0.50))) + " p95=" + str(float(np.quantile(a, 0.95))) + " max=" + str(float(a.max())))

    print("")
    print("== Executor Summary ==")
    print("run_seconds:", float(run_seconds))
    print("control_hz:", float(control_hz))
    print("open_loop_horizon:", int(open_loop_horizon))
    print("hold_ticks:", int(hold_ticks))
    _summary_stats("tick_dt_ms", tick_dt_ms)
    _summary_stats("jitter_ms", jitter_ms)
    _summary_stats("chunk_rtt_ms", chunk_rtt_ms)
    _summary_stats("send_to_exec_ms", send_to_exec_ms)
    _summary_stats("recv_to_exec_ms", recv_to_exec_ms)

    unique_policies = sorted(set(policy_list))
    if unique_policies:
        print("unique policies:", int(len(unique_policies)))
        for p in unique_policies[:10]:
            print("  ", p)

    keys = []
    for p in policy_list:
        k = _parse_policy_key(p)
        if k is not None and str(k).strip():
            keys.append(str(k))
    print("unique key count:", int(len(set(keys))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--json", default="/home/wangyuhan/MPPI/data/test/data.json")
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--max-steps", "--steps", dest="max_steps", type=int, default=50, help="0 means all")
    ap.add_argument("--sleep-s", type=float, default=0.1)
    ap.add_argument("--gripper", type=float, default=0.0)
    ap.add_argument("--pcd-npz", type=str, default="", help="Optional .npz with 'points' (and optional 'colors'); reused for all frames")
    ap.add_argument("--pcd-a", type=str, default="", help="Optional .npz (A) with 'points' (and optional 'colors')")
    ap.add_argument("--pcd-b", type=str, default="", help="Optional .npz (B) with 'points' (and optional 'colors')")
    ap.add_argument("--ab-alternate", action="store_true", help="Alternate between --pcd-a and --pcd-b")
    ap.add_argument("--executor", action="store_true")
    ap.add_argument("--run-seconds", type=float, default=60.0)
    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--open-loop-horizon", type=int, default=8)
    ap.add_argument("--initial-q", type=str, default="")
    ap.add_argument("--print-actions", action="store_true")
    args = ap.parse_args()

    pcd_back_cam = _load_pcd_npz(str(args.pcd_npz)) if str(args.pcd_npz).strip() else None
    pcd_back_cam_a = _load_pcd_npz(str(args.pcd_a)) if str(args.pcd_a).strip() else None
    pcd_back_cam_b = _load_pcd_npz(str(args.pcd_b)) if str(args.pcd_b).strip() else None

    if bool(args.ab_alternate) and (pcd_back_cam_a is None or pcd_back_cam_b is None):
        raise ValueError("--ab-alternate requires both --pcd-a and --pcd-b")

    if bool(args.executor):
        if str(args.initial_q).strip():
            q0 = _parse_q_csv(str(args.initial_q))
        else:
            data = _load_json(str(args.json))
            if int(args.start_idx) < 0 or int(args.start_idx) >= len(data):
                raise ValueError(f"start_idx out of range: {args.start_idx} / {len(data)}")
            q0 = _extract_q(data[int(args.start_idx)])

        asyncio.run(
            run_executor(
                url=str(args.url),
                initial_q=q0,
                run_seconds=float(args.run_seconds),
                control_hz=float(args.control_hz),
                open_loop_horizon=int(args.open_loop_horizon),
                gripper=float(args.gripper),
                pcd_back_cam=pcd_back_cam,
                pcd_back_cam_a=pcd_back_cam_a,
                pcd_back_cam_b=pcd_back_cam_b,
                ab_alternate=bool(args.ab_alternate),
            )
        )
    else:
        asyncio.run(
            run_playback(
                url=str(args.url),
                json_path=str(args.json),
                start_idx=int(args.start_idx),
                max_steps=int(args.max_steps),
                sleep_s=float(args.sleep_s),
                gripper=float(args.gripper),
                print_actions=bool(args.print_actions),
                pcd_back_cam=pcd_back_cam,
                pcd_back_cam_a=pcd_back_cam_a,
                pcd_back_cam_b=pcd_back_cam_b,
                ab_alternate=bool(args.ab_alternate),
            )
        )


if __name__ == "__main__":
    main()