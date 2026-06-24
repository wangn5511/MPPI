from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from mppi.protocol.msgpack_codec import decode_message, encode_message
from mppi.protocol.types import InferRequestV1, ObsV1, SCHEMA_VERSION_V1


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
    run_seconds: float = 60.0
    control_hz: float = 20.0
    open_loop_horizon: int = 8
    gripper: float = 0.0


def _load_pcd_npz(path: str) -> Any:
    z = np.load(path, allow_pickle=False)
    if "points" not in z:
        raise ValueError(f"{path} missing 'points'")
    pts = np.asarray(z["points"], dtype=np.float32)
    cols = None
    if "colors" in z:
        cols = np.asarray(z["colors"], dtype=np.uint8)
    return {"points": pts, "colors": cols} if cols is not None else {"points": pts}


async def _infer_once(
    ws: Any,
    *,
    q: list[float],
    gripper: float,
    step_id: int,
    pcd_back_cam: Optional[Any],
    timeout_s: float,
) -> tuple[Dict[str, Any], int]:
    request_id = str(uuid.uuid4())
    obs = ObsV1(
        t_client_send_ns=time.time_ns(),
        step_id=int(step_id),
        q=list(q),
        gripper=float(gripper),
        pcd_back_cam=pcd_back_cam,
    )
    req = InferRequestV1(request_id=request_id, obs=obs).to_envelope()
    payload = encode_message(req)

    await ws.send(payload)
    data = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    t_client_recv_ns = time.time_ns()

    if isinstance(data, str):
        raise RuntimeError("Expected binary msgpack payload, got text frame.")
    resp = decode_message(data)
    if int(resp.get("schema_version", -1)) != SCHEMA_VERSION_V1:
        raise RuntimeError(f"Unsupported schema_version: {resp.get('schema_version')}")
    if resp.get("type") == "error":
        raise RuntimeError(f"Server error: {resp}")
    payload_dict = resp.get("payload", {})
    if not isinstance(payload_dict, dict):
        raise RuntimeError("Bad response payload")
    return dict(payload_dict), int(t_client_recv_ns)


async def run_client(
    cfg: ClientConfig,
    *,
    initial_q: list[float],
    pcd_npz: Optional[str] = None,
    pcd_a_npz: Optional[str] = None,
    pcd_b_npz: Optional[str] = None,
    ab_alternate: bool = False,
) -> None:
    websockets = _require_websockets()

    pcd_fixed = _load_pcd_npz(str(pcd_npz)) if pcd_npz else None
    pcd_a = _load_pcd_npz(str(pcd_a_npz)) if pcd_a_npz else None
    pcd_b = _load_pcd_npz(str(pcd_b_npz)) if pcd_b_npz else None
    if bool(ab_alternate) and (pcd_a is None or pcd_b is None):
        raise ValueError("ab_alternate requires both pcd_a_npz and pcd_b_npz")

    dt = 1.0 / max(1e-6, float(cfg.control_hz))
    H = int(cfg.open_loop_horizon)
    if H <= 0:
        raise ValueError("open_loop_horizon must be > 0")

    tick_dt_ms: list[float] = []
    jitter_ms: list[float] = []
    chunk_rtt_ms: list[float] = []
    send_to_exec_ms: list[float] = []
    recv_to_exec_ms: list[float] = []

    hold_ticks = 0
    policy_list: list[str] = []

    q_cur = np.asarray(initial_q, dtype=np.float32).reshape(7)
    step_id = 0
    req_count = 0

    current_actions: Optional[np.ndarray] = None
    chunk_step = 0
    chunk_send_ns = 0
    chunk_recv_ns = 0

    pending: Optional[asyncio.Task] = None

    def _pcd_for_request(k: int) -> Optional[Any]:
        if bool(ab_alternate) and (pcd_a is not None) and (pcd_b is not None):
            return pcd_a if (k % 2 == 0) else pcd_b
        return pcd_fixed

    async with websockets.connect(cfg.url, max_size=None) as ws:
        t_start = time.perf_counter()
        t_next = t_start
        t_prev_tick = None

        pending = asyncio.create_task(
            _infer_once(
                ws,
                q=q_cur.tolist(),
                gripper=float(cfg.gripper),
                step_id=int(step_id),
                pcd_back_cam=_pcd_for_request(req_count),
                timeout_s=float(cfg.request_timeout_s),
            )
        )
        req_count += 1

        while True:
            now = time.perf_counter()
            if now - t_start >= float(cfg.run_seconds):
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

            used_action = False
            if current_actions is not None and chunk_step < min(H, int(current_actions.shape[0])):
                q_cmd = np.asarray(current_actions[chunk_step, 0:7], dtype=np.float32).reshape(7)
                q_cur = q_cmd
                used_action = True
            else:
                hold_ticks += 1

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
                        gripper=float(cfg.gripper),
                        step_id=int(step_id),
                        pcd_back_cam=_pcd_for_request(req_count),
                        timeout_s=float(cfg.request_timeout_s),
                    )
                )
                req_count += 1

    def _summary_stats(name: str, vals: list[float]) -> None:
        a = np.asarray(vals, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            print(name + ": n=0")
            return
        print(
            name
            + ": n="
            + str(int(a.shape[0]))
            + " mean="
            + str(float(a.mean()))
            + " p50="
            + str(float(np.quantile(a, 0.50)))
            + " p95="
            + str(float(np.quantile(a, 0.95)))
            + " max="
            + str(float(a.max()))
        )

    print("")
    print("== Client Executor Summary ==")
    print("run_seconds:", float(cfg.run_seconds))
    print("control_hz:", float(cfg.control_hz))
    print("open_loop_horizon:", int(cfg.open_loop_horizon))
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
        for token in str(p).split("+"):
            if token.startswith("key") and len(token) > 3:
                keys.append(str(token[3:]))
                break
    print("unique key count:", int(len(set([k for k in keys if str(k).strip()]))))


def main(
    *,
    url: str,
    run_seconds: float = 60.0,
    control_hz: float = 20.0,
    open_loop_horizon: int = 8,
    request_timeout_s: float = 2.0,
    gripper: float = 0.0,
    initial_q: Optional[list[float]] = None,
    pcd_npz: Optional[str] = None,
    pcd_a_npz: Optional[str] = None,
    pcd_b_npz: Optional[str] = None,
    ab_alternate: bool = False,
) -> None:
    cfg = ClientConfig(
        url=str(url),
        request_timeout_s=float(request_timeout_s),
        run_seconds=float(run_seconds),
        control_hz=float(control_hz),
        open_loop_horizon=int(open_loop_horizon),
        gripper=float(gripper),
    )
    q0 = initial_q if initial_q is not None else [0.0] * 7
    asyncio.run(
        run_client(
            cfg,
            initial_q=q0,
            pcd_npz=pcd_npz,
            pcd_a_npz=pcd_a_npz,
            pcd_b_npz=pcd_b_npz,
            ab_alternate=bool(ab_alternate),
        )
    )


if __name__ == "__main__":
    import argparse

    def _parse_q_csv(s: str) -> list[float]:
        parts = [p.strip() for p in str(s).split(",") if p.strip()]
        if len(parts) != 7:
            raise ValueError("--initial-q must be 7 comma-separated floats")
        return [float(x) for x in parts]

    parser = argparse.ArgumentParser(prog="ws_client_sync")
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--run-seconds", type=float, default=60.0)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--request-timeout-s", type=float, default=2.0)
    parser.add_argument("--gripper", type=float, default=0.0)
    parser.add_argument("--initial-q", type=str, default="")
    parser.add_argument("--pcd-npz", type=str, default="")
    parser.add_argument("--pcd-a", type=str, default="")
    parser.add_argument("--pcd-b", type=str, default="")
    parser.add_argument("--ab-alternate", action="store_true")
    args = parser.parse_args()

    q0 = _parse_q_csv(str(args.initial_q)) if str(args.initial_q).strip() else None

    main(
        url=str(args.url),
        run_seconds=float(args.run_seconds),
        control_hz=float(args.control_hz),
        open_loop_horizon=int(args.open_loop_horizon),
        request_timeout_s=float(args.request_timeout_s),
        gripper=float(args.gripper),
        initial_q=q0,
        pcd_npz=(str(args.pcd_npz) if str(args.pcd_npz).strip() else None),
        pcd_a_npz=(str(args.pcd_a) if str(args.pcd_a).strip() else None),
        pcd_b_npz=(str(args.pcd_b) if str(args.pcd_b).strip() else None),
        ab_alternate=bool(args.ab_alternate),
    )