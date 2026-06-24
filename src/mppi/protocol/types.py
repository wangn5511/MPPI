from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict

SCHEMA_VERSION_V1 = 1


class Envelope(TypedDict):
    schema_version: int
    type: str
    request_id: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class ObsV1:
    t_client_send_ns: int
    step_id: int
    q: List[float]
    gripper: float
    dq: Optional[List[float]] = None
    ee_pose: Optional[Dict[str, Any]] = None
    pcd_back_cam: Optional[Any] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "t_client_send_ns": int(self.t_client_send_ns),
            "step_id": int(self.step_id),
            "q": list(self.q),
            "gripper": float(self.gripper),
        }
        if self.dq is not None:
            payload["dq"] = list(self.dq)
        if self.ee_pose is not None:
            payload["ee_pose"] = dict(self.ee_pose)
        if self.pcd_back_cam is not None:
            payload["pcd_back_cam"] = self.pcd_back_cam
        return payload

    @staticmethod
    def from_payload(payload: Dict[str, Any]) -> "ObsV1":
        return ObsV1(
            t_client_send_ns=int(payload["t_client_send_ns"]),
            step_id=int(payload["step_id"]),
            q=list(payload["q"]),
            gripper=float(payload["gripper"]),
            dq=list(payload["dq"]) if "dq" in payload and payload["dq"] is not None else None,
            ee_pose=dict(payload["ee_pose"]) if "ee_pose" in payload and payload["ee_pose"] is not None else None,
            pcd_back_cam=payload.get("pcd_back_cam"),
        )


@dataclass(frozen=True)
class InferRequestV1:
    request_id: str
    obs: ObsV1

    def to_envelope(self) -> Envelope:
        return {
            "schema_version": SCHEMA_VERSION_V1,
            "type": "infer_request",
            "request_id": self.request_id,
            "payload": self.obs.to_payload(),
        }


class PolicyNameLiteral(TypedDict):
    name: str


@dataclass(frozen=True)
class ServerTimingV1:
    infer_ms: float
    queue_ms: float
    policy: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "infer_ms": float(self.infer_ms),
            "queue_ms": float(self.queue_ms),
            "policy": str(self.policy),
        }


@dataclass(frozen=True)
class ActionChunkV1:
    t_server_recv_ns: int
    t_server_send_ns: int
    t_client_send_ns_echo: int
    open_loop_horizon: int
    actions: Any
    server_timing: ServerTimingV1

    def to_payload(self) -> Dict[str, Any]:
        return {
            "t_server_recv_ns": int(self.t_server_recv_ns),
            "t_server_send_ns": int(self.t_server_send_ns),
            "t_client_send_ns_echo": int(self.t_client_send_ns_echo),
            "open_loop_horizon": int(self.open_loop_horizon),
            "actions": self.actions,
            "server_timing": self.server_timing.to_dict(),
        }


@dataclass(frozen=True)
class InferResponseV1:
    request_id: str
    action_chunk: ActionChunkV1

    def to_envelope(self) -> Envelope:
        return {
            "schema_version": SCHEMA_VERSION_V1,
            "type": "infer_response",
            "request_id": self.request_id,
            "payload": self.action_chunk.to_payload(),
        }


@dataclass(frozen=True)
class ErrorV1:
    request_id: str
    code: str
    message: str
    t_server_send_ns: int

    def to_envelope(self) -> Envelope:
        return {
            "schema_version": SCHEMA_VERSION_V1,
            "type": "error",
            "request_id": self.request_id,
            "payload": {
                "request_id": self.request_id,
                "code": str(self.code),
                "message": str(self.message),
                "t_server_send_ns": int(self.t_server_send_ns),
            },
        }