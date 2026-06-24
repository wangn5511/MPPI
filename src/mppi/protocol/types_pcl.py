from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

SCHEMA_VERSION_PCL = 100


class EnvelopePCL(TypedDict):
    schema_version: int
    type: str
    request_id: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class ObsPCL:
    t_client_send_ns: int
    step_id: int
    q: List[float]
    gripper: float

    rgb_codec: Optional[str] = None
    rgb_bytes: Optional[Any] = None
    rgb_shape_hw: Optional[List[int]] = None

    depth_codec: Optional[str] = None
    depth_bytes: Optional[Any] = None
    depth_shape_hw: Optional[List[int]] = None

    rgb_back: Optional[Any] = None
    depth_back: Optional[Any] = None

    cam_id: Optional[str] = None
    intrinsics: Optional[Dict[str, Any]] = None
    depth_unit_scale: Optional[float] = None
    T_base_cam: Optional[Any] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "t_client_send_ns": int(self.t_client_send_ns),
            "step_id": int(self.step_id),
            "q": list(self.q),
            "gripper": float(self.gripper),
        }

        if self.rgb_codec is not None:
            payload["rgb_codec"] = str(self.rgb_codec)
        if self.rgb_bytes is not None:
            payload["rgb_bytes"] = self.rgb_bytes
        if self.rgb_shape_hw is not None:
            payload["rgb_shape_hw"] = [int(x) for x in self.rgb_shape_hw]

        if self.depth_codec is not None:
            payload["depth_codec"] = str(self.depth_codec)
        if self.depth_bytes is not None:
            payload["depth_bytes"] = self.depth_bytes
        if self.depth_shape_hw is not None:
            payload["depth_shape_hw"] = [int(x) for x in self.depth_shape_hw]

        if self.rgb_back is not None:
            payload["rgb_back"] = self.rgb_back
        if self.depth_back is not None:
            payload["depth_back"] = self.depth_back

        if self.cam_id is not None:
            payload["cam_id"] = str(self.cam_id)
        if self.intrinsics is not None:
            payload["intrinsics"] = dict(self.intrinsics)
        if self.depth_unit_scale is not None:
            payload["depth_unit_scale"] = float(self.depth_unit_scale)
        if self.T_base_cam is not None:
            payload["T_base_cam"] = self.T_base_cam
        return payload

    @staticmethod
    def from_payload(payload: Dict[str, Any]) -> "ObsPCL":
        return ObsPCL(
            t_client_send_ns=int(payload["t_client_send_ns"]),
            step_id=int(payload["step_id"]),
            q=list(payload["q"]),
            gripper=float(payload["gripper"]),
            rgb_codec=(str(payload["rgb_codec"]) if "rgb_codec" in payload and payload["rgb_codec"] is not None else None),
            rgb_bytes=(payload.get("rgb_bytes", None)),
            rgb_shape_hw=(
                [int(x) for x in payload["rgb_shape_hw"]]
                if "rgb_shape_hw" in payload and isinstance(payload["rgb_shape_hw"], list)
                else None
            ),
            depth_codec=(str(payload["depth_codec"]) if "depth_codec" in payload and payload["depth_codec"] is not None else None),
            depth_bytes=(payload.get("depth_bytes", None)),
            depth_shape_hw=(
                [int(x) for x in payload["depth_shape_hw"]]
                if "depth_shape_hw" in payload and isinstance(payload["depth_shape_hw"], list)
                else None
            ),
            rgb_back=(payload.get("rgb_back", None)),
            depth_back=(payload.get("depth_back", None)),
            cam_id=(str(payload["cam_id"]) if "cam_id" in payload and payload["cam_id"] is not None else None),
            intrinsics=(dict(payload["intrinsics"]) if "intrinsics" in payload and payload["intrinsics"] is not None else None),
            depth_unit_scale=(
                float(payload["depth_unit_scale"]) if "depth_unit_scale" in payload and payload["depth_unit_scale"] is not None else None
            ),
            T_base_cam=(payload.get("T_base_cam", None)),
        )


@dataclass(frozen=True)
class InferRequestPCL:
    request_id: str
    obs: ObsPCL

    def to_envelope(self) -> EnvelopePCL:
        return {
            "schema_version": SCHEMA_VERSION_PCL,
            "type": "infer_request_pcl",
            "request_id": self.request_id,
            "payload": self.obs.to_payload(),
        }


@dataclass(frozen=True)
class ServerTimingPCL:
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
class ActionChunkPCL:
    t_server_recv_ns: int
    t_server_send_ns: int
    t_client_send_ns_echo: int
    open_loop_horizon: int
    actions: Any
    server_timing: ServerTimingPCL

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
class InferResponsePCL:
    request_id: str
    action_chunk: ActionChunkPCL

    def to_envelope(self) -> EnvelopePCL:
        return {
            "schema_version": SCHEMA_VERSION_PCL,
            "type": "infer_response_pcl",
            "request_id": self.request_id,
            "payload": self.action_chunk.to_payload(),
        }


@dataclass(frozen=True)
class ErrorPCL:
    request_id: str
    code: str
    message: str
    t_server_send_ns: int

    def to_envelope(self) -> EnvelopePCL:
        return {
            "schema_version": SCHEMA_VERSION_PCL,
            "type": "error_pcl",
            "request_id": self.request_id,
            "payload": {
                "request_id": self.request_id,
                "code": str(self.code),
                "message": str(self.message),
                "t_server_send_ns": int(self.t_server_send_ns),
            },
        }