from __future__ import annotations

import base64
from typing import Any, Dict, Tuple


def _require_msgpack():
    try:
        import msgpack  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: msgpack. Install it in the container env.") from e
    return msgpack


def _optional_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    return np


def _is_ndarray(obj: Any) -> bool:
    np = _optional_numpy()
    if np is None:
        return False
    return isinstance(obj, np.ndarray)


def _pack_ndarray(arr: Any) -> Dict[str, Any]:
    data = arr.tobytes(order="C")
    return {
        "__ndarray__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(data).decode("ascii"),
    }


def _unpack_ndarray(obj: Dict[str, Any]) -> Any:
    np = _optional_numpy()
    if np is None:
        raise RuntimeError("Received ndarray payload but numpy is not installed in this env.")
    dtype = np.dtype(obj["dtype"])
    shape = tuple(int(x) for x in obj["shape"])
    raw = base64.b64decode(obj["data_b64"].encode("ascii"))
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(shape)


def _to_wire(obj: Any) -> Any:
    if _is_ndarray(obj):
        return _pack_ndarray(obj)
    if isinstance(obj, dict):
        return {k: _to_wire(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_wire(v) for v in obj]
    return obj


def _from_wire(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__ndarray__") is True:
        return _unpack_ndarray(obj)
    if isinstance(obj, dict):
        return {k: _from_wire(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_wire(v) for v in obj]
    return obj


def encode_message(envelope: Dict[str, Any]) -> bytes:
    msgpack = _require_msgpack()
    wire_obj = _to_wire(envelope)
    return msgpack.packb(wire_obj, use_bin_type=True)


def decode_message(data: bytes) -> Dict[str, Any]:
    msgpack = _require_msgpack()
    obj = msgpack.unpackb(data, raw=False)
    return _from_wire(obj)