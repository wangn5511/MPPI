from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointWorldCostConfig:
    mode: str = "flow_l2"
    use_model_confidence: bool = True
    use_track_confidence: bool = True
    min_confidence: float = 0.0
    ignore_t0: bool = True
    eps: float = 1e-6


def reduce_pointworld_cost(
    *,
    scene_relative: np.ndarray,
    scene_exists: np.ndarray,
    model_confidence: np.ndarray | None = None,
    track_confidence: np.ndarray | None = None,
    cfg: PointWorldCostConfig | None = None,
) -> np.ndarray:
    ccfg = cfg if cfg is not None else PointWorldCostConfig()

    rel = np.asarray(scene_relative, dtype=np.float32)
    exists = np.asarray(scene_exists, dtype=bool)
    if rel.ndim != 4 or rel.shape[-1] != 3:
        raise ValueError(f"scene_relative must be (B,T,N,3), got {rel.shape}")
    if exists.shape != rel.shape[:3]:
        raise ValueError(f"scene_exists shape {exists.shape} must match scene_relative[:3]={rel.shape[:3]}")

    rel_eval = rel[:, 1:] if bool(ccfg.ignore_t0) and rel.shape[1] > 1 else rel
    exists_eval = exists[:, 1:] if bool(ccfg.ignore_t0) and exists.shape[1] > 1 else exists

    if str(ccfg.mode) == "flow_l1":
        point_cost = np.linalg.norm(rel_eval, axis=-1)
    elif str(ccfg.mode) == "final_flow_l2":
        point_cost = np.sum(rel_eval[:, -1:, :, :] * rel_eval[:, -1:, :, :], axis=-1)
    else:
        point_cost = np.sum(rel_eval * rel_eval, axis=-1)

    weight = exists_eval.astype(np.float32)

    if bool(ccfg.use_model_confidence) and model_confidence is not None:
        mc = np.asarray(model_confidence, dtype=np.float32)
        mc = mc[:, 1:] if bool(ccfg.ignore_t0) and mc.shape[1] > 1 else mc
        if mc.shape != point_cost.shape:
            raise ValueError(f"model_confidence shape {mc.shape} must match point cost shape {point_cost.shape}")
        weight *= np.clip(mc, float(ccfg.min_confidence), 1.0)

    if bool(ccfg.use_track_confidence) and track_confidence is not None:
        tc = np.asarray(track_confidence, dtype=np.float32)
        tc = tc[:, 1:] if bool(ccfg.ignore_t0) and tc.shape[1] > 1 else tc
        if tc.shape != point_cost.shape:
            raise ValueError(f"track_confidence shape {tc.shape} must match point cost shape {point_cost.shape}")
        weight *= np.clip(tc, float(ccfg.min_confidence), 1.0)

    numer = np.sum(point_cost * weight, axis=(1, 2)).astype(np.float32)
    denom = np.sum(weight, axis=(1, 2)).astype(np.float32)
    denom = np.maximum(denom, float(ccfg.eps))
    return (numer / denom).astype(np.float32)


def reduce_pointworld_cost_torch(
    *,
    scene_relative,
    scene_exists,
    model_confidence=None,
    track_confidence=None,
    cfg: PointWorldCostConfig | None = None,
):
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: torch") from e

    ccfg = cfg if cfg is not None else PointWorldCostConfig()

    rel = torch.as_tensor(scene_relative, dtype=torch.float32)
    exists = torch.as_tensor(scene_exists, device=rel.device, dtype=torch.bool)
    if rel.ndim != 4 or rel.shape[-1] != 3:
        raise ValueError(f"scene_relative must be (B,T,N,3), got {tuple(rel.shape)}")
    if tuple(exists.shape) != tuple(rel.shape[:3]):
        raise ValueError(f"scene_exists shape {tuple(exists.shape)} must match scene_relative[:3]={tuple(rel.shape[:3])}")

    rel_eval = rel[:, 1:] if bool(ccfg.ignore_t0) and rel.shape[1] > 1 else rel
    exists_eval = exists[:, 1:] if bool(ccfg.ignore_t0) and exists.shape[1] > 1 else exists

    if str(ccfg.mode) == "flow_l1":
        point_cost = torch.linalg.norm(rel_eval, dim=-1)
    elif str(ccfg.mode) == "final_flow_l2":
        point_cost = torch.sum(rel_eval[:, -1:, :, :] * rel_eval[:, -1:, :, :], dim=-1)
    else:
        point_cost = torch.sum(rel_eval * rel_eval, dim=-1)

    weight = exists_eval.to(dtype=torch.float32)

    if bool(ccfg.use_model_confidence) and model_confidence is not None:
        mc = torch.as_tensor(model_confidence, device=rel.device, dtype=torch.float32)
        mc = mc[:, 1:] if bool(ccfg.ignore_t0) and mc.shape[1] > 1 else mc
        if tuple(mc.shape) != tuple(point_cost.shape):
            raise ValueError(f"model_confidence shape {tuple(mc.shape)} must match point cost shape {tuple(point_cost.shape)}")
        weight = weight * torch.clamp(mc, min=float(ccfg.min_confidence), max=1.0)

    if bool(ccfg.use_track_confidence) and track_confidence is not None:
        tc = torch.as_tensor(track_confidence, device=rel.device, dtype=torch.float32)
        tc = tc[:, 1:] if bool(ccfg.ignore_t0) and tc.shape[1] > 1 else tc
        if tuple(tc.shape) != tuple(point_cost.shape):
            raise ValueError(f"track_confidence shape {tuple(tc.shape)} must match point cost shape {tuple(point_cost.shape)}")
        weight = weight * torch.clamp(tc, min=float(ccfg.min_confidence), max=1.0)

    numer = torch.sum(point_cost * weight, dim=(1, 2))
    denom = torch.sum(weight, dim=(1, 2)).clamp_min(float(ccfg.eps))
    return (numer / denom).to(dtype=torch.float32)
