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
    scene_p0: np.ndarray | None = None,
    task_point_indices: np.ndarray | None = None,
    task_goal_positions: np.ndarray | None = None,
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

    mode = str(ccfg.mode)

    if mode in {"task_point_goal_l2", "final_task_point_goal_l2"}:
        if scene_p0 is None or task_point_indices is None or task_goal_positions is None:
            raise ValueError("task_point_goal_l2 requires scene_p0, task_point_indices, task_goal_positions")

        p0 = np.asarray(scene_p0, dtype=np.float32)
        if p0.ndim != 3 or p0.shape[0] != rel.shape[0] or p0.shape[2] != 3:
            raise ValueError(f"scene_p0 must be (B,N,3), got {p0.shape}")

        idx = np.asarray(task_point_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return np.zeros((int(rel.shape[0]),), dtype=np.float32)

        N = int(rel.shape[2])
        if int(np.min(idx)) < 0 or int(np.max(idx)) >= N:
            raise ValueError(f"task_point_indices out of range [0,{N}), got min={int(np.min(idx))}, max={int(np.max(idx))}")

        goal = np.asarray(task_goal_positions, dtype=np.float32)
        if goal.shape == (3,):
            goal_k3 = np.repeat(goal.reshape(1, 3), int(idx.size), axis=0)
        elif goal.ndim == 2 and int(goal.shape[0]) == int(idx.size) and int(goal.shape[1]) == 3:
            goal_k3 = goal
        else:
            raise ValueError(f"task_goal_positions must be (3,) or (K,3) with K={int(idx.size)}, got {goal.shape}")

        rel_task = rel_eval[:, -1:] if mode.startswith("final_") else rel_eval
        exists_task = exists_eval[:, -1:] if mode.startswith("final_") else exists_eval

        p = p0[:, None, :, :] + rel_task
        p_sel = p[:, :, idx, :]
        diff = p_sel - goal_k3[None, None, :, :]
        point_cost = np.sum(diff * diff, axis=-1)

        weight = exists_task[:, :, idx].astype(np.float32)

        if bool(ccfg.use_model_confidence) and model_confidence is not None:
            mc = np.asarray(model_confidence, dtype=np.float32)
            mc = mc[:, 1:] if bool(ccfg.ignore_t0) and mc.shape[1] > 1 else mc
            mc = mc[:, -1:] if mode.startswith("final_") and mc.shape[1] > 1 else mc
            if mc.shape != exists.shape:
                raise ValueError(f"model_confidence shape {mc.shape} must match scene_exists shape {exists.shape}")
            weight *= np.clip(mc[:, :, idx], float(ccfg.min_confidence), 1.0)

        if bool(ccfg.use_track_confidence) and track_confidence is not None:
            tc = np.asarray(track_confidence, dtype=np.float32)
            tc = tc[:, 1:] if bool(ccfg.ignore_t0) and tc.shape[1] > 1 else tc
            tc = tc[:, -1:] if mode.startswith("final_") and tc.shape[1] > 1 else tc
            if tc.shape != exists.shape:
                raise ValueError(f"track_confidence shape {tc.shape} must match scene_exists shape {exists.shape}")
            weight *= np.clip(tc[:, :, idx], float(ccfg.min_confidence), 1.0)

        numer = np.sum(point_cost * weight, axis=(1, 2)).astype(np.float32)
        denom = np.sum(weight, axis=(1, 2)).astype(np.float32)
        denom = np.maximum(denom, float(ccfg.eps))
        return (numer / denom).astype(np.float32)

    if mode == "flow_l1":
        point_cost = np.linalg.norm(rel_eval, axis=-1)
    elif mode == "final_flow_l2":
        point_cost = np.sum(rel_eval[:, -1:, :, :] * rel_eval[:, -1:, :, :], axis=-1)
    else:
        point_cost = np.sum(rel_eval * rel_eval, axis=-1)

    weight = exists_eval.astype(np.float32)

    if bool(ccfg.use_model_confidence) and model_confidence is not None:
        mc = np.asarray(model_confidence, dtype=np.float32)
        mc_eval = mc[:, 1:] if bool(ccfg.ignore_t0) and mc.shape[1] > 1 else mc
        mc_task = mc_eval[:, -1:] if mode.startswith("final_") and mc_eval.shape[1] > 1 else mc_eval
        if mc_task.shape != exists_task.shape:
            raise ValueError(f"model_confidence shape {mc_task.shape} must match exists shape {exists_task.shape}")
        weight *= np.clip(mc_task, float(ccfg.min_confidence), 1.0)

    if bool(ccfg.use_track_confidence) and track_confidence is not None:
        tc = np.asarray(track_confidence, dtype=np.float32)
        tc_eval = tc[:, 1:] if bool(ccfg.ignore_t0) and tc.shape[1] > 1 else tc
        tc_task = tc_eval[:, -1:] if mode.startswith("final_") and tc_eval.shape[1] > 1 else tc_eval
        if tc_task.shape != exists_task.shape:
            raise ValueError(f"track_confidence shape {tc_task.shape} must match exists shape {exists_task.shape}")
        weight *= np.clip(tc_task, float(ccfg.min_confidence), 1.0)

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
    scene_p0=None,
    task_point_indices=None,
    task_goal_positions=None,
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

    mode = str(ccfg.mode)

    if mode in {"task_point_goal_l2", "final_task_point_goal_l2"}:
        if scene_p0 is None or task_point_indices is None or task_goal_positions is None:
            raise ValueError("task_point_goal_l2 requires scene_p0, task_point_indices, task_goal_positions")

        p0 = torch.as_tensor(scene_p0, device=rel.device, dtype=torch.float32)
        if p0.ndim != 3 or int(p0.shape[0]) != int(rel.shape[0]) or int(p0.shape[2]) != 3:
            raise ValueError(f"scene_p0 must be (B,N,3), got {tuple(p0.shape)}")

        idx = torch.as_tensor(task_point_indices, device=rel.device, dtype=torch.long).reshape(-1)
        if int(idx.numel()) == 0:
            return torch.zeros((int(rel.shape[0]),), device=rel.device, dtype=torch.float32)

        N = int(rel.shape[2])
        if int(torch.min(idx)) < 0 or int(torch.max(idx)) >= N:
            raise ValueError(f"task_point_indices out of range [0,{N})")

        goal = torch.as_tensor(task_goal_positions, device=rel.device, dtype=torch.float32)
        if tuple(goal.shape) == (3,):
            goal_k3 = goal.reshape(1, 3).repeat(int(idx.numel()), 1)
        elif goal.ndim == 2 and int(goal.shape[0]) == int(idx.numel()) and int(goal.shape[1]) == 3:
            goal_k3 = goal
        else:
            raise ValueError(f"task_goal_positions must be (3,) or (K,3) with K={int(idx.numel())}, got {tuple(goal.shape)}")

        rel_task = rel_eval[:, -1:] if mode.startswith("final_") else rel_eval
        exists_task = exists_eval[:, -1:] if mode.startswith("final_") else exists_eval

        p = p0[:, None, :, :] + rel_task
        p_sel = p.index_select(dim=2, index=idx)
        diff = p_sel - goal_k3[None, None, :, :]
        point_cost = torch.sum(diff * diff, dim=-1)

        weight = exists_task.index_select(dim=2, index=idx).to(dtype=torch.float32)

        if bool(ccfg.use_model_confidence) and model_confidence is not None:
            mc = torch.as_tensor(model_confidence, device=rel.device, dtype=torch.float32)
            mc = mc[:, 1:] if bool(ccfg.ignore_t0) and mc.shape[1] > 1 else mc
            mc = mc[:, -1:] if mode.startswith("final_") and mc.shape[1] > 1 else mc
            if tuple(mc.shape) != tuple(exists.shape):
                raise ValueError(f"model_confidence shape {tuple(mc.shape)} must match scene_exists shape {tuple(exists.shape)}")
            weight = weight * torch.clamp(mc.index_select(dim=2, index=idx), min=float(ccfg.min_confidence), max=1.0)

        if bool(ccfg.use_track_confidence) and track_confidence is not None:
            tc = torch.as_tensor(track_confidence, device=rel.device, dtype=torch.float32)
            tc = tc[:, 1:] if bool(ccfg.ignore_t0) and tc.shape[1] > 1 else tc
            tc = tc[:, -1:] if mode.startswith("final_") and tc.shape[1] > 1 else tc
            if tuple(tc.shape) != tuple(exists.shape):
                raise ValueError(f"track_confidence shape {tuple(tc.shape)} must match scene_exists shape {tuple(exists.shape)}")
            weight = weight * torch.clamp(tc.index_select(dim=2, index=idx), min=float(ccfg.min_confidence), max=1.0)

        numer = torch.sum(point_cost * weight, dim=(1, 2))
        denom = torch.sum(weight, dim=(1, 2)).clamp_min(float(ccfg.eps))
        return (numer / denom).to(dtype=torch.float32)

    if mode == "flow_l1":
        point_cost = torch.linalg.norm(rel_eval, dim=-1)
    elif mode == "final_flow_l2":
        point_cost = torch.sum(rel_eval[:, -1:, :, :] * rel_eval[:, -1:, :, :], dim=-1)
    else:
        point_cost = torch.sum(rel_eval * rel_eval, dim=-1)

    weight = exists_eval.to(dtype=torch.float32)

    if bool(ccfg.use_model_confidence) and model_confidence is not None:
        mc = torch.as_tensor(model_confidence, device=rel.device, dtype=torch.float32)
        mc_eval = mc[:, 1:] if bool(ccfg.ignore_t0) and mc.shape[1] > 1 else mc
        mc_task = mc_eval[:, -1:] if mode.startswith("final_") and mc_eval.shape[1] > 1 else mc_eval
        if tuple(mc_task.shape) != tuple(exists_task.shape):
            raise ValueError(f"model_confidence shape {tuple(mc_task.shape)} must match exists shape {tuple(exists_task.shape)}")
        weight = weight * torch.clamp(mc_task, min=float(ccfg.min_confidence), max=1.0)

    if bool(ccfg.use_track_confidence) and track_confidence is not None:
        tc = torch.as_tensor(track_confidence, device=rel.device, dtype=torch.float32)
        tc_eval = tc[:, 1:] if bool(ccfg.ignore_t0) and tc.shape[1] > 1 else tc
        tc_task = tc_eval[:, -1:] if mode.startswith("final_") and tc_eval.shape[1] > 1 else tc_eval
        if tuple(tc_task.shape) != tuple(exists_task.shape):
            raise ValueError(f"track_confidence shape {tuple(tc_task.shape)} must match exists shape {tuple(exists_task.shape)}")
        weight = weight * torch.clamp(tc_task, min=float(ccfg.min_confidence), max=1.0)

    numer = torch.sum(point_cost * weight, dim=(1, 2))
    denom = torch.sum(weight, dim=(1, 2)).clamp_min(float(ccfg.eps))
    return (numer / denom).to(dtype=torch.float32)
