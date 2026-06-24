from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from mppi.pointworld_ext.scene_flow_builder import SceneFlowBuildOutput


def _ratio(x: np.ndarray) -> float:
    a = np.asarray(x).astype(np.float32)
    if a.size == 0:
        return 0.0
    return float(np.mean(a))


def print_scene_flow_stats(scene: SceneFlowBuildOutput) -> Dict[str, Any]:
    flows = np.asarray(scene.scene_flows, dtype=np.float32)
    exists = np.asarray(scene.scene_exists, dtype=bool)
    conf = np.asarray(scene.scene_track_confidence, dtype=np.float32)

    if flows.ndim != 3 or flows.shape[2] != 3:
        raise ValueError(f"scene_flows must be (T,N,3), got {flows.shape}")
    if exists.shape != flows.shape[:2]:
        raise ValueError(f"scene_exists must be (T,N)={flows.shape[:2]}, got {exists.shape}")
    if conf.shape != flows.shape[:2]:
        raise ValueError(f"scene_track_confidence must be (T,N)={flows.shape[:2]}, got {conf.shape}")

    T, N = int(flows.shape[0]), int(flows.shape[1])
    valid_ratio_per_t = exists.mean(axis=1).astype(np.float32) if N > 0 else np.zeros((T,), dtype=np.float32)

    stats: Dict[str, Any] = {
        "T": T,
        "N": N,
        "exists_ratio_mean": float(valid_ratio_per_t.mean()) if valid_ratio_per_t.size else 0.0,
        "exists_ratio_per_t": valid_ratio_per_t,
        "confidence_mean_on_exists": float(conf[exists].mean()) if np.any(exists) else 0.0,
        "cameras_used": tuple(scene.cameras_used),
    }

    print(f"scene_flows: {tuple(flows.shape)}")
    print(f"scene_exists mean: {stats['exists_ratio_mean']:.4f}")
    if valid_ratio_per_t.size:
        print(f"scene_exists per_t: {[float(x) for x in valid_ratio_per_t.tolist()]}")
    print(f"scene_track_confidence mean(on exists): {stats['confidence_mean_on_exists']:.4f}")
    print(f"cameras_used: {stats['cameras_used']}")

    return stats


def print_track_dropout_stats(scene: SceneFlowBuildOutput) -> Dict[str, Any]:
    exists = np.asarray(scene.scene_exists, dtype=bool)
    if exists.ndim != 2:
        raise ValueError(f"scene_exists must be (T,N), got {exists.shape}")

    T, N = int(exists.shape[0]), int(exists.shape[1])
    if T == 0 or N == 0:
        stats = {"dropout_ratio": 0.0, "ever_visible_ratio": 0.0}
        print(f"track_dropout: {stats}")
        return stats

    ever = np.any(exists, axis=0)
    always = np.all(exists, axis=0)
    dropout = ever & (~always)

    stats = {
        "ever_visible_ratio": float(np.mean(ever.astype(np.float32))),
        "always_visible_ratio": float(np.mean(always.astype(np.float32))),
        "dropout_ratio": float(np.mean(dropout.astype(np.float32))),
    }

    print(f"track_ever_visible_ratio: {stats['ever_visible_ratio']:.4f}")
    print(f"track_always_visible_ratio: {stats['always_visible_ratio']:.4f}")
    print(f"track_dropout_ratio: {stats['dropout_ratio']:.4f}")

    return stats


def print_camera_concat_stats(scene: SceneFlowBuildOutput) -> Dict[str, Any]:
    slices = tuple(scene.camera_track_slices)
    cams = tuple(scene.cameras_used)
    ids = np.asarray(scene.camera_track_ids, dtype=np.int32)

    if len(slices) != len(cams):
        raise ValueError("camera_track_slices length must match cameras_used")

    contrib = []
    for i, (s0, s1) in enumerate(slices):
        s0i, s1i = int(s0), int(s1)
        contrib.append((cams[i], max(0, s1i - s0i), (s0i, s1i)))

    stats = {
        "num_cameras": len(cams),
        "contrib": contrib,
        "camera_track_ids_unique": int(np.unique(ids).size) if ids.size else 0,
    }

    print(f"camera_concat num_cameras: {stats['num_cameras']}")
    for name, n, sl in contrib:
        print(f"camera_concat {name}: N={n}, slice={sl}")
    return stats


def print_workspace_filter_stats(*, exists_before: np.ndarray, exists_after: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(exists_before).astype(bool)
    b = np.asarray(exists_after).astype(bool)
    if a.shape != b.shape:
        raise ValueError("exists_before/exists_after shape mismatch")

    removed = a & (~b)
    stats = {
        "removed_ratio": _ratio(removed),
        "kept_ratio_after": _ratio(b),
    }
    print(f"workspace_filter removed_ratio: {stats['removed_ratio']:.4f}")
    print(f"workspace_filter kept_ratio_after: {stats['kept_ratio_after']:.4f}")
    return stats


def print_robot_filter_stats(*, exists_before: np.ndarray, exists_after: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(exists_before).astype(bool)
    b = np.asarray(exists_after).astype(bool)
    if a.shape != b.shape:
        raise ValueError("exists_before/exists_after shape mismatch")

    removed = a & (~b)
    stats = {
        "removed_ratio": _ratio(removed),
        "kept_ratio_after": _ratio(b),
    }
    print(f"robot_filter removed_ratio: {stats['removed_ratio']:.4f}")
    print(f"robot_filter kept_ratio_after: {stats['kept_ratio_after']:.4f}")
    return stats


def _max_true_run_per_track(mask_tn: np.ndarray) -> np.ndarray:
    m = np.asarray(mask_tn).astype(bool)
    if m.ndim != 2:
        raise ValueError(f"mask_tn must be (T,N), got {m.shape}")
    T, N = int(m.shape[0]), int(m.shape[1])
    cur = np.zeros((N,), dtype=np.int32)
    mx = np.zeros((N,), dtype=np.int32)
    for t in range(T):
        cur = (cur + 1) * m[t].astype(np.int32)
        mx = np.maximum(mx, cur)
    return mx


def print_window_stability_stats(
    scene: SceneFlowBuildOutput,
    *,
    ratio_thresh: float = 0.9,
    run_len_thresh: int = 8,
) -> Dict[str, Any]:
    exists = np.asarray(scene.scene_exists, dtype=bool)
    if exists.ndim != 2:
        raise ValueError(f"scene_exists must be (T,N), got {exists.shape}")
    T, N = int(exists.shape[0]), int(exists.shape[1])
    if T == 0 or N == 0:
        stats = {
            "T": T,
            "N": N,
            "stable_ratio": 0.0,
            "ratio_thresh": float(ratio_thresh),
            "run_len_thresh": int(run_len_thresh),
        }
        print(f"window_stability: {stats}")
        return stats

    ratio = exists.astype(np.float32).mean(axis=0)
    run = _max_true_run_per_track(exists)
    stable = (ratio >= float(ratio_thresh)) & (run >= int(run_len_thresh))

    q = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float32)
    ratio_q = np.quantile(ratio, q).astype(np.float32)
    run_q = np.quantile(run.astype(np.float32), q).astype(np.float32)

    stats: Dict[str, Any] = {
        "T": T,
        "N": N,
        "ratio_thresh": float(ratio_thresh),
        "run_len_thresh": int(run_len_thresh),
        "stable_ratio": float(np.mean(stable.astype(np.float32))),
        "ratio_quantiles": ratio_q,
        "run_len_quantiles": run_q,
    }

    print(f"window_stability ratio_thresh={float(ratio_thresh):.3f} run_len_thresh={int(run_len_thresh)}")
    print(f"window_stability stable_ratio: {stats['stable_ratio']:.4f}")
    print(f"window_stability exists_ratio q(0,0.1,0.5,0.9,1): {[float(x) for x in ratio_q.tolist()]}")
    print(f"window_stability exists_runlen q(0,0.1,0.5,0.9,1): {[float(x) for x in run_q.tolist()]}")

    slices = tuple(scene.camera_track_slices)
    cams = tuple(scene.cameras_used)
    if len(slices) == len(cams) and N > 0:
        per_cam = []
        for i, (s0, s1) in enumerate(slices):
            a, b = int(s0), int(s1)
            if a < 0:
                a = 0
            if b > N:
                b = N
            if b <= a:
                per_cam.append((cams[i], 0, 0.0))
            else:
                per_cam.append((cams[i], int(b - a), float(np.mean(stable[a:b].astype(np.float32)))))
        stats["stable_ratio_per_camera"] = per_cam
        for name, ncam, sr in per_cam:
            print(f"window_stability {name}: N={ncam}, stable_ratio={sr:.4f}")

    return stats


def print_pointworld_batch_stats(batch: Dict[str, Any]) -> Dict[str, Any]:
    flows = np.asarray(batch.get("scene_flows"))
    exists = np.asarray(batch.get("scene_exists"))
    robot = np.asarray(batch.get("robot_flows"))

    stats: Dict[str, Any] = {
        "scene_flows_shape": tuple(flows.shape),
        "scene_exists_shape": tuple(exists.shape),
        "robot_flows_shape": tuple(robot.shape),
    }

    if exists.size:
        stats["scene_exists_mean"] = float(np.asarray(exists, dtype=np.float32).mean())

    cams = batch.get("cameras_used")
    if cams is not None:
        stats["cameras_used"] = tuple(np.asarray(cams).tolist())

    print(f"batch scene_flows: {stats['scene_flows_shape']}")
    print(f"batch scene_exists: {stats['scene_exists_shape']}")
    print(f"batch robot_flows: {stats['robot_flows_shape']}")
    if "scene_exists_mean" in stats:
        print(f"batch scene_exists mean: {float(stats['scene_exists_mean']):.4f}")
    if "cameras_used" in stats:
        print(f"batch cameras_used: {stats['cameras_used']}")

    return stats