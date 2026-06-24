from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class QueryPointManagerConfig:
    max_query_points_per_camera: int
    min_track_confidence: float = 0.0
    rng_seed: int = 0

    def __post_init__(self) -> None:
        if int(self.max_query_points_per_camera) < 1:
            raise ValueError("max_query_points_per_camera must be >= 1")
        c = float(self.min_track_confidence)
        if not (0.0 <= c <= 1.0):
            raise ValueError("min_track_confidence must be in [0,1]")


class QueryPointManager:
    def __init__(self, *, cfg: QueryPointManagerConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(int(cfg.rng_seed))
        self._q: Dict[str, np.ndarray] = {}

    def reset(self, camera_name: Optional[str] = None) -> None:
        if camera_name is None:
            self._q.clear()
        else:
            self._q.pop(str(camera_name), None)

    def get_or_create(
        self,
        camera_name: str,
        *,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
        workspace_mask: Optional[np.ndarray] = None,
        robot_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        name = str(camera_name)
        if name in self._q:
            return self._q[name]
        q = self.sample_initial(
            rgb=rgb,
            depth=depth,
            valid_mask=valid_mask,
            workspace_mask=workspace_mask,
            robot_mask=robot_mask,
            n=self.cfg.max_query_points_per_camera,
        )
        self._q[name] = q
        return q

    def sample_initial(
        self,
        *,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
        workspace_mask: Optional[np.ndarray] = None,
        robot_mask: Optional[np.ndarray] = None,
        n: Optional[int] = None,
    ) -> np.ndarray:
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Expected rgb shape (H,W,3), got {img.shape}")
        H, W = int(img.shape[0]), int(img.shape[1])

        keep = np.ones((H, W), dtype=bool)
        if valid_mask is not None:
            m = np.asarray(valid_mask).astype(bool)
            if m.shape != (H, W):
                raise ValueError(f"valid_mask shape {m.shape} must be (H,W)={(H,W)}")
            keep &= m
        if workspace_mask is not None:
            m = np.asarray(workspace_mask).astype(bool)
            if m.shape != (H, W):
                raise ValueError(f"workspace_mask shape {m.shape} must be (H,W)={(H,W)}")
            keep &= m
        if robot_mask is not None:
            m = np.asarray(robot_mask).astype(bool)
            if m.shape != (H, W):
                raise ValueError(f"robot_mask shape {m.shape} must be (H,W)={(H,W)}")
            keep &= ~m
        if depth is not None:
            d = np.asarray(depth)
            if d.shape != (H, W):
                raise ValueError(f"depth shape {d.shape} must be (H,W)={(H,W)}")
            keep &= np.isfinite(d)

        n0 = int(self.cfg.max_query_points_per_camera if n is None else n)
        ys, xs = np.nonzero(keep)
        if ys.size == 0:
            q = np.zeros((n0, 2), dtype=np.float32)
            q[:, 0] = float(W // 2)
            q[:, 1] = float(H // 2)
            return q

        idx = np.arange(ys.size)
        if ys.size >= n0:
            sel = self._rng.choice(idx, size=n0, replace=False)
        else:
            sel = self._rng.choice(idx, size=n0, replace=True)
        q = np.stack([xs[sel].astype(np.float32), ys[sel].astype(np.float32)], axis=1)
        return q

    def advance_window(
        self,
        camera_name: str,
        *,
        uv_tracks: np.ndarray,
        visibility: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        stable_mask0: Optional[np.ndarray] = None,
        new_query_index: int = 0,
        rgb0: Optional[np.ndarray] = None,
        depth0: Optional[np.ndarray] = None,
        valid_mask0: Optional[np.ndarray] = None,
        workspace_mask0: Optional[np.ndarray] = None,
        robot_mask0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        name = str(camera_name)
        uv = np.asarray(uv_tracks, dtype=np.float32)
        vis = np.asarray(visibility).astype(bool)
        if uv.ndim != 3 or uv.shape[-1] != 2:
            raise ValueError(f"Expected uv_tracks shape (T,N,2), got {uv.shape}")
        if vis.shape != uv.shape[:2]:
            raise ValueError(f"visibility shape {vis.shape} must match (T,N)={uv.shape[:2]}")
        conf = np.ones_like(vis, dtype=np.float32) if confidence is None else np.asarray(confidence, dtype=np.float32)
        if conf.shape != uv.shape[:2]:
            raise ValueError(f"confidence shape {conf.shape} must match (T,N)={uv.shape[:2]}")

        t = int(new_query_index)
        if t < 0 or t >= uv.shape[0]:
            raise ValueError("new_query_index out of range")

        ok = vis[t] & (conf[t] >= float(self.cfg.min_track_confidence))
        if stable_mask0 is not None:
            sm = np.asarray(stable_mask0).astype(bool)
            if sm.shape != (uv.shape[1],):
                raise ValueError(f"stable_mask0 shape {sm.shape} must be (N,)={(uv.shape[1],)}")
            ok &= sm
        kept = uv[t][ok]
        n0 = int(self.cfg.max_query_points_per_camera)
        if kept.shape[0] >= n0:
            q = kept[:n0].astype(np.float32)
        else:
            if rgb0 is None:
                extra = np.tile(kept[:1], (n0 - kept.shape[0], 1)).astype(np.float32) if kept.shape[0] > 0 else np.zeros((n0 - kept.shape[0], 2), dtype=np.float32)
            else:
                extra = self.sample_initial(
                    rgb=rgb0,
                    depth=depth0,
                    valid_mask=valid_mask0,
                    workspace_mask=workspace_mask0,
                    robot_mask=robot_mask0,
                    n=n0 - kept.shape[0],
                )
            q = np.concatenate([kept.astype(np.float32), extra], axis=0)
        self._q[name] = q
        return q