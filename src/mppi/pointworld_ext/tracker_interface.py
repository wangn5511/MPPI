from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from mppi.utils.paths import ensure_sys_path_for_runtime


POINTWORLD_WINDOW_LEN = 11


@dataclass(frozen=True)
class TrackWindowOutput:
    uv_tracks: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray


class OnlinePointTracker(ABC):
    @abstractmethod
    def track_window(self, frames: Union[np.ndarray, Sequence[np.ndarray]], query_points: np.ndarray) -> TrackWindowOutput:
        raise NotImplementedError

    def update(self, frame: np.ndarray) -> None:
        raise NotImplementedError("update() is not implemented for this tracker")


def _require_torch():
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: torch") from e

    return torch


def _infer_cotracker_window_len(checkpoint: str) -> int | None:
    torch = _require_torch()
    with open(str(checkpoint), "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    if not isinstance(state_dict, dict):
        return None
    time_emb = state_dict.get("time_emb")
    if time_emb is None:
        return None
    try:
        shape = tuple(int(x) for x in time_emb.shape)
    except Exception:
        return None
    if len(shape) >= 2 and int(shape[1]) > 0:
        return int(shape[1])
    return None


def _as_frames_array(frames: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
    if isinstance(frames, np.ndarray):
        arr = np.asarray(frames)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"Expected frames shape (T,H,W,3), got {arr.shape}")
        return arr

    stack = [np.asarray(f) for f in frames]
    if not stack:
        raise ValueError("frames must be non-empty")
    if any(f.ndim != 3 or f.shape[-1] != 3 for f in stack):
        shapes = [getattr(f, "shape", None) for f in stack]
        raise ValueError(f"Expected each frame shape (H,W,3), got {shapes}")
    return np.stack(stack, axis=0)


class CoTrackerOnlinePointTracker(OnlinePointTracker):
    def __init__(
        self,
        *,
        checkpoint: str,
        window_len: int = 16,
        device: Optional[str] = None,
        v2: bool = False,
    ) -> None:
        torch = _require_torch()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        ckpt = str(checkpoint)
        ensure_sys_path_for_runtime()
        try:
            from cotracker.predictor import CoTrackerOnlinePredictor
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import cotracker. Install MPPI third_party/co-tracker with: pip install -e /home/wangyuhan/MPPI/third_party/co-tracker"
            ) from e

        model_window_len = _infer_cotracker_window_len(ckpt) or int(window_len)
        predictor = CoTrackerOnlinePredictor(checkpoint=ckpt, window_len=int(model_window_len), v2=bool(v2))

        self._torch = torch
        self._predictor = predictor.to(device)
        self._device = str(device)
        self._model_window_len = int(model_window_len)

    def track_window(self, frames: Union[np.ndarray, Sequence[np.ndarray]], query_points: np.ndarray) -> TrackWindowOutput:
        torch = self._torch
        pred = self._predictor

        fr = _as_frames_array(frames)
        if fr.shape[0] != int(POINTWORLD_WINDOW_LEN):
            raise ValueError(f"PointWorld requires window length T={POINTWORLD_WINDOW_LEN}, got T={fr.shape[0]}")

        q = np.asarray(query_points, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError(f"Expected query_points shape (N,2), got {q.shape}")

        video = torch.from_numpy(fr).to(device=torch.device(self._device))
        video = video.float().permute(0, 3, 1, 2)[None]

        queries = torch.from_numpy(np.concatenate([np.zeros((q.shape[0], 1), dtype=np.float32), q], axis=1)[None])
        queries = queries.to(device=video.device, dtype=video.dtype)

        pred(video, is_first_step=True, queries=queries)

        if hasattr(pred, "interp_shape") and hasattr(pred, "model") and hasattr(pred, "queries"):
            import torch.nn.functional as F

            B, T, C, H, W = video.shape
            video_rs = video.reshape(B * T, C, H, W)
            video_rs = F.interpolate(video_rs, tuple(pred.interp_shape), mode="bilinear", align_corners=True)
            video_rs = video_rs.reshape(B, T, 3, pred.interp_shape[0], pred.interp_shape[1])

            out = pred.model(video=video_rs, queries=pred.queries, iters=6, is_online=True)
            if len(out) >= 3:
                tracks_t, vis_t, conf_t = out[0], out[1], out[2]
            else:
                tracks_t, vis_t = out[0], out[1]
                conf_t = vis_t

            if tracks_t.shape[-1] != 2:
                raise RuntimeError(f"Unexpected tracks shape: {tuple(tracks_t.shape)}")

            if vis_t.ndim == 4 and vis_t.shape[-1] == 1:
                vis_t = vis_t[..., 0]
            if conf_t is not None and conf_t.ndim == 4 and conf_t.shape[-1] == 1:
                conf_t = conf_t[..., 0]

            if conf_t is None:
                conf_t = vis_t

            vis_conf = vis_t.to(dtype=torch.float32) * conf_t.to(dtype=torch.float32)
            thr = 0.6
            vis_bool = vis_conf > float(thr)

            scale = tracks_t.new_tensor([(W - 1) / (pred.interp_shape[1] - 1), (H - 1) / (pred.interp_shape[0] - 1)])
            tracks_t = tracks_t * scale

            uv_tracks = tracks_t[0].detach().cpu().numpy().astype(np.float32)
            visibility = vis_bool[0].detach().cpu().numpy().astype(bool)
            confidence = vis_conf[0].detach().cpu().numpy().astype(np.float32)

            return TrackWindowOutput(uv_tracks=uv_tracks, visibility=visibility, confidence=confidence)

        tracks, visibility = pred(video, is_first_step=False)
        if tracks is None or visibility is None:
            raise RuntimeError("CoTracker did not return tracking results")

        tr = tracks[0].detach().cpu().numpy().astype(np.float32)
        vis = visibility[0]
        if hasattr(vis, "detach"):
            vis = vis.detach().cpu().numpy()
        vis = np.asarray(vis).astype(bool)
        conf = vis.astype(np.float32)

        return TrackWindowOutput(uv_tracks=tr, visibility=vis, confidence=conf)
