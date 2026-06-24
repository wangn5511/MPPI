from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from mppi.pointworld_ext.geometry import PinholeIntrinsics


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: PinholeIntrinsics
    extrinsics: np.ndarray


@dataclass(frozen=True)
class WindowStep:
    cameras: Dict[str, CameraFrame]
    joint_positions: np.ndarray
    gripper_positions: np.ndarray
    timestamp: float


class PointWorldWindowBuffer:
    def __init__(self, *, window_size: int) -> None:
        if int(window_size) < 1:
            raise ValueError("window_size must be >= 1")
        if int(window_size) != 11:
            raise ValueError("PointWorld requires window_size == 11")
        self.window_size = int(window_size)
        self._steps: Deque[WindowStep] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._steps.clear()

    def __len__(self) -> int:
        return len(self._steps)

    def is_ready(self) -> bool:
        return len(self._steps) >= self.window_size

    def push_frame(
        self,
        *,
        cameras: Dict[str, CameraFrame],
        joint_positions: np.ndarray,
        gripper_positions: np.ndarray,
        timestamp: float,
    ) -> None:
        if not cameras:
            raise ValueError("cameras must be non-empty")
        cams: Dict[str, CameraFrame] = {str(k): v for k, v in cameras.items()}
        jp = np.asarray(joint_positions, dtype=np.float32)
        gp = np.asarray(gripper_positions, dtype=np.float32)
        ts = float(timestamp)
        self._steps.append(WindowStep(cameras=cams, joint_positions=jp, gripper_positions=gp, timestamp=ts))

    def get_window(self) -> Tuple[WindowStep, ...]:
        if not self.is_ready():
            raise RuntimeError(f"Window not ready: {len(self._steps)}/{self.window_size}")
        return tuple(self._steps)

    def get_available_cameras(self) -> Tuple[str, ...]:
        if not self._steps:
            return ()
        names = set(self._steps[0].cameras.keys())
        for s in self._steps:
            names &= set(s.cameras.keys())
        return tuple(sorted(names))