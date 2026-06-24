from __future__ import annotations

from typing import Tuple

import numpy as np

from mppi.robots.franka_kinematics import FrankaFK


def ee_pos_cost(fk: FrankaFK, q_batch: np.ndarray, goal: Tuple[float, float, float]) -> np.ndarray:
    pos = fk.fk_pos(q_batch)
    goal_np = np.asarray(goal, dtype=np.float32).reshape(1, 3)
    d = pos - goal_np
    return np.sum(d * d, axis=1).astype(np.float32)