from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_torch_pk():
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: torch") from e

    try:
        import pytorch_kinematics as pk
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: pytorch_kinematics") from e

    return torch, pk


@dataclass(frozen=True)
class FrankaFK:
    urdf_path: str
    ee_link: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        torch, pk = _require_torch_pk()
        with open(self.urdf_path, "rb") as f:
            urdf_bytes = f.read()

        chain = pk.build_chain_from_urdf(bytes(urdf_bytes))
        chain = chain.to(device=torch.device(self.device), dtype=torch.float32)

        object.__setattr__(self, "_torch", torch)
        object.__setattr__(self, "_chain", chain)
        object.__setattr__(self, "_frame_indices", chain.get_frame_indices(str(self.ee_link)))

        joint_names = None
        if hasattr(chain, "get_joint_parameter_names"):
            joint_names = list(chain.get_joint_parameter_names())
        elif hasattr(chain, "get_joint_names"):
            joint_names = list(chain.get_joint_names())

        if not joint_names:
            raise RuntimeError("Failed to query joint names from pytorch_kinematics chain")

        object.__setattr__(self, "_joint_names", tuple(str(n) for n in joint_names))

    def fk_pos(self, q_batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        chain = self._chain

        q = np.asarray(q_batch, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 7:
            raise ValueError(f"Expected q_batch shape (B,7), got {q.shape}")

        q_t = torch.from_numpy(q).to(device=torch.device(self.device), dtype=torch.float32)
        zeros = torch.zeros((q_t.shape[0],), device=q_t.device, dtype=q_t.dtype)
        joint_dict = {name: zeros for name in self._joint_names}
        joint_dict.update(
            {
                "panda_joint1": q_t[:, 0],
                "panda_joint2": q_t[:, 1],
                "panda_joint3": q_t[:, 2],
                "panda_joint4": q_t[:, 3],
                "panda_joint5": q_t[:, 4],
                "panda_joint6": q_t[:, 5],
                "panda_joint7": q_t[:, 6],
            }
        )

        with torch.no_grad():
            tf = chain.forward_kinematics(joint_dict, frame_indices=self._frame_indices)
            T = tf[str(self.ee_link)].get_matrix()
            pos = T[:, :3, 3]
        return pos.detach().cpu().numpy()