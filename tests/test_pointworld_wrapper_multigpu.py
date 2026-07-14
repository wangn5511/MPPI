from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    nn_stub = types.ModuleType("torch.nn")

    class _Module:
        pass

    class _Linear:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    nn_stub.Module = _Module
    nn_stub.Linear = _Linear
    torch_stub.nn = nn_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub

from mppi.pointworld_ext.wrapper import PointWorldCostModel


def test_pointworld_cost_model_records_replica_sample_ranges() -> None:
    model = PointWorldCostModel.__new__(PointWorldCostModel)
    model._replicas = (
        SimpleNamespace(device="cuda:0"),
        SimpleNamespace(device="cuda:1"),
    )

    def fake_eval(*, replica, q_traj, pointworld_obs, gripper):
        del pointworld_obs, gripper
        fill = 0.0 if str(replica.device) == "cuda:0" else 1.0
        return np.full((q_traj.shape[0],), fill, dtype=np.float32)

    model._evaluate_cost_on_replica = fake_eval

    out = PointWorldCostModel.evaluate_cost(
        model,
        q_traj=np.zeros((5, 11, 7), dtype=np.float32),
        pointworld_obs={},
        gripper=None,
    )

    assert out.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert model.last_eval_ranges == (
        {"device": "cuda:0", "start": 0, "end": 3, "samples": 3},
        {"device": "cuda:1", "start": 3, "end": 5, "samples": 2},
    )
