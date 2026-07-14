from __future__ import annotations

import numpy as np

import mppi.mpc.solver as solver_mod
from mppi.mpc.solver import (
    JointMPPIConfig,
    _curobo_batch_distance,
    _expand_device_list,
    _first_device,
    _make_sample_ranges,
)


def test_curobo_device_list_helpers() -> None:
    assert _expand_device_list("cuda:0, cuda:1", fallback="cuda:0") == ("cuda:0", "cuda:1")
    assert _expand_device_list("", fallback="cuda:0") == ("cuda:0",)
    assert _first_device("cuda:2,cuda:3") == "cuda:2"
    assert _make_sample_ranges(5, 2) == [(0, 3), (3, 5)]
    assert _make_sample_ranges(2, 8) == [(0, 1), (1, 2)]


def test_curobo_batch_distance_splits_samples_across_devices(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    class _FakeChecker:
        def __init__(self, device: str) -> None:
            self.device = str(device)

        def batch_distance(self, q_traj, *, scene_cuboids=None):
            del scene_cuboids
            q = np.asarray(q_traj, dtype=np.float32)
            device_idx = int(self.device.split(":")[-1])
            calls.append((self.device, tuple(int(x) for x in q[:, 0, 0].tolist())))

            B, H, _ = q.shape
            d_scene = np.zeros((B, H, 2), dtype=np.float32)
            d_self = np.zeros((B, H, 1), dtype=np.float32)
            d_scene[:, :, 0] = float(device_idx)
            d_scene[:, :, 1] = q[:, :1, 0]
            d_self[:, :, 0] = 10.0 + q[:, :1, 0]
            return d_scene, d_self

    def fake_get_curobo_collision_checker(cfg):
        return _FakeChecker(str(cfg.device))

    monkeypatch.setattr(solver_mod, "get_curobo_collision_checker", fake_get_curobo_collision_checker)

    cfg = JointMPPIConfig(curobo_device="cuda:0,cuda:1")
    q = np.zeros((5, 3, 7), dtype=np.float32)
    q[:, :, 0] = np.arange(5, dtype=np.float32)[:, None]

    d_scene, d_self, ranges = _curobo_batch_distance(cfg, q, scene_cuboids=[{"center": [0, 0, 0], "dims": [1, 1, 1]}])

    assert ranges == (
        {"device": "cuda:0", "start": 0, "end": 3, "samples": 3},
        {"device": "cuda:1", "start": 3, "end": 5, "samples": 2},
    )
    assert calls == [("cuda:0", (0, 1, 2)), ("cuda:1", (3, 4))]
    assert d_scene[:, 0, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert d_scene[:, 0, 1].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert d_self[:, 0, 0].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
