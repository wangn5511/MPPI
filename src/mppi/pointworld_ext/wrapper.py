from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch.nn as nn

from mppi.costs.pointworld_cost import (
    PointWorldCostConfig,
    reduce_pointworld_cost_torch,
)
from mppi.pointworld_ext.flows import (
    RobotFlowAdapter,
    build_robot_inputs,
    build_scene_features_torch,
    prepare_scene_inputs,
)
from mppi.utils.paths import default_pointworld_root, default_urdf_path, ensure_sys_path_for_runtime


@dataclass(frozen=True)
class _PointWorldReplica:
    device: str
    model: Any
    robot: RobotFlowAdapter


@dataclass(frozen=True)
class PointWorldModelConfig:
    checkpoint_path: str
    device: str = "cuda"
    domain: Optional[str] = None
    urdf_path: str = field(default_factory=default_urdf_path)
    max_scene_points: Optional[int] = None
    max_robot_points: Optional[int] = None
    robot_sampler_device: Optional[str] = None
    robot_gripper_only: bool = True
    seed: int = 1
    disable_compile: bool = True
    eval_batch_size: int = 32
    cost: PointWorldCostConfig = field(default_factory=PointWorldCostConfig)


class PointWorldCostModel:
    def __init__(self, cfg: PointWorldModelConfig) -> None:
        self.cfg = cfg
        self._torch = self._require_torch()
        self._checkpoint = self._load_checkpoint(cfg.checkpoint_path)
        self._model_contract, self._data_contract = self._read_contract(self._checkpoint)
        self._domain = str(cfg.domain or self._data_contract["domains"][0])
        self._devices = self._expand_device_list(str(cfg.device), fallback="cuda")
        self._robot_devices = self._resolve_robot_devices()

        self.max_scene_points = int(
            cfg.max_scene_points
            if cfg.max_scene_points is not None
            else min(int(self._model_contract["max_scene_points"]), 1024)
        )
        self.max_robot_points = int(
            cfg.max_robot_points
            if cfg.max_robot_points is not None
            else min(int(self._model_contract["max_robot_points"]), 256)
        )

        self._replicas = [
            self._build_replica(device=model_device, robot_device=robot_device)
            for model_device, robot_device in zip(self._devices, self._robot_devices)
        ]

    def __call__(
        self,
        *,
        q_traj: np.ndarray,
        u_traj: np.ndarray,
        pointworld_obs: dict[str, Any],
        gripper: Optional[float],
    ) -> np.ndarray:
        del u_traj
        return self.evaluate_cost(q_traj=q_traj, pointworld_obs=pointworld_obs, gripper=gripper)

    def _require_torch(self):
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Missing dependency: torch") from e
        return torch

    def _load_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        ckpt = str(checkpoint_path).strip()
        if not ckpt:
            raise ValueError("PointWorld checkpoint_path is required")
        return self._torch.load(ckpt, map_location="cpu", weights_only=False)

    def _expand_device_list(self, raw: str, *, fallback: str) -> list[str]:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if parts:
            return parts
        return [str(fallback)]

    def _resolve_robot_devices(self) -> list[str]:
        raw = str(self.cfg.robot_sampler_device or "").strip()
        parts = self._expand_device_list(raw, fallback=self._devices[0]) if raw else list(self._devices)
        if len(parts) == 1 and len(self._devices) > 1:
            parts = parts * len(self._devices)
        if len(parts) != len(self._devices):
            raise ValueError(
                f"robot_sampler_device count {len(parts)} must match model devices {len(self._devices)} or be a single device"
            )
        return parts

    def _read_contract(self, checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        ensure_sys_path_for_runtime()
        from pointworld.checkpoint_contract import read_checkpoint_contract

        return read_checkpoint_contract(checkpoint, context=f"PointWorld checkpoint '{self.cfg.checkpoint_path}'")

    def _resolve_norm_stats_path(self, raw_path: str) -> str:
        p = Path(str(raw_path))
        if p.is_absolute():
            return str(p)
        return str(Path(default_pointworld_root()) / p)

    def _ensure_dinov3_weights(self) -> None:
        root = Path(default_pointworld_root())
        dinov3_root = root / "third_party" / "dinov3"
        if not dinov3_root.is_dir():
            return

        ckpt_dir = dinov3_root / "checkpoints"
        if ckpt_dir.is_dir() and any(ckpt_dir.glob("dinov3_vitl16_pretrain*.pth")):
            return

        source = Path("/home/models/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
        if not source.is_file():
            return

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        target = ckpt_dir / source.name
        if target.exists():
            return
        target.symlink_to(source)

    def _build_replica(self, *, device: str, robot_device: str) -> _PointWorldReplica:
        return _PointWorldReplica(
            device=str(device),
            model=self._build_model(device=str(device)),
            robot=RobotFlowAdapter(
                urdf_path=str(self.cfg.urdf_path),
                max_robot_points=int(self.max_robot_points),
                device=str(robot_device),
                seed=int(self.cfg.seed),
                gripper_only=bool(self.cfg.robot_gripper_only),
            ),
        )

    def _build_args(self, *, device: str) -> SimpleNamespace:
        mc = self._model_contract
        domains = list(self._data_contract["domains"])
        return SimpleNamespace(
            device=str(device),
            distributed=False,
            predictor_dim=int(mc["predictor_dim"]),
            ptv3_size=str(mc["ptv3_size"]),
            ptv3_patch_size=int(mc["ptv3_patch_size"]),
            grid_size=float(mc["grid_size"]),
            depth_threshold=float(mc["depth_threshold"]),
            norm_stats_path=self._resolve_norm_stats_path(str(mc["norm_stats_path"])),
            disable_compile=bool(self.cfg.disable_compile),
            domains=domains,
            robot_features=[
                "robot_flows",
                "robot_colors",
                "robot_normals",
                "gripper_open",
                "robot_velocity",
                "robot_acceleration",
            ],
            scene_features=[
                "scene_flows",
                "scene_colors",
                "scene_normals",
                "gripper_open",
                "dist2robot",
            ],
            seed=int(self.cfg.seed),
            deterministic_train=False,
            deterministic_algorithms=False,
            amp=True,
            dynamics_head_init_scale=(1.0 if any(str(d).startswith("droid") for d in domains) else 0.0),
        )

    def _infer_feature_dims(self, checkpoint: dict[str, Any]) -> dict[str, int]:
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise KeyError("PointWorld checkpoint missing 'model' state dict")
        scene_key = "scene_feature_encoder.scene_raw_feat_proj.weight"
        robot_key = "robot_proj.fc1.weight"
        if scene_key not in state or robot_key not in state:
            raise KeyError("PointWorld checkpoint is missing required projection weights")
        return {
            "scene_features_dim": int(state[scene_key].shape[1]),
            "robot_features_dim": int(state[robot_key].shape[1]),
        }

    def _build_model(self, *, device: str):
        ensure_sys_path_for_runtime()
        self._ensure_dinov3_weights()
        args = self._build_args(device=str(device))
        data_info = self._infer_feature_dims(self._checkpoint)
        state = self._checkpoint.get("model")
        if not isinstance(state, dict):
            raise KeyError("PointWorld checkpoint missing 'model' state dict")

        try:
            from pointworld.base import BaseModel

            model = BaseModel(args, data_info, rank=0, cpu_pg=None)
            model.load_state_dict(state, strict=True)
            model.to(self._torch.device(str(device)))
            model.eval()
            return model
        except Exception as exc:
            if "DINOv3" not in str(exc) and "dinov3" not in str(exc):
                raise
            if os.getenv("MPPI_PW_ALLOW_RAW_SCENE_FALLBACK", "0").strip().lower() not in {"1", "true", "yes"}:
                raise
            return self._build_model_without_dinov3(args=args, data_info=data_info, state=state, device=str(device))

    def _build_model_without_dinov3(self, *, args: Any, data_info: dict[str, int], state: dict[str, Any], device: str):
        ensure_sys_path_for_runtime()
        import scene_featurizer
        from pointworld.base import BaseModel

        predictor_dim = int(self._model_contract["predictor_dim"])
        feat_proj_w = state.get("scene_feature_encoder.scene_encoder.feat_proj.weight")
        if feat_proj_w is None:
            raise KeyError("Checkpoint missing scene_feature_encoder.scene_encoder.feat_proj.weight")
        dino_in_dim = int(feat_proj_w.shape[1])

        class _ZeroSceneEncoder2D(nn.Module):
            def __init__(self, args, channels, data_info_dict, rank: int = 0):  # noqa: ARG002
                super().__init__()
                self.args = args
                self.rank = rank
                self.device = args.device
                self.channels = channels
                self.feat_proj = nn.Linear(dino_in_dim, channels)

            def forward(self, scene_coord, scene_exists, camera_data):  # noqa: ARG002
                import torch

                B, Ns, _ = scene_coord.shape
                out = torch.zeros((B, Ns, predictor_dim), device=scene_coord.device, dtype=scene_coord.dtype)
                out[~scene_exists] = 0.0
                return out

            def _extract_camera_data(self, data_dict):
                return {}

        original_cls = scene_featurizer.SceneEncoder2D
        scene_featurizer.SceneEncoder2D = _ZeroSceneEncoder2D
        try:
            model = BaseModel(args, data_info, rank=0, cpu_pg=None)
        finally:
            scene_featurizer.SceneEncoder2D = original_cls

        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = {
            "scene_feature_encoder.scene_encoder.feat_proj.weight",
            "scene_feature_encoder.scene_encoder.feat_proj.bias",
        }
        extra_missing = [k for k in missing if k not in allowed_missing]
        if extra_missing:
            raise RuntimeError(f"Unexpected missing keys in PointWorld fallback load: {extra_missing}")
        if not all(k.startswith("scene_feature_encoder.scene_encoder.dinov3.") for k in unexpected):
            raise RuntimeError(f"Unexpected non-DINO keys in PointWorld fallback load: {unexpected}")

        model.to(self._torch.device(str(device)))
        model.eval()
        return model

    def _encode_scene_raw_only(self, scene_features_t: Any, *, model: Any, device: str) -> Any:
        torch = self._torch

        B = int(scene_features_t.shape[0])
        device_t = torch.device(str(device))
        idx = [model._domain_to_index[self._domain] for _ in range(B)]
        model._current_domain_indices = torch.as_tensor(idx, device=device_t, dtype=torch.long)

        sfe = model.scene_feature_encoder
        raw = model.normalize_scene_features(scene_features_t)
        zeros = torch.zeros((B, raw.shape[1], model.channels), device=device_t, dtype=raw.dtype)
        fused = torch.cat(
            [
                sfe.scene_encoder_norm(zeros),
                sfe.scene_raw_norm(sfe.scene_raw_feat_proj(raw)),
            ],
            dim=-1,
        )
        return sfe.scene_proj(fused)

    def _prepare_batch(
        self,
        *,
        q_traj: np.ndarray,
        pointworld_obs: dict[str, Any],
        gripper: Optional[float],
        robot: RobotFlowAdapter,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        q = np.asarray(q_traj, dtype=np.float32)
        if q.ndim != 3 or q.shape[-1] != 7:
            raise ValueError(f"q_traj must be (B,T,7), got {q.shape}")
        B, T, _ = q.shape

        scene = prepare_scene_inputs(
            scene_flows=np.asarray(pointworld_obs["scene_flows"], dtype=np.float32),
            scene_colors=np.asarray(pointworld_obs.get("scene_colors"), dtype=np.uint8),
            scene_exists=np.asarray(pointworld_obs["scene_exists"], dtype=bool),
            scene_track_confidence=pointworld_obs.get("scene_track_confidence"),
            batch_size=1,
            max_scene_points=int(self.max_scene_points),
        )
        if int(scene["scene_flows"].shape[1]) != int(T):
            raise ValueError(
                f"PointWorld scene window T={int(scene['scene_flows'].shape[1])} must match q_traj horizon T={int(T)}"
            )

        if gripper is None:
            gripper_arr = np.asarray(pointworld_obs.get("gripper_positions_window"), dtype=np.float32).reshape(1, -1)
            if gripper_arr.shape[1] != T:
                if gripper_arr.shape[1] == 0:
                    gripper_arr = np.zeros((1, T), dtype=np.float32)
                else:
                    gripper_arr = np.repeat(gripper_arr[:, :1], T, axis=1)
            gripper_arr = np.repeat(gripper_arr, B, axis=0)
        else:
            gripper_arr = np.full((B, T), float(gripper), dtype=np.float32)

        robot_flows, robot_colors, robot_normals = robot.build(q_traj=q, gripper_positions=gripper_arr)
        robot = build_robot_inputs(
            robot_flows=robot_flows,
            robot_colors=robot_colors,
            robot_normals=robot_normals,
            gripper_positions=gripper_arr,
            max_robot_points=int(self.max_robot_points),
        )
        batch = {
            "gripper_positions": gripper_arr,
            "robot_flows": robot["robot_flows"],
            "robot_features": robot["robot_features"],
            "robot_exists": robot["robot_exists"],
        }
        return scene, batch

    def _numpy_batch_to_torch(self, batch: dict[str, Any], *, device: str) -> dict[str, Any]:
        torch = self._torch
        device_t = torch.device(str(device))
        out: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                if value.dtype == np.bool_:
                    out[key] = torch.as_tensor(value, device=device_t, dtype=torch.bool)
                else:
                    out[key] = torch.as_tensor(value, device=device_t, dtype=torch.float32)
            else:
                out[key] = value
        return out

    def _slice_batch(self, batch: dict[str, Any], start: int, end: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1 and int(value.shape[0]) == int(batch["scene_flows"].shape[0]):
                out[key] = value[start:end]
            elif isinstance(value, list) and len(value) == int(batch["scene_flows"].shape[0]):
                out[key] = value[start:end]
            else:
                out[key] = value
        return out

    def _evaluate_batch_on_replica(
        self,
        *,
        replica: _PointWorldReplica,
        scene_np: dict[str, np.ndarray],
        batch_np: dict[str, np.ndarray],
    ) -> np.ndarray:
        B = int(batch_np["robot_flows"].shape[0])
        chunk_size = max(1, min(int(self.cfg.eval_batch_size), B))
        costs: list[np.ndarray] = []
        torch = self._torch

        if replica.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(replica.device))

        scene_t = self._numpy_batch_to_torch(scene_np, device=replica.device)

        for start in range(0, B, chunk_size):
            end = min(B, start + chunk_size)
            batch_chunk = self._slice_batch(batch_np, start, end)
            robot_t = self._numpy_batch_to_torch(batch_chunk, device=replica.device)
            chunk_b = int(end - start)

            scene_flows_t = scene_t["scene_flows"].expand(chunk_b, -1, -1, -1)
            scene_colors_t = scene_t["scene_colors"].expand(chunk_b, -1, -1, -1)
            scene_exists_t = scene_t["scene_exists"].expand(chunk_b, -1, -1)
            scene_features_t = build_scene_features_torch(
                scene_flows=scene_flows_t,
                scene_colors=scene_colors_t,
                gripper_positions=robot_t["gripper_positions"],
            )
            batch_t = {
                "scene_flows": scene_flows_t,
                "scene_exists": scene_exists_t,
                "scene_features": scene_features_t,
                "robot_flows": robot_t["robot_flows"],
                "robot_features": robot_t["robot_features"],
                "robot_exists": robot_t["robot_exists"],
                "__domain__": [self._domain] * chunk_b,
            }

            with torch.no_grad():
                encoded_scene = self._encode_scene_raw_only(
                    batch_t["scene_features"][:, 0],
                    model=replica.model,
                    device=replica.device,
                )
                outputs = replica.model(batch_t, training=False, encoded_scene_feat0=encoded_scene)

            model_conf = outputs.get("confidence")
            track_conf_t = None
            if "scene_track_confidence" in scene_t:
                track_conf_t = scene_t["scene_track_confidence"].expand(chunk_b, -1, -1)

            costs.append(
                reduce_pointworld_cost_torch(
                    scene_relative=outputs["scene_relative"],
                    scene_exists=batch_t["scene_exists"],
                    model_confidence=model_conf,
                    track_confidence=track_conf_t,
                    cfg=self.cfg.cost,
                ).detach().cpu().numpy().astype(np.float32)
            )

        return np.concatenate(costs, axis=0).astype(np.float32, copy=False)

    def _evaluate_cost_on_replica(
        self,
        *,
        replica: _PointWorldReplica,
        q_traj: np.ndarray,
        pointworld_obs: dict[str, Any],
        gripper: Optional[float],
    ) -> np.ndarray:
        scene_np, batch_np = self._prepare_batch(
            q_traj=q_traj,
            pointworld_obs=pointworld_obs,
            gripper=gripper,
            robot=replica.robot,
        )
        return self._evaluate_batch_on_replica(replica=replica, scene_np=scene_np, batch_np=batch_np)

    def _make_ranges(self, total: int, parts: int) -> list[tuple[int, int]]:
        if total <= 0 or parts <= 0:
            return []
        parts = min(int(parts), int(total))
        base = total // parts
        rem = total % parts
        out: list[tuple[int, int]] = []
        start = 0
        for idx in range(parts):
            width = base + (1 if idx < rem else 0)
            end = start + width
            out.append((start, end))
            start = end
        return out

    def evaluate_cost(
        self,
        *,
        q_traj: np.ndarray,
        pointworld_obs: dict[str, Any],
        gripper: Optional[float],
    ) -> np.ndarray:
        q = np.asarray(q_traj, dtype=np.float32)
        if q.ndim != 3 or q.shape[-1] != 7:
            raise ValueError(f"q_traj must be (B,T,7), got {q.shape}")
        B = int(q.shape[0])
        if B == 0:
            return np.zeros((0,), dtype=np.float32)

        if len(self._replicas) == 1:
            return self._evaluate_cost_on_replica(
                replica=self._replicas[0],
                q_traj=q,
                pointworld_obs=pointworld_obs,
                gripper=gripper,
            )

        ranges = self._make_ranges(B, len(self._replicas))
        costs = np.zeros((B,), dtype=np.float32)

        def _worker(replica: _PointWorldReplica, start: int, end: int) -> tuple[int, int, np.ndarray]:
            return (
                int(start),
                int(end),
                self._evaluate_cost_on_replica(
                    replica=replica,
                    q_traj=q[start:end],
                    pointworld_obs=pointworld_obs,
                    gripper=gripper,
                ),
            )

        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [
                pool.submit(_worker, self._replicas[idx], start, end)
                for idx, (start, end) in enumerate(ranges)
            ]
            for fut in futures:
                start, end, arr = fut.result()
                costs[start:end] = arr

        return costs
