from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class CameraConfig:
    name: str
    extrinsic_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("CameraConfig.name must be non-empty")


@dataclass(frozen=True)
class TrackingConfig:
    max_query_points_per_camera: int
    min_track_confidence: float = 0.0

    def __post_init__(self) -> None:
        if int(self.max_query_points_per_camera) < 1:
            raise ValueError("max_query_points_per_camera must be >= 1")
        c = float(self.min_track_confidence)
        if not (0.0 <= c <= 1.0):
            raise ValueError("min_track_confidence must be in [0, 1]")


@dataclass(frozen=True)
class WorkspaceFilterConfig:
    workspace_min: Vec3
    workspace_max: Vec3

    stability_ws_ratio_thresh: float = 0.9
    stability_ws_run_len_thresh: int = 8
    stability_apply_to_exists: bool = False
    stability_apply_to_confidence: bool = True

    def __post_init__(self) -> None:
        mn = tuple(float(x) for x in self.workspace_min)
        mx = tuple(float(x) for x in self.workspace_max)
        if len(mn) != 3 or len(mx) != 3:
            raise ValueError("workspace_min/workspace_max must be 3D tuples")
        if not (mn[0] < mx[0] and mn[1] < mx[1] and mn[2] < mx[2]):
            raise ValueError("workspace_min must be strictly smaller than workspace_max")

        r = float(self.stability_ws_ratio_thresh)
        if not (0.0 <= r <= 1.0):
            raise ValueError("stability_ws_ratio_thresh must be in [0, 1]")
        if int(self.stability_ws_run_len_thresh) < 1:
            raise ValueError("stability_ws_run_len_thresh must be >= 1")
        if int(self.stability_ws_run_len_thresh) > 11:
            raise ValueError("stability_ws_run_len_thresh must be <= 11")


Sphere = Tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotFilterConfig:
    robot_mask_margin_m: float = 0.0

    ee_filter_enabled: bool = False
    ee_filter_link: str = "panda_link7"
    ee_filter_spheres: Tuple[Sphere, ...] = ()
    ee_filter_margin_m: float = 0.0

    def __post_init__(self) -> None:
        if float(self.robot_mask_margin_m) < 0.0:
            raise ValueError("robot_mask_margin_m must be >= 0")
        if float(self.ee_filter_margin_m) < 0.0:
            raise ValueError("ee_filter_margin_m must be >= 0")
        if bool(self.ee_filter_enabled) and not str(self.ee_filter_link):
            raise ValueError("ee_filter_link must be non-empty when ee_filter_enabled is True")
        for s in self.ee_filter_spheres:
            if len(s) != 4:
                raise ValueError("ee_filter_spheres entries must be (x,y,z,r)")
            if float(s[3]) <= 0.0:
                raise ValueError("ee_filter_spheres radius must be > 0")


@dataclass(frozen=True)
class PointWorldInputConfig:
    window_size: int
    tracking: TrackingConfig
    workspace_filter: WorkspaceFilterConfig
    robot_filter: RobotFilterConfig = RobotFilterConfig()

    urdf_path: Optional[str] = None
    seed_robot_mask_enabled: bool = True
    robot_mask_seed: Optional[int] = None

    camera_names: Optional[Tuple[str, ...]] = None
    camera_extrinsic_paths: Optional[Dict[str, str]] = None
    camera_selection: str = "all_available"
    min_cameras: int = 1

    def __post_init__(self) -> None:
        if int(self.window_size) < 1:
            raise ValueError("window_size must be >= 1")
        if int(self.window_size) != 11:
            raise ValueError("PointWorld requires window_size == 11")
        if int(self.min_cameras) < 1:
            raise ValueError("min_cameras must be >= 1")

        if (bool(self.seed_robot_mask_enabled) or bool(self.robot_filter.ee_filter_enabled)) and not self.urdf_path:
            raise ValueError("urdf_path is required when seed_robot_mask_enabled or ee_filter_enabled is True")

        if self.robot_mask_seed is not None:
            object.__setattr__(self, "robot_mask_seed", int(self.robot_mask_seed))

        if (bool(self.seed_robot_mask_enabled) or bool(self.robot_filter.ee_filter_enabled)) and not self.urdf_path:
            raise ValueError("urdf_path is required when seed_robot_mask_enabled or ee_filter_enabled is True")

        if self.robot_mask_seed is not None:
            object.__setattr__(self, "robot_mask_seed", int(self.robot_mask_seed))

        if self.camera_names is not None:
            names = tuple(str(x) for x in self.camera_names)
            if len(names) == 0:
                raise ValueError("camera_names must be non-empty if provided")
            if len(set(names)) != len(names):
                raise ValueError("camera_names must be unique")
            object.__setattr__(self, "camera_names", names)

        if self.camera_extrinsic_paths is not None:
            paths = {str(k): str(v) for k, v in dict(self.camera_extrinsic_paths).items()}
            if self.camera_names is not None:
                missing = [n for n in self.camera_names if n not in paths]
                if missing:
                    raise ValueError(f"camera_extrinsic_paths missing keys: {missing}")
            object.__setattr__(self, "camera_extrinsic_paths", paths)

        if str(self.camera_selection) not in {"all_available", "subset"}:
            raise ValueError('camera_selection must be "all_available" or "subset"')

    def select_cameras(self, available: Sequence[str]) -> Tuple[str, ...]:
        avail = tuple(str(x) for x in available)
        if self.camera_selection == "all_available":
            used = avail
        else:
            if self.camera_names is None:
                used = avail
            else:
                s = set(avail)
                used = tuple(n for n in self.camera_names if n in s)

        if len(used) < int(self.min_cameras):
            raise RuntimeError(
                f"Not enough cameras available: used={len(used)} < min_cameras={self.min_cameras}"
            )
        return used

    def iter_camera_configs(self, available: Sequence[str]) -> Tuple[CameraConfig, ...]:
        used = self.select_cameras(available)
        paths = self.camera_extrinsic_paths or {}
        return tuple(CameraConfig(name=n, extrinsic_path=paths.get(n)) for n in used)