from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def repo_path(*parts: str) -> str:
    return str(repo_root().joinpath(*parts))


def _candidate_urdf_paths() -> list[str]:
    env_path = os.getenv("MPPI_URDF_PATH", "").strip()
    root = repo_root()
    candidates = [
        env_path,
        "/workspace/pointworld/assets/franka_description/franka_panda_robotiq_2f85.urdf",
        str(root.parent / "PointWorld" / "assets" / "franka_description" / "franka_panda_robotiq_2f85.urdf"),
        "/home/wangning/PointWorld/assets/franka_description/franka_panda_robotiq_2f85.urdf",
        "/home/wangyuhan/PointWorld/assets/franka_description/franka_panda_robotiq_2f85.urdf",
    ]
    out: list[str] = []
    for path in candidates:
        p = str(path).strip()
        if p and p not in out:
            out.append(p)
    return out


def default_urdf_path() -> str:
    for path in _candidate_urdf_paths():
        if Path(path).is_file():
            return path
    candidates = _candidate_urdf_paths()
    return candidates[0] if candidates else "/workspace/pointworld/assets/franka_description/franka_panda_robotiq_2f85.urdf"


def _candidate_pointworld_roots() -> list[str]:
    root = repo_root()
    candidates = [
        os.getenv("POINTWORLD_ROOT", "").strip(),
        "/workspace/pointworld",
        str(root.parent / "PointWorld"),
        "/home/wangning/PointWorld",
        "/home/wangyuhan/PointWorld",
    ]
    out: list[str] = []
    for path in candidates:
        p = str(path).strip()
        if p and p not in out:
            out.append(p)
    return out


def default_pointworld_root() -> str:
    for path in _candidate_pointworld_roots():
        if Path(path).is_dir():
            return path
    candidates = _candidate_pointworld_roots()
    return candidates[0] if candidates else "/workspace/pointworld"


def default_cotracker_root() -> str:
    candidates = [
        os.getenv("COTRACKER_ROOT", "").strip(),
        repo_path("third_party", "co-tracker"),
        str(Path(default_pointworld_root()) / "third_party" / "co-tracker"),
    ]
    out: list[str] = []
    for path in candidates:
        p = str(path).strip()
        if p and p not in out:
            out.append(p)
    for path in out:
        if Path(path).is_dir():
            return path
    return out[0] if out else repo_path("third_party", "co-tracker")


def default_curobo_root() -> str:
    candidates = [
        os.getenv("CUROBO_ROOT", "").strip(),
        repo_path("third_party", "curobo"),
        "/home/wangyuhan/curobo",
    ]
    out: list[str] = []
    for path in candidates:
        p = str(path).strip()
        if p and p not in out:
            out.append(p)
    for path in out:
        marker = Path(path) / "curobo" / "__init__.py"
        if marker.is_file():
            return path
    return out[0] if out else repo_path("third_party", "curobo")


def ensure_sys_path_for_runtime() -> None:
    roots = [
        default_pointworld_root(),
        default_cotracker_root(),
        default_curobo_root(),
        str(repo_root()),
        repo_path("src"),
    ]
    for path in roots:
        if Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)
