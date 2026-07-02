#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
离线调试 robot mask（仅使用 back 视角）

用法：
  bash /home/wangyuhan/MPPI/scripts/debug_robot_mask.sh [选项]

常用选项：
  --frame-idx N                 选择 data.json 中第 N 条样本，默认 0
  --data-root PATH              数据根目录，默认 /home/datasets/FrankaNav/robot_test
  --json PATH                   标注 json，默认 /home/wangyuhan/MPPI/data/test/data.json
  --out-dir PATH                输出目录，默认 /home/wangyuhan/MPPI/data/robot_mask_debug
  --cam-info PATH               相机内参，默认 /home/wangyuhan/MPPI/configs/back_cam_info.yaml
  --t-base-cam PATH             base<-cam 外参，默认 /home/wangyuhan/MPPI/configs/T_base_cam.yaml
  --device DEV                  cuRobo 设备，默认 cuda:0
  --robot-mask-margin-m X       robot spheres 膨胀半径，默认 0.02
  --depth-min-m X               最小深度，默认 0.07
  --depth-max-m X               最大深度，默认 2.0
  --voxel-size-m X              体素大小，默认 0.01
  --roi-min x,y,z               ROI 最小值
  --roi-max x,y,z               ROI 最大值
  --show                        生成后调用 visual_npy.py 显示点云

输出文件：
  frame_<frame_id>_base.npz         原始 base 点云
  frame_<frame_id>_base.ply
  frame_<frame_id>_masked.npz       去除机器人后的环境点云
  frame_<frame_id>_masked.ply
  frame_<frame_id>_robot_only.npz   被 mask 掉的机器人点
  frame_<frame_id>_robot_only.ply
  frame_<frame_id>_spheres.npz      cuRobo 生成的 robot spheres
EOF
}

FRAME_IDX=0
DATA_ROOT="/home/datasets/FrankaNav/test"
JSON_PATH="/home/wangyuhan/MPPI/data/test/data.json"
OUT_DIR="/home/wangyuhan/MPPI/data/robot_mask_debug"
CAM_INFO="/home/wangyuhan/MPPI/configs/back_cam_info.yaml"
T_BASE_CAM="/home/wangyuhan/MPPI/configs/T_base_cam.yaml"
URDF_PATH="/home/wangyuhan/PointWorld/assets/franka_description/franka_panda_robotiq_2f85.urdf"
ROBOT_YAML="franka.yml"
TOOL_FRAME="robotiq_85_base_link"
CUROBO_DEVICE="cuda:0"

DEPTH_UNIT_SCALE="1.0"
DEPTH_MIN_M="0.07"
DEPTH_MAX_M="2.0"
VOXEL_SIZE_M="0.01"
ROI_MIN="-0.1,-0.7,-0.05"
ROI_MAX="1.2,0.7,1.2"
ROBOT_MASK_MARGIN_M="0.1"
SHOW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --frame-idx) FRAME_IDX="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --json) JSON_PATH="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --cam-info) CAM_INFO="$2"; shift 2 ;;
    --t-base-cam) T_BASE_CAM="$2"; shift 2 ;;
    --device) CUROBO_DEVICE="$2"; shift 2 ;;
    --robot-mask-margin-m) ROBOT_MASK_MARGIN_M="$2"; shift 2 ;;
    --depth-min-m) DEPTH_MIN_M="$2"; shift 2 ;;
    --depth-max-m) DEPTH_MAX_M="$2"; shift 2 ;;
    --voxel-size-m) VOXEL_SIZE_M="$2"; shift 2 ;;
    --roi-min) ROI_MIN="$2"; shift 2 ;;
    --roi-max) ROI_MAX="$2"; shift 2 ;;
    --show) SHOW=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

export PYTHONPATH="/home/wangyuhan/MPPI/src:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR"

mapfile -t META < <(
python3 - "$JSON_PATH" "$DATA_ROOT" "$FRAME_IDX" <<'PY'
import json, os, re, sys

json_path, data_root, frame_idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
item = data[frame_idx]

def normalize_rel(p: str) -> str:
    p = str(p).replace("\\", "/")
    p = re.sub(r"^ep_[^/]+/", "", p)
    return p

depth_rel = normalize_rel(item["depths"]["back_depth"])
rgb_rel = normalize_rel(item["images"]["back"])
q = item["/franka/joint_states"]["position"]

depth = os.path.join(data_root, depth_rel)
rgb = os.path.join(data_root, rgb_rel)

if not os.path.isfile(depth):
    raise FileNotFoundError(f"depth not found: {depth}")
if not os.path.isfile(rgb):
    raise FileNotFoundError(f"rgb not found: {rgb}")
if len(q) != 7:
    raise ValueError(f"expected 7 joint positions, got {len(q)}")

print(str(item.get("frame_id", frame_idx)))
print(depth)
print(os.path.dirname(rgb))
print(",".join(str(float(x)) for x in q))
PY
)

FRAME_ID="${META[0]}"
DEPTH="${META[1]}"
RGB_DIR="${META[2]}"
Q="${META[3]}"

FRAME_TAG="$(printf "%04d" "$FRAME_ID")"
BASE_NPZ="$OUT_DIR/frame_${FRAME_TAG}_base.npz"
BASE_PLY="$OUT_DIR/frame_${FRAME_TAG}_base.ply"
MASKED_NPZ="$OUT_DIR/frame_${FRAME_TAG}_masked.npz"
MASKED_PLY="$OUT_DIR/frame_${FRAME_TAG}_masked.ply"
ROBOT_ONLY_NPZ="$OUT_DIR/frame_${FRAME_TAG}_robot_only.npz"
ROBOT_ONLY_PLY="$OUT_DIR/frame_${FRAME_TAG}_robot_only.ply"
SPHERES_NPZ="$OUT_DIR/frame_${FRAME_TAG}_spheres.npz"

echo "== Frame Metadata =="
echo "frame_id: $FRAME_ID"
echo "depth:    $DEPTH"
echo "rgb_dir:  $RGB_DIR"
echo "q:        $Q"
echo

echo "== Step 1/2: 生成 base 点云 =="
python3 /home/wangyuhan/MPPI/src/mppi/curobo_ext/check_depth.py \
  --depth "$DEPTH" \
  --rgb-dir "$RGB_DIR" \
  --cam-info "$CAM_INFO" \
  --t-base-cam "$T_BASE_CAM" \
  --depth-unit-scale "$DEPTH_UNIT_SCALE" \
  --depth-min-m "$DEPTH_MIN_M" \
  --depth-max-m "$DEPTH_MAX_M" \
  --roi-min="$ROI_MIN" \
  --roi-max="$ROI_MAX" \
  --voxel-size-m "$VOXEL_SIZE_M" \
  --merge \
  --out "$BASE_NPZ" \
  --save-ply "$BASE_PLY"

echo
echo "== Step 2/2: 用 cuRobo spheres 做 robot mask =="
python3 - "$BASE_NPZ" "$MASKED_NPZ" "$MASKED_PLY" "$ROBOT_ONLY_NPZ" "$ROBOT_ONLY_PLY" "$SPHERES_NPZ" "$Q" "$ROBOT_MASK_MARGIN_M" "$CUROBO_DEVICE" "$ROBOT_YAML" "$URDF_PATH" "$TOOL_FRAME" <<'PY'
import os
import sys
import numpy as np

from mppi.curobo_ext.collision_checker import CuRoboCollisionConfig, get_curobo_collision_checker

pc_in, masked_npz, masked_ply, robot_npz, robot_ply, spheres_npz, q_csv, margin_m, device, robot_yaml, urdf_path, tool_frame = sys.argv[1:]
margin_m = float(margin_m)
q = np.fromstring(q_csv, sep=",", dtype=np.float32)
if q.shape != (7,):
    raise ValueError(f"expected q shape (7,), got {q.shape}")

z = np.load(pc_in, allow_pickle=False)
pts = z["points"].astype(np.float32)
cols = z["colors"].astype(np.uint8) if "colors" in z.files else None

ccfg = CuRoboCollisionConfig(
    device=device,
    robot_yaml=robot_yaml,
    urdf_path=urdf_path,
    tool_frame=tool_frame,
    with_world=False,
)
checker = get_curobo_collision_checker(ccfg)
spheres = checker.get_robot_spheres_base(q, margin_m=margin_m)
np.savez_compressed(spheres_npz, spheres=spheres)

keep = np.ones((pts.shape[0],), dtype=bool)
for c, r in zip(spheres[:, :3], spheres[:, 3]):
    d2 = np.sum((pts - c[None, :]) ** 2, axis=1)
    keep &= d2 > float(r * r)

pts_masked = pts[keep]
pts_robot = pts[~keep]
cols_masked = cols[keep] if cols is not None else None
cols_robot = cols[~keep] if cols is not None else None

def save_npz(path, points, colors=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if colors is None:
        np.savez_compressed(path, points=np.asarray(points, dtype=np.float32))
    else:
        np.savez_compressed(
            path,
            points=np.asarray(points, dtype=np.float32),
            colors=np.asarray(colors, dtype=np.uint8),
        )

def save_ply(path, points, colors=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    p = np.asarray(points, dtype=np.float32)
    c = None if colors is None else np.asarray(colors, dtype=np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {p.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if c is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if c is None:
            for xyz in p:
                f.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n")
        else:
            for xyz, rgb in zip(p, c):
                f.write(f"{xyz[0]} {xyz[1]} {xyz[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")

save_npz(masked_npz, pts_masked, cols_masked)
save_ply(masked_ply, pts_masked, cols_masked)
save_npz(robot_npz, pts_robot, cols_robot)
save_ply(robot_ply, pts_robot, cols_robot)

print("robot_spheres:", spheres.shape)
print("points_before:", int(pts.shape[0]))
print("points_after_mask:", int(pts_masked.shape[0]))
print("robot_points_removed:", int(pts_robot.shape[0]))
print("saved:", masked_npz)
print("saved:", masked_ply)
print("saved:", robot_npz)
print("saved:", robot_ply)
print("saved:", spheres_npz)
PY

echo
echo "== 输出文件 =="
echo "$BASE_NPZ"
echo "$BASE_PLY"
echo "$MASKED_NPZ"
echo "$MASKED_PLY"
echo "$ROBOT_ONLY_NPZ"
echo "$ROBOT_ONLY_PLY"
echo "$SPHERES_NPZ"

if [[ "$SHOW" == "1" ]]; then
  echo
  echo "== 可视化 base 点云 =="
  python3 /home/wangyuhan/MPPI/src/mppi/curobo_ext/visual_npy.py --input "$BASE_NPZ" --show
  echo "== 可视化 masked 点云 =="
  python3 /home/wangyuhan/MPPI/src/mppi/curobo_ext/visual_npy.py --input "$MASKED_NPZ" --show
  echo "== 可视化 robot-only 点云 =="
  python3 /home/wangyuhan/MPPI/src/mppi/curobo_ext/visual_npy.py --input "$ROBOT_ONLY_NPZ" --show
fi