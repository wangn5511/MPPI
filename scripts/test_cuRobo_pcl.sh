#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
POINTWORLD_ROOT="${POINTWORLD_ROOT:-/workspace/pointworld}"

detect_visible_cuda_devices() {
  python3 - <<'PY'
import subprocess

try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    count = max(1, len(lines))
except Exception:
    count = 1

print(",".join(f"cuda:{i}" for i in range(count)))
PY
}

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/co-tracker:${REPO_ROOT}/third_party/curobo:${PYTHONPATH:-}"
if [ -d "${POINTWORLD_ROOT}" ]; then
  export PYTHONPATH="${POINTWORLD_ROOT}:${PYTHONPATH}"
fi

export MPPI_URDF_PATH="${MPPI_URDF_PATH:-/workspace/pointworld/assets/franka_description/franka_panda_robotiq_2f85.urdf}"

export MPPI_USE_CUROBO_COLLISION="${MPPI_USE_CUROBO_COLLISION:-1}"
export MPPI_W_SCENE_COLLISION="${MPPI_W_SCENE_COLLISION:-1}"
export MPPI_SCENE_FROM_PCD_BACK_CAM="${MPPI_SCENE_FROM_PCD_BACK_CAM:-1}"

export MPPI_SCENE_PCD_SCALE="${MPPI_SCENE_PCD_SCALE:-1}"
export MPPI_SCENE_PCD_IN_BASE="${MPPI_SCENE_PCD_IN_BASE:-1}"

export MPPI_SCENE_ADD_TABLE="${MPPI_SCENE_ADD_TABLE:-1}"
export MPPI_SCENE_TABLE_DIMS="${MPPI_SCENE_TABLE_DIMS:-2.0,2.0,0.2}"
export MPPI_SCENE_TABLE_CENTER="${MPPI_SCENE_TABLE_CENTER:-0.4,0.0,-0.1}"

export MPPI_SCENE_REMOVE_TABLE_POINTS="${MPPI_SCENE_REMOVE_TABLE_POINTS:-1}"
export MPPI_SCENE_TABLE_EPS_M="${MPPI_SCENE_TABLE_EPS_M:-0.01}"

export MPPI_SCENE_REMOVE_WALL_POINTS="${MPPI_SCENE_REMOVE_WALL_POINTS:-1}"
export MPPI_SCENE_WALL_DIMS="${MPPI_SCENE_WALL_DIMS:-2.5,0.5,2.0}"
export MPPI_SCENE_WALL_CENTER="${MPPI_SCENE_WALL_CENTER:-0.5,0.5,-0.5}"
export MPPI_SCENE_WALL_MARGIN_M="${MPPI_SCENE_WALL_MARGIN_M:-0.05}"

export MPPI_T_BASE_CAM_BACK_PATH="${MPPI_T_BASE_CAM_BACK_PATH:-${REPO_ROOT}/configs/T_base_cam.yaml}"
export MPPI_SCENE_ROI_MIN="${MPPI_SCENE_ROI_MIN:--0.1,-0.7,-0.05}"
export MPPI_SCENE_ROI_MAX="${MPPI_SCENE_ROI_MAX:-1.2,0.7,1.2}"
export MPPI_SCENE_VOXEL_SIZE_M="${MPPI_SCENE_VOXEL_SIZE_M:-0.01}"
export MPPI_SCENE_PADDING_M="${MPPI_SCENE_PADDING_M:-0.02}"
export MPPI_SCENE_MAX_CUBOIDS="${MPPI_SCENE_MAX_CUBOIDS:-20}"
export MPPI_SCENE_ROBOT_MASK_MARGIN_M="${MPPI_SCENE_ROBOT_MASK_MARGIN_M:-0.1}"
export MPPI_SCENE_MIN_CLUSTER_VOXELS="${MPPI_SCENE_MIN_CLUSTER_VOXELS:-20}"

export MPPI_SCENE_TRACK_ALPHA="${MPPI_SCENE_TRACK_ALPHA:-0.6}"
export MPPI_SCENE_TRACK_REMOVE_AFTER_MISSES="${MPPI_SCENE_TRACK_REMOVE_AFTER_MISSES:-5}"
export MPPI_SCENE_TRACK_MAX_TRACKS="${MPPI_SCENE_TRACK_MAX_TRACKS:-20}"
export MPPI_SCENE_TRACK_MATCH_CENTER_DIST_M="${MPPI_SCENE_TRACK_MATCH_CENTER_DIST_M:-0.05}"
export MPPI_SCENE_TRACK_MATCH_IOU_MIN="${MPPI_SCENE_TRACK_MATCH_IOU_MIN:-0.05}"

MPPI_INFER_BUDGET_MS_WAS_SET="${MPPI_INFER_BUDGET_MS+x}"
export MPPI_INFER_BUDGET_MS="${MPPI_INFER_BUDGET_MS:-50}"
export MPPI_BUDGET_MAX_DYNAMIC_CUBOIDS="${MPPI_BUDGET_MAX_DYNAMIC_CUBOIDS:-6}"

export MPPI_PCL_CAM_INFO_BACK_PATH="${MPPI_PCL_CAM_INFO_BACK_PATH:-${REPO_ROOT}/configs/back_cam_info.yaml}"
export MPPI_PCL_T_BASE_CAM_BACK_PATH="${MPPI_PCL_T_BASE_CAM_BACK_PATH:-${REPO_ROOT}/configs/T_base_cam.yaml}"
export MPPI_PCL_DEPTH_UNIT_SCALE="${MPPI_PCL_DEPTH_UNIT_SCALE:-1.0}"
export MPPI_PCL_DEPTH_MIN_M="${MPPI_PCL_DEPTH_MIN_M:-0.05}"
export MPPI_PCL_DEPTH_MAX_M="${MPPI_PCL_DEPTH_MAX_M:-2.0}"
export MPPI_PCL_STRIDE="${MPPI_PCL_STRIDE:-1}"

export MPPI_PCL_SAVE_PCD="${MPPI_PCL_SAVE_PCD:-0}"
export MPPI_PCL_SAVE_PCD_OUT="${MPPI_PCL_SAVE_PCD_OUT:-${REPO_ROOT}/data/test/real_pcl.npz}"
export MPPI_PCL_ROBOT_MASK_VIS_POINTS_PER_SPHERE="${MPPI_PCL_ROBOT_MASK_VIS_POINTS_PER_SPHERE:-800}"
export MPPI_PCL_ROBOT_MASK_VIS_MAX_POINTS="${MPPI_PCL_ROBOT_MASK_VIS_MAX_POINTS:-200000}"

export MPPI_PW_ENABLE="${MPPI_PW_ENABLE:-0}"
export MPPI_USE_POINTWORLD_COST="${MPPI_USE_POINTWORLD_COST:-0}"
export MPPI_W_POINTWORLD="${MPPI_W_POINTWORLD:-1.0}"
export MPPI_PW_DISABLE_COMPILE="${MPPI_PW_DISABLE_COMPILE:-1}"
export MPPI_PW_MODEL_DEVICE="${MPPI_PW_MODEL_DEVICE:-cuda}"
export MPPI_PW_COTRACKER_DEVICE="${MPPI_PW_COTRACKER_DEVICE:-cuda}"
export MPPI_PW_MODEL_PATH="${MPPI_PW_MODEL_PATH:-/home/models/PointWorld/PointWorld_models/large-droid/model-best.pt}"
export MPPI_PW_COTRACKER_CKPT="${MPPI_PW_COTRACKER_CKPT:-/home/models/Co-tracker/scaled_online.pth}"
export MPPI_PW_MODEL_DOMAIN="${MPPI_PW_MODEL_DOMAIN:-droid}"
export MPPI_PW_ALLOW_RAW_SCENE_FALLBACK="${MPPI_PW_ALLOW_RAW_SCENE_FALLBACK:-0}"

OPEN_LOOP_HORIZON="${MPPI_OPEN_LOOP_HORIZON:-8}"
if [ "${MPPI_PW_ENABLE}" = "1" ] || [ "${MPPI_USE_POINTWORLD_COST}" = "1" ]; then
  OPEN_LOOP_HORIZON="${MPPI_OPEN_LOOP_HORIZON:-11}"
fi

if [ "${MPPI_PW_ENABLE}" = "1" ] || [ "${MPPI_USE_POINTWORLD_COST}" = "1" ]; then
  if [ "${MPPI_PW_MODEL_DEVICE:-cuda}" = "cuda" ]; then
    export MPPI_PW_MODEL_DEVICE="$(detect_visible_cuda_devices)"
  fi
  export MPPI_PW_ROBOT_SAMPLER_DEVICE="${MPPI_PW_ROBOT_SAMPLER_DEVICE:-${MPPI_PW_MODEL_DEVICE}}"
  if [ -z "${MPPI_INFER_BUDGET_MS_WAS_SET}" ]; then
    export MPPI_INFER_BUDGET_MS=1500
  fi
  export MPPI_PW_EVAL_BATCH_SIZE="${MPPI_PW_EVAL_BATCH_SIZE:-32}"
  if [ ! -f "${REPO_ROOT}/third_party/dinov3/hubconf.py" ]; then
    echo "Missing ${REPO_ROOT}/third_party/dinov3. Prepare the DINOv3 source checkout first." >&2
    exit 1
  fi

  if [ ! -f "${POINTWORLD_ROOT}/third_party/dinov3/hubconf.py" ]; then
    rm -rf "${POINTWORLD_ROOT}/third_party/dinov3"
    ln -s "${REPO_ROOT}/third_party/dinov3" "${POINTWORLD_ROOT}/third_party/dinov3"
  fi

  mkdir -p "${POINTWORLD_ROOT}/third_party/dinov3/checkpoints"
  if [ ! -e "${POINTWORLD_ROOT}/third_party/dinov3/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" ]; then
    ln -sf /home/models/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
      "${POINTWORLD_ROOT}/third_party/dinov3/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
  fi
fi

python3 -m mppi.comm.ws_server_async_pcl \
  --host 0.0.0.0 \
  --port 9011 \
  --open-loop-horizon "${OPEN_LOOP_HORIZON}" \
  --policy "${MPPI_POLICY:-mppi_joint}" \
  --cam-id "${MPPI_CAM_ID:-back}"
