#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-all}"          # server | replay | all
PROFILE="${2:-obs_infl}"    # no_pw(baseline:disable PW cost) | obs_only | obs_infl

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

DEFAULT_POINTWORLD_ROOT="${REPO_ROOT}/../PointWorld"
if [[ ! -d "${DEFAULT_POINTWORLD_ROOT}" && -d /workspace/pointworld ]]; then
  DEFAULT_POINTWORLD_ROOT="/workspace/pointworld"
elif [[ ! -d "${DEFAULT_POINTWORLD_ROOT}" && -d /home/wangning/PointWorld ]]; then
  DEFAULT_POINTWORLD_ROOT="/home/wangning/PointWorld"
elif [[ ! -d "${DEFAULT_POINTWORLD_ROOT}" && -d /home/wangyuhan/PointWorld ]]; then
  DEFAULT_POINTWORLD_ROOT="/home/wangyuhan/PointWorld"
fi
POINTWORLD_ROOT="${POINTWORLD_ROOT:-${DEFAULT_POINTWORLD_ROOT}}"
DINOv3_ROOT="${DINOv3_ROOT:-${POINTWORLD_ROOT}/third_party/dinov3}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/co-tracker:${REPO_ROOT}/third_party/curobo:${PYTHONPATH:-}"
if [[ -d "${POINTWORLD_ROOT}" ]]; then
  export PYTHONPATH="${POINTWORLD_ROOT}:${PYTHONPATH}"
fi
if [[ -d "${DINOv3_ROOT}" ]]; then
  export PYTHONPATH="${DINOv3_ROOT}:${PYTHONPATH}"
fi

PORT="${PORT:-9011}"
URL="${URL:-ws://127.0.0.1:${PORT}}"
PRIMARY_CAM_ID="${PRIMARY_CAM_ID:-back}"
SERVER_WAIT_S="${SERVER_WAIT_S:-0}"
SERVER_WAIT_TIMEOUT_S="${SERVER_WAIT_TIMEOUT_S:-180}"
SERVER_WAIT_INTERVAL_S="${SERVER_WAIT_INTERVAL_S:-0.5}"
PW_MULTI_GPU_DEVICES="${MPPI_PW_MULTI_GPU_DEVICES:-}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/pw_acceptance/${PROFILE}}"
ACCEPT_DIR="${MPPI_PW_ACCEPTANCE_DUMP_DIR:-${OUT_ROOT}/server}"
REPORT_JSON="${REPORT_JSON:-${OUT_ROOT}/client_report.json}"

JSON_PATH="${JSON_PATH:-}"
DATA_ROOT="${DATA_ROOT:-/home/datasets/FrankaNav/test}"
EPISODE_DIR="${EPISODE_DIR:-}"
DUAL_VIEW="${DUAL_VIEW:-1}"
START_IDX="${START_IDX:-0}"
MAX_STEPS="${MAX_STEPS:-16}"
SLEEP_S="${SLEEP_S:-0.0}"
GRIPPER="${GRIPPER:-0.0}"
DEPTH_UNIT_SCALE="${DEPTH_UNIT_SCALE:-1.0}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-30.0}"

mkdir -p "${OUT_ROOT}"
rm -rf "${ACCEPT_DIR}"
mkdir -p "${ACCEPT_DIR}"

case "${PROFILE}" in
  no_pw)
    export MPPI_PW_TASK_ABLATION="no_pw"
    ;;
  obs_only)
    export MPPI_PW_TASK_ABLATION="obs_only"
    ;;
  obs_infl)
    export MPPI_PW_TASK_ABLATION="obs_infl"
    ;;
  *)
    echo "Unknown PROFILE=${PROFILE}, expected: no_pw | obs_only | obs_infl" >&2
    exit 2
    ;;
esac

export MPPI_PW_ENABLE="${MPPI_PW_ENABLE:-1}"

if [[ "${PROFILE}" == "no_pw" ]]; then
  export MPPI_USE_POINTWORLD_COST="${MPPI_USE_POINTWORLD_COST:-0}"
  export MPPI_W_POINTWORLD="${MPPI_W_POINTWORLD:-0.0}"
else
  export MPPI_USE_POINTWORLD_COST="${MPPI_USE_POINTWORLD_COST:-1}"
  export MPPI_W_POINTWORLD="${MPPI_W_POINTWORLD:-1.0}"
fi

export MPPI_PW_COST_MODE="${MPPI_PW_COST_MODE:-task_point_goal_l2}"
export MPPI_PW_AABB_CONFIG_PATH="${MPPI_PW_AABB_CONFIG_PATH:-${REPO_ROOT}/configs/pointworld_static_aabbs.json}"
export MPPI_PW_TASK_W_OBS="${MPPI_PW_TASK_W_OBS:-1.0}"
export MPPI_PW_TASK_W_INFL="${MPPI_PW_TASK_W_INFL:-0.5}"
export MPPI_PW_ACCEPTANCE_DUMP_DIR="${ACCEPT_DIR}"

run_server() {
  export MPPI_PCL_VERBOSE="${MPPI_PCL_VERBOSE:-1}"
  export MPPI_PCL_PRINT_EVERY="${MPPI_PCL_PRINT_EVERY:-1}"

  export MPPI_PCL_CAM_INFO_BACK_PATH="${MPPI_PCL_CAM_INFO_BACK_PATH:-${REPO_ROOT}/configs/back_cam_info.yaml}"
  export MPPI_PCL_T_BASE_CAM_BACK_PATH="${MPPI_PCL_T_BASE_CAM_BACK_PATH:-${REPO_ROOT}/configs/T_base_cam_back.yaml}"
  export MPPI_PCL_CAM_INFO_SIDE_PATH="${MPPI_PCL_CAM_INFO_SIDE_PATH:-${REPO_ROOT}/configs/side_cam_info.yaml}"
  export MPPI_PCL_T_BASE_CAM_SIDE_PATH="${MPPI_PCL_T_BASE_CAM_SIDE_PATH:-${REPO_ROOT}/configs/T_base_cam_side.yaml}"

  export MPPI_PCL_DEPTH_UNIT_SCALE="${MPPI_PCL_DEPTH_UNIT_SCALE:-1.0}"
  export MPPI_PCL_DEPTH_MIN_M="${MPPI_PCL_DEPTH_MIN_M:-0.05}"
  export MPPI_PCL_DEPTH_MAX_M="${MPPI_PCL_DEPTH_MAX_M:-4.0}"
  export MPPI_PCL_STRIDE="${MPPI_PCL_STRIDE:-1}"

  export MPPI_POLICY="${MPPI_POLICY:-mppi_joint}"
  export MPPI_OPEN_LOOP_HORIZON="${MPPI_OPEN_LOOP_HORIZON:-11}"

  export MPPI_PW_MODEL_PATH="${MPPI_PW_MODEL_PATH:-/home/models/PointWorld/PointWorld_models/large-droid/model-best.pt}"
  export MPPI_PW_COTRACKER_CKPT="${MPPI_PW_COTRACKER_CKPT:-/home/models/Co-tracker/scaled_online.pth}"
  export MPPI_PW_MODEL_DEVICE="${MPPI_PW_MODEL_DEVICE:-cuda:0}"
  export MPPI_PW_COTRACKER_DEVICE="${MPPI_PW_COTRACKER_DEVICE:-cuda:0}"
  export MPPI_PW_ROBOT_SAMPLER_DEVICE="${MPPI_PW_ROBOT_SAMPLER_DEVICE:-${MPPI_PW_MODEL_DEVICE}}"
  export MPPI_PW_DIST2ROBOT_MODE="${MPPI_PW_DIST2ROBOT_MODE:-t0_repeat}"
  if [[ -n "${PW_MULTI_GPU_DEVICES}" ]]; then
    export MPPI_PW_MODEL_DEVICE="${PW_MULTI_GPU_DEVICES}"
    export MPPI_PW_COTRACKER_DEVICE="${PW_MULTI_GPU_DEVICES}"
    export MPPI_PW_ROBOT_SAMPLER_DEVICE="${PW_MULTI_GPU_DEVICES}"
  fi
  export MPPI_PW_MODEL_DOMAIN="${MPPI_PW_MODEL_DOMAIN:-droid}"

  export MPPI_URDF_PATH="${MPPI_URDF_PATH:-${POINTWORLD_ROOT}/assets/franka_description/franka_panda_robotiq_2f85.urdf}"
  export MPPI_PW_URDF_PATH="${MPPI_PW_URDF_PATH:-${MPPI_URDF_PATH}}"

  if [[ ! -f "${DINOv3_ROOT}/hubconf.py" ]]; then
    echo "Missing DINOv3 hubconf.py at ${DINOv3_ROOT}. Expected: /home/wangyuhan/PointWorld/third_party/dinov3" >&2
    exit 1
  fi

  if [[ -f /home/models/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth ]]; then
    mkdir -p "${DINOv3_ROOT}/checkpoints"
    ln -sf /home/models/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
      "${DINOv3_ROOT}/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" || true
  fi

  echo "[acceptance] pw_model_device=${MPPI_PW_MODEL_DEVICE}"
  echo "[acceptance] pw_cotracker_device=${MPPI_PW_COTRACKER_DEVICE}"
  echo "[acceptance] pw_robot_sampler_device=${MPPI_PW_ROBOT_SAMPLER_DEVICE}"
  echo "[acceptance] pw_dist2robot_mode=${MPPI_PW_DIST2ROBOT_MODE}"

  python3 -u -m mppi.comm.ws_server_async_pcl \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --open-loop-horizon "${MPPI_OPEN_LOOP_HORIZON}" \
    --policy "${MPPI_POLICY}" \
    --cam-id "${PRIMARY_CAM_ID}"
}

_wait_for_tcp_port() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  local interval_s="$4"

  python3 - "$host" "$port" "$timeout_s" "$interval_s" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout_s = float(sys.argv[3])
interval_s = float(sys.argv[4])

deadline = time.time() + timeout_s
last_err = None
while time.time() < deadline:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect((host, port))
        s.close()
        print(f"[acceptance] server_ready host={host} port={port}")
        raise SystemExit(0)
    except Exception as e:
        last_err = e
    finally:
        try:
            s.close()
        except Exception:
            pass
    time.sleep(interval_s)

print(f"[acceptance] server_not_ready host={host} port={port} timeout_s={timeout_s} last_err={last_err}")
raise SystemExit(1)
PY
}

run_replay() {
  local -a cmd
  cmd=(
    python3 "${REPO_ROOT}/tests/pw_replay_acceptance.py"
    --url "${URL}"
    --primary-cam-id "${PRIMARY_CAM_ID}"
    --start-idx "${START_IDX}"
    --max-steps "${MAX_STEPS}"
    --sleep-s "${SLEEP_S}"
    --gripper "${GRIPPER}"
    --depth-unit-scale "${DEPTH_UNIT_SCALE}"
    --request-timeout-s "${REQUEST_TIMEOUT_S}"
    --report-json "${REPORT_JSON}"
  )

  if [[ -n "${EPISODE_DIR}" ]]; then
    cmd+=(--episode-dir "${EPISODE_DIR}")
  elif [[ -n "${JSON_PATH}" ]]; then
    cmd+=(--json "${JSON_PATH}" --data-root "${DATA_ROOT}")
  else
    echo "Set either EPISODE_DIR or JSON_PATH before replay." >&2
    exit 2
  fi

  if [[ "${DUAL_VIEW}" == "1" ]]; then
    cmd+=(--dual-view)
  fi

  "${cmd[@]}"
}

summarize_acceptance() {
  python3 - "${ACCEPT_DIR}" "${PROFILE}" "${REPORT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

accept_dir = Path(sys.argv[1])
profile = sys.argv[2]
report_json = Path(sys.argv[3])

files = sorted(accept_dir.glob("*.json"))
if not files:
    print("FAIL: no server acceptance json generated")
    raise SystemExit(1)

rows = [json.loads(p.read_text(encoding="utf-8")) for p in files]

checks = [
    ("scene_flows", all(bool(r.get("has_scene_flows")) for r in rows)),
    ("scene_visibility", all(bool(r.get("has_scene_visibility")) for r in rows)),
    ("scene_depth_valid_mask", all(bool(r.get("has_scene_depth_valid_mask")) for r in rows)),
    ("runtime_policy", all(bool(r.get("has_runtime_policy")) for r in rows)),
]

if profile != "no_pw":
    checks.append(("task_n_obs", all(bool(r.get("has_task_n_obs")) for r in rows)))
if profile == "obs_infl":
    checks.append(("task_n_infl", all(bool(r.get("has_task_n_infl")) for r in rows)))

print("")
print("== Acceptance Summary ==")
print("server_rows:", len(rows))
for name, ok in checks:
    print(f"{name}: {'PASS' if ok else 'FAIL'}")

if report_json.is_file():
    rep = json.loads(report_json.read_text(encoding="utf-8"))
    infer = rep.get("infer_ms", {})
    print("client_steps:", rep.get("steps"))
    print("client_infer_ms_mean:", infer.get("mean"))
    print("client_infer_ms_p95:", infer.get("p95"))
    print("client_unique_policy_count:", rep.get("unique_policy_count"))
else:
    print("client_report: MISSING")

failed = [name for name, ok in checks if not ok]
if failed:
    print("FINAL: FAIL", ",".join(failed))
    raise SystemExit(1)

print("FINAL: PASS")
PY
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

case "${ACTION}" in
  server)
    run_server
    ;;
  replay)
    run_replay
    summarize_acceptance
    ;;
  all)
    run_server &
    SERVER_PID=$!

    if [[ "${SERVER_WAIT_S}" != "0" ]]; then
      echo "[acceptance] legacy_sleep_s=${SERVER_WAIT_S}" >&2
      sleep "${SERVER_WAIT_S}"
    fi

    _wait_for_tcp_port "127.0.0.1" "${PORT}" "${SERVER_WAIT_TIMEOUT_S}" "${SERVER_WAIT_INTERVAL_S}"

    run_replay
    summarize_acceptance
    ;;
  *)
    echo "Unknown ACTION=${ACTION}, expected: server | replay | all" >&2
    exit 2
    ;;
esac
