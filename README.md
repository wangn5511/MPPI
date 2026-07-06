# MPPI

本仓库提供一个面向 Franka 机械臂的 MPPI（Model Predictive Path Integral）关节空间推理服务。当前主链路收敛为：

- PCL 链路（schema_version=100，端口 9011）：客户端发送 RGB+Depth（可压缩）+ 相机标识/参数；服务器端在线解码、反投影生成 base 点云，并接入 PointWorld 在线 window/tracking 以产出 `scene_flows` 等观测字段。

本 README 的目标是提供一条可复制的“本地回放验收”标准入口，并指出关键文件位置与清理建议。

---

## 1. 快速上手（PCL + PointWorld：本地回放验收）

### 1.1 基础：设置 PYTHONPATH
```bash
export PYTHONPATH=/home/wangyuhan/MPPI/src:$PYTHONPATH
```

### 1.2 一键：起 server + 回放 + 验收（推荐）
- 标准入口：`tests/run_pw_replay_acceptance.sh`
- profile：`no_pw` / `obs_only` / `obs_infl`

#### A) 原生 episode（双视角 back+side）
```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh all obs_infl
```

#### B) data.json（单视角 back）
```bash
JSON_PATH=/home/wangyuhan/MPPI/data/test/data.json \
DATA_ROOT=/home/datasets/FrankaNav/test \
DUAL_VIEW=0 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh all obs_only
```

### 1.3 单次请求（PCL client）
```bash
python3 -m mppi.comm.ws_client_sync_pcl \
  --url ws://127.0.0.1:9011 \
  --rgb <rgb_path> \
  --depth <depth_path> \
  --cam-id back \
  --print-actions
```

关键约束：
- PointWorld window 强制 `open-loop-horizon=11`（验收脚本已固化）
- 静态 AABB 默认：`configs/pointworld_static_aabbs.json`

---

## 2. 目录结构与职责

### 2.1 顶层目录
- `src/`：核心 Python 包（mppi），推理/通信/场景构建均在这里
- `scripts/`：启动/测试/回放脚本（用于复现实验与性能压测）
- `configs/`：相机与外参标定、变换矩阵等配置（YAML）
- `data/`：示例数据（npz/ply/json 等）
- `tests/`：实验记录与少量测试脚本/占位内容
- `third_party/`：外部第三方代码（例如 co-tracker）

### 2.2 `src/mppi/`（核心包）
- `src/mppi/cli.py`
  - 统一入口：`mppi server` / `mppi client` / `mppi curobo-smoke`
  - 同事做“通信功能检测”时优先用这里的 server/client 快速验证链路
- `src/mppi/comm/`
  - 通信层（websocket + msgpack）
  - **PCL：**
    - `ws_server_async_pcl.py`：服务端（schema_version=100），接收 RGB/Depth，在线反投影生成点云，并接入 PointWorld window/tracking
    - `ws_client_sync_pcl.py`：客户端（把 RGB 编成 JPEG，把 Depth 编成 npy+zlib 发给服务端）
- `src/mppi/protocol/`
  - `types_pcl.py`：PCL 协议数据结构（ObsPCL/InferRequestPCL/InferResponsePCL/ErrorPCL）
  - `msgpack_codec.py`：msgpack 编解码
- `src/mppi/mpc/solver.py`
  - `JointMPPISolver`：MPPI 核心推理（采样、代价、退化策略、时间预算）
  - 场景构建入口：`build_scene_cuboids_from_pcd_back_cam(...)`
  - cuRobo 碰撞入口：`get_curobo_collision_checker(...).batch_distance(...)` / `collision_penalty(...)`
  - 同事做“curobo 碰撞箱 GPU 加速/验证”主要会改这里与 `curobo_ext/`
- `src/mppi/curobo_ext/`
  - `collision_checker.py`
    - cuRobo 的封装与缓存（按 scene key 缓存 RobotCollisionChecker）
    - `batch_distance()` 会把 q_traj 放到 GPU 并取回距离矩阵
    - `get_robot_spheres_base()` 可生成 robot mask spheres（用于点云去掉机器人本体点）
  - `scene_builder.py`
    - 点云 -> base 坐标 -> ROI crop -> 去桌面/去墙 -> voxel downsample -> robot mask
    - 点云聚类得到 AABB，再转成 cuboids（给 cuRobo 当 world obstacles）
  - `check_depth_pcl.py`
    - PCL 链路的 RGBD->点云：`rgbd_to_pointcloud_base(...)`
    - 解析 intrinsics / T_base_cam（来自 ObsPCL 或环境变量）
  - `check_depth.py`
    - 读入/预处理 depth/rgb 的工具函数（PCL client 与调试脚本会用）
- `src/mppi/utils/`
  - `pointcloud.py`：纯 numpy 点云工具（voxel、AABB、聚类、mask 等）；在线点云构建的 CPU 热点很可能在这里
  - `se3.py`：SE(3) 相关工具（坐标变换）
- `src/mppi/robots/`
  - `franka_kinematics.py`：Franka 正运动学（代价项 ee/link7 位置等）
- `src/mppi/costs/`
  - `ee_pose.py`：末端/指定 link 的位置代价（MPPI cost term）

### 2.3 `configs/`（相机/外参）
- `configs/back_cam_info.yaml`：相机内参（PCL server 用来解析 intrinsics）
- `configs/T_base_cam.yaml`：base->camera 外参 4x4（PCL server 用来把点云变到 base）
- `configs/T_ee_wrist.yaml`、`configs/wrist_can_info.yaml`：其他标定信息（按实际 pipeline 使用）

### 2.4 `scripts/` / `tests/`（用于复现与验收）
- 标准验收入口：`tests/run_pw_replay_acceptance.sh`
  - 一键：起 server + 回放 + 验收（`no_pw/obs_only/obs_infl` 三档）
- 回放 client：`tests/pw_replay_acceptance.py`
  - 发送 PCL 请求（支持 data.json 或原生 episode_dir；支持 back 或 back+side）
- PCL server 启动脚本：`scripts/test_cuRobo_pcl.sh`
  - 负责把 PCL + cuRobo + PointWorld 的运行期环境变量固化起来
- 离线 AABB 检测：`scripts/debug_robot_mask.sh`、`scripts/visual_cub.py`
  - 输出点云/聚类结果，并固化到 `configs/pointworld_static_aabbs.json`

### 2.5 Legacy（计划删除的旧入口）
- `scripts/playback_client_pcl.py`
  - 功能已被 `tests/pw_replay_acceptance.py` 覆盖，README 不再使用
- `scripts/run_server_gpu.sh`
  - 仅是 `test_cuRobo_pcl.sh` 的薄封装，README 不再使用
- V1（端口 9010）：`scripts/test_cuRobo.sh`、`scripts/playback_client.py`、`scripts/run_client_cpu.sh`
  - 仅在你仍需要 V1 baseline/通信测试时保留；默认路径已迁移到 PCL+PointWorld 验收

---

## 3. 通信协议（PCL）

### 3.1 PCL（schema_version = 100）
- 请求 envelope：
  - `type = "infer_request_pcl"`
  - payload = `ObsPCL`：支持 `rgb_bytes(jpeg)` + `depth_bytes(npy_zlib)`，以及 `cam_id`/`intrinsics`/`T_base_cam`
- 响应 envelope：
  - `type = "infer_response_pcl"`
- 错误：
  - `type = "error_pcl"`

代码位置：
- `src/mppi/protocol/types_pcl.py`
- server：`src/mppi/comm/ws_server_async_pcl.py`
- client：`src/mppi/comm/ws_client_sync_pcl.py`

---

## 4. 本地回放验收（收口到一个入口）

目标：从“本地 episode / data.json 回放”稳定产出 PointWorld 观测关键字段，并用脚本自动判定 PASS/FAIL。

### 4.1 标准入口
- 入口脚本：`tests/run_pw_replay_acceptance.sh`
- 回放 client：`tests/pw_replay_acceptance.py`

三档 profile（同一启动方式切换）：
- `no_pw`：不使用 task term（用于对照）
- `obs_only`：只启用 `I_obs`
- `obs_infl`：启用 `I_obs + I_infl`

### 4.2 验收项（server 端必须稳定产出）
- `scene_flows`
- `scene_visibility`
- `scene_depth_valid_mask`
- `task_n_obs`
- `task_n_infl`（仅 `obs_infl`）
- `runtime_policy`

### 4.3 输出
- server 摘要：`${MPPI_PW_ACCEPTANCE_DUMP_DIR}/*.json`（由 server 在每个 step 落盘）
- client 汇总：`${REPORT_JSON}`（由回放脚本输出）

验收脚本结束会输出：`FINAL: PASS` 或 `FINAL: FAIL`。

---

## 5. 运行期配置（收口到两个脚本）

- server 启动配置：`scripts/test_cuRobo_pcl.sh`
  - PCL server + cuRobo + PointWorld 的环境变量都在这里
- 回放与验收配置：`tests/run_pw_replay_acceptance.sh`
  - 固化：AABB 配置、三档 profile 切换、验收输出目录

核心约束：
- `open-loop-horizon=11`（PointWorld window 硬性要求；验收脚本默认已设置）
- Workspace ROI 必须对齐 PWM_Data：`[0.0,-0.38,-0.30] ~ [0.8,0.30,1.20]`
- 静态障碍物 AABB 默认：`configs/pointworld_static_aabbs.json`

更完整的环境变量说明：见 `protocol-and-parameters.md`。

---

## 6. 清理建议（只保留 PCL + PointWorld）

建议直接删除下列冗余/历史入口，避免仓库入口爆炸：
- V1 链路（端口 9010）相关脚本与代码：
  - `scripts/test_cuRobo.sh`
  - `scripts/run_client_cpu.sh`
  - `scripts/playback_client.py`
  - `src/mppi/comm/ws_server_async.py`
  - `src/mppi/comm/ws_client_sync.py`
  - `src/mppi/protocol/types.py`
- PCL 老回放脚本（已被验收脚本覆盖）：
  - `scripts/playback_client_pcl.py`
- 冗余 wrapper：
  - `scripts/run_server_gpu.sh`
- 历史待办：
  - `tests/real_todo.md`