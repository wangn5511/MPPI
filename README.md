# MPPI

本仓库提供一个面向 Franka 机械臂的 MPPI 关节空间推理服务。

当前主链路收敛到 PCL 协议（schema_version=100，ws://<HOST>:9011）：
- client 上行：双视角 back+side 的 RGB/Depth（可压缩）+ 当前关节 `q(7)` + `step_id/t_client_send_ns`
- server 侧：在线解码与反投影得到 base 点云；可选接入 PointWorld window/tracking 产出 `scene_flows` 等观测；用 MPPI 求解并回包 `actions(T,8)`

本 README 只讲两条主线，并给出可直接复制的启动命令：
- 线 A：server 本机“数据回放/验收模式”（离线 episode_dir → 回放 client → `FINAL: PASS/FAIL`）
- 线 B：Franka↔server “通信闭环模式”（先 shadow：只采集+回包校验，不执行；再 execute：安全门控后执行 `actions[0]`）

---

## 1) 快速上手

### 1.1 基础：设置 PYTHONPATH
```bash
export PYTHONPATH=/home/wangyuhan/MPPI/src:$PYTHONPATH
```

### 1.2 线 A：本机回放验收（推荐标准入口）
- 唯一推荐入口：`tests/run_pw_replay_acceptance.sh`
- profile：`no_pw` / `obs_only` / `obs_infl`

原生 episode_dir（双视角 back+side）
```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh all obs_infl
```

验收通过口径：脚本最终输出 `FINAL: PASS`，并在 `${MPPI_PW_ACCEPTANCE_DUMP_DIR}` 下产生 server 摘要 json。

### 1.3 线 B：Franka↔server 通信（先 dummy_hold，再 mppi_joint）

0) 云端/本机启动 server（只测通信稳定性：dummy_hold）
```bash
cd /home/wangyuhan/MPPI

MPPI_PCL_CAM_INFO_BACK_PATH=/home/wangyuhan/MPPI/configs/back_cam_info.yaml \
MPPI_PCL_T_BASE_CAM_BACK_PATH=/home/wangyuhan/MPPI/configs/T_base_cam_back.yaml \
MPPI_PCL_CAM_INFO_SIDE_PATH=/home/wangyuhan/MPPI/configs/side_cam_info.yaml \
MPPI_PCL_T_BASE_CAM_SIDE_PATH=/home/wangyuhan/MPPI/configs/T_base_cam_side.yaml \
MPPI_PCL_VERBOSE=1 MPPI_PCL_PRINT_EVERY=10 MPPI_PCL_HEARTBEAT_S=10.0 \
MPPI_PW_ENABLE=0 MPPI_USE_POINTWORLD_COST=0 \
PYTHONPATH=/home/wangyuhan/MPPI/src \
python3 -m mppi.comm.ws_server_async_pcl \
  --host 0.0.0.0 \
  --port 9011 \
  --open-loop-horizon 8 \
  --policy dummy_hold \
  --cam-id back
```

1) Franka 侧先做单次 smoke（只验证协议/回包；必须双视角）
```bash
PYTHONPATH=/home/wangyuhan/MPPI/src \
python3 -m mppi.comm.ws_client_sync_pcl \
  --url ws://<CLOUD_IP>:9011 \
  --rgb-back /tmp/back.jpg \
  --depth-back /tmp/back_depth.npy \
  --rgb-side /tmp/side.jpg \
  --depth-side /tmp/side_depth.npy \
  --depth-unit-scale 1.0 \
  --step-id 0 \
  --request-timeout-s 10 \
  --print-actions
```

2) 通信稳定后再切到 mppi_joint（并按需打开 cuRobo / PointWorld）
- 只开 mppi_joint：`--policy mppi_joint`
- 打开 PointWorld 时：必须 `--open-loop-horizon 11`，且 client timeout 建议 ≥120s（PointWorld 很慢）

---

## 2. 目录结构（建议先看表）

| 目录 | 作用 | 你会直接用到的入口 |
|---|---|---|
| `src/` | 核心 Python 包 `mppi`（通信/推理/PointWorld/curobo） | `python3 -m mppi.comm.ws_server_async_pcl`、`python3 -m mppi.comm.ws_client_sync_pcl` |
| `tests/` | 标准回放验收与测试记录（收口 PASS/FAIL） | `tests/run_pw_replay_acceptance.sh`、`tests/pw_replay_acceptance.py` |
| `scripts/` | 辅助脚本（启动/复现/调试） | `scripts/test_cuRobo_pcl.sh` |
| `configs/` | 相机内参/外参、PointWorld AABB 等配置 | `configs/*_cam_info.yaml`、`configs/T_base_cam_*.yaml`、`configs/pointworld_static_aabbs.json` |
| `data/` | 示例数据与验收落盘输出 | `data/pw_acceptance/<profile>/...` |
| `third_party/` | 外部依赖（例如 co-tracker） | 按需安装/引用 |

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