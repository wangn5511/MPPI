# cuRobo MPC 框架搭建方案（对齐 PointWorld MPC）—规划文档

目标：在“端侧无 NVIDIA 驱动/CUDA（仅 CPU，负责连接 Franka） + 近端同网段低时延 GPU 服务器（负责全部重计算）”的部署形态下，用 cuRobo 构建可实时运行的 MPC/MPPI 框架，并预留 PointWorld rollout + cost 的对齐路径。

部署原则：

- 端侧只做观测采集与执行；不做任何 GPU 计算
- 远程 GPU 服务器统一承载：MPPI（采样与更新）+ cuRobo（IK/碰撞/约束）+ PointWorld（rollout）
- WebSocket 仅负责传输：观测数据 -> 动作块

两套采样策略用于对比：

- 版本 A：关节空间采样（Joint-space MPPI）
- 版本 B：EE Pose 空间采样 + IK（SE(3) MPPI + IK，对齐 PointWorld 论文描述）

通信框架复用 openpi 的 websocket client/server：

- client（端侧）周期性采集观测并请求动作块，按固定频率执行（见 `Franka_pi0_client/franka_openpi_client.py`）
- server（GPU 服务器）收到观测后运行 MPPI+PointWorld+cuRobo 并返回动作块（见 `openpi_client/websocket_policy_server.py`）

## 1. 论文对齐目标（PointWorld MPC 关键点）

PointWorld 论文中 MPC 的核心是：

- 采样式 MPC：使用 MPPI 做滚动规划（sampling-based MPC / MPPI）
- 决策变量：规划长度为 T 的末端执行器 SE(3) 位姿目标序列
- 代价函数：定义在世界模型的“状态空间”（点云/点流表征）上
- 世界模型 rollout：PointWorld 以 chunked multi-step 方式预测未来 H=10 步，且每步 0.1s（10Hz 预测粒度）

我们要实现的“对齐功能集合”：

- MPPI 求解器（可配置采样数 K、温度 λ、噪声协方差、规划步长等）
- 两种动作参数化（关节 / EE pose）
- cuRobo 提供的 IK/可行性约束/碰撞（用于过滤或惩罚不可行采样）
- 与 openpi websocket 协议兼容：输入 obs → 输出 action\_chunk（协议在本项目内实现，便于独立运行）
- 预留 PointWorld rollout 与 cost 接口（第一阶段可先用几何/目标距离的 cost 做 baseline）

## 2. MPPI 项目目录结构（以本工作空间为基准）

项目根目录：`/home/wangyuhan/MPPI/`（当前仅有 `plan.md`）。

外部依赖位置（不纳入本项目目录树）：

- PointWorld：`/home/wangyuhan/PointWorld/`（第二阶段 world model rollout + 点云代价）
- cuRobo：GPU 环境中安装（IK/碰撞/约束）

完整目录结构（建议）：

- `MPPI/`
  - `plan.md`
  - `pyproject.toml`
  - `requirements/`
    - `cpu.txt`
    - `gpu.txt`
  - `configs/`（运行期配置，YAML）
    - `websocket.yaml`（host/port、超时、chunk 长度等）
    - `robot_franka.yaml`（关节限位、速度/加速度上限、默认 home）
    - `mppi_joint.yaml`（版本A：关节空间 MPPI 参数）
    - `mppi_se3_ik.yaml`（版本B：SE(3)+IK MPPI 参数）
    - `pointworld.yaml`（PointWorld checkpoint 路径、rollout 配置）
  - `src/`
    - `mppi/`
      - `__init__.py`
      - `cli.py`（统一命令行入口：server/client/benchmark）
      - `protocol/`（消息与序列化，保证端侧/服务端一致）
        - `types.py`（Obs/ActionChunk/Timing 的 dataclass 或 TypedDict）
        - `msgpack_codec.py`（msgpack + numpy 编解码）
      - `comm/`（通信层：WebSocket）
        - `ws_client_sync.py`（端侧同步 client：obs -> action\_chunk）
        - `ws_server_async.py`（GPU 侧 asyncio server：infer -> action\_chunk）
      - `mpc/`（MPPI 核心）
        - `solver.py`（MPPI 主循环：采样/加权/更新）
        - `sampler.py`（噪声采样与控制参数化：joint / SE(3)）
        - `rollout_base.py`（rollout 抽象：FK/约束/PointWorld rollout 钩子）
        - `time_budget.py`（infer\_ms 预算与降级策略）
      - `costs/`（代价项）
        - `base.py`
        - `smoothness.py`
        - `ee_pose.py`（baseline：FK 末端误差）
        - `pointworld_cost.py`（第二阶段：点云/点流代价）
      - `robots/`（机器人模型与通用 FK）
        - `franka_kinematics.py`
        - `limits.py`
      - `curobo_ext/`（对 cuRobo 的薄封装）
        - `ik.py`
        - `collision_checker.py`
        - `feasibility.py`
      - `pointworld_ext/`（对 PointWorld 的薄封装）
        - `wrapper.py`（加载 checkpoint、batch rollout）
        - `flows.py`（从关节/EE 轨迹生成 robot point flows）
      - `utils/`
        - `timing.py`
        - `se3.py`
  - `scripts/`（一键启动脚本）
    - `run_server_gpu.sh`
    - `run_client_cpu.sh`
    - `bench_latency.sh`
  - `data/`（运行产物，不要求入库）
    - `logs/.gitkeep`
    - `checkpoints/.gitkeep`
    - `scenes/.gitkeep`
  - `tests/`
    - `test_protocol.py`
    - `test_mppi_shapes.py`
    - `test_time_budget.py`

约定：server 侧无论版本 A/B，都输出 `actions: [N, 8]`（7 关节 + 夹爪），以便端侧统一执行。

## 3. 部署拓扑与运行方式

### 3.1 端侧（无 NVIDIA 驱动/CUDA，连接 Franka）

- 运行：`scripts/run_client_cpu.sh`（或 `python -m mppi.comm.ws_client_sync --config configs/websocket.yaml`）
- 职责：采集观测（图像/关节/夹爪）并发送到 websocket；接收动作块并按 `control.frequency` 执行
- 端侧不安装 CUDA/torch-cuda/curobo/PointWorld，仅保留运行 client 必需依赖
- 每执行 `open_loop_horizon` 步重新请求一次动作块（当前配置下，server 调用频率约为 `frequency/open_loop_horizon`）

### 3.2 GPU 服务器（同网段低时延，承载全部 MPC 计算）

- 运行：`scripts/run_server_gpu.sh`（或 `python -m mppi.comm.ws_server_async --config configs/mppi_joint.yaml`）
- 职责：接收观测 -> 运行 MPPI（采样/更新）+ cuRobo（IK/碰撞/约束）+ PointWorld（rollout，可选阶段）-> 输出动作块
- 对外暴露 websocket 端口（例如 9010），与端侧 `remote_server.host/port` 对应

### 3.3 可行性与实时性约束（WebSocket 仅传输观测/动作）

该方案在“同网段低时延且稳定”的前提下可行，关键约束是让端侧不会因同步 `infer()` 阻塞而失频：

- 端侧控制频率由本地执行循环保障；server 每次返回一个动作块，端侧开环执行该 chunk
- `open_loop_horizon` 越大，server 调用频率越低，对网络抖动越鲁棒；但反馈变弱
- server 端需要保证 `infer_ms` 有上界（可通过降低 K/T、简化 cost、批处理 rollout 实现）

建议迭代节奏：

- 第 1 阶段：GPU server 只做 cuRobo MPPI baseline（不接 PointWorld），先验证闭环稳定与时延上界
- 第 2 阶段：引入 PointWorld rollout + cost，仍沿用同一 MPPI 框架，对齐论文表述

## 4. 依赖与环境

### 4.1 端侧（仅 CPU，连接 Franka）

- Linux x86\_64
- Python（与现有 openpi client 保持一致）
- franky（或你当前 Franka 控制栈）+ 相机/夹爪依赖
- websockets（同步 client）+ msgpack / msgpack\_numpy
- 不要求 NVIDIA 驱动、CUDA、torch-cuda

### 4.2 GPU 服务器（承载 MPPI+PointWorld+cuRobo）

- Linux x86\_64
- NVIDIA 驱动 + CUDA（版本按 cuRobo / PointWorld 所需）
- Python（建议 3.10 或 3.11，按 cuRobo / PointWorld 建议）
- PyTorch（带 CUDA 支持）
- websockets（asyncio server）+ msgpack / msgpack\_numpy

### 4.3 cuRobo 依赖（GPU 服务器）

- cuRobo（建议锁定版本）
- 可能依赖：warp、triton、torch 版本匹配（以 cuRobo 文档/发布版本为准）
  参考：
- cuRobo 文档：<https://nvlabs.github.io/curobo/latest/>
- cuRobo 仓库：<https://github.com/NVlabs/curobo>

### 4.4 PointWorld（GPU 服务器，第二阶段接入）

- PointWorld 仓库（模型推理 + checkpoint）
- 推理侧需要：batch rollout（一次评估 K 条轨迹）
  参考：
- PointWorld 仓库：<https://github.com/NVlabs/PointWorld>
- 论文：<https://arxiv.org/abs/2601.03782>

## 5. MPC “功能清单”与对齐里程碑

### 5.1 必做（MVP，可闭环跑通）

- WebSocket server（GPU 服务器）可接收端侧 obs，并返回 action chunk（N x 8）
- 端侧 client 不依赖 CUDA：仅负责“采集观测 + 执行动作块”
- 版本A（关节 MPPI）可输出平滑的关节目标序列（优先用于打通闭环）
- cuRobo 可行性/约束支持（在 GPU 服务器侧执行）：
  - 关节限位、速度/加速度正则
  - 自碰/场景碰撞（至少自碰先跑通）
- 真实场景 → cuRobo scene\_model（先单 back 视角跑通，后续再加 wrist 融合；贴合 PointWorld 的“单帧/少帧 RGB-D capture”假设）：
  - 目标：从观测深度构建保守的环境几何，用于 cuRobo 环境碰撞惩罚；不做重建/建图，避免破坏实时性
  - 坐标系约定（全部在 panda 基坐标系下）：
    - base：panda\_link0
    - wrist EE：panda\_link7
    - 背景相机：cam\_back
    - 腕部相机：cam\_wrist
  - 外参来源（YAML 的 frame\_id → child\_frame\_id，T 为 4x4 row-major）：
    - base→cam\_back：configs/T\_base\_cam.yaml（panda\_link0 → cam\_back）
    - panda\_link7→cam\_wrist：configs/T\_ee\_wrist.yaml（panda\_link7 → cam\_wrist）
  - 变换链路（方案B0：只用背景相机）：
    - 对背景相机点：p\_base = T\_base\_cam\_back · p\_cam\_back
  - 深度→点云（方案B0）：
    - 输入：深度 D(u,v)（米）与内参 fx, fy, cx, cy
    - back-project：x=(u-cx)/fx*z, y=(v-cy)/fy*z, z=D(u,v)
    - 丢弃无效/超范围深度（min/max depth）
  - 点云清洗（方案B0）：
    - ROI 裁剪：限定到工作空间盒子（例如 1m×1m×1m，或按任务自定义）
    - 体素下采样：voxel size 建议 5–10mm（权衡稳定性与细节）
    - 机器人 mask：用 cuRobo robot spheres（当前 q\_t）剔除机器人附近点，避免把机器人当作障碍
  - 点云→scene\_model（先用 primitives，避免在线 ESDF 复杂度）：
    - 体素占据/聚类 → 连通域/聚类 → 每簇 AABB/OBB → cuboids
    - 对 cuboid 做 padding（例如 1–2cm）以抵抗标定/深度噪声
    - 控制数量：只保留 top-N（例如 10–30 个）或仅保留末端附近的障碍
  - 更新策略（低频、保守、可复现）：
    - 建议 1–2Hz 或每个 action\_chunk 更新一次；不要每个 MPPI tick 都重建
    - 新增快、删除慢：避免点云抖动导致障碍闪烁
    - 对障碍位姿做 EMA 平滑，降低碰撞代价的高频噪声
  - 后续增强（加入 wrist 视角以减少遮挡盲区）：
    - 对腕部相机点：p\_base = T\_base\_link7(q\_t) · T\_link7\_cam\_wrist · p\_cam\_wrist
    - 其中 T\_base\_link7(q\_t) 来自当前关节 q\_t 的 FK（URDF：/home/wangyuhan/PointWorld/assets/franka\_description/franka\_panda\_robotiq\_2f85.urdf）
    - 融合策略：先各自 back-project + 变换到 base，再拼接后统一 ROI/下采样/mask
  - 对齐检查（上线前必须完成）：
    - 双视角融合一致性：同一桌面边缘/静态物体在 base 坐标系下应重合（误差在厘米级内）
    - 深度尺度检查：桌面高度/已知距离物体的 z 值应符合实际
    - URDF 一致性检查：端侧与 server/cuRobo 使用同一 URDF 文件路径与内容（hash 校验）
- 代价函数 baseline（先不接 PointWorld）：
  - 关节平滑/能量正则
  - 与目标 EE pose 的误差（用 FK 计算，占位验证 MPC 行为）
- 统计与监控：
  - server\_timing：infer\_ms、采样数、有效样本比例
  - 端到端时延：端侧发送→接收→执行的时间戳；推理超时与回退策略统计

### 5.2 对齐 PointWorld MPC（论文功能）

- 版本B（EE pose MPPI + IK）：
  - 决策变量为 SE(3) 轨迹（T 段）
  - IK 将 SE(3) 轨迹投影成关节轨迹（不可达时强惩罚或重采样）
- PointWorld rollout + cost（第二阶段）：
  - 从 obs（RGB-D）构建 scene points
  - 从候选动作构建 robot point flows
  - PointWorld batch rollout 预测未来点云流
  - cost 在点云/点流上定义（任务相关的 moved-points 代价、目标点到达、接触/避障等）

## 6. 两套采样策略设计（用于对比）

共同点（两版都用 MPPI 框架）：

- 规划长度：T（可与端侧 open\_loop\_horizon 对齐，也可用更短 T 做高频重规划）
- 采样条数：K（例如 256/512/1024）
- MPPI 温度：λ
- 噪声：高斯噪声，支持时间相关/独立
- 输出：action\_chunk（关节目标序列 + 夹爪）

### 6.1 版本 A：关节空间采样（Joint-space MPPI）

适用原因：你端侧能直接获取关节状态，并且端侧执行接口天然是关节目标。

- 状态（来自端侧 obs）：
  - q\_t：7 关节角
- 控制变量 u\_t：
  - 方案 A1：采样“关节增量” Δq（更稳定）
- rollout：
  - 用 cuRobo 的轨迹结构 + 限幅（推荐）
- 代价函数（baseline）：
  - 平滑：∑ ||Δq\_k||^2 或 ||q\_{k+1}-q\_k||^2
  - 速度/加速度约束：软惩罚
  - 碰撞：自碰/环境碰撞惩罚（curobo collision checker）
  - 任务项（第一阶段）：EE pose 误差（通过 FK 计算）
  - 任务项（第二阶段）：PointWorld 点云 cost（用该关节序列生成 robot flows）
- 优缺点：
  - 优点：实现快、接口一致、稳定性通常更好（无 IK 失败分支）
  - 缺点：与 PointWorld 论文“直接规划 SE(3) 目标序列”不完全一致；在任务代价上需要从关节轨迹推导 robot flows

对比指标重点：

- 端到端时延稳定性
- 关节抖动/平滑性
- IK 不存在，因此成功率受可行性约束影响较小

### 6.2 版本 B：EE Pose 采样 + IK（SE(3) MPPI + IK）

目标：贴合 PointWorld 论文：MPPI 直接规划 SE(3) 末端位姿目标序列。

- 状态：
  - 当前 EE pose（由端侧 q\_t 经 FK 或端侧直接给出）
  - 当前关节 q\_t（用于 IK seed、关节连续性）
- 控制变量 u\_t（SE(3)）：
  - 方案 B1：采样 EE pose 增量（dx,dy,dz + droll,dpitch,dyaw）
  - 方案 B2：采样绝对 EE pose target
- IK 投影（关键差异点）：
  - 对每条采样轨迹：将 EE pose 序列逐步 IK → 得到关节序列
  - IK 失败处理：
    - 失败步开始后整条轨迹加大惩罚
    - 或对失败步做 fallback（例如保持上一步关节）
- rollout：
  - 在关节空间执行可行性检查、碰撞、平滑正则
  - 同时保留 EE pose 序列用于任务 cost（更直观）
- 代价函数（baseline 与 PointWorld 对齐）：
  - baseline：EE pose 目标误差、碰撞、平滑
  - PointWorld 对齐：使用 EE pose 序列生成 robot point flows，调用 PointWorld rollout，在点云上计算任务 cost
- 优缺点：
  - 优点：与论文动作表征一致，更容易直接复用 PointWorld 的“robot point flows from URDF+FK”的思路
  - 缺点：IK 失败/多解/数值不稳会显著影响 MPPI 的有效样本比例，工程复杂度更高

对比指标重点：

- 有效样本比例（IK 成功 + 无碰撞）
- 任务完成率（尤其是需要精细接触的任务）
- 计算开销（IK 成本可能成为瓶颈）

## 7. 实验对比设计（建议）

至少做三组对比，保证结论可信：

- A（Joint MPPI） vs B（EE+IK MPPI），在相同：
  - T、K、λ、控制频率、端侧 open\_loop\_horizon
  - 碰撞开关、平滑权重、限速参数
- 指标：
  - 控制质量：轨迹平滑、最大关节跳变、末端抖动
  - 安全性：碰撞次数/最小距离、关节限位触发次数
  - 性能：server infer\_ms 的均值/方差、端到端延迟、丢包/超时
  - 成功率：任务成功率、失败模式统计（IK fail / collision / tracking fail）
- 记录：
  - server 返回 `server_timing`（你当前框架已支持在 server 侧附加 timing 字段）
  - 端侧记录 send/receive 时间戳（你当前 client 已有 chunk 级别时间戳记录逻辑）

## 8. 与现有 openpi 接口的兼容性约束

端侧当前执行逻辑假定：

- response\["actions"] 是一个二维数组：`[N, 8]`
- 每行：`[q1..q7, gripper]`
- N 至少为 1，通常为 open\_loop\_horizon

因此无论 A/B 版本，server 都需要输出关节目标序列：

- 版本A：直接输出采样后的关节目标序列
- 版本B：EE pose 采样后经 IK 得到关节序列，再输出

## 9. 后续实现顺序（建议）

1. 在 GPU 服务器上跑通 `WebsocketPolicyServer + dummy policy`（固定姿态输出），验证端到端通信与频率稳定
2. 实现版本A（Joint MPPI）baseline（无 PointWorld），先只做平滑 + 关节限位 + 目标 EE pose（FK）
3. 引入 cuRobo 自碰/环境碰撞 cost，观察稳定性与实时性
4. 实现版本B（EE pose MPPI + IK），先在无 PointWorld cost 下完成对比
5. 接入 PointWorld rollout + cost（batch 推理），完成论文对齐版本

