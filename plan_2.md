# Joint-space MPPI → PointWorld 接入待办清单（基于当前 MVP）

目标：在当前 `_pcl` MVP（WebSocket 输入 q+RGB+Depth，server 端 RGBD→点云 + cuRobo scene collision + Joint-space MPPI）基础上，把 PointWorld 作为本地 world model 接入到 MPPI 的 cost/rollout 中，实现“用 robot point flow 条件化预测 scene point flow，再据此打分采样轨迹”。

前提约束（已确认）：
- `Depth` 单位为 `float32(m)`（与录制数据一致）
- `/franka/end_effector_pose ≈ panda_link8`（用于对齐校验）
- gripper 在任务中默认闭合不动（可固定 `finger_joint` 为常数）
- PointWorld 本机仓库：`/home/wangyuhan/PointWorld`
- 机器人 URDF：`/home/wangyuhan/PointWorld/assets/franka_description/franka_panda_robotiq_2f85.urdf`

---

## 0. 关键接口与形状契约（必须对齐）

PointWorld `BaseModel.forward(data_dict)` 需要以下键（推理同样需要）：
- `scene_flows`: (B, T, Ns, 3) 其中 t=0 的坐标用于建模（见 pointworld/base.py）
- `scene_features`: (B, T, Ns, Ds) 实际使用 t=0（scene_feature_encoder）
- `scene_exists`: (B, T, Ns) bool
- `robot_flows`: (B, T, Nr, 3)
- `robot_features`: (B, T, Nr, Dr)
- `robot_exists`: (B, T, Nr) bool
- `__domain__`: list[str]，长度为 B（例如 `["droid"]*B`）

参考：
- `scene_coord0 = data_dict["scene_flows"][:,0]`、`robot_coord_seq = data_dict["robot_flows"]`：
  /home/wangyuhan/PointWorld/pointworld/base.py

---

## 1) P0：本地 PointWorld 推理“可调用化”（先跑通一次 forward）

### 1.1 环境与依赖
- GPU server 环境能 `import pointworld`
- 配置 checkpoint（DROID/Panda 对应 droid domain），能加载 `BaseModel`

### 1.2 最小输入字典跑通
- 构造全零但 shape 正确的 `data_dict`，确保 `model(data_dict, training=False)` 可返回：
  - `outputs["scene_flows"]` (B,T,Ns,3)
  - `outputs["confidence"]` 等

为什么：
- 先验证 “模型加载 + data_dict 契约” 没问题，避免后续把时间花在排查 import/checkpoint/shape/domain 上。

---

## 2) P1：Robot Flow 生成（Joint-space MPPI 的核心桥梁）

Joint-space MPPI 的采样/rollout给的是未来关节轨迹 `q_{t:t+T}`，但 PointWorld 需要 robot point flow（机器人几何点在 3D 中的轨迹）。

### 2.1 固定 URDF 与 tool frame 语义
- URDF：`franka_panda_robotiq_2f85.urdf`
- 注意末端挂载链：
  - `panda_link7 -> panda_link8` 固定关节 `xyz=0,0,0.107`
  - `panda_link8 -> camera_mount_link -> robotiq_85_base_link` 固定关节链（含 rpy 旋转）
- gripper 固定闭合：`finger_joint` 固定常数，不需要从数据推开合。

为什么：
- 之前 robot mask/可视化出现系统性错位的根因就是 link7/link8/tool_frame 语义不一致；robot flow 同样必须严格对齐 frame，否则 PointWorld 条件输入失真。

### 2.2 复用 PointWorld 官方 robot sampler（推荐，不要自写）
优先直接复用 PointWorld 仓库的：
- `RobotSampler`（GPU 加速、支持 robotiq mimic joints）：
  `/home/wangyuhan/PointWorld/robot_sampler.py`
- 数据管线里生成 robot_flows 的逻辑：
  `/home/wangyuhan/PointWorld/dataset_components/robot.py` (`_get_robot_flows_droid`, `gather_features`)

输出目标：
- 对每条候选关节轨迹生成 `robot_flows` shape (B=K, T, Nr, 3)
- 同时生成 `robot_normals` 等可选特征（如果 checkpoint 需要）

为什么：
- PointWorld 训练/评估用的 robot flow 就是这样生成的；复用可最大限度保证分布一致、减少“喂对 shape 但模型无效”的风险。

### 2.3 对齐自检（强烈建议先离线验证）
用你录制数据的同一帧：
- `q` 做 FK 求 `T_base_link8_fk`
- 与 `/franka/end_effector_pose ≈ T_base_link8_meas` 对比
- 必须消除 link7/link8 0.107m 的系统误差后，才进入下一步。

---

## 3) P2：Scene State 构建（从 MVP 的 RGBD 输出一个 PointWorld 分支）

你现在已有 RGBD→点云 + ROI/voxel/robot mask/table/wall 的 cuRobo 场景构建链路。接 PointWorld 时建议：
- 保持“给 cuRobo 的 cuboids”链路不变（负责碰撞约束）
- 为 PointWorld 单独输出一个 scene points 表示（不要压成 cuboids）

### 3.1 scene_flows（输入）
- `scene_flows[:,0]` 放当前帧 scene 点坐标（base frame）
- 其余时间步可重复 t=0（推理阶段可简化），保证 shape (B,T,Ns,3)

### 3.2 scene_features（输入）
PointWorld 默认 scene_features 由以下项拼接（见 dataset_components/robot.py:gather_features）：
- `scene_flows`（本身也作为 feature 之一）
- `scene_colors`（RGB 采样，建议归一化到 [0,1]）
- `scene_normals`（建议补 normals；MVP 可先置零但效果可能下降）
- `gripper_open`（你任务固定闭合 -> 全零）
- `dist2robot`（按 PointWorld 逻辑计算：scene 点到 robot 点距离特征）

### 3.3 exists mask
- `scene_exists`: 全 True
- `robot_exists`: 全 True

为什么：
- PointWorld forward 明确依赖这些特征张量；按其数据管线构造特征，才能最大程度对齐训练分布。

---

## 4) P3：把 PointWorld rollout 接到 MPPI cost（让 world model 真正起作用）

### 4.1 每次 infer 的建议结构（一次场景，多条轨迹批处理）
- 从当前帧 RGBD 构建 `scene_state`（一次）
- MPPI 采样产生 K 条候选关节序列 `q_{t:t+T}`（已有）
- 批量生成 `robot_flows(K,T,Nr,3)` 与 `robot_features(K,T,Nr,Dr)`
- 调一次 PointWorld：得到 `pred_scene_traj(K,T,Ns,3)`（输出 `scene_flows`）
- 在预测的 scene 轨迹上定义 task cost（推/拉/避障/到达等）
- 与现有 cost（平滑、限位、cuRobo self/scene collision）加权合并

为什么：
- PointWorld 的价值在“预测未来世界如何动”；只有进入 cost，MPPI 才会利用它。

### 4.2 预算与缓存（必须做，否则 infer_ms 会爆）
- scene 编码（如 2D encoder 特征）每帧只算一次，K 共享
- robot flow 必须 batch 生成
- 当超预算时按顺序降级：
  1) 降 K（采样数）
  2) 降 Ns（scene 点数）
  3) 降 Nr（robot 点数）
  4) 降 T/horizon（PointWorld 预测长度）
  5) 最后才考虑冻结 scene（避免行为闪烁）

---

## 5) 与现有 MVP 的对接点（你需要新增/修改的模块）

### 5.1 MPPI 仓库侧（/home/wangyuhan/MPPI）
- 新增/实现（目前文件为空）：
  - `src/mppi/pointworld_ext/wrapper.py`：封装 PointWorld 模型加载、forward、scene 编码缓存
  - `src/mppi/pointworld_ext/flows.py`：封装 joint trajectories → robot_flows 的批处理接口（尽量复用 PointWorld RobotSampler）
  - `src/mppi/costs/pointworld_cost.py`：把 PointWorld rollout 的输出转成 task cost（先做最简单的 cost 验证链路）
- 修改 solver 的评估路径（Joint-space MPPI rollout/cost 处）：
  - 在每次 infer 内：构建一次 scene_state + batch 生成 robot_flows + PointWorld forward + cost

为什么：
- 你当前 MVP 已具备 websocket、RGBD→点云、cuRobo 碰撞；缺的是 “robot flow 生成 + PointWorld 调用 + cost 注入”。

---

## 6) 建议的落地顺序（最少返工、最快可验证）

1. 离线：用 data.json 的 q + end_effector_pose 做 FK 对齐检查（消除 link7/link8/tool_frame 错位风险）
2. 离线：用 PointWorld RobotSampler 生成一段 robot_flows，确认 shape/可视化合理（gripper 固定）
3. 本地：构造最小 data_dict 跑通 PointWorld forward（先不接 MPPI）
4. 在线：在 server 内加入 PointWorld wrapper，固定一条轨迹测试输出差异
5. 接入 MPPI：用最简单 task cost（验证能区分轨迹）
6. 性能优化：缓存/批处理/降级策略与 Ns/Nr/T/K 的可控开关

---

## 7) 关键风险点（需要提前规避）
- frame 语义不一致（link7 vs link8 vs robotiq_base）：会导致 robot flow 与 scene 对不上，模型条件输入失真
- scene_features 分布不对（颜色范围、缺 normals、dist2robot 未构造）：shape 对了但模型表现会崩
- 逐轨迹 Python 循环 FK：infer_ms 会远超预算；必须 batch
- 把 PointWorld scene 输入压成 cuboids：丢信息，世界模型的预测价值大幅下降
