# MVP 归档：PCL(WebSocket) 通信协议 & test_cuRobo_pcl.sh 环境变量说明

本文件用于归档当前 MVP（_pcl 分支）在线推理链路的：
- 通信协议（msgpack over WebSocket）
- 启动脚本 /home/wangyuhan/MPPI/scripts/test_cuRobo_pcl.sh 中环境变量含义与调参方向

---

## A. PCL 通信协议（SCHEMA_VERSION_PCL=100）

### A1. 传输层
- 传输：WebSocket 二进制帧
- 编码：msgpack
- 约定：payload 内禁止直接传 `np.ndarray`（会触发 base64 打包，体积膨胀与 CPU 抖动），推荐传 `bytes`（msgpack bin）

### A2. Envelope（所有消息通用）
| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| schema_version | int | 是 | `100` |
| type | str | 是 | `infer_request_pcl` / `infer_response_pcl` / `error_pcl` |
| request_id | str | 是 | UUID 字符串 |
| payload | dict | 是 | 根据 type 不同而不同 |

### A3. infer_request_pcl.payload = ObsPCL
#### 业务字段（每帧必须）
| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| t_client_send_ns | int | 是 | client 发送时间戳（ns） |
| step_id | int | 是 | 帧号/时间步（建议与数据集 frame_id 对齐） |
| q | list[float] | 是 | 7 维关节角（rad），必须与该帧 RGB/Depth 同步 |
| gripper | float | 是 | 夹爪状态（约定由上层定义） |

#### 相机参数（推荐 cam_id 方案，最小侵入）
| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| cam_id | str | 推荐 | 例如 `back`；server 用固定配置加载 intrinsics 和 T_base_cam |
| intrinsics | dict | 否 | 逐帧内参（不推荐，除非你必须动态变化） |
| T_base_cam | any | 否 | 逐帧外参 4x4（不推荐） |

#### 原始输入（压缩传输推荐）
Depth 必须是 **float32(m)**（与录制数据单位对齐）。

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| rgb_codec | str | 是 | `"jpeg"` |
| rgb_bytes | bytes | 是 | JPEG bytes（msgpack bin） |
| rgb_shape_hw | list[int] | 推荐 | `[H, W]`，用于一致性校验 |
| depth_codec | str | 是 | `"npy_zlib"` |
| depth_bytes | bytes | 是 | `zlib.compress(npy_bytes)`，其中 npy 内为 `float32(H,W)`，单位 m |
| depth_shape_hw | list[int] | 推荐 | `[H, W]`，用于一致性校验 |
| depth_unit_scale | float | 推荐 | 由于 depth 已是米制 float32，推荐固定为 `1.0` |

#### 兼容字段（历史版本 / 回退用）
| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| rgb_back | ndarray | 否 | 旧版直接传 ndarray（慢，不推荐） |
| depth_back | ndarray | 否 | 同上 |

兼容策略（server 侧解码优先级）：
1) 若 `rgb_bytes`/`depth_bytes` 存在，则走压缩解码
2) 否则 fallback 使用 `rgb_back`/`depth_back`

### A4. infer_response_pcl.payload = ActionChunkPCL
| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| t_server_recv_ns | int | 是 | server 收到请求时间戳（ns） |
| t_server_send_ns | int | 是 | server 发送响应时间戳（ns） |
| t_client_send_ns_echo | int | 是 | 回显 client 的 t_client_send_ns |
| open_loop_horizon | int | 是 | 返回 action chunk 的长度 |
| actions | any | 是 | 动作数组（实现上通常是 `float32(H,8)`） |
| server_timing | dict | 是 | `{infer_ms, queue_ms, policy}` |

---

## B. test_cuRobo_pcl.sh 环境变量说明（MVP）

脚本位置：/home/wangyuhan/MPPI/scripts/test_cuRobo_pcl.sh  
该脚本主要配置三件事：
1) cuRobo 是否启用、权重与预算
2) scene builder 的裁剪/降采样/聚类与过滤策略（robot mask / table / wall）
3) PCL（RGBD->点云）参数与调试保存/可视化

说明约定：
- “数值越大/越小”只对连续数值生效；布尔值用 0/1。
- 主要单位：米（m）、毫秒（ms）。

### B1. 基础运行
| 变量 | 示例值 | 含义 | 值更大 | 值更小 |
|---|---:|---|---|---|
| PYTHONPATH | /home/wangyuhan/MPPI/src:$PYTHONPATH | 让 python 能找到 mppi 包 | - | - |

启动命令（非 env）：
- `--open-loop-horizon 8`：一次返回多少步动作。更大：动作 chunk 更长、通讯更省，但单次推理更重；更小反之。
- `--policy mppi_joint`：使用 MPPI+cuRobo。
- `--port 9011`：服务端端口。

### B2. cuRobo / 代价权重开关
| 变量 | 示例值 | 含义 | 值更大 | 值更小 |
|---|---:|---|---|---|
| MPPI_USE_CUROBO_COLLISION | 1 | 是否启用 cuRobo 碰撞 | 1=启用，开销上升 | 0=关闭，开销下降 |
| MPPI_W_SCENE_COLLISION | 1 | 场景碰撞项权重（scene cuboids） | 更避障但更易触发“构建场景”的路径 | 更不避障，甚至不需要建场景 |
| MPPI_SCENE_FROM_PCD_BACK_CAM | 1 | 是否从点云构建动态场景 cuboids | 1=每帧会尝试构建/更新场景 | 0=不从点云建场景（cub=0） |

补充：
- 实际触发“从点云建场景”需要同时满足：`USE_CUROBO_COLLISION=1`、`SCENE_FROM_PCD_BACK_CAM=1`、`W_SCENE_COLLISION>0`。

### B3. 点云语义与坐标
| 变量 | 示例值 | 含义 | 值更大 | 值更小 |
|---|---:|---|---|---|
| MPPI_SCENE_PCD_SCALE | 1 | 点云尺度缩放 | 放大点云（慎用） | 缩小点云（慎用） |
| MPPI_SCENE_PCD_IN_BASE | 1 | 输入点云是否已在 base 坐标系 | 1=无需再做 cam->base | 0=需要再变换一次（依赖外参） |
| MPPI_T_BASE_CAM_BACK_PATH | .../T_base_cam.yaml | base<-cam 外参（cam->base） | - | - |

建议：
- 如果 server 端已输出 base 点云，`MPPI_SCENE_PCD_IN_BASE` 必须为 1，否则会被重复变换导致 mask/cuboids 全错位。

### B4. ROI / 体素 / 聚类（决定 cub 数量、稳定性与耗时）
| 变量 | 示例值 | 含义 | 值更大 | 值更小 |
|---|---:|---|---|---|
| MPPI_SCENE_ROI_MIN | -0.1,-0.7,-0.05 | ROI 下界（base） | 范围更大（包含更多点，耗时↑，误检↑） | 范围更小（点更少，耗时↓，漏检↑） |
| MPPI_SCENE_ROI_MAX | 1.2,0.7,1.2 | ROI 上界（base） | 同上 | 同上 |
| MPPI_SCENE_VOXEL_SIZE_M | 0.01 | 体素下采样/聚类体素大小 | 更大：点/体素更少，速度↑，细节↓ | 更小：更细，速度↓，噪声敏感↑ |
| MPPI_SCENE_MIN_CLUSTER_VOXELS | 20 | 最小簇体素数（过滤小物体） | 更大：过滤更多小簇，cub 数↓，更稳定但漏检↑ | 更小：更敏感，cub 数↑，抖动↑ |
| MPPI_SCENE_MAX_CUBOIDS | 20 | 最多输出多少个动态 cuboids（不含 table 时） | 更大：允许更多障碍，开销↑ | 更小：更快，但可能漏障碍 |
| MPPI_SCENE_PADDING_M | 0.02 | cuboid padding（AABB 膨胀） | 更保守，避障更强，误碰风险↓ | 更激进，避障更弱，误碰风险↑ |

### B5. Robot mask（影响“机械臂点是否剔除干净”）
| 变量 | 示例值 | 含义 | 值更大 | 值更小 |
|---|---:|---|---|---|
| MPPI_SCENE_ROBOT_MASK_MARGIN_M | 0.1 | robot spheres 半径膨胀（m） | mask 更大：机器人点更容易被剔除，但可能误删邻近环境点 | mask 更小：误删↓，但容易残留机器人点 |

说明：
- robot mask 使用 cuRobo 根据 q 生成的一组球体（每个球半径会加 margin），再把点云中落入球体的点剔除。

### B6. 桌面与墙体过滤（主要影响聚类质量与 cub 数）
| 变量 | 示例值 | 含义 | 值更大 | 值更小 |
|---|---:|---|---|---|
| MPPI_SCENE_ADD_TABLE | 1 | 是否把 table cuboid 加入最终 scene_cuboids | 1=policy 中 tab=1 | 0=tab=0 |
| MPPI_SCENE_TABLE_DIMS | 2.0,2.0,0.2 | 桌面 cuboid 尺寸（m） | 更大：table 覆盖更广 | 更小：table 覆盖更窄 |
| MPPI_SCENE_TABLE_CENTER | 0.4,0.0,-0.1 | 桌面 cuboid 中心（base） | 位置偏移会改变剔除阈值与避障体 | 同左 |
| MPPI_SCENE_REMOVE_TABLE_POINTS | 1 | 是否剔除桌面点（z<th） | 1=更利于正确聚类 | 0=桌面点可能桥接导致聚类错 |
| MPPI_SCENE_TABLE_EPS_M | 0.01 | 桌面剔除裕量（m） | 更大：剔除更多靠近桌面的点（可能删掉物体底部） | 更小：保留更多桌面附近点（可能残留桌面） |
| MPPI_SCENE_REMOVE_WALL_POINTS | 1 | 是否剔除墙体盒内点 | 1=减少墙点干扰 | 0=墙点可能被聚类成障碍 |
| MPPI_SCENE_WALL_DIMS | 2.5,0.5,2.0 | 墙体剔除盒尺寸（m） | 更大：剔除更多区域 | 更小：剔除更少区域 |
| MPPI_SCENE_WALL_CENTER | 0.5,0.5,-0.5 | 墙体剔除盒中心（base） | 位置偏移会改变剔除区域 | 同左 |
| MPPI_SCENE_WALL_MARGIN_M | 0.05 | 墙体剔除盒额外膨胀（m） | 更大：剔除更 aggressive | 更小：剔除更保守 |

### B7. 动态场景跟踪（稳定性 vs 响应速度）
| 变量 | 示例值 | 含义 | 值更大 | 值更小 |
|---|---:|---|---|---|
| MPPI_SCENE_TRACK_ALPHA | 0.6 | track 滤波系数（0-1） | 更大：更“跟当前帧”，响应快但抖动↑ | 更小：更“平滑”，更稳但延迟↑ |
| MPPI_SCENE_TRACK_REMOVE_AFTER_MISSES | 5 | 连续 miss 几次后移除 track | 更大：更不易消失（残留风险↑） | 更小：更快清理（闪烁风险↑） |
| MPPI_SCENE_TRACK_MAX_TRACKS | 20 | 最大 track 数 | 更大：更多障碍跟踪，开销↑ | 更小：更快但可能漏 |
| MPPI_SCENE_TRACK_MATCH_CENTER_DIST_M | 0.05 | center 距离匹配阈值（m） | 更大：更容易匹配为同一物体（错配↑） | 更小：更严格（断 track ↑） |
| MPPI_SCENE_TRACK_MATCH_IOU_MIN | 0.05 | AABB IoU 匹配阈值 | 更大：更严格（断 track ↑） | 更小：更宽松（错配↑） |

### B8. 预算降级策略（避免超时）
| 变量 | 示例值 | 含义 | 值更大 | 值更小 |
|---|---:|---|---|---|
| MPPI_INFER_BUDGET_MS | 50 | 推理预算（ms），超预算会触发降级（freeze_scene/limit_cuboids/reduce_samples/hold） | 更大：更少降级，更稳定但可能延迟更高 | 更小：更易降级，延迟更可控但策略可能变保守/停顿 |
| MPPI_BUDGET_MAX_DYNAMIC_CUBOIDS | 6 | 降级到 limit_cuboids 时最多保留几个动态 cuboids | 更大：更保守避障但开销↑ | 更小：更快但漏避障风险↑ |

### B9. PCL（RGBD->点云）参数
| 变量 | 示例值 | 含义 | 值更大 | 值更小 |
|---|---:|---|---|---|
| MPPI_PCL_CAM_INFO_BACK_PATH | .../back_cam_info.yaml | back 相机内参 | - | - |
| MPPI_PCL_T_BASE_CAM_BACK_PATH | .../T_base_cam.yaml | back 相机外参（base<-cam） | - | - |
| MPPI_PCL_DEPTH_UNIT_SCALE | 1.0 | 深度单位缩放（float32(m) 时应为 1.0） | >1 会放大深度（慎用） | <1 会缩小深度（慎用） |
| MPPI_PCL_DEPTH_MIN_M | 0.05 | 反投影最小深度（m） | 更大：近处点更少，噪声↓，机器人残留↓/或漏近物体↑ | 更小：保留更多近点，噪声↑ |
| MPPI_PCL_DEPTH_MAX_M | 2.0 | 反投影最大深度（m） | 更大：远点更多，开销↑ | 更小：远点更少，开销↓ |
| MPPI_PCL_STRIDE | 1 | 深度采样步长（1=全分辨率） | 更大：点数约按 1/stride^2 下降，速度↑，细节↓ | 更小：更密，速度↓ |

### B10. 保存/可视化（调试用，通常应关闭）
| 变量 | 示例值 | 含义 | 值更大 | 值更小 |
|---|---:|---|---|---|
| MPPI_PCL_SAVE_PCD | 0 | 是否保存调试点云 npz | 1=保存（耗时↑、抖动↑） | 0=不保存（推荐） |
| MPPI_PCL_SAVE_PCD_OUT | .../real_pcl.npz | 保存路径（.npz 结尾会覆盖） | - | - |
| MPPI_PCL_ROBOT_MASK_VIS_POINTS_PER_SPHERE | 800 | robot mask 可视化每个 sphere 采样点数 | 更大：显示更密，写盘/体积↑ | 更小：更稀疏 |
| MPPI_PCL_ROBOT_MASK_VIS_MAX_POINTS | 200000 | robot mask 可视化最大总点数 | 更大：更完整但更重 | 更小：更轻但可能截断 |

---

## C. 常见现象与定位建议（MVP）
- cub 数量异常/抖动：
  - 先确认 q 与该帧 RGBD 同步；不同步会导致 robot mask 与聚类都错位。
  - 再检查：ROI、VOXEL_SIZE、MIN_CLUSTER_VOXELS、TABLE/WALL 过滤是否按预期生效。
- infer_ms 出现大尖峰：
  - 若传 ndarray（base64）：会有明显抖动；
  - 若改为 bytes 压缩：尖峰通常显著减少；
  - 保存点云（写盘）也会引入尖峰，应默认关闭。
