下面这版待办清单按你的要求重写： 不在现有点云协议/文件上加字段 ，而是 重建一套只接受 RGB+NPY(depth) 作为原始输入 的 pipeline，并且所有新版本文件都用 _pcl 后缀，方便你“纯净版”重建与并行对照测试。（路径里你写的 \home\... 在 Linux 实际是 /home/... ）
# 1) 新建协议类型：types_pcl.py（协议分叉的根）

## 要做：从 types.py 复制出 src/mppi/protocol/types_pcl.py
  - 定义 SCHEMA_VERSION_PCL 、 ObsPCL 、 InferRequestPCL 、 InferResponsePCL
  - ObsPCL 只包含： t_client_send_ns, step_id, q, gripper, rgb_back, depth_back ，以及相机参数（两种选一）
    - A) 每帧带： intrinsics(fx,fy,cx,cy,w,h) + depth_unit_scale + T_base_cam
    - B) 只带 cam_id ，server 用固定配置（推荐）
## 为什么：你要“纯净版只吃 RGB+NPY”，就必须让 schema 本身断开 pcd_back_cam ，否则会隐性回到旧逻辑；协议分叉也保证旧 client/server 不受影响。
# 2) 新建 RGBD→点云核心模块：check_depth_pcl.py（server 内存版）

## 要做：从 check_depth.py 复制出 src/mppi/curobo_ext/check_depth_pcl.py
  - 保留并复用现有函数能力（你已经有这些依赖与实现）：
    - backproject_depth_to_points_with_uv （深度反投影+uv）
    - load_pinhole_intrinsics_from_cam_info （从 yaml 读内参）
    - transform_points （cam→base）
    - _voxel_downsample_indices 、ROI crop、颜色采样（ rgb[v,u] ）
  - 新增/整理成“纯函数入口”（供 server 调用，而不是 argparse CLI）：
    - rgbd_to_pointcloud_base(depth, rgb, intr, T_base_cam, depth_unit_scale, depth_min/max, stride, roi_min/max, voxel_size) -> {"points":..., "colors":...}
## 为什么：server 端要实时处理，必须直接吃 ndarray，而不是像离线脚本那样走“读文件路径+argparse”；复制成 _pcl 可以做到“纯净版不改动现有 check_depth CLI”。
# 3) 新建 server：ws_server_async_pcl.py（只接受 RGB+NPY 请求）

## 要做：从 ws_server_async.py 复制出 src/mppi/comm/ws_server_async_pcl.py
  - 只处理 type="infer_request_pcl" + SCHEMA_VERSION_PCL
  - ObsPCL 解码后：调用 check_depth_pcl.rgbd_to_pointcloud_base(...) 得到 pcd_base
  - 将 pcd_base 组装成现有 solver 可吃的 pcd_back_cam 字典（语义上就是“点云输入”，你现有 solver 已支持 scene_pcd_in_base=1 ）
  - 继续复用现有 JointMPPISolver.infer_actions(q0, gripper, pcd_back_cam=...)
  - server 启动时一次性加载相机配置（若你选 cam_id 方案）： back_cam_info.yaml + T_base_cam.yaml
## 为什么：你的“纯净版”核心就是把 /home/wangyuhan/MPPI/scripts/debug_robot_mask.sh 的 **Step1（RGBD→点云）**搬进 server，并保证后续 robot mask / 背景删除 / cuboids / table 都走你现有稳定实现（solver+scene_builder+collision_checker）。
# 4) 新建最小 client：ws_client_sync_pcl.py（用于在线单次/快速调试）

## 要做：从 ws_client_sync.py 复制出 src/mppi/comm/ws_client_sync_pcl.py
  - 不再支持 --pcd-npz
  - 支持 --rgb PATH --depth PATH （client 本地读入为 ndarray，然后发 rgb_back/depth_back ）
  - 构造并发送 InferRequestPCL
## 为什么：你需要一个“非 playback”的最小闭环工具来快速确认 server_pcl 协议/解码/点云生成没问题，避免每次都跑长序列。
# 5) 新建 playback：playback_client_pcl.py（用 data.json 做端到端回归）

## 要做：从 playback_client.py 复制出 scripts/playback_client_pcl.py
  - 从 /home/wangyuhan/MPPI/data/test/data.json 逐帧读取 images.back 与 depths.back_depth （你离线脚本里已经验证过 key 路径）
  - 复用你离线脚本中的路径 normalize 逻辑（去掉可能的 ep_xxx/ 前缀、处理 \ → / ）
  - 读入 rgb/depth ndarray，发送 InferRequestPCL
  - 继续解析回包的 policy 字符串里 tab/cub/sph ，用于确认 table/cuboids/spheres 数量稳定且正确
## 为什么：你强调“RGB+NPY 已 ROS2 时间对齐封装在 data.json”，playback 是验证在线 _pcl pipeline 正确性的最快方式，也能直接对比你旧“传点云版”的性能/稳定性。
# 6) 新建启动脚本：test_cuRobo_pcl.sh（你已创建空文件，补全它）

## 要做：补全 test_cuRobo_pcl.sh
  - 启动 python3 -m mppi.comm.ws_server_async_pcl --policy mppi_joint ... （或直接 python3 /.../ws_server_async_pcl.py ）
  - 环境变量沿用旧版（scene/table/wall/cluster/robot_mask_margin 等），另外增加/固定相机配置路径（若用 cam_id 方案）
## 为什么：你要并行保留旧 pipeline 做 A/B，对照测试必须“一键切换 server 版本”，脚本是最稳定的入口。
# 7) 明确 server 内处理顺序（必须固化，不然会回到你之前的坑）

## 要做：在 server_pcl 生成点云后，后续处理顺序固定为：
  - cam→base（只一次）→ ROI → voxel → robot spheres mask → table 点剔除 → wall box 剔除 → 聚类→cuboids → scene_cuboids=[table]+dynamic
## 为什么：你已验证“桌/墙不剔除会桥接导致聚类错”；robot mask 必须在聚类前，否则机械臂会被当环境障碍；并且 cam/base 语义必须唯一，避免重复变换。
# 8) 传输格式与大小策略（PCL 版必须考虑，否则很容易卡带宽/内存）

## 要做（MVP 推荐）：直接发送 numpy ndarray（你现有 msgpack_codec.py 已支持 ndarray 自动 base64）
  - rgb_back: uint8(H,W,3) ， depth_back: float32(H,W) 或 uint16(H,W)+depth_unit_scale
## 为什么：这是最省依赖、最少代码的“纯净版”起步；后续如果要提性能，再把 rgb/depth 改成压缩 bytes（JPEG/PNG/NPY bytes）也不影响整体结构。