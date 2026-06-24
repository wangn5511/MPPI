import os, json, numpy as np
import xml.etree.ElementTree as ET

BASE_NPZ = "/home/wangyuhan/MPPI/data/robot_mask_debug/frame_0000_base.npz"
JSON_PATH = "/home/wangyuhan/MPPI/data/test/data.json"
FRAME_IDX = 0
URDF_PATH = "/home/wangyuhan/PointWorld/assets/franka_description/franka_panda_robotiq_2f85.urdf"

OUT_NPZ = "/home/wangyuhan/MPPI/data/robot_mask_debug/frame_0000_base_plus_gripper_spheres.npz"
N_PER_SPHERE = 4000

try:
    import torch
    import pytorch_kinematics as pk
except Exception as e:
    raise SystemExit(f"Missing dependency for FK (torch+pytorch_kinematics): {e}")

z = np.load(BASE_NPZ, allow_pickle=False)
pts = z["points"].astype(np.float32)
cols = z["colors"].astype(np.uint8) if "colors" in z.files else None

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
q = np.asarray(data[FRAME_IDX]["/franka/joint_states"]["position"], dtype=np.float32).reshape(7)

with open(URDF_PATH, "rb") as f:
    urdf_bytes = f.read()
chain = pk.build_chain_from_urdf(bytes(urdf_bytes)).to(device=torch.device("cpu"), dtype=torch.float32)
frame_indices = chain.get_frame_indices("panda_link7")

joint_names = None
if hasattr(chain, "get_joint_parameter_names"):
    joint_names = list(chain.get_joint_parameter_names())
elif hasattr(chain, "get_joint_names"):
    joint_names = list(chain.get_joint_names())
if not joint_names:
    raise RuntimeError("Failed to query joint names from pytorch_kinematics chain")

q_t = torch.from_numpy(q.reshape(1, 7)).to(device=torch.device("cpu"), dtype=torch.float32)
zeros = torch.zeros((1,), dtype=torch.float32)
joint_dict = {str(n): zeros for n in joint_names}
joint_dict.update(
    {
        "panda_joint1": q_t[:, 0],
        "panda_joint2": q_t[:, 1],
        "panda_joint3": q_t[:, 2],
        "panda_joint4": q_t[:, 3],
        "panda_joint5": q_t[:, 4],
        "panda_joint6": q_t[:, 5],
        "panda_joint7": q_t[:, 6],
    }
)

with torch.no_grad():
    tf = chain.forward_kinematics(joint_dict, frame_indices=frame_indices)
    T = tf["panda_link7"].get_matrix()[0].detach().cpu().numpy().astype(np.float32)

def sphere_surface_points(center, radius, n):
    i = np.arange(n, dtype=np.float32) + 0.5
    phi = np.arccos(1 - 2*i/n)
    theta = np.pi * (1 + 5**0.5) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    p = np.stack([x, y, z], axis=1) * float(radius)
    return p + np.asarray(center, dtype=np.float32).reshape(1, 3)

def link7_offset_center(offset_z_m):
    p_l7 = np.asarray([0.0, 0.0, float(offset_z_m), 1.0], dtype=np.float32)  # “下方”按 -Z
    p_b = (T @ p_l7)[:3]
    return p_b

c1 = link7_offset_center(0.05); r1 = 0.17
c2 = link7_offset_center(0.20); r2 = 0.10

print("Added sphere A (base): center=", c1.tolist(), "r=", r1)
print("Added sphere B (base): center=", c2.tolist(), "r=", r2)

p1 = sphere_surface_points(c1, r1, N_PER_SPHERE)
p2 = sphere_surface_points(c2, r2, N_PER_SPHERE)

pts2 = np.concatenate([pts, p1, p2], axis=0).astype(np.float32, copy=False)

red = np.asarray([255, 0, 0], dtype=np.uint8)
if cols is None:
    cols0 = np.tile(np.asarray([180, 180, 180], dtype=np.uint8)[None, :], (pts.shape[0], 1))
else:
    cols0 = cols
cols2 = np.concatenate(
    [cols0, np.tile(red[None, :], (p1.shape[0] + p2.shape[0], 1))],
    axis=0,
).astype(np.uint8, copy=False)

os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
np.savez_compressed(OUT_NPZ, points=pts2, colors=cols2)
print("Saved:", OUT_NPZ, "points:", pts2.shape[0])