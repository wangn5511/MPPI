import os
import numpy as np

from mppi.curobo_ext.scene_builder import SceneBuildConfig, build_scene_cuboids_from_pcd_back_cam

def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def _env_vec3(name: str, default: str):
    return tuple(float(x) for x in os.environ.get(name, default).split(","))


PCD = os.environ.get("MPPI_SCENE_PCD_NPZ", "/home/wangyuhan/MPPI/data/robot_mask_debug/frame_0000_base.npz")
OUT = os.environ.get("MPPI_SCENE_PCD_NPZ_OUT", "/home/wangyuhan/MPPI/data/robot_mask_debug/scene_points_with_cuboids.npz")

VOXEL_SIZE_M = float(os.environ.get("MPPI_SCENE_VOXEL_SIZE_M", "0.01"))
PADDING_M = float(os.environ.get("MPPI_SCENE_PADDING_M", "0.02"))
MAX_CUBOIDS = int(os.environ.get("MPPI_SCENE_MAX_CUBOIDS", "20"))
MIN_CLUSTER_VOXELS = int(os.environ.get("MPPI_SCENE_MIN_CLUSTER_VOXELS", "30"))

ROI_MIN = _env_vec3("MPPI_SCENE_ROI_MIN", "-0.1,-0.7,-0.05")
ROI_MAX = _env_vec3("MPPI_SCENE_ROI_MAX", "1.2,0.7,1.2")

TABLE_DIMS = _env_vec3("MPPI_SCENE_TABLE_DIMS", "2.0,2.0,0.2")
TABLE_CENTER = _env_vec3("MPPI_SCENE_TABLE_CENTER", "0.4,0.0,-0.1")

REMOVE_TABLE_POINTS = _env_bool("MPPI_SCENE_REMOVE_TABLE_POINTS", "1")
TABLE_EPS_M = float(os.environ.get("MPPI_SCENE_TABLE_EPS_M", "0.01"))

REMOVE_WALL_POINTS = _env_bool("MPPI_SCENE_REMOVE_WALL_POINTS", "0")
WALL_DIMS = _env_vec3("MPPI_SCENE_WALL_DIMS", "2.5,0.5,2.0")
WALL_CENTER = _env_vec3("MPPI_SCENE_WALL_CENTER", "0.5,0.5,-0.5")
WALL_MARGIN_M = float(os.environ.get("MPPI_SCENE_WALL_MARGIN_M", "0.05"))

z = np.load(PCD, allow_pickle=False)
pts = z["points"].astype(np.float32)
cols = z["colors"].astype(np.uint8) if "colors" in z.files else None

table_top_z = float(TABLE_CENTER[2]) + 0.5 * float(TABLE_DIMS[2])

sb_cfg = SceneBuildConfig(
    t_base_cam_back_path=os.environ.get("MPPI_T_BASE_CAM_BACK_PATH", "/home/wangyuhan/MPPI/configs/T_base_cam.yaml"),
    roi_min=ROI_MIN,
    roi_max=ROI_MAX,
    voxel_size_m=VOXEL_SIZE_M,
    padding_m=PADDING_M,
    max_cuboids=MAX_CUBOIDS,
    robot_mask_margin_m=0.0,
    min_cluster_voxels=MIN_CLUSTER_VOXELS,
    remove_table_points=bool(REMOVE_TABLE_POINTS),
    table_top_z_m=float(table_top_z),
    table_eps_m=float(TABLE_EPS_M),
    remove_wall_points=bool(REMOVE_WALL_POINTS),
    wall_center=WALL_CENTER,
    wall_dims=WALL_DIMS,
    wall_margin_m=float(WALL_MARGIN_M),
)

dynamic = build_scene_cuboids_from_pcd_back_cam(
    pts,
    cfg=sb_cfg,
    pcd_scale=1.0,
    pcd_in_base=True,
    robot_spheres=None,
)

table = {"center": [float(TABLE_CENTER[0]), float(TABLE_CENTER[1]), float(TABLE_CENTER[2])],
         "dims": [float(TABLE_DIMS[0]), float(TABLE_DIMS[1]), float(TABLE_DIMS[2])]}

cuboids = [table] + list(dynamic)

wall_box = {
    "center": [float(WALL_CENTER[0]), float(WALL_CENTER[1]), float(WALL_CENTER[2])],
    "dims": [float(WALL_DIMS[0]) + 2.0 * float(WALL_MARGIN_M), float(WALL_DIMS[1]) + 2.0 * float(WALL_MARGIN_M), float(WALL_DIMS[2]) + 2.0 * float(WALL_MARGIN_M)],
}

print("dynamic cuboids:", len(dynamic))
for i, c in enumerate(cuboids):
    print(i, "center=", c["center"], "dims=", c["dims"])
if bool(REMOVE_WALL_POINTS):
    print("wall_filter_box:", "center=", wall_box["center"], "dims=", wall_box["dims"])

def cuboid_wire_points(center, dims, n=30):
    c = np.asarray(center, dtype=np.float32)
    d = np.asarray(dims, dtype=np.float32) * 0.5
    xs = [c[0]-d[0], c[0]+d[0]]
    ys = [c[1]-d[1], c[1]+d[1]]
    zs = [c[2]-d[2], c[2]+d[2]]
    corners = np.array([[x,y,z] for x in xs for y in ys for z in zs], dtype=np.float32)
    edges = [
        (0,1),(0,2),(0,4),
        (3,1),(3,2),(3,7),
        (5,1),(5,4),(5,7),
        (6,2),(6,4),(6,7),
    ]
    pts=[]
    for a,b in edges:
        pa, pb = corners[a], corners[b]
        t = np.linspace(0,1,n, dtype=np.float32)[:,None]
        pts.append(pa[None,:]*(1-t) + pb[None,:]*t)
    return np.concatenate(pts, axis=0)

wire_all=[]
col_all=[]
for i, c in enumerate(cuboids):
    w = cuboid_wire_points(c["center"], c["dims"], n=25)
    wire_all.append(w)
    if i == 0:
        color = np.array([255, 0, 0], dtype=np.uint8)
    else:
        color = np.array([0, 255, 0], dtype=np.uint8)
    col_all.append(np.tile(color[None, :], (w.shape[0], 1)))

if bool(REMOVE_WALL_POINTS):
    w = cuboid_wire_points(wall_box["center"], wall_box["dims"], n=25)
    wire_all.append(w)
    col_all.append(np.tile(np.array([0, 0, 255], dtype=np.uint8)[None, :], (w.shape[0], 1)))

wire = np.concatenate(wire_all, axis=0).astype(np.float32)
wire_c = np.concatenate(col_all, axis=0).astype(np.uint8)

if cols is None:
    cols0 = np.tile(np.array([180,180,180],dtype=np.uint8)[None,:], (pts.shape[0],1))
else:
    cols0 = cols

pts_out = np.concatenate([pts, wire], axis=0)
cols_out = np.concatenate([cols0, wire_c], axis=0)

np.savez_compressed(OUT, points=pts_out, colors=cols_out)
print("saved:", OUT)