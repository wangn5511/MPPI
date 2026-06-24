import glob, os, numpy as np, yaml
from mppi.curobo_ext.scene_builder import PinholeIntrinsics, backproject_depth_to_points

data_root="/home/datasets/FrankaNav/test"
p=sorted(glob.glob(os.path.join(data_root,"NPY","*.npy")))[0]
d=np.load(p).astype(np.float32)

cam=yaml.safe_load(open("/home/wangyuhan/MPPI/configs/back_cam_info.yaml","r"))
K=cam["camera_matrix"]["data"]
intr=PinholeIntrinsics(fx=float(K[0]), fy=float(K[4]), cx=float(K[2]), cy=float(K[5]))

pts=backproject_depth_to_points(d, intr=intr, depth_scale=1.0, depth_min_m=0.1, depth_max_m=15.0, stride=4)
print("pts_cam", pts.shape)
print("z min/max", pts[:,2].min(), pts[:,2].max())