"""合成 RGB-D 仿真器共享库。

作用：替代 RealSense D435，用离屏渲染在已知 GT 位姿下生成 RGB / 深度 / mask。
这是"仿真级复现"的核心：没有相机也能驱动整条 6DoF 管线并量化评估。

坐标系约定（与原仓库 M3T 保持一致）：
  - 物体系 = models/object/Str3D.obj 的原始坐标（M3T body yaml 中 geometry2body_pose=I）
  - 相机系 = camera_color_optical_frame（OpenCV 约定：x右 y下 z前）
  - GT 位姿 T_cam_obj 为 4x4 齐次矩阵
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path("/home/yangyuting/summer_plan/ros2_ws/src/RCIA_Benmao_Vision")
MESH_PATH = REPO_ROOT / "models" / "object" / "Str3D.obj"

# 作者 D435 标定内参（见 scripts/linemod_detect_demo.py 默认值），仿真相机沿用同一套，
# 保证 LINEMOD 模板 / M3T / 评估脚本全部一致。
FX, FY, CX, CY = 606.1544849121768, 606.2304160969798, 320.71911360697, 239.7231533262084
WIDTH, HEIGHT = 640, 480


def camera_matrix() -> np.ndarray:
    return np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)


def load_mesh_vertices_faces(mesh_path: Path = MESH_PATH):
    """读取 OBJ 的顶点和面（复用仓库的解析器，保证和 LINEMOD 一致）。"""
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.linemod_silhouette_renderer import load_obj_mesh

    return load_obj_mesh(mesh_path)


def euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """R = Rz @ Ry @ Rx（角度制）。"""
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    sx, cx_ = np.sin(rx), np.cos(rx)
    sy, cy_ = np.sin(ry), np.cos(ry)
    sz, cz_ = np.sin(rz), np.cos(rz)
    Rx = np.array([[1, 0, 0], [0, cx_, -sx], [0, sx, cx_]])
    Ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    Rz = np.array([[cz_, -sz, 0], [sz, cz_, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_pose(rx, ry, rz, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = euler_xyz_to_matrix(rx, ry, rz)
    T[:3, 3] = t
    return T


def project_points(points_cam: np.ndarray, K: np.ndarray | None = None) -> np.ndarray:
    K = camera_matrix() if K is None else K
    p = (K @ points_cam.T).T
    return p[:, :2] / p[:, 2:3]


def draw_pose_overlay(
    image_bgr: np.ndarray,
    vertices_model: np.ndarray,
    T_cam_obj: np.ndarray,
    K: np.ndarray | None = None,
    subsample: int = 12,
    axes_length: float = 0.05,
    contour_color=(0, 255, 0),
    with_axes: bool = True,
) -> np.ndarray:
    """把模型点云投影回图像画轮廓 + 画物体坐标轴（位姿可视化）。"""
    out = image_bgr.copy()
    pts_cam = (T_cam_obj[:3, :3] @ vertices_model.T).T + T_cam_obj[:3, 3]
    if np.any(pts_cam[:, 2] <= 0):
        return out
    uv = project_points(pts_cam, K)
    for u, v in uv[::subsample]:
        cv2.circle(out, (int(round(u)), int(round(v))), 1, contour_color, -1, cv2.LINE_AA)
    if with_axes:
        origin = T_cam_obj[:3, 3]
        # 物体系 bbox 中心放轴，直观看姿态
        c_model = (vertices_model.min(0) + vertices_model.max(0)) / 2.0
        origin = (T_cam_obj[:3, :3] @ c_model) + T_cam_obj[:3, 3]
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x红 y绿 z蓝
        for i, col in enumerate(colors):
            axis_model = np.zeros(3)
            axis_model[i] = axes_length
            tip = (T_cam_obj[:3, :3] @ (c_model + axis_model)) + T_cam_obj[:3, 3]
            p0 = project_points(origin[None], K)[0]
            p1 = project_points(tip[None], K)[0]
            cv2.arrowedLine(out, tuple(p0.astype(int)), tuple(p1.astype(int)), col, 3, cv2.LINE_AA)
    return out


def load_gt(root: Path):
    with open(root / "gt_poses.json") as f:
        data = json.load(f)
    poses = [np.asarray(p, dtype=np.float64) for p in data["poses"]]
    return data, poses


def load_frame(root: Path, idx: int):
    rgb = cv2.imread(str(root / "rgb" / f"{idx:06d}.png"), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(root / "depth" / f"{idx:06d}.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(root / "mask" / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return rgb, depth_mm, mask
