#!/usr/bin/env python3
"""生成合成 RGB-D 序列（仿真级复现的数据源）。

对每帧 i：
  1. 采样 GT 位姿 T_cam_obj（平移轨迹 + 姿态轨迹，圆柱轴大致竖直 + 自转 + 摆动）
  2. pyrender EGL 离屏渲染：完整场景(物体+背景杂物) → RGB/深度；仅物体 → GT mask
  3. 后处理：高斯噪声 + 亮度抖动（模拟真实传感器）

输出 data/synthetic_v1/{rgb,depth,mask}/%06d.png + gt_poses.json + preview 网格图
用法: python3 render_synthetic_seq.py [--frames 120] [--out ../../data/synthetic_v1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))
from sim_lib import (  # noqa: E402
    CX, CY, FX, FY, HEIGHT, MESH_PATH, WIDTH, camera_matrix, load_mesh_vertices_faces,
)

T_GL_CAM = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV 光心系 -> OpenGL 相机系
# 注意约定：场景世界系 = OpenCV 光心系（camera_color_optical_frame）。
# 相机节点姿态 = T_GL_CAM；物体/背景节点姿态直接用光心系下的 4x4，不要再左乘！


def build_object_meshes():
    """把 Str3D.obj 按区域拆成两个网格：黑本体 + 白色(顶帽/底帽/侧面环)。

    原 OBJ 引用的 mtl 文件缺失，且实测 pyrender 传自定义材质/顶点色路径均不可靠，
    因此按面拆分后各用 baseColorFactor 上色（模拟真实能量单元黑白外观——README
    health gate 提到目标是低饱和黑白分布；此处为近似假设，已记录在 notes）。
    返回 [(trimesh, rgb_factor), ...]
    """
    import sys

    sys.path.insert(0, str(SIM_DIR))
    from sim_lib import load_mesh_vertices_faces

    v, f = load_mesh_vertices_faces()
    z = v[:, 2]
    white_vert = (z > -0.008) | (z < -0.142) | ((z > -0.088) & (z < -0.068))
    # 面标白：3 个顶点里至少 2 个属于白色区域（避免边界锯齿）
    white_face = white_vert[f].sum(axis=1) >= 2

    dark = (0.13, 0.13, 0.15, 1.0)
    white = (0.86, 0.85, 0.81, 1.0)
    meshes = []
    for mask, factor in ((~white_face, dark), (white_face, white)):
        if mask.sum() == 0:
            continue
        sub = trimesh.Trimesh(vertices=v, faces=f[mask], process=False)
        meshes.append((sub, factor))
    return meshes


def build_background(scene: pyrender.Scene) -> None:
    """静态背景：墙 + 地板 + 两个杂物箱（都在相机系下，深度 1.2~1.6m）。"""
    wall = trimesh.creation.box(extents=(2.4, 1.8, 0.02))
    wall.visual.face_colors = np.array([120, 118, 112, 255], dtype=np.uint8)
    tw = np.eye(4)
    tw[:3, 3] = (0.0, 0.0, 1.6)
    scene.add(pyrender.Mesh.from_trimesh(wall, smooth=False), pose=tw)

    floor = trimesh.creation.box(extents=(2.4, 0.02, 1.6))
    floor.visual.face_colors = np.array([80, 78, 74, 255], dtype=np.uint8)
    tf = np.eye(4)
    tf[:3, 3] = (0.0, -0.42, 0.9)
    scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False), pose=tf)

    for i, (dx, dy, w) in enumerate([(-0.28, 0.10, 0.14), (0.30, -0.05, 0.10)]):
        box = trimesh.creation.box(extents=(w, w, w))
        box.visual.face_colors = np.array(
            [[150, 140, 90], [90, 100, 150]][i] + [255], dtype=np.uint8)
        tb = np.eye(4)
        tb[:3, 3] = (dx, dy, 1.25)
        scene.add(pyrender.Mesh.from_trimesh(box, smooth=False), pose=tb)

    # 平行光：节点姿态取 T_GL_CAM（其局部 -z 即光心系 +z，朝向场景）
    light_rot = T_GL_CAM.copy()
    scene.add(pyrender.DirectionalLight(intensity=1.1), pose=light_rot)
    lut = T_GL_CAM.copy()
    lut[0, 3], lut[1, 3], lut[2, 3] = -0.15, -0.35, -0.2  # 光心系下左上方
    scene.add(pyrender.PointLight(intensity=2.2), pose=lut)


def gt_pose(i: int, n: int, c_model: np.ndarray) -> np.ndarray:
    """GT 轨迹：物体 bbox 中心在相机系的位置 p(i)，姿态 = 竖直基准 + 倾摆 + 自转。"""
    t = i / max(n - 1, 1)
    # 平移：来回移动 + 前后距离变化（0.50m -> 0.40m -> 0.48m）
    px = 0.085 * np.sin(2 * np.pi * 1.1 * t)
    py = 0.045 * np.sin(2 * np.pi * 0.7 * t + 0.6)
    pz = 0.49 - 0.09 * np.sin(np.pi * t)
    p = np.array([px, py, pz])
    # 姿态：Rx(90°) 让圆柱轴(z)竖直朝上；再加缓慢倾摆和绕轴自转
    tilt_x = 90.0 + 14.0 * np.sin(2 * np.pi * 0.5 * t)
    tilt_y = 10.0 * np.sin(2 * np.pi * 0.35 * t + 1.0)
    spin_z = 360.0 * 1.5 * t
    from sim_lib import make_pose

    T = make_pose(tilt_x, tilt_y, spin_z, np.zeros(3))
    # 平移按"bbox 中心落在 p"反解：t = p - R @ c_model
    T[:3, 3] = p - T[:3, :3] @ c_model
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=str, default=str(SIM_DIR.parent.parent / "data" / "synthetic_v1"))
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    for sub in ("rgb", "depth", "mask"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    verts, _faces = load_mesh_vertices_faces()
    c_model = (verts.min(0) + verts.max(0)) / 2.0

    scene = pyrender.Scene(ambient_light=[0.13, 0.13, 0.14], bg_color=[0.0, 0.0, 0.0])
    build_background(scene)

    # 黑白两个子网格：同一姿态、两个节点
    obj_nodes = []
    for sub, factor in build_object_meshes():
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.05, roughnessFactor=0.85,
            baseColorFactor=list(factor), alphaMode="OPAQUE")
        obj_nodes.append(scene.add(
            pyrender.Mesh.from_trimesh(sub, material=material, smooth=True)))

    # 仅物体的场景（白光照满）用于渲染 GT mask / GT 深度
    scene_objonly = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
    objonly_nodes = []
    for sub, _factor in build_object_meshes():
        objonly_nodes.append(scene_objonly.add(pyrender.Mesh.from_trimesh(sub)))

    cam = pyrender.IntrinsicsCamera(fx=FX, fy=FY, cx=CX, cy=CY, znear=0.05, zfar=4.0)
    scene.add(cam, pose=T_GL_CAM)
    scene_objonly.add(
        pyrender.IntrinsicsCamera(fx=FX, fy=FY, cx=CX, cy=CY, znear=0.05, zfar=4.0),
        pose=T_GL_CAM)

    renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)

    poses = []
    for i in range(args.frames):
        T_cam_obj = gt_pose(i, args.frames, c_model)
        # 世界系=光心系：物体节点直接用 T_cam_obj；不要再左乘 T_GL_CAM
        for node in obj_nodes:
            scene.set_pose(node, T_cam_obj)
        for node in objonly_nodes:
            scene_objonly.set_pose(node, T_cam_obj)
        color, depth_full = renderer.render(scene)
        _color_o, depth_obj = renderer.render(scene_objonly)

        mask = (depth_obj > 0).astype(np.uint8) * 255

        # 传感器仿真：噪声 + 亮度抖动 + 轻微深度量化(1mm 已由 uint16 保证)
        color = color.astype(np.float32)
        color += rng.normal(0, 2.0, color.shape)
        color *= rng.uniform(0.96, 1.04)
        color = np.clip(color, 0, 255).astype(np.uint8)

        # 深度加一点噪声（仅物体区域）
        depth_m = depth_full.copy()
        obj_px = mask > 0
        depth_m[obj_px] += rng.normal(0, 0.0008, int(obj_px.sum()))

        cv2.imwrite(str(out / "rgb" / f"{i:06d}.png"),
                    cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "depth" / f"{i:06d}.png"),
                    np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(out / "mask" / f"{i:06d}.png"), mask)
        poses.append(T_cam_obj.tolist())

        if i % 20 == 0:
            print(f"[{i+1}/{args.frames}] t={T_cam_obj[:3,3].round(3)} done")

    renderer.delete()

    with open(out / "gt_poses.json", "w") as f:
        json.dump({
            "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY,
                            "width": WIDTH, "height": HEIGHT},
            "fps": args.fps,
            "mesh": str(MESH_PATH),
            "model_center": c_model.tolist(),
            "poses": poses,
        }, f, indent=2)

    # 预览网格图（直接给人看的第一份证据）
    idxs = np.linspace(0, args.frames - 1, 9).astype(int)
    grid = []
    for r in range(3):
        row = []
        for c in range(3):
            k = idxs[r * 3 + c]
            row.append(cv2.imread(str(out / "rgb" / f"{k:06d}.png")))
        grid.append(np.hstack(row))
    cv2.imwrite(str(out / "preview_grid.png"), np.vstack(grid))
    print("saved:", out, "| preview:", out / "preview_grid.png")


if __name__ == "__main__":
    main()
