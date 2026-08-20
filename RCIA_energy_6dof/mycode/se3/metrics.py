"""SO(3)/SE(3) 数学工具 + 6DoF 位姿评估指标（手写最小实现）。

这是原仓库学习计划 mycode/se3 任务的核心：
  - so3: 旋转表示互转（旋转矩阵/四元数/欧拉角/轴角/李代数 so(3)）
  - metrics: ADD / ADD-S / 旋转误差 / 平移误差

为什么圆柱体要用 ADD-S（答辩要点）：
  ADD   = mean || (R x_i + t) - (R' x_i + t') ||      逐点对应，要求姿态唯一
  ADD-S = mean_i min_j || (R x_i + t) - (R' x_j + t') ||  最近点，容忍对称歧义
  圆柱绕对称轴旋转位姿不唯一：视觉上一样的姿态有无穷多个，逐点 ADD 会虚高；
  ADD-S 把"对到模型任意等价点"都算对，才能反映真实定位质量。
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- SO(3) 工具
def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """R(3x3) -> 单位四元数 [w, x, y, z]。"""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:  # 180° 旋转，w≈0
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        q = np.zeros(4)
        q[1 + i] = np.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) / 2
        q[1 + j] = (R[j, i] + R[i, j]) / (4 * q[1 + i])
        q[1 + k] = (R[k, i] + R[i, k]) / (4 * q[1 + i])
        return q
    return np.array([w, x, y, z])


def so3_log(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 轴角向量（李代数 so(3)，theta = ||v||）。"""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-8:
        return np.zeros(3)
    if np.isclose(np.pi, theta):  # 接近 180°，用对称部分反解
        A = (R + np.eye(3)) / 2
        axis = np.sqrt(np.maximum(np.diag(A), 0))
        axis /= np.linalg.norm(axis)
        return theta * axis
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * axis / (2 * np.sin(theta))


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """旋转误差（度）：|| Log(R_gt^T R_est) ||。"""
    v = so3_log(R_gt.T @ R_est)
    return float(np.degrees(np.linalg.norm(v)))


def translation_error_m(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(t_est) - np.asarray(t_gt)))


# ---------------------------------------------------------------- 位姿指标
def add_metric(T_est, T_gt, model_points: np.ndarray) -> float:
    """ADD（非对称物体）：模型点逐一变换后求平均欧氏距离（米）。"""
    pe = (T_est[:3, :3] @ model_points.T).T + T_est[:3, 3]
    pg = (T_gt[:3, :3] @ model_points.T).T + T_gt[:3, 3]
    return float(np.mean(np.linalg.norm(pe - pg, axis=1)))


def add_s_metric(T_est, T_gt, model_points: np.ndarray, chunk: int = 512) -> float:
    """ADD-S（对称物体）：估计点 到 GT 变换点集 的最近距离均值。

    复杂度 N^2，分块计算防内存爆。模型 1690 点 -> 1690^2*8B ≈ 23MB，其实可以
    直接算，chunk 只是为了对大模型也通用。
    """
    pe = (T_est[:3, :3] @ model_points.T).T + T_est[:3, 3]
    pg = (T_gt[:3, :3] @ model_points.T).T + T_gt[:3, 3]
    total = 0.0
    for s in range(0, len(pe), chunk):
        d = np.linalg.norm(pe[s:s + chunk, None, :] - pg[None, :, :], axis=2)  # (c, N)
        total += float(d.min(axis=1).sum())
    return total / len(pe)


def evaluate_pose(T_est, T_gt, model_points: np.ndarray) -> dict:
    """一次算齐：ADD / ADD-S / 旋转误差 / 平移误差。"""
    return {
        "add_m": round(add_metric(T_est, T_gt, model_points), 5),
        "add_s_m": round(add_s_metric(T_est, T_gt, model_points), 5),
        "rot_err_deg": round(rotation_error_deg(T_est[:3, :3], T_gt[:3, :3]), 2),
        "trans_err_m": round(translation_error_m(T_est[:3, 3], T_gt[:3, 3]), 5),
    }
