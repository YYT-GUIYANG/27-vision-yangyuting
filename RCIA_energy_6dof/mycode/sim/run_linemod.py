#!/usr/bin/env python3
"""管线第 3 环（对比方案）：LINEMOD/Line2D 模板匹配 6DoF 检测。

复现原仓库保留的经典方法路线（scripts/linemod_*.py），在合成序列上：
  1. 加载模板库（由仓库官方 linemod_template_builder.py 生成）
  2. 每帧 Line2D 梯度方向匹配 -> 最优模板 -> 该视角下的候选位姿
  3. 用深度图在匹配中心反解平移（replace_pose_translation_from_depth）
  4. ADD / ADD-S / 旋转误差 / 平移误差 量化评估（mycode/se3）

这是"同方向不同方案对比"的经典组：LINEMOD(2012 模板匹配) vs 深度学习方法。
输出：results_v1/linemod/%06d.png + linemod_eval.json + 误差曲线图
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SIM_DIR = Path(__file__).resolve().parent
LEARN_ROOT = SIM_DIR.parent.parent
sys.path.insert(0, str(SIM_DIR)); sys.path.insert(0, str(SIM_DIR.parent))
sys.path.insert(0, "/home/yangyuting/summer_plan/ros2_ws/src/RCIA_Benmao_Vision")

from sim_lib import (  # noqa: E402
    REPO_ROOT, camera_matrix, draw_pose_overlay, load_frame, load_gt,
    load_mesh_vertices_faces,
)
from se3.metrics import evaluate_pose  # noqa: E402

TEMPLATES_DIR = REPO_ROOT / "runtime" / "linemod_templates_compact"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(LEARN_ROOT / "data" / "synthetic_v1"))
    ap.add_argument("--out", default=str(LEARN_ROOT / "results_v1" / "linemod"))
    ap.add_argument("--threshold", type=float, default=70.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--step", type=int, default=1, help="every step-th frame")
    args = ap.parse_args()

    from scripts.linemod_line2d import load_line2d_templates, match_line2d_templates
    from scripts.linemod_pose_utils import replace_pose_translation_from_depth
    from scripts.linemod_silhouette_renderer import render_silhouette
    from scripts.linemod_template_store import find_template_metadata, load_template_metadata

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_frames = len(list((data / "rgb").glob("*.png")))

    templates = load_line2d_templates(TEMPLATES_DIR / "Str3D_line2d_templates.json")
    metadata = load_template_metadata(TEMPLATES_DIR / "Str3D_template_poses.json")
    verts, faces = load_mesh_vertices_faces()
    K = camera_matrix()
    _data, gt_poses = load_gt(data)

    records = []
    t_match_ms: list[float] = []
    for i in range(0, n_frames, args.step):
        rgb, depth_mm, _mask = load_frame(data, i)
        import time

        t0 = time.perf_counter()
        matches = match_line2d_templates(rgb, templates, threshold=args.threshold,
                                          stride=args.stride)
        t_match_ms.append((time.perf_counter() - t0) * 1000)

        rec = {"frame": i, "detected": bool(matches)}
        overlay = rgb.copy()
        if matches:
            best = matches[0]
            template = {(t.class_id, t.template_id): t for t in templates}[
                (best.class_id, best.template_id)]
            pose = find_template_metadata(
                metadata, best.class_id, best.template_id).object_in_camera.copy()

            # 深度反解平移：匹配中心处读合成深度图
            cu = best.x + template.width / 2.0
            cv_ = best.y + template.height / 2.0
            d_px = depth_mm[int(cv_), int(cu)]
            depth_m = float(d_px) / 1000.0 if d_px > 0 else None
            if depth_m:
                pose = replace_pose_translation_from_depth(
                    pose, center_u=cu, center_v=cv_, depth_m=depth_m, k_matrix=K)

            rec.update(evaluate_pose(pose, gt_poses[i], verts))
            rec["score"] = round(float(best.similarity), 1)

            # 可视化：绿框=检测bbox，黄点=估计位姿投影轮廓，红色轴=GT
            cv2.rectangle(overlay, (best.x, best.y),
                          (best.x + template.width, best.y + template.height),
                          (0, 255, 0), 2)
            _sil_color, sil_mask = render_silhouette(
                verts, faces, pose, K, (rgb.shape[1], rgb.shape[0]))
            overlay[sil_mask > 0] = 0.4 * overlay[sil_mask > 0] \
                + 0.6 * np.array([0, 255, 255])
            overlay = draw_pose_overlay(overlay, verts, gt_poses[i],
                                        contour_color=(0, 0, 255), with_axes=True,
                                        subsample=999)  # GT 只画轴不画点云
            txt = (f"LINEMOD score={best.similarity:.0f} ADD-S={rec['add_s_m']*100:.1f}cm "
                   f"rot={rec['rot_err_deg']:.1f}deg")
            cv2.putText(overlay, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(overlay, "LINEMOD: no match", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{i:06d}.png"), overlay)
        records.append(rec)
        print(f"[{i+1}/{n_frames}] detected={rec['detected']}", flush=True)

    det = [r for r in records if r["detected"]]
    summary = {
        "frames": n_frames,
        "evaluated_frames": len(records),
        "detection_rate": round(len(det) / max(len(records), 1), 3),
        "mean_match_ms": round(float(np.mean(t_match_ms)), 1),
        "mean_add_s_cm": round(float(np.mean([r["add_s_m"] for r in det])) * 100, 2)
        if det else None,
        "median_add_s_cm": round(float(np.median([r["add_s_m"] for r in det])) * 100, 2)
        if det else None,
        "mean_rot_err_deg": round(float(np.mean([r["rot_err_deg"] for r in det])), 1)
        if det else None,
        "mean_trans_err_cm": round(float(np.mean([r["trans_err_m"] for r in det])) * 100, 2)
        if det else None,
    }
    with open(out / "linemod_eval.json", "w") as f:
        json.dump({"summary": summary, "frames": records}, f, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
