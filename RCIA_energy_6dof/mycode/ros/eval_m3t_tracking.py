#!/usr/bin/env python3
"""M3T 跟踪 vs GT 离线评估（全链仿真的核心量化结果）。

输入：m3t_eval_recorder.py 录的 jsonl（/energy/m3t_pose + /energy/synthetic_gt_pose）
方法：按 header stamp 最近邻配对（容差 50ms），逐帧算 ADD/ADD-S/旋转/平移误差。
输出：results_v1/m3t_tracking/m3t_eval.json + perframe 曲线图
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROS_DIR = Path(__file__).resolve().parent
LEARN_ROOT = ROS_DIR.parent.parent
sys.path.insert(0, str(LEARN_ROOT / "mycode"))
sys.path.insert(0, str(LEARN_ROOT / "mycode" / "sim"))
from se3.metrics import evaluate_pose  # noqa: E402
from sim_lib import load_mesh_vertices_faces  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(LEARN_ROOT / "results_v1" / "ros_logs" / "m3t_vs_gt.jsonl"))
    ap.add_argument("--tol", type=float, default=0.05, help="配对时间容差(s)")
    ap.add_argument("--out", default=str(LEARN_ROOT / "results_v1" / "m3t_tracking"))
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.jsonl)]
    m3t = sorted((r for r in recs if r["type"] == "m3t"), key=lambda r: r["stamp"])
    gt = sorted((r for r in recs if r["type"] == "gt"), key=lambda r: r["stamp"])
    if not m3t or not gt:
        raise SystemExit("no m3t or gt records")

    verts, _ = load_mesh_vertices_faces()
    c_model = (verts.min(0) + verts.max(0)) / 2.0  # bbox 中心（物体系）
    gt_stamps = np.array([r["stamp"] for r in gt])

    def symmetric_extras(T_m, T_g):
        """对称物体专用指标：轴向误差 + bbox中心误差（原点/旋转矩阵会被对称歧义污染）。"""
        axis_e = T_m[:3, :3] @ np.array([0, 0, 1.0])
        axis_g = T_g[:3, :3] @ np.array([0, 0, 1.0])
        cosang = np.clip(np.dot(axis_e, axis_g) /
                         (np.linalg.norm(axis_e) * np.linalg.norm(axis_g)), -1, 1)
        # 圆柱轴无方向性(两端对调等价)，取 [0,90]
        axis_err = float(np.degrees(np.arccos(abs(cosang))))
        ce = T_m[:3, :3] @ c_model + T_m[:3, 3]
        cg = T_g[:3, :3] @ c_model + T_g[:3, 3]
        return round(axis_err, 2), round(float(np.linalg.norm(ce - cg)) * 100, 3)

    rows = []
    for m in m3t:
        j = int(np.argmin(np.abs(gt_stamps - m["stamp"])))
        if abs(gt_stamps[j] - m["stamp"]) > args.tol:
            continue
        T_m, T_g = np.asarray(m["T"]), np.asarray(gt[j]["T"])
        ev = evaluate_pose(T_m, T_g, verts)
        ev["axis_err_deg"], ev["center_err_cm"] = symmetric_extras(T_m, T_g)
        ev["t"] = round(m["stamp"] - m3t[0]["stamp"], 2)
        rows.append(ev)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adds = np.array([r["add_s_m"] for r in rows]) * 100
    rot = np.array([r["rot_err_deg"] for r in rows])
    axis = np.array([r["axis_err_deg"] for r in rows])
    cen = np.array([r["center_err_cm"] for r in rows])
    t = np.array([r["t"] for r in rows])

    # 跟丢检测：ADD-S 连续超阈值(10cm)视为跟丢
    lost = adds > 10.0
    summary = {
        "matched_frames": len(rows),
        "m3t_frames": len(m3t),
        "gt_frames": len(gt),
        "add_s_cm_mean": round(float(adds.mean()), 2),
        "add_s_cm_median": round(float(np.median(adds)), 2),
        "add_s_cm_p95": round(float(np.percentile(adds, 95)), 2),
        "axis_err_deg_median(对称物体有效旋转指标)": round(float(np.median(axis)), 1),
        "center_err_cm_median(对称物体有效平移指标)": round(float(np.median(cen)), 2),
        "rot_err_deg_median(含对称歧义虚高,仅参考)": round(float(np.median(rot)), 1),
        "tracking_success_rate(<10cm)": round(float((~lost).mean()), 3),
    }
    with open(out / "m3t_eval.json", "w") as f:
        json.dump({"summary": summary, "frames": rows}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "sans-serif"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    ax[0].plot(t, adds, lw=0.8, color="#27ae60", label="ADD-S")
    ax[0].axhline(10, color="r", ls="--", label="lost threshold 10cm")
    ax[0].set_ylabel("ADD-S (cm)"); ax[0].legend()
    ax[0].set_title(f"M3T tracking vs GT (synthetic seq, {len(rows)} frames, "
                    f"success {summary['tracking_success_rate(<10cm)']*100:.1f}%)")
    ax[1].plot(t, axis, lw=0.8, color="#8e44ad", label="axis error (symmetry-aware)")
    ax[1].plot(t, rot, lw=0.8, color="#c0392b", alpha=0.5,
               label="full rot err (symmetry-inflated)")
    ax[1].set_ylabel("rotation (deg)"); ax[1].legend()
    ax[2].plot(t, cen, lw=0.8, color="#16a085", label="center err (symmetry-aware)")
    ax[2].set_ylabel("center err (cm)"); ax[2].set_xlabel("time (s)"); ax[2].legend()
    fig.tight_layout()
    fig.savefig(out / "m3t_perframe.png", dpi=150)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
