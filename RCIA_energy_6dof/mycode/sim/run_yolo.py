#!/usr/bin/env python3
"""管线第 1 环：YOLO 2D 检测（合成帧）。

复现原仓库 foundationpose_init_node.py 中 YOLO 的角色：
  - 初始化前提供目标 bbox
  - 给 SAM 分割提示
  - 给 supervisor 一个"目标可能存在"的观测信号

同时按原仓库 tip.md 的默认参数 conf=0.75 跑，另测一组 conf=0.25 观察域差影响。
输出：results_v1/yolo/%06d.png 叠加图 + yolo_detections.json + 检出率统计
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))
from sim_lib import REPO_ROOT, load_frame  # noqa: E402

YOLO_WEIGHTS = REPO_ROOT / "models" / "yolo" / "best-v2.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(SIM_DIR.parent.parent / "data" / "synthetic_v1"))
    ap.add_argument("--out", default=str(SIM_DIR.parent.parent / "results_v1" / "yolo"))
    ap.add_argument("--conf", type=float, default=0.75, help="原仓库 tip.md 默认 0.75")
    ap.add_argument("--probe-conf", type=float, default=0.25, help="低阈值探针")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from ultralytics import YOLO

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_frames = len(list((data / "rgb").glob("*.png")))

    model = YOLO(str(YOLO_WEIGHTS))
    records = []
    best_low = []  # 低阈值下每帧最大置信度（用于分析域差）

    for i in range(n_frames):
        rgb, _depth, gt_mask = load_frame(data, i)
        res_hi = model.predict(rgb, conf=args.conf, device=args.device, verbose=False)[0]
        res_lo = model.predict(rgb, conf=args.probe_conf, device=args.device, verbose=False)[0]

        max_lo = float(res_lo.boxes.conf.max()) if len(res_lo.boxes) else 0.0
        best_low.append(max_lo)

        rec = {"frame": i, "conf_threshold": args.conf, "max_low_conf": round(max_lo, 4),
               "boxes": []}
        overlay = rgb.copy()
        for box, conf in zip(res_hi.boxes.xyxy.cpu().numpy(), res_hi.boxes.conf.cpu().numpy()):
            x0, y0, x1, y1 = box.tolist()
            rec["boxes"].append({"xyxy": [round(v, 1) for v in box.tolist()],
                                  "conf": round(float(conf), 3)})
            cv2.rectangle(overlay, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            cv2.putText(overlay, f"YOLO {conf:.2f}", (int(x0), max(18, int(y0) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # 检出框与 GT bbox 的 IoU（评估定位质量）
        ys, xs = np.where(gt_mask > 0)
        if len(xs) and rec["boxes"]:
            gx0, gy0, gx1, gy1 = xs.min(), ys.min(), xs.max(), ys.max()
            b = rec["boxes"][0]["xyxy"]
            ix0, iy0 = max(b[0], gx0), max(b[1], gy0)
            ix1, iy1 = min(b[2], gx1), min(b[3], gy1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union = (b[2]-b[0])*(b[3]-b[1]) + (gx1-gx0)*(gy1-gy0) - inter
            rec["bbox_iou_vs_gt"] = round(inter / union, 3) if union > 0 else 0.0
        cv2.imwrite(str(out / f"{i:06d}.png"), overlay)
        records.append(rec)

    det_rate = sum(bool(r["boxes"]) for r in records) / len(records)
    ious = [r["bbox_iou_vs_gt"] for r in records if "bbox_iou_vs_gt" in r]
    summary = {
        "frames": n_frames,
        "conf_threshold": args.conf,
        "detection_rate": round(det_rate, 3),
        "mean_bbox_iou_vs_gt": round(float(np.mean(ious)), 3) if ious else None,
        "mean_max_low_conf": round(float(np.mean(best_low)), 4),
        "frames_with_lowconf_hit": int(sum(c >= args.probe_conf for c in best_low)),
    }
    with open(out / "yolo_detections.json", "w") as f:
        json.dump({"summary": summary, "frames": records}, f, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
