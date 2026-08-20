#!/usr/bin/env python3
"""管线第 2 环：SAM 分割（bbox 提示 → mask）。

复现原仓库链路中 SAM 的角色：YOLO bbox 作 prompt 生成目标 mask，
限制 FoundationPose 注册区域。因 YOLO 在合成域失效（域差，见 yolo_detections.json），
默认用 GT bbox 作 prompt（oracle 模式），量化 SAM 本身的分割质量。

输出：results_v1/sam/%06d.png + sam_masks.json（每帧与 GT mask 的 IoU）
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

SAM_WEIGHTS = REPO_ROOT / "models" / "sam" / "sam_vit_b_01ec64.pth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(SIM_DIR.parent.parent / "data" / "synthetic_v1"))
    ap.add_argument("--out", default=str(SIM_DIR.parent.parent / "results_v1" / "sam"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from segment_anything import SamPredictor, sam_model_registry

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_frames = len(list((data / "rgb").glob("*.png")))

    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_WEIGHTS)).to(args.device)
    predictor = SamPredictor(sam)

    records = []
    for i in range(n_frames):
        rgb, _depth, gt_mask = load_frame(data, i)
        ys, xs = np.where(gt_mask > 0)
        prompt_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.int32)

        predictor.set_image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(box=prompt_box[None], multimask_output=False)
        mask = masks[0].astype(np.uint8)

        inter = int(((mask > 0) & (gt_mask > 0)).sum())
        union = int(((mask > 0) | (gt_mask > 0)).sum())
        iou = round(inter / union, 4) if union else 0.0

        overlay = rgb.copy()
        overlay[mask > 0] = 0.5 * overlay[mask > 0] + \
            0.5 * np.array([0, 200, 255], dtype=np.float32)
        edge = cv2.Canny(mask * 255, 50, 150)
        overlay[edge > 0] = (0, 0, 255)
        cv2.rectangle(overlay, tuple(prompt_box[:2]), tuple(prompt_box[2:]), (255, 0, 0), 1)
        cv2.putText(overlay, f"SAM IoU={iou:.3f} score={float(scores[0]):.2f}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{i:06d}.png"), overlay)
        records.append({"frame": i, "iou_vs_gt": iou,
                        "sam_score": round(float(scores[0]), 3)})

    ious = [r["iou_vs_gt"] for r in records]
    summary = {"frames": n_frames, "prompt": "gt_bbox(oracle)",
               "mean_iou": round(float(np.mean(ious)), 4),
               "min_iou": round(float(np.min(ious)), 4),
               "p95_iou": round(float(np.percentile(ious, 95)), 4)}
    with open(out / "sam_masks.json", "w") as f:
        json.dump({"summary": summary, "frames": records}, f, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
