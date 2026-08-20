#!/usr/bin/env python3
"""实时可视化：RGB + GT 位姿叠加 + M3T 位姿叠加（同一窗口左右对比）。

不依赖 cv_bridge（规避 OpenCV 版本冲突）。按 q / ESC 退出。
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mycode"))
sys.path.insert(0, str(ROOT / "mycode" / "sim"))
from se3.metrics import rotation_error_deg, translation_error_m  # noqa: E402
from sim_lib import draw_pose_overlay, load_mesh_vertices_faces  # noqa: E402


def _imgmsg_to_bgr(msg: Image) -> np.ndarray:
    h, w = msg.height, msg.width
    if msg.encoding in ("bgr8", "rgb8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        return arr if msg.encoding == "bgr8" else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    raise RuntimeError(f"unsupported encoding: {msg.encoding}")


def _pose_to_T(msg: PoseStamped) -> np.ndarray:
    q = msg.pose.orientation
    w, x, y, z = q.w, q.x, q.y, q.z
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    return T


class LiveViewer(Node):
    def __init__(self) -> None:
        super().__init__("rcia_live_viewer")
        self.lock = threading.Lock()
        self.color = None
        self.T_gt = None
        self.T_m3t = None
        self.state = "NO_TRACK"
        self.verts, _ = load_mesh_vertices_faces()
        self.create_subscription(Image, "/camera/color/image_raw", self._on_color, 10)
        self.create_subscription(PoseStamped, "/energy/synthetic_gt_pose", self._on_gt, 10)
        self.create_subscription(PoseStamped, "/energy/m3t_pose", self._on_m3t, 10)
        # trusted pose if available
        self.create_subscription(PoseStamped, "/energy/trusted_pose", self._on_m3t, 10)
        self.create_subscription(String, "/energy/tracking_state", self._on_state, 10)
        self.timer = self.create_timer(1.0 / 30.0, self._tick)
        self.get_logger().info("live viewer ready — press q/ESC to quit")

    def _on_color(self, msg: Image) -> None:
        with self.lock:
            self.color = _imgmsg_to_bgr(msg)

    def _on_gt(self, msg: PoseStamped) -> None:
        with self.lock:
            self.T_gt = _pose_to_T(msg)

    def _on_m3t(self, msg: PoseStamped) -> None:
        with self.lock:
            self.T_m3t = _pose_to_T(msg)

    def _on_state(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.state = str(data.get("state", data.get("action", msg.data)))
        except Exception:
            self.state = msg.data[:40]

    def _tick(self) -> None:
        with self.lock:
            if self.color is None:
                return
            img = self.color.copy()
            T_gt, T_m3t = self.T_gt, self.T_m3t
            state = self.state

        left = img.copy()
        right = img.copy()
        if T_gt is not None:
            left = draw_pose_overlay(left, self.verts, T_gt, contour_color=(0, 255, 0))
            cv2.putText(left, "GT pose", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(left, "GT waiting...", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        if T_m3t is not None:
            right = draw_pose_overlay(right, self.verts, T_m3t, contour_color=(0, 140, 255))
            cv2.putText(right, "M3T track", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        else:
            cv2.putText(right, "M3T waiting...", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        panel = np.hstack([left, right])
        bar = np.zeros((40, panel.shape[1], 3), np.uint8)
        info = f"state={state}"
        if T_gt is not None and T_m3t is not None:
            info += (f" | rot={rotation_error_deg(T_m3t[:3,:3], T_gt[:3,:3]):.1f}deg"
                     f"  trans={translation_error_m(T_m3t[:3,3], T_gt[:3,3])*100:.1f}cm")
        cv2.putText(bar, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        vis = np.vstack([panel, bar])
        cv2.imshow("RCIA Energy 6DoF Live (GT | M3T)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.get_logger().info("quit requested")
            rclpy.shutdown()


def main() -> None:
    rclpy.init()
    node = LiveViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
