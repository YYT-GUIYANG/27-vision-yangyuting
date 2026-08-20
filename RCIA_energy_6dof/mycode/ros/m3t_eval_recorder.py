#!/usr/bin/env python3
"""录 /energy/m3t_pose 与 /energy/synthetic_gt_pose，供离线 ADD-S 评估。"""
import argparse
import json
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String


def pose_to_T(msg: PoseStamped):
    import numpy as np
    q = msg.pose.orientation
    p = msg.pose.position
    w, x, y, z = q.w, q.x, q.y, q.z
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [p.x, p.y, p.z]
    return T.tolist()


class Recorder(Node):
    def __init__(self, out_path: str, duration: float) -> None:
        super().__init__("m3t_eval_recorder")
        qos = QoSProfile(depth=50)
        self.out = open(out_path, "w")
        self.t0 = time.time()
        self.duration = duration
        self.done = False
        self.n_m3t = self.n_gt = 0
        self.create_subscription(PoseStamped, "/energy/m3t_pose", self.on_m3t, qos)
        self.create_subscription(PoseStamped, "/energy/synthetic_gt_pose", self.on_gt, qos)
        self.create_subscription(String, "/energy/m3t_status", self.on_status, qos)
        self.create_subscription(String, "/energy/tracking_state", self.on_state, qos)
        self.create_timer(2.0, self.report)

    def _stamp(self, msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def on_m3t(self, msg: PoseStamped) -> None:
        self.n_m3t += 1
        self.out.write(json.dumps({"t": time.time(), "stamp": self._stamp(msg),
                                   "type": "m3t", "T": pose_to_T(msg)}) + "\n")

    def on_gt(self, msg: PoseStamped) -> None:
        self.n_gt += 1
        self.out.write(json.dumps({"t": time.time(), "stamp": self._stamp(msg),
                                   "type": "gt", "T": pose_to_T(msg)}) + "\n")

    def on_status(self, msg: String) -> None:
        self.out.write(json.dumps({"t": time.time(), "type": "m3t_status",
                                   "data": msg.data}) + "\n")

    def on_state(self, msg: String) -> None:
        self.out.write(json.dumps({"t": time.time(), "type": "tracking_state",
                                   "data": msg.data}) + "\n")

    def report(self) -> None:
        left = self.t0 + self.duration - time.time()
        self.get_logger().info(
            f"m3t_poses={self.n_m3t} gt_poses={self.n_gt} ({left:.0f}s left)")
        if left <= 0:
            self.done = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=25.0)
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = Recorder(args.out, args.duration)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        try:
            node.out.close()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f"[recorder] done m3t={node.n_m3t} gt={node.n_gt}")


if __name__ == "__main__":
    main()
