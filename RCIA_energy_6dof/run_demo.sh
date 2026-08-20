#!/usr/bin/env bash
# RCIA 能量单元 6DoF — 一键仿真演示（实时窗口：左 GT / 右 M3T）
# 用法:
#   bash run_demo.sh              # 实时可视化（默认）
#   bash run_demo.sh --build      # 先编译 ROS 包再跑
#   bash run_demo.sh --headless 30  # 无窗口，只跑 N 秒并录评测
set -o pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="${RCIA_REPO:-/home/yangyuting/summer_plan/ros2_ws/src/RCIA_Benmao_Vision}"
WS="${ROS_WS:-/home/yangyuting/summer_plan/ros2_ws}"
DATA="$ROOT/data/synthetic_v1"
MESH="$REPO/models/object/Str3D.obj"
LOG="$ROOT/run_logs"
mkdir -p "$LOG"

BUILD=0
HEADLESS=0
DUR=0
for a in "$@"; do
  case "$a" in
    --build) BUILD=1 ;;
    --headless) HEADLESS=1; DUR=30 ;;
    [0-9]*) DUR="$a" ;;
  esac
done

source /opt/ros/humble/setup.bash
# 清掉可能干扰 ROS 的 conda / 用户 site OpenCV
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

if [[ ! -d "$DATA/rgb" ]]; then
  echo "[error] 缺少合成数据: $DATA"
  echo "        先生成:  (unset PYTHONNOUSERSITE; python3 mycode/sim/render_synthetic_seq.py)"
  exit 1
fi
if [[ ! -f "$MESH" ]]; then
  echo "[error] 找不到模型: $MESH  (设置 RCIA_REPO=原仓库路径)"
  exit 1
fi

if [[ "$BUILD" == "1" ]]; then
  echo "[build] rcia_energy_trace_ros ..."
  cd "$WS"
  colcon build --packages-select rcia_energy_trace_ros \
    --base-paths "$REPO/ros2/rcia_energy_trace_ros"
fi
# shellcheck disable=SC1091
source "$WS/install/setup.bash"

cleanup() {
  for pat in synthetic_rgbd_publisher m3t_ros_tracking tracking_supervisor \
             rcia_live_viewer live_viewer m3t_eval_recorder; do
    pkill -f "$pat" 2>/dev/null || true
  done
}
trap cleanup EXIT
cleanup
sleep 0.5

echo "[run] publisher + M3T + supervisor + live viewer"
ros2 run rcia_energy_trace_ros synthetic_rgbd_publisher_node.py \
  --ros-args -p data_dir:="$DATA" -p fps:=30.0 -p loop:=true \
  >"$LOG/publisher.log" 2>&1 &

# 实时 M3T 原生窗口（可选）；主可视化靠 live_viewer
ros2 run rcia_energy_trace_ros m3t_ros_tracking_node \
  --ros-args -p show_m3t_window:=false \
  >"$LOG/m3t.log" 2>&1 &

ros2 run rcia_energy_trace_ros tracking_supervisor_node.py \
  --ros-args \
  -p mesh_file:="$MESH" \
  -p candidate_match_tolerance_ms:=500.0 \
  >"$LOG/supervisor.log" 2>&1 &

sleep 2

if [[ "$HEADLESS" == "1" ]]; then
  echo "[run] headless record ${DUR}s"
  python3 "$ROOT/mycode/ros/m3t_eval_recorder.py" \
    --out "$LOG/m3t_vs_gt.jsonl" --duration "$DUR"
  export MPLBACKEND=Agg
  timeout 60 python3 "$ROOT/mycode/ros/eval_m3t_tracking.py" \
    --jsonl "$LOG/m3t_vs_gt.jsonl" --out "$LOG/m3t_eval" || true
  echo "[done] logs -> $LOG"
  if [[ -f "$LOG/m3t_eval/m3t_eval.json" ]]; then
    python3 -c "import json;print(json.load(open('$LOG/m3t_eval/m3t_eval.json'))['summary'])"
  fi
else
  echo "[viz] OpenCV 窗口: 左=GT位姿  右=M3T跟踪  | 按 q 或 ESC 退出"
  python3 "$ROOT/mycode/ros/live_viewer.py"
fi
