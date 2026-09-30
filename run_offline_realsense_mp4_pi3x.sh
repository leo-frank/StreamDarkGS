#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_offline_realsense_mp4_pi3x.sh [--prepare-only] [--save-visuals]

Environment overrides:
  VIDEO         Input RGB MP4
  CAMERA_INFO   Matching RealSense color camera JSON
  START_SEC     Starting time in seconds (default: 0)
  DURATION_SEC  Video seconds to use (default: 60; 0 means until the end)
  OUTPUT_FPS    Selected frames per video second (default: 1)
  REPLAY_FPS    Replay throttle (default: 0, as fast as reconstruction allows)
  ROOT_OUT      New output directory (default: generated under StreamDarkGS/output)
EOF
}

PREPARE_ONLY=0
SAVE_VISUALS=0
while (($#)); do
  case "$1" in
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --save-visuals) SAVE_VISUALS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=/home/zxn/anaconda3/envs/mvinverse-gsplat-cu118/bin/python
VIDEO="${VIDEO:-$PROJECT_ROOT/input/rgb.mp4}"
CAMERA_INFO="${CAMERA_INFO:-$PROJECT_ROOT/input/20260507_171419_camera_color.json}"
PI3_ROOT=/home/zxn/disk/Pi3-main
PI3X_ROOT="${PI3X_ROOT:-/home/zxn/disk/Pi3X-main}"
PI3X_CKPT="${PI3X_CKPT:-/home/zxn/disk/Pi3X-main/model.safetensors}"
MVINVERSE_CKPT=/home/zxn/.cache/huggingface/hub/models--maddog241--mvinverse/snapshots/ac2d62d9ab2d8e23370dc4de5e6543cd52662c0e
SAM2_CKPT=/home/zxn/disk/sam2/checkpoints/sam2.1_hiera_base_plus.pt
START_SEC="${START_SEC:-0}"
DURATION_SEC="${DURATION_SEC:-60}"
OUTPUT_FPS="${OUTPUT_FPS:-1}"
REPLAY_FPS="${REPLAY_FPS:-0}"

for required in "$PYTHON" "$VIDEO" "$CAMERA_INFO"; do
  [[ -e "$required" ]] || { echo "Missing: $required" >&2; exit 1; }
done
if [[ "$PREPARE_ONLY" == 0 ]]; then
  for required in "$PI3X_ROOT" "$PI3X_CKPT" "$MVINVERSE_CKPT" "$SAM2_CKPT"; do
    [[ -e "$required" ]] || { echo "Missing: $required" >&2; exit 1; }
  done
fi

mkdir -p "$PROJECT_ROOT/output"
if [[ -n "${ROOT_OUT:-}" ]]; then
  RUN_OUT="$ROOT_OUT"
  if [[ -e "$RUN_OUT" ]]; then
    echo "Output already exists: $RUN_OUT" >&2; exit 1
  fi
  mkdir -p "$RUN_OUT"
else
  RUN_OUT=$(mktemp -d "$PROJECT_ROOT/output/offline_realsense_mp4_XXXXXXXX")
fi

echo "[output] $RUN_OUT"
"$PYTHON" "$PROJECT_ROOT/tools/prepare_offline_realsense_mp4.py" \
  --video "$VIDEO" --camera-info "$CAMERA_INFO" \
  --output-dir "$RUN_OUT/input_frames" \
  --start-sec "$START_SEC" --duration-sec "$DURATION_SEC" \
  --output-fps "$OUTPUT_FPS"

"$PYTHON" - "$RUN_OUT" "$OUTPUT_FPS" "$SAVE_VISUALS" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
fps = float(sys.argv[2])
frames = sum(1 for _ in (out / 'selection.jsonl').open(encoding='utf-8'))
config = {
    'window': 3, 'window_stride': 2, 'create_stride': 2,
    'timing': 'wall', 'mode': 'replay', 'fps': fps,
    'seconds': frames / fps, 'online_optimization': True,
    'online_steps': 10, 'online_interval': 1,
    'online_window_multiplier': 5, 'low_latency_pipeline': True,
    'preview_noncreation_frames': True, 'save_viewer': bool(int(sys.argv[3])),
    'note': 'Offline RealSense MP4; one highest-detail frame per interval.',
}
(out / 'case.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
PY

if [[ "$PREPARE_ONLY" == 1 ]]; then
  echo "[prepared] $RUN_OUT"
  exit 0
fi

VISUAL_ARGS=()
if [[ "$SAVE_VISUALS" == 1 ]]; then
  VISUAL_ARGS=(--save_viewer_dir "$RUN_OUT/viewer_frames")
fi
echo '[reconstruction] model=pi3x intrinsics=sidecar input_mode=replay'
echo '[viewer] http://127.0.0.1:8765/profile?poll=50 (available during reconstruction)'
cd "$PROJECT_ROOT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" live_gaussians_from_rgbd.py \
  --input_mode replay --replay_fps "$REPLAY_FPS" \
  --serve_replay_viewer --stream_host 127.0.0.1 --stream_port 8765 \
  --profile_exit_after_stream \
  --image_dir "$RUN_OUT/input_frames" \
  --geometry_model pi3x --pi3x_intrinsics sidecar \
  --pi3_root "$PI3_ROOT" \
  --pi3x_root "$PI3X_ROOT" --pi3x_ckpt "$PI3X_CKPT" \
  --mvinverse_ckpt "$MVINVERSE_CKPT" \
  --output_path "$RUN_OUT/gaussian_map.pt" \
  --window_size 3 --window_stride 2 --input_frame_stride 1 --fusion_frame_stride 2 \
  --mvinverse_overlap_policy first --pi3_overlap_policy first \
  --no_mvinverse_window_align_to_overlap \
  --creation_material_align_to_map --creation_material_align_region_source sam \
  --creation_material_align_sam_max_color_distance 0.15 \
  --creation_material_align_global_max_log_offset 0.05 \
  --creation_material_align_cluster_max_log_offset 0.8 \
  --creation_material_align_min_valid_ratio 0.5 \
  --creation_material_align_cluster_min_pixels 512 \
  --sam2_ckpt "$SAM2_CKPT" \
  --sam2_config configs/sam2.1/sam2.1_hiera_b+.yaml --sam2_device cuda \
  --global_optimization --global_optimization_steps 0 \
  --online_global_optimization --online_global_optimization_steps 10 \
  --online_global_optimization_interval 1 \
  --online_global_optimization_window_multiplier 5 \
  --low_latency_pipeline --preview_noncreation_frames --profile_timing wall \
  "${VISUAL_ARGS[@]}" 2>&1 | tee "$RUN_OUT/run.log"

"$PYTHON" tools/summarize_go2_run.py --output-dir "$RUN_OUT"
echo "[done] $RUN_OUT"
