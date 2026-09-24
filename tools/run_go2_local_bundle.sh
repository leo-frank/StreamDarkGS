#!/usr/bin/env bash
set -euo pipefail

SAVE_VISUALS=0
if [[ "${1:-}" == "--save-visuals" && "$#" == 1 ]]; then
  SAVE_VISUALS=1
elif [[ "$#" != 0 ]]; then
  echo 'Usage: bash run_go2_local_bundle.sh [--save-visuals]' >&2
  exit 2
fi

BUNDLE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$BUNDLE_ROOT/StreamDarkGS"
PYTHON="$BUNDLE_ROOT/env/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Environment is not unpacked. Follow README.md section 4 first." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/output"
RUN_OUT=$(mktemp -d "$PROJECT_ROOT/output/go2_local_XXXXXXXX")
VISUAL_ARGS=()
if [[ "$SAVE_VISUALS" == 1 ]]; then
  VISUAL_ARGS=(--save_viewer_dir "$RUN_OUT/viewer_frames")
fi

echo "[output] $RUN_OUT"
echo '[viewer] http://127.0.0.1:8765/profile?poll=50'
cd "$PROJECT_ROOT"
PYTHONPATH="$BUNDLE_ROOT/sam2-runtime:$BUNDLE_ROOT/gsplat-runtime:${PYTHONPATH:-}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" live_gaussians_from_rgbd.py \
  --input_mode stream --image_dir "$RUN_OUT/input_frames" \
  --stream_host 127.0.0.1 --stream_port 8765 --stream_capture_fps 1 \
  --stream_queue_size 256 --stream_bootstrap_frames 0 --profile_exit_after_stream \
  --pi3_root "$BUNDLE_ROOT/Pi3-runtime" \
  --pi3_ckpt "$BUNDLE_ROOT/models/pi3/model.safetensors" \
  --mvinverse_ckpt "$BUNDLE_ROOT/models/mvinverse" \
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
  --sam2_ckpt "$BUNDLE_ROOT/models/sam2/sam2.1_hiera_base_plus.pt" \
  --sam2_config configs/sam2.1/sam2.1_hiera_b+.yaml --sam2_device cuda \
  --global_optimization --global_optimization_steps 0 \
  --online_global_optimization --online_global_optimization_steps 10 \
  --online_global_optimization_interval 1 \
  --online_global_optimization_window_multiplier 5 \
  --low_latency_pipeline --preview_noncreation_frames --profile_timing wall \
  "${VISUAL_ARGS[@]}" 2>&1 | tee "$RUN_OUT/run.log"

"$PYTHON" tools/summarize_go2_run.py --output-dir "$RUN_OUT"
echo "[done] $RUN_OUT"
