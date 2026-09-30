#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT="$BUNDLE_ROOT/StreamDarkGS"
PYTHON="$BUNDLE_ROOT/env/bin/python"
if ! command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_BIN="${FFMPEG_BIN:-/home/igrape/anaconda3/envs/ros1/bin/ffmpeg}"
  if [[ ! -x "$FFMPEG_BIN" ]]; then
    echo "ffmpeg missing; set FFMPEG_BIN to the installed executable" >&2; exit 1
  fi
  export PATH="$(dirname -- "$FFMPEG_BIN"):$PATH"
fi
GEOMETRY_MODEL="${GEOMETRY_MODEL:-pi3x}"
CASE_SELECTOR="${CASE_SELECTOR:-all}"
IMAGE_DIR="${IMAGE_DIR:-$PROJECT_ROOT/output/go2_local_0ci0xxeF/stream_capture_20260924_220301}"
ROOT_OUT="${ROOT_OUT:-$PROJECT_ROOT/output/${GEOMETRY_MODEL}_window_compare_go2_bright_$(date +%Y%m%d_%H%M%S)}"

if [[ "$GEOMETRY_MODEL" != pi3 && "$GEOMETRY_MODEL" != pi3x ]]; then
  echo 'GEOMETRY_MODEL must be pi3 or pi3x' >&2; exit 2
fi
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Image directory missing: $IMAGE_DIR" >&2; exit 1
fi
if [[ "$GEOMETRY_MODEL" == pi3x && ! -s "$BUNDLE_ROOT/models/pi3x/model.safetensors" ]]; then
  echo 'Pi3X checkpoint missing: models/pi3x/model.safetensors' >&2; exit 1
fi
cd "$PROJECT_ROOT"

run_case() {
  local NAME="$1"
  local WINDOW_SIZE="$2"
  local WINDOW_STRIDE="$3"
  local INPUT_STRIDE="$4"
  local OUT="${ROOT_OUT}/${NAME}"
  mkdir -p "${OUT}"
  echo "===== ${NAME}: window=${WINDOW_SIZE}, stride=${WINDOW_STRIDE}, input_stride=${INPUT_STRIDE} ====="

  PYTHONPATH="$BUNDLE_ROOT/sam2-runtime:$BUNDLE_ROOT/gsplat-runtime:${PYTHONPATH:-}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" live_gaussians_from_rgbd.py \
    --input_mode replay \
    --replay_fps 0 \
    --image_dir "${IMAGE_DIR}" \
    --geometry_model "$GEOMETRY_MODEL" \
    --pi3_root "$BUNDLE_ROOT/Pi3-runtime" \
    --pi3_ckpt "$BUNDLE_ROOT/models/pi3/model.safetensors" \
    --pi3x_root "$BUNDLE_ROOT/Pi3X-runtime" \
    --pi3x_ckpt "$BUNDLE_ROOT/models/pi3x/model.safetensors" \
    --mvinverse_ckpt "$BUNDLE_ROOT/models/mvinverse" \
    --output_path "${OUT}/gaussian_map.pt" \
    --global_optimization \
    --creation_material_align_to_map \
    --creation_material_align_roughness_mode region_constant \
    --creation_material_align_region_source sam \
    --creation_material_align_sam_max_color_distance 0.15 \
    --creation_material_align_global_max_log_offset 0.05 \
    --creation_material_align_cluster_max_log_offset 0.8 \
    --sam2_ckpt "$BUNDLE_ROOT/models/sam2/sam2.1_hiera_base_plus.pt" \
    --sam2_config "configs/sam2.1/sam2.1_hiera_b+.yaml" \
    --debug_creation_mvinverse_dir "${OUT}/material_debug" \
    --sam2_device cuda \
    --export_relit_after_fusion \
    --export_relit_output_dir "${OUT}/relit_flash_full" \
    --window_size "${WINDOW_SIZE}" \
    --window_stride "${WINDOW_STRIDE}" \
    --mvinverse_overlap_policy first \
    --input_frame_stride "${INPUT_STRIDE}" \
    --pi3_overlap_policy first \
    --fusion_frame_stride 2 \
    --creation_material_align_cluster_min_pixels 512 \
    --creation_material_align_min_valid_ratio 0.5 \
    --online_global_optimization \
    --online_global_optimization_steps 10 \
    --online_global_optimization_interval 1 \
    --online_global_optimization_window_multiplier 5 \
    --global_optimization_steps 0 \
    --global_optimization_normal_weight 1.0 \
    --global_optimization_surface_normal_weight 0.1 \
    --no_mvinverse_window_align_to_overlap \
    --global_optimization_alpha_weight 1.0 \
    2>&1 | tee "${OUT}/run.log"
}

case "$CASE_SELECTOR" in
  all)
    run_case w10s9_stride1 10 9 1
    run_case w3s2_stride1 3 2 1
    run_case w3s2_stride4_same_span 3 2 4 ;;
  w10s9_stride1) run_case w10s9_stride1 10 9 1 ;;
  w3s2_stride1) run_case w3s2_stride1 3 2 1 ;;
  w3s2_stride4_same_span) run_case w3s2_stride4_same_span 3 2 4 ;;
  *) echo "Unknown CASE_SELECTOR: $CASE_SELECTOR" >&2; exit 2 ;;
esac

echo "All $GEOMETRY_MODEL window comparison results are in: ${ROOT_OUT}"
