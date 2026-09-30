#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

# Override these from the command line when replaying another capture, e.g.
# GO2_INPUT_DIR=/path/to/capture bash run_go2_field_material_debug.sh
SOURCE_DIR=${GO2_INPUT_DIR:-"$PROJECT_ROOT/input/stream_capture_20260924_220301"}
PYTHON=${GO2_PYTHON:-/home/zxn/anaconda3/envs/mvinverse-gsplat-cu118/bin/python}
PI3_ROOT=${GO2_PI3_ROOT:-/home/zxn/disk/Pi3-main}
PI3_CKPT=${GO2_PI3_CKPT:-/home/zxn/.cache/huggingface/hub/models--yyfz233--Pi3/snapshots/ae722e7039287d0c8fde9f11f197f804f44b510c/model.safetensors}
MVINVERSE_CKPT=${GO2_MVINVERSE_CKPT:-/home/zxn/.cache/huggingface/hub/models--maddog241--mvinverse/snapshots/ac2d62d9ab2d8e23370dc4de5e6543cd52662c0e}
SAM2_CKPT=${GO2_SAM2_CKPT:-/home/zxn/disk/sam2/checkpoints/sam2.1_hiera_base_plus.pt}

for required in "$SOURCE_DIR" "$PYTHON" "$PI3_ROOT/pi3" "$PI3_CKPT" \
  "$MVINVERSE_CKPT/model.safetensors" "$MVINVERSE_CKPT/config.json" "$SAM2_CKPT"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input is missing: $required" >&2
    exit 1
  fi
done

mkdir -p "$PROJECT_ROOT/output"
RUN_OUT=$(mktemp -d "$PROJECT_ROOT/output/go2_field_material_debug_XXXXXXXX——10")
CLEAN_INPUT="$RUN_OUT/input_frames"
mkdir -p "$CLEAN_INPUT"

# Ignore macOS AppleDouble files such as ._stream_*.jpg without modifying the
# original field capture. Natural filename sorting is performed by the pipeline.
while IFS= read -r -d '' image; do
  name=$(basename -- "$image")
  [[ "$name" == ._* ]] && continue
  case "${name,,}" in
    *.jpg|*.jpeg|*.png) ln -s -- "$(realpath -- "$image")" "$CLEAN_INPUT/$name" ;;
  esac
done < <(find "$SOURCE_DIR" -maxdepth 1 -type f -print0)

FRAME_COUNT=$(find "$CLEAN_INPUT" -maxdepth 1 -type l | wc -l)
if (( FRAME_COUNT < 3 )); then
  echo "Need at least 3 valid images, found $FRAME_COUNT in $SOURCE_DIR" >&2
  exit 1
fi

echo "[input]  $SOURCE_DIR"
echo "[frames] $FRAME_COUNT"
echo "[output] $RUN_OUT"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" live_gaussians_from_rgbd.py \
  --input_mode replay \
  --replay_fps 0 \
  --image_dir "$CLEAN_INPUT" \
  --pi3_root "$PI3_ROOT" \
  --pi3_ckpt "$PI3_CKPT" \
  --mvinverse_ckpt "$MVINVERSE_CKPT" \
  --output_path "$RUN_OUT/gaussian_map.pt" \
  --window_size 10 \
  --window_stride 9 \
  --input_frame_stride 1 \
  --fusion_frame_stride 2 \
  --mvinverse_overlap_policy first \
  --pi3_overlap_policy first \
  --no_mvinverse_window_align_to_overlap \
  --creation_material_align_to_map \
  --creation_material_align_region_source sam \
  --creation_material_align_sam_max_color_distance 0.15 \
  --creation_material_align_global_max_log_offset 0.05 \
  --creation_material_align_cluster_max_log_offset 0.8 \
  --creation_material_align_min_valid_ratio 0.5 \
  --creation_material_align_cluster_min_pixels 512 \
  --sam2_ckpt "$SAM2_CKPT" \
  --sam2_config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_device cuda \
  --debug_creation_mvinverse_dir "$RUN_OUT/material_debug" \
  --debug_optimization_inputs_dir "$RUN_OUT/optimization_inputs" \
  --debug_global_optimization_dir "$RUN_OUT/global_optimization_debug" \
  --debug_global_optimization_interval 0 \
  --online_global_optimization \
  --online_global_optimization_steps 10 \
  --online_global_optimization_interval 1 \
  --online_global_optimization_window_multiplier 5 \
  --global_optimization \
  --global_optimization_steps 1000 \
  --global_optimization_normal_weight 1.0 \
  --global_optimization_surface_normal_weight 0.1 \
  --global_optimization_alpha_weight 1.0 \
  --export_relit_after_fusion \
  --export_relit_output_dir "$RUN_OUT/relit_flash_full" \
  --profile_timing wall \
  2>&1 | tee "$RUN_OUT/run.log"

echo "[done] $RUN_OUT"
echo "[material] $RUN_OUT/material_debug"
echo "[relit] $RUN_OUT/relit_flash_full"
