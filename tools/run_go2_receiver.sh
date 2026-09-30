#!/usr/bin/env bash
set -euo pipefail
GO2_SAVE_VISUALS=0
if [[ "${1:-}" == "--save-visuals" && "$#" == 1 ]]; then
  GO2_SAVE_VISUALS=1
elif [[ "$#" != 0 ]]; then
  echo 'Usage: bash tools/run_go2_receiver.sh [--save-visuals]' >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p output
GO2_RUN_OUT=$(mktemp -d "$PWD/output/go2_live_XXXXXXXX")
echo "[output] ${GO2_RUN_OUT}"
echo '[viewer] GPU host: http://127.0.0.1:8765/profile?poll=50'
echo '[security] Receiver listens ONLY on GPU-host loopback; use the SSH tunnel.'
GO2_VISUAL_ARGS=()
if [[ "$GO2_SAVE_VISUALS" == 1 ]]; then
  GO2_VISUAL_ARGS=(--save_viewer_dir "${GO2_RUN_OUT}/viewer_frames")
fi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zxn/anaconda3/envs/mvinverse-gsplat-cu118/bin/python live_gaussians_from_rgbd.py \
  --input_mode stream --image_dir "${GO2_RUN_OUT}/input_frames" \
  --stream_host 127.0.0.1 --stream_port 8765 --stream_capture_fps 1 \
  --stream_queue_size 256 --stream_bootstrap_frames 0 --profile_exit_after_stream \
  --pi3_root /home/zxn/disk/Pi3-main \
  --pi3_ckpt /home/zxn/.cache/huggingface/hub/models--yyfz233--Pi3/snapshots/ae722e7039287d0c8fde9f11f197f804f44b510c/model.safetensors \
  --mvinverse_ckpt /home/zxn/.cache/huggingface/hub/models--maddog241--mvinverse/snapshots/ac2d62d9ab2d8e23370dc4de5e6543cd52662c0e \
  --output_path "${GO2_RUN_OUT}/gaussian_map.pt" \
  --window_size 3 --window_stride 2 --input_frame_stride 1 --fusion_frame_stride 2 \
  --mvinverse_overlap_policy first --pi3_overlap_policy first \
  --no_mvinverse_window_align_to_overlap \
  --creation_material_align_to_map --creation_material_align_region_source sam \
  --creation_material_align_sam_max_color_distance 0.15 \
  --creation_material_align_global_max_log_offset 0.05 \
  --creation_material_align_cluster_max_log_offset 0.8 \
  --creation_material_align_min_valid_ratio 0.5 --creation_material_align_cluster_min_pixels 512 \
  --sam2_ckpt ../sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  --sam2_config configs/sam2.1/sam2.1_hiera_b+.yaml --sam2_device cuda \
  --global_optimization --global_optimization_steps 0 \
  --online_global_optimization --online_global_optimization_steps 10 \
  --online_global_optimization_interval 1 --online_global_optimization_window_multiplier 5 \
  --low_latency_pipeline --preview_noncreation_frames --profile_timing wall "${GO2_VISUAL_ARGS[@]}" \
  2>&1 | tee "${GO2_RUN_OUT}/run.log"
/home/zxn/anaconda3/envs/mvinverse-gsplat-cu118/bin/python tools/summarize_go2_run.py --output-dir "${GO2_RUN_OUT}"
echo "[done] ${GO2_RUN_OUT}"
