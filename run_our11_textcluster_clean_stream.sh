#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR="/home/zxn/disk/Pi3-main/cvpr/our/11/images/"
TEST_RUN_DIR="output/color_cluster/11_sam_full_alpha_debugtext-newnewnew4-deoth1-nopose-stream"

PI3_ROOT="/home/zxn/disk/Pi3-main"
PI3_CKPT="/home/zxn/.cache/huggingface/hub/models--yyfz233--Pi3/snapshots/ae722e7039287d0c8fde9f11f197f804f44b510c/model.safetensors"
MVINVERSE_CKPT="/home/zxn/.cache/huggingface/hub/models--maddog241--mvinverse/snapshots/ac2d62d9ab2d8e23370dc4de5e6543cd52662c0e"

STREAM_HOST="0.0.0.0"
STREAM_PORT=8765
STREAM_CAPTURE_FPS=1
STREAM_JPEG_QUALITY=85
STREAM_QUEUE_SIZE=120

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zxn/anaconda3/envs/mvinverse-gsplat-cu118/bin/python live_gaussians_from_rgbd.py \
  --input_mode stream \
  --image_dir "${IMAGE_DIR}" \
  --stream_host "${STREAM_HOST}" \
  --stream_port "${STREAM_PORT}" \
  --stream_capture_fps "${STREAM_CAPTURE_FPS}" \
  --stream_jpeg_quality "${STREAM_JPEG_QUALITY}" \
  --stream_queue_size "${STREAM_QUEUE_SIZE}" \
  --pi3_root "${PI3_ROOT}" \
  --pi3_ckpt "${PI3_CKPT}" \
  --mvinverse_ckpt "${MVINVERSE_CKPT}" \
  --output_path "${TEST_RUN_DIR}/gaussian_map.pt" \
  --global_optimization \
  --creation_material_align_to_map \
  --creation_material_align_roughness_mode region_constant \
  --creation_material_align_region_source sam \
  --creation_material_align_sam_max_color_distance 0.15 \
  --sam2_ckpt "third_party/texturesam/checkpoints/sam2.1_hiera_small_0.3.pt" \
  --sam2_config "configs/sam2.1/sam2.1_hiera_s.yaml" \
  --sam2_device cuda \
  --no_sam2_apply_postprocessing \
  --sam2_points_per_side 16 \
  --sam2_pred_iou_thresh 0.8 \
  --sam2_stability_score_thresh 0.2 \
  --sam2_mask_threshold 0.0 \
  --sam2_min_mask_region_area 0 \
  --no_sam2_multimask_output \
  --debug_creation_mvinverse_dir "${TEST_RUN_DIR}/mature_material_debug" \
  --export_relit_after_fusion \
  --export_relit_output_dir "${TEST_RUN_DIR}/relit_flash_full" \
  --window_size 10 \
  --window_stride 8 \
  --mvinverse_overlap_policy latest \
  --input_frame_stride 1 \
  --pi3_overlap_policy first \
  --fusion_frame_stride 4 --creation_material_align_cluster_min_pixels 512 \
  --online_global_optimization \
  --online_global_optimization_steps 10 \
  --online_global_optimization_interval 1 \
  --online_global_optimization_window_multiplier 5 \
  --global_optimization_steps 0 \
  --global_optimization_normal_weight 1.0 \
  --global_optimization_surface_normal_weight 0.1 \
  --no_mvinverse_window_align_to_overlap \
  --debug_global_optimization_dir "${TEST_RUN_DIR}/global_opt_render_debug" \
  --stream_certfile "certs/streamdarkgs-server.crt" \
  --stream_keyfile "certs/streamdarkgs-server.key" \
  --global_optimization_alpha_weight 1.0
