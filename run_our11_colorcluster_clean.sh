#!/usr/bin/env bash
set -euo pipefail

#IMAGE_DIR="/home/zxn/disk/darkgs-main/lab1/images"
#SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-output/color_cluster/9_sam_full_alpha_debug23}"
#TEST_RUN_DIR="${TEST_RUN_DIR:-output/color_cluster/9_sam_full_alpha_debug23}"


IMAGE_DIR="/home/zxn/disk/Pi3-main/cvpr/our/11/images/"
#RUN_DIR="output/color_cluster/11_sam_full"
TEST_RUN_DIR="output/color_cluster/11_sam_full_alpha_debug3"
#IMAGE_DIR="/home/zxn/disk/Pi3-main/cvpr/our1/9-new/images/"
#RUN_DIR="output/color_cluster/9-new_sam_full"

PI3_ROOT="/home/zxn/disk/Pi3-main"
PI3_CKPT="/home/zxn/.cache/huggingface/hub/models--yyfz233--Pi3/snapshots/ae722e7039287d0c8fde9f11f197f804f44b510c/model.safetensors"
MVINVERSE_CKPT="/home/zxn/.cache/huggingface/hub/models--maddog241--mvinverse/snapshots/ac2d62d9ab2d8e23370dc4de5e6543cd52662c0e"




PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zxn/anaconda3/envs/mvinverse-gsplat-cu118/bin/python live_gaussians_from_rgbd.py \
  --image_dir "${IMAGE_DIR}" \
  --pi3_root "${PI3_ROOT}" \
  --pi3_ckpt "${PI3_CKPT}" \
  --mvinverse_ckpt "${MVINVERSE_CKPT}" \
  --output_path "${TEST_RUN_DIR}/gaussian_map.pt" \
  --global_optimization \
  --creation_material_align_to_map \
  --creation_material_align_roughness_mode region_constant \
  --creation_material_align_region_source sam \
  --sam2_ckpt "/home/zxn/disk/sam2/checkpoints/sam2.1_hiera_base_plus.pt" \
  --sam2_config "configs/sam2.1/sam2.1_hiera_b+.yaml" \
  --sam2_device cuda \
  --debug_creation_mvinverse_dir "${TEST_RUN_DIR}/mature_material_debug" \
  --export_relit_after_fusion \
  --export_relit_output_dir "${TEST_RUN_DIR}/relit_flash_full" \
  --window_size 10 \
  --window_stride 8 \
  --mvinverse_overlap_policy latest \
  --pi3_overlap_policy first \
  --fusion_frame_stride 2 --creation_material_align_cluster_min_pixels 512 \
  --online_global_optimization \
  --online_global_optimization_steps 10 \
  --online_global_optimization_interval 1 \
  --global_optimization_steps 0 \
  --global_optimization_normal_weight 1.0 \
  --global_optimization_surface_normal_weight 0.1 \
  --global_optimization_alpha_weight 1.0
#  --debug_global_optimization_interval 500 \
#  --skip_online_inference
#  --debug_optimization_inputs_dir "${TEST_RUN_DIR}/optimization_supervision_debug" \
#  --debug_global_optimization_dir "${TEST_RUN_DIR}/global_opt_render_debug" \
