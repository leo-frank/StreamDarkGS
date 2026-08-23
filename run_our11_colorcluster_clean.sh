#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR="data_examples/darkgs_lab1/images"
RUN_DIR="output/color_cluster/darkgs_lab1_clean"
RUN_DIR="output/color_cluster/darkgs_lab1_sam_512"

# IMAGE_DIR="data_examples/home"
# RUN_DIR="output/color_cluster/home_sam_albedo_512"

PI3_ROOT="/home/pdl/liusidun_3d/lwy_3d/pi3_recent"
PI3_CKPT="/home/pdl/liusidun_3d/lwy_3d/Pi3/pi3-model.safetensors"
MVINVERSE_CKPT="/home/pdl/liusidun_3d/lwy_3d/mvinverse/hf_ckpts"


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python live_gaussians_from_rgbd.py \
  --image_dir "${IMAGE_DIR}" \
  --pi3_root "${PI3_ROOT}" \
  --pi3_ckpt "${PI3_CKPT}" \
  --mvinverse_ckpt "${MVINVERSE_CKPT}" \
  --output_path "${RUN_DIR}/gaussian_map.pt" \
  --creation_material_align_to_map \
  --creation_material_align_region_source sam \
  --sam2_ckpt "third_party/sam2/sam2.1_hiera_large.pt" \
  --sam2_config "configs/sam2.1/sam2.1_hiera_l.yaml" \
  --sam2_device cuda \
  --debug_creation_mvinverse_dir "${RUN_DIR}/mature_material_debug" \
  --export_relit_after_fusion \
  --export_relit_output_dir "${RUN_DIR}/relit_flash_full" \
  --no_export_relit_apply_tonemap \
  --window_size 10 \
  --window_stride 8 \
  --mvinverse_overlap_policy latest \
  --fusion_frame_stride 10 --creation_material_align_cluster_min_pixels 512
