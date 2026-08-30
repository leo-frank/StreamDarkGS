#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR="${IMAGE_DIR:-/home/zxn/disk/Pi3-main/cvpr/our/11/images/}"
TEST_RUN_DIR="${TEST_RUN_DIR:-output/color_cluster/11_sam_full_alpha_debug3_profile_2}"
PROFILE_WORK_DIR="${PROFILE_WORK_DIR:-/tmp/streamdarkgs_profile_online}"
MAX_IMAGES="${MAX_IMAGES:-10}"
PROFILE_MODE="${PROFILE_MODE:-full}"

PI3_ROOT="${PI3_ROOT:-/home/zxn/disk/Pi3-main}"
PI3_CKPT="${PI3_CKPT:-/home/zxn/.cache/huggingface/hub/models--yyfz233--Pi3/snapshots/ae722e7039287d0c8fde9f11f197f804f44b510c/model.safetensors}"
MVINVERSE_CKPT="${MVINVERSE_CKPT:-/home/zxn/.cache/huggingface/hub/models--maddog241--mvinverse/snapshots/ac2d62d9ab2d8e23370dc4de5e6543cd52662c0e}"

PYTHON_BIN="${PYTHON_BIN:-/home/zxn/anaconda3/envs/mvinverse-gsplat-cu118/bin/python}"

INPUT_DIR="${PROFILE_WORK_DIR}/images_${MAX_IMAGES}"
LOG_PATH="${TEST_RUN_DIR}/profile.log"
SUMMARY_PATH="${TEST_RUN_DIR}/profile_summary.txt"

rm -rf "${INPUT_DIR}"
mkdir -p "${INPUT_DIR}" "${TEST_RUN_DIR}"

mapfile -t PROFILE_IMAGES < <(
  find "${IMAGE_DIR}" -maxdepth 1 -type f \
    \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.JPG' -o -name '*.PNG' -o -name '*.JPEG' \) \
    | sort \
    | sed -n "1,${MAX_IMAGES}p"
)
if [[ "${#PROFILE_IMAGES[@]}" -eq 0 ]]; then
  echo "No images found in ${IMAGE_DIR}" >&2
  exit 2
fi
for image_path in "${PROFILE_IMAGES[@]}"; do
  ln -sf "${image_path}" "${INPUT_DIR}/"
done

COMMON_ARGS=(
  live_gaussians_from_rgbd.py
  --image_dir "${INPUT_DIR}"
  --pi3_root "${PI3_ROOT}"
  --pi3_ckpt "${PI3_CKPT}"
  --mvinverse_ckpt "${MVINVERSE_CKPT}"
  --output_path "${TEST_RUN_DIR}/gaussian_map.pt"
  --creation_material_align_to_map
  --creation_material_align_roughness_mode region_constant
  --creation_material_align_region_source sam
  --sam2_ckpt "/home/zxn/disk/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
  --sam2_config "configs/sam2.1/sam2.1_hiera_b+.yaml"
  --sam2_device cuda
  --window_size 10
  --window_stride 8
  --mvinverse_overlap_policy latest
  --pi3_overlap_policy first
  --fusion_frame_stride 2
  --creation_material_align_cluster_min_pixels 512
  --online_global_optimization
  --online_global_optimization_steps 10
  --online_global_optimization_interval 1
  --global_optimization_steps 0
  --global_optimization_normal_weight 1.0
  --global_optimization_surface_normal_weight 0.1
  --global_optimization_alpha_weight 1.0
)

if [[ "${PROFILE_MODE}" == "full" ]]; then
  COMMON_ARGS+=(
    --global_optimization
    --debug_creation_mvinverse_dir "${TEST_RUN_DIR}/mature_material_debug"
    --export_relit_after_fusion
    --export_relit_output_dir "${TEST_RUN_DIR}/relit_flash_full"
  )
elif [[ "${PROFILE_MODE}" == "online_only" ]]; then
  :
else
  echo "PROFILE_MODE must be full or online_only, got: ${PROFILE_MODE}" >&2
  exit 2
fi

echo "[profile-run] mode=${PROFILE_MODE} max_images=${MAX_IMAGES} input=${INPUT_DIR} output=${TEST_RUN_DIR}"
echo "[profile-run] log=${LOG_PATH}"

set +e
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/usr/bin/time -p "${PYTHON_BIN}" "${COMMON_ARGS[@]}" 2>&1 \
  | "${PYTHON_BIN}" -u -c '
import sys
import time

start = time.perf_counter()
for line in sys.stdin:
    elapsed = time.perf_counter() - start
    sys.stdout.write(f"[t={elapsed:9.3f}] {line}")
    sys.stdout.flush()
' | tee "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e

"${PYTHON_BIN}" - "${LOG_PATH}" "${SUMMARY_PATH}" "${MAX_IMAGES}" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
max_images = int(sys.argv[3])

timestamp_re = re.compile(r"^\[t=\s*([0-9.]+)\]\s*(.*)$")
online_re = re.compile(r"\[online-global-opt\].*?observations=(\d+).*?steps=(\d+).*?gaussians=(\d+).*?elapsed=([0-9.]+)s")
window_re = re.compile(r"\[window\]\s+index=(\d+)")
fusion_re = re.compile(r"\[fusion\]\s+(\S+).*?created=(\d+).*?formal=(\d+)")
material_align_re = re.compile(r"\[material-align\]\s+(\S+).*?valid=(\d+).*?clusters=(\d+)")

events = []
real_time = None
for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    match = timestamp_re.match(raw)
    if not match:
        continue
    t = float(match.group(1))
    msg = match.group(2)
    events.append((t, msg))
    if msg.startswith("real "):
        try:
            real_time = float(msg.split()[1])
        except (IndexError, ValueError):
            pass

def first_time(needle: str):
    for t, msg in events:
        if needle in msg:
            return t
    return None

def all_times(needle: str):
    return [(t, msg) for t, msg in events if needle in msg]

window_times = []
current_window = None
for t, msg in events:
    m = window_re.search(msg)
    if m:
        current_window = {"index": int(m.group(1)), "start": t, "pi3_done": None, "mvinverse_done": None}
        window_times.append(current_window)
        continue
    if current_window is not None and "[pi3]" in msg:
        current_window["pi3_done"] = t
        continue
    if current_window is not None and "[mvinverse]" in msg:
        current_window["mvinverse_done"] = t
        current_window = None

online = []
for t, msg in events:
    m = online_re.search(msg)
    if m:
        online.append(
            {
                "t": t,
                "observations": int(m.group(1)),
                "steps": int(m.group(2)),
                "gaussians": int(m.group(3)),
                "elapsed": float(m.group(4)),
            }
        )

fusions = []
for t, msg in events:
    m = fusion_re.search(msg)
    if m:
        fusions.append(
            {
                "t": t,
                "frame": m.group(1),
                "created": int(m.group(2)),
                "formal": int(m.group(3)),
            }
        )

material_aligns = []
for t, msg in events:
    m = material_align_re.search(msg)
    if m:
        material_aligns.append(
            {
                "t": t,
                "frame": m.group(1),
                "valid": int(m.group(2)),
                "clusters": int(m.group(3)),
            }
        )

program_end = real_time if real_time is not None else (events[-1][0] if events else 0.0)
sam_loaded = first_time("[sam2]")
first_window = first_time("[window]")
first_pi3 = first_time("[pi3]")
first_mvinverse = first_time("[mvinverse]")
save_final = first_time("[save] ")

online_total = sum(item["elapsed"] for item in online)
online_avg = online_total / len(online) if online else 0.0

lines = []
lines.append("Profile Summary")
lines.append(f"log: {log_path}")
lines.append(f"frames: {max_images}")
lines.append(f"total_seconds: {program_end:.3f}")
lines.append(f"seconds_per_input_frame: {program_end / max(max_images, 1):.3f}")
if first_window is not None:
    lines.append(f"startup_to_first_window_seconds: {first_window:.3f}")
if sam_loaded is not None:
    lines.append(f"sam_loaded_at_seconds: {sam_loaded:.3f}")
if save_final is not None:
    lines.append(f"first_save_log_at_seconds: {save_final:.3f}")

lines.append("")
lines.append("Window Timing")
for item in window_times:
    pi3_time = None
    mvinverse_time = None
    if item["pi3_done"] is not None:
        pi3_time = item["pi3_done"] - item["start"]
    if item["pi3_done"] is not None and item["mvinverse_done"] is not None:
        mvinverse_time = item["mvinverse_done"] - item["pi3_done"]
    lines.append(
        "window={index} start={start:.3f} pi3_seconds={pi3} mvinverse_seconds={mvinverse}".format(
            index=item["index"],
            start=item["start"],
            pi3=f"{pi3_time:.3f}" if pi3_time is not None else "NA",
            mvinverse=f"{mvinverse_time:.3f}" if mvinverse_time is not None else "NA",
        )
    )

lines.append("")
lines.append("Online Optimization")
lines.append(f"runs: {len(online)}")
lines.append(f"total_seconds_reported: {online_total:.3f}")
lines.append(f"avg_seconds_per_run: {online_avg:.3f}")
lines.append(f"seconds_per_input_frame: {online_total / max(max_images, 1):.3f}")
if online:
    lines.append("runs_detail: " + ", ".join(f"obs={x['observations']}: {x['elapsed']:.3f}s" for x in online))

lines.append("")
lines.append("Fusion")
lines.append(f"creation_frames: {len(fusions)}")
if fusions:
    lines.append(f"last_gaussians: {fusions[-1]['formal']}")
    lines.append("frames: " + ", ".join(f"{x['frame']}(created={x['created']}, formal={x['formal']})" for x in fusions))

lines.append("")
lines.append("Material Align")
lines.append(f"align_calls: {len(material_aligns)}")
if material_aligns:
    lines.append("frames: " + ", ".join(f"{x['frame']}(valid={x['valid']}, clusters={x['clusters']})" for x in material_aligns))

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

exit "${status}"
