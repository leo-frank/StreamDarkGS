#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 1 ]]; then
  echo "Usage: bash tools/prepare_go2_local_bundle.sh /absolute/path/go2_local_bundle" >&2
  exit 2
fi

DEST="$1"
if [[ "$DEST" != /* ]]; then
  echo "Destination must be an absolute path." >&2
  exit 2
fi
if [[ -e "$DEST" ]]; then
  echo "Refusing to overwrite existing path: $DEST" >&2
  exit 2
fi

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PI3_ROOT=/home/zxn/disk/Pi3-main
SAM2_ROOT=/home/zxn/disk/sam2
GSPLAT_ROOT=/home/zxn/disk/mvinverse-main/third_party/gsplat_legacy_cu118
CONDA_PACK=/home/zxn/anaconda3/bin/conda-pack
ENV_NAME=mvinverse-gsplat-cu118
PI3_WEIGHT=/home/zxn/.cache/huggingface/hub/models--yyfz233--Pi3/snapshots/ae722e7039287d0c8fde9f11f197f804f44b510c/model.safetensors
MV_SNAPSHOT=/home/zxn/.cache/huggingface/hub/models--maddog241--mvinverse/snapshots/ac2d62d9ab2d8e23370dc4de5e6543cd52662c0e
SAM2_WEIGHT="$SAM2_ROOT/checkpoints/sam2.1_hiera_base_plus.pt"

for path in "$PROJECT_ROOT/live_gaussians_from_rgbd.py" "$PI3_ROOT/pi3" \
  "$SAM2_ROOT/sam2" "$SAM2_ROOT/sam2/configs" "$GSPLAT_ROOT/gsplat" \
  "$PI3_WEIGHT" "$MV_SNAPSHOT/model.safetensors" "$MV_SNAPSHOT/config.json" \
  "$SAM2_WEIGHT" "$CONDA_PACK"; do
  if [[ ! -e "$path" ]]; then
    echo "Required source is missing: $path" >&2
    exit 1
  fi
done

mkdir -p "$DEST/StreamDarkGS" "$DEST/Pi3-runtime" "$DEST/sam2-runtime" \
  "$DEST/gsplat-runtime" "$DEST/models/mvinverse" "$DEST/models/pi3" \
  "$DEST/models/sam2"

# The destination is often an exFAT/NTFS portable drive. Dereference source
# symlinks and do not require Unix ownership/permission metadata there.
RSYNC_PORTABLE=(-rltL --no-perms --no-owner --no-group)

rsync "${RSYNC_PORTABLE[@]}" \
  --exclude output --exclude .git --exclude .idea --exclude __pycache__ \
  --exclude certs --exclude '*.log' --exclude zhuan --exclude third_party \
  "$PROJECT_ROOT/" "$DEST/StreamDarkGS/"
rsync "${RSYNC_PORTABLE[@]}" --exclude __pycache__ \
  "$PI3_ROOT/pi3/" "$DEST/Pi3-runtime/pi3/"
rsync "${RSYNC_PORTABLE[@]}" \
  --exclude .git --exclude .idea --exclude checkpoints --exclude __pycache__ \
  "$SAM2_ROOT/" "$DEST/sam2-runtime/"
rsync "${RSYNC_PORTABLE[@]}" \
  --exclude .git --exclude docs --exclude examples --exclude assets \
  --exclude __pycache__ "$GSPLAT_ROOT/" "$DEST/gsplat-runtime/"

# Hugging Face snapshots contain symlinks. -L copies the real multi-GB blobs.
cp -L "$PI3_WEIGHT" "$DEST/models/pi3/model.safetensors"
cp -L "$MV_SNAPSHOT/model.safetensors" "$DEST/models/mvinverse/model.safetensors"
cp -L "$MV_SNAPSHOT/config.json" "$DEST/models/mvinverse/config.json"
cp -L "$SAM2_WEIGHT" "$DEST/models/sam2/sam2.1_hiera_base_plus.pt"

"$CONDA_PACK" -n "$ENV_NAME" --ignore-editable-packages --ignore-missing-files \
  -o "$DEST/mvinverse-gsplat-cu118.tar.gz"

cp "$PROJECT_ROOT/tools/run_go2_local_bundle.sh" "$DEST/run_go2_local_bundle.sh"
cp "$PROJECT_ROOT/docs/go2_local/readme.md" "$DEST/README.md"

(
  cd "$DEST"
  sha256sum \
    mvinverse-gsplat-cu118.tar.gz \
    models/pi3/model.safetensors \
    models/mvinverse/model.safetensors \
    models/mvinverse/config.json \
    models/sam2/sam2.1_hiera_base_plus.pt > SHA256SUMS
)

du -sh "$DEST"
echo "Bundle ready: $DEST"
echo "Copy the whole directory, including hidden files and SHA256SUMS."
