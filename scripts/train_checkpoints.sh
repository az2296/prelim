#!/usr/bin/env bash
#
# Trains every checkpoint wired into m3m4/compare_distributions.ipynb.
#
#   12 diffusion : {M3,M4} x {mlp,film,deep} x {adamw,muon}
#   24 wgan      : {M3,M4} x {mlp,film,deep} x {adamw,muon} x recon {0,0.1}
#   12 wgan      : {M3,M4} x {mlp,film,deep} x recon {0,0.1}, Muon momentum 0
#   ----
#   48 checkpoints total, seed 1.
#
# All WGAN runs use leaky-slope 0.01 and n-critic 1. The "muon m=0" WGANs use
# Muon with momentum 0 on both generator and critic.
#
# Resumable: any checkpoint already on disk is skipped, so it is safe to
# re-run after an interruption. Trains one job at a time.
#
#     bash scripts/train_checkpoints.sh
#     DEVICE=cuda bash scripts/train_checkpoints.sh   # override device (default: mps)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-mps}"
SEED=1
DIFF_DIR="m3m4/diffusion/notebook_ckpts"
WGAN_DIR="m3m4/wgan/notebook_ckpts"
mkdir -p "$DIFF_DIR" "$WGAN_DIR"

# train_diffusion <model> <arch> <optimizer>
train_diffusion() {
  local model="$1" arch="$2" opt="$3"
  local m; m="$(echo "$model" | tr 'A-Z' 'a-z')"
  local ckpt="$DIFF_DIR/${m}_${arch}_${opt}.pt"
  if [ -f "$ckpt" ]; then
    echo "[skip] exists: $ckpt"
    return
  fi
  echo ">>> diffusion | model=$model arch=$arch opt=$opt -> $ckpt"
  python m3m4/diffusion/train.py \
    --model "$model" --arch "$arch" --optimizer "$opt" \
    --start-seed "$SEED" --end-seed "$SEED" \
    --device "$DEVICE" \
    --ckpt "$ckpt" --csv "${ckpt%.pt}.csv" --no-hf-upload
}

# train_wgan <model> <arch> <optimizer> <muon-momentum> <recon-weight> <recon-tag> <filename-tag>
train_wgan() {
  local model="$1" arch="$2" opt="$3" mom="$4" rweight="$5" rtag="$6" tag="$7"
  local m; m="$(echo "$model" | tr 'A-Z' 'a-z')"
  local ckpt="$WGAN_DIR/${m}_${arch}_${tag}_recon${rtag}.pt"
  if [ -f "$ckpt" ]; then
    echo "[skip] exists: $ckpt"
    return
  fi
  echo ">>> wgan | model=$model arch=$arch opt=$tag recon=$rweight -> $ckpt"
  python m3m4/wgan/train.py \
    --model "$model" --arch "$arch" \
    --optimizer-g "$opt" --optimizer-c "$opt" \
    --muon-momentum-g "$mom" --muon-momentum-c "$mom" \
    --recon-weight "$rweight" \
    --n-critic 1 --leaky-slope 0.01 \
    --start-seed "$SEED" --end-seed "$SEED" \
    --device "$DEVICE" \
    --ckpt "$ckpt" --csv "${ckpt%.pt}.csv" --no-hf-upload
}

# --- diffusion: arch x optimizer ------------------------------------------
for model in M3 M4; do
  for arch in mlp film deep; do
    for opt in adamw muon; do
      train_diffusion "$model" "$arch" "$opt"
    done
  done
done

# --- wgan: arch x optimizer-variant x recon-weight ------------------------
# Optimizer variants encoded as "optimizer:muon-momentum:filename-tag":
#   adamw  -> plain AdamW           (momentum arg is unused by AdamW)
#   muon   -> Muon, momentum 0.95   (Muon default)
#   muonm0 -> Muon, momentum 0      (no-momentum)
for model in M3 M4; do
  for arch in mlp film deep; do
    for variant in "adamw:0.95:adamw" "muon:0.95:muon" "muon:0:muonm0"; do
      opt="${variant%%:*}"
      rest="${variant#*:}"
      mom="${rest%%:*}"
      tag="${rest##*:}"
      for recon in "0:0" "0.1:01"; do
        rweight="${recon%%:*}"
        rtag="${recon##*:}"
        train_wgan "$model" "$arch" "$opt" "$mom" "$rweight" "$rtag" "$tag"
      done
    done
  done
done

echo "Done: all 48 checkpoints present in"
echo "  $DIFF_DIR"
echo "  $WGAN_DIR"
