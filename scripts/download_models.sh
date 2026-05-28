#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p ckpts data

cleanup_hf_locks() {
  find ckpts data -path '*/.cache/huggingface/download/*.lock' -type f -delete 2>/dev/null || true
}

cleanup_hf_locks

echo "Downloading Wan2.2-Animate model files..."
hf download Wan-AI/Wan2.2-Animate-14B \
  --include "diffusion_pytorch_model*.safetensors" \
  --include "models_t5_umt5-xxl-enc-bf16.pth" \
  --include "Wan2.1_VAE.pth" \
  --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --local-dir ckpts/Wan-AI/Wan2.2-Animate-14B

echo "Downloading DiffSynth Wan tokenizer files..."
cleanup_hf_locks
hf download google/umt5-xxl \
  --include "tokenizer*" \
  --include "special_tokens_map.json" \
  --include "spiece.model" \
  --local-dir ckpts/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl

echo "Downloading Wav2Vec processor files used by the training pipeline..."
cleanup_hf_locks
hf download Wan-AI/Wan2.2-S2V-14B \
  --include "wav2vec2-large-xlsr-53-english/**" \
  --local-dir ckpts/Wan-AI/Wan2.2-S2V-14B

echo "Downloading EverAnimate LoRA checkpoints..."
cleanup_hf_locks
hf download epfl-vita/everanimate \
  --repo-type model \
  --include "ckpts/everanimate-v1-lora32/*.safetensors" \
  --local-dir .

echo "Downloading EverAnimate demo data..."
cleanup_hf_locks
hf download epfl-vita/everanimate \
  --repo-type model \
  --include "data/**" \
  --local-dir .

echo "Checking downloaded files..."

required_files=(
  "ckpts/Wan-AI/Wan2.2-Animate-14B/models_t5_umt5-xxl-enc-bf16.pth"
  "ckpts/Wan-AI/Wan2.2-Animate-14B/Wan2.1_VAE.pth"
  "ckpts/Wan-AI/Wan2.2-Animate-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  "ckpts/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json"
  "ckpts/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json"
  "ckpts/everanimate-v1-lora32/stage1_480p.safetensors"
  "ckpts/everanimate-v1-lora32/stage2_480p.safetensors"
  "data/test/frames/demo_000001.png"
  "data/test/poses/demo_000001_poses.mp4"
  "data/test/faces/demo_000001_faces.mp4"
  "data/train/metadata.csv"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

if ! compgen -G "ckpts/Wan-AI/Wan2.2-Animate-14B/diffusion_pytorch_model*.safetensors" >/dev/null; then
  echo "Missing Wan2.2-Animate diffusion_pytorch_model*.safetensors files." >&2
  exit 1
fi

if ! compgen -G "ckpts/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/*" >/dev/null; then
  echo "Missing Wav2Vec processor files under ckpts/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/." >&2
  exit 1
fi

cat <<'MSG'

Model files and demo data are ready.
For offline runs, use:

  export DIFFSYNTH_MODEL_BASE_PATH=$PWD/ckpts
  export DIFFSYNTH_SKIP_DOWNLOAD=True

MSG
