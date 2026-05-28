#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$PWD/ckpts}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-True}"
export TOKENIZERS_PARALLELISM=false

TOKENIZER_DIR="$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
if [[ ! -f "$TOKENIZER_DIR/tokenizer_config.json" ]]; then
  echo "Missing Wan tokenizer files under: $TOKENIZER_DIR" >&2
  echo "Run: bash scripts/download_models.sh" >&2
  exit 1
fi

python scripts/everanimate_inference.py \
  --input_image "${INPUT_IMAGE:-data/test/frames/demo_000001.png}" \
  --pose_video "${POSE_VIDEO:-data/test/poses/demo_000001_poses.mp4}" \
  --face_video "${FACE_VIDEO:-data/test/faces/demo_000001_faces.mp4}" \
  --output "${OUTPUT_PATH:-outputs/test/demo_000001.mp4}" \
  --resume_state_dir "${RESUME_STATE_DIR:-resume_state/test/demo_000001}" \
  --lora_path "${LORA_PATH:-ckpts/everanimate-v1-lora32/stage2_480p.safetensors}" \
  --num_clips "${NUM_CLIPS:-10}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-20}" \
  --use_pingpong \
  --num_overlap_frame 4 \
  --height "${HEIGHT:-480}" \
  --width "${WIDTH:-832}" \
  --frames_per_clip 77 \
  --sigma_shift 5 \
  --use_image_anchor \
  --num_video_anchor_latents 4 \
  --use_random_frame_anchor \
  --random_anchor_with_user_first \
  --random_anchor_frames 3 \
  --num_motion_latents 1 \
  --resume
