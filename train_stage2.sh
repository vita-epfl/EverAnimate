#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$PWD/ckpts}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_PATH:-experiments/stage2}" logs

accelerate launch --num_processes "${NUM_PROCESSES:-1}" examples/wanvideo/model_training/train_svi.py \
  --dataset_base_path "${DATASET_BASE_PATH:-data/train}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH:-data/train/metadata.csv}" \
  --data_file_keys "video,animate_pose_video,animate_face_video" \
  --max_pixels "${MAX_PIXELS:-400000}" \
  --num_frames "${NUM_FRAMES:-81}" \
  --dataset_repeat "${DATASET_REPEAT:-1}" \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-Animate-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-Animate-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-Animate-14B:Wan2.1_VAE.pth,Wan-AI/Wan2.2-Animate-14B:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --num_epochs "${NUM_EPOCHS:-2}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH:-experiments/stage2}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "${LORA_RANK:-32}" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --num_overlap_frames 4 \
  --num_motion_latents 1 \
  --num_video_anchor_latents 4 \
  --use_aux_video \
  --sigma_shift 5 \
  --rand_aug \
  --video_aug_prob 0.8 \
  --aug_anchor \
  --trajectory_correction_schedule gaussian_timestep \
  --trajectory_correction_weight 1.0 \
  --same_augmentation \
  --lora_checkpoint "${LORA_CHECKPOINT:-ckpts/everanimate-v1-lora32/stage1_480p.safetensors}" \
  --extra_inputs "input_clip,animate_pose_video,animate_face_video,anchor,auxiliary_video"
