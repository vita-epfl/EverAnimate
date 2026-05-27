from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
import numpy as np
import os
import argparse
import random
import glob
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video_svi import WanVideoSviPipeline, ModelConfig


class StreamingAnimateVideoProcessor:
    def __init__(self, lora_path="", use_anchor=False, seed_multiplier=42, use_pingpong=False, num_motion_frame=1, num_motion_latents=2, num_overlap_frame=1, use_face_anchor=False, debug_save_latents=False, debug_output_dir="debug/latents", num_video_anchor_latents=4, use_zero_padding=False, add_noise_to_motion_latent=False, motion_latent_shared_noise=False, use_image_anchor=False, remove_pose=False, mask_anchor_motion=False, bi_sink=False, use_repeat_anchor=False, enable_anchor_key_focus=False, use_random_frame_anchor=False, random_anchor_frames=3, random_anchor_with_user_first=False):
        self.lora_path = lora_path
        self.pipe = None
        self.initialize_pipeline()
        self.pipe.num_video_anchor_latents = num_video_anchor_latents
        self.pipe.use_zero_padding = use_zero_padding
        self.pipe.add_noise_to_motion_latent = add_noise_to_motion_latent
        self.pipe.motion_latent_shared_noise = motion_latent_shared_noise
        self.pipe.bi_sink = bi_sink
        self.pipe.enable_anchor_key_focus = enable_anchor_key_focus
        if self.pipe.dit is not None:
            self.pipe.dit.enable_anchor_key_focus = enable_anchor_key_focus
        if getattr(self.pipe, "dit2", None) is not None:
            self.pipe.dit2.enable_anchor_key_focus = enable_anchor_key_focus
        
        # Configuration
        self.frames_per_clip = 77  # Output frames per clip (21 latents generated, first anchor latent skipped → 20 latents decoded = 77 frames)
        self.height = 480
        self.width = 832
        self.fps = 25
        self.num_clips = 10  # Default number of clips
        self.use_anchor = use_anchor
        self.use_repeat_anchor = use_repeat_anchor
        self.seed_multiplier = seed_multiplier
        self.use_pingpong = use_pingpong  # Enable ping-pong (forward-backward) looping
        self.num_inference_steps = 20
        self.sigma_shift = 5.0
        self.cfg_scale = 1.0
        self.num_motion_frame = num_motion_frame
        self.num_motion_latents = num_motion_latents
        self.num_overlap_frame = num_overlap_frame
        self.use_face_anchor = use_face_anchor
        self.debug_save_latents = debug_save_latents
        self.debug_output_dir = debug_output_dir
        self.num_video_anchor_latents = num_video_anchor_latents
        self.use_zero_padding = use_zero_padding
        self.add_noise_to_motion_latent = add_noise_to_motion_latent
        self.motion_latent_shared_noise = motion_latent_shared_noise
        self.use_image_anchor = use_image_anchor
        self.enable_anchor_key_focus = enable_anchor_key_focus
        self.use_random_frame_anchor = use_random_frame_anchor
        self.random_anchor_frames = max(0, int(random_anchor_frames))
        self.random_anchor_with_user_first = random_anchor_with_user_first
        self.pipe.remove_pose = remove_pose
        self.pipe.mask_anchor_motion = mask_anchor_motion

    def _encode_user_anchor_single_latent(self, anchor_image):
        """Encode user anchor image into a single temporal latent [C, 1, H, W]."""
        frame = self._to_pil_rgb(anchor_image).resize((self.width, self.height))
        frame_tensor = self.pipe.preprocess_image(frame).to(self.pipe.device).transpose(0, 1).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        frame_latent = self.pipe.vae.encode(
            [frame_tensor],
            device=self.pipe.device,
            tiled=False,
        )[0].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        return frame_latent[:, :1]

    def _encode_frame_to_single_latent(self, frame):
        """Encode one frame into a single temporal VAE latent [C, 1, H, W]."""
        frame = self._to_pil_rgb(frame).resize((self.width, self.height))
        frame_tensor = self.pipe.preprocess_image(frame).to(self.pipe.device).transpose(0, 1).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        frame_latent = self.pipe.vae.encode(
            [frame_tensor],
            device=self.pipe.device,
            tiled=True,
        )[0].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        return frame_latent[:, :1]

    def _normalize_anchor_latent_count(self, latent_slices, target_count):
        """Pad or trim a list of [C,1,H,W] latents so concatenation always matches target_count."""
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        if len(latent_slices) == 0:
            raise ValueError("latent_slices must contain at least one latent")

        normalized = list(latent_slices[:target_count])
        while len(normalized) < target_count:
            normalized.append(normalized[-1].clone())
        return normalized

    def _frames_to_uint8_tensor(self, frames):
        arrays = [np.asarray(self._to_pil_rgb(frame), dtype=np.uint8) for frame in frames]
        if not arrays:
            return torch.empty((0, self.height, self.width, 3), dtype=torch.uint8)
        return torch.from_numpy(np.stack(arrays, axis=0).copy())

    def _uint8_tensor_to_frames(self, frames_tensor):
        frames_tensor = frames_tensor.detach().cpu()
        return [Image.fromarray(frame.numpy()).convert("RGB") for frame in frames_tensor]

    def _capture_rng_state(self):
        state = {
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        if not state:
            return
        if "python_random_state" in state:
            random.setstate(state["python_random_state"])
        if "numpy_random_state" in state:
            np.random.set_state(state["numpy_random_state"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if "torch_cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])

    def _torch_load(self, path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _resume_config(self, input_image_path, pose_video_path, face_video_path, prompt, output_path):
        return {
            "input_image_path": os.path.abspath(input_image_path),
            "pose_video_path": os.path.abspath(pose_video_path),
            "face_video_path": os.path.abspath(face_video_path),
            "prompt": prompt,
            "frames_per_clip": self.frames_per_clip,
            "height": self.height,
            "width": self.width,
            "fps": self.fps,
            "num_clips": self.num_clips,
            "use_pingpong": self.use_pingpong,
            "seed_multiplier": self.seed_multiplier,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "cfg_scale": self.cfg_scale,
            "num_motion_frame": self.num_motion_frame,
            "num_motion_latents": self.num_motion_latents,
            "num_overlap_frame": self.num_overlap_frame,
            "use_face_anchor": self.use_face_anchor,
            "num_video_anchor_latents": self.num_video_anchor_latents,
            "use_image_anchor": self.use_image_anchor,
            "use_repeat_anchor": self.use_repeat_anchor,
            "use_random_frame_anchor": self.use_random_frame_anchor,
            "random_anchor_frames": self.random_anchor_frames,
            "random_anchor_with_user_first": self.random_anchor_with_user_first,
            "add_noise_to_motion_latent": self.add_noise_to_motion_latent,
            "motion_latent_shared_noise": self.motion_latent_shared_noise,
        }

    def _check_resume_config(self, saved_config, current_config):
        if not saved_config:
            return
        mismatches = []
        for key, current_value in current_config.items():
            saved_value = saved_config.get(key)
            if saved_value != current_value:
                mismatches.append((key, saved_value, current_value))
        if mismatches:
            details = "\n".join(
                f"  {key}: checkpoint={saved!r}, current={current!r}"
                for key, saved, current in mismatches
            )
            raise ValueError(
                "Resume checkpoint was created with different generation settings:\n"
                f"{details}\nUse a different --resume_state_dir or restart from clip 0."
            )

    def _frame_chunk_path(self, frame_state_dir, clip_idx):
        return os.path.join(frame_state_dir, f"clip_{clip_idx:03d}_frames.pt")

    def _resume_checkpoint_path(self, resume_state_dir, next_clip_idx):
        return os.path.join(resume_state_dir, f"resume_next_clip_{next_clip_idx:03d}.pt")

    def _save_frame_chunk(self, frame_state_dir, clip_idx, video_frames):
        os.makedirs(frame_state_dir, exist_ok=True)
        path = self._frame_chunk_path(frame_state_dir, clip_idx)
        tmp_path = f"{path}.tmp"
        torch.save({"clip_idx": clip_idx, "video_frames": self._frames_to_uint8_tensor(video_frames)}, tmp_path)
        os.replace(tmp_path, path)
        print(f"Saved resume frame chunk: {path}")

    def _load_frame_chunk(self, frame_state_dir, clip_idx):
        path = self._frame_chunk_path(frame_state_dir, clip_idx)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing resume frame chunk for clip {clip_idx}: {path}")
        chunk = self._torch_load(path)
        return self._uint8_tensor_to_frames(chunk["video_frames"])

    def _latest_resume_checkpoint(self, resume_state_dir):
        if not resume_state_dir or not os.path.isdir(resume_state_dir):
            return None
        candidates = sorted(glob.glob(os.path.join(resume_state_dir, "resume_next_clip_*.pt")))
        if not candidates:
            return None
        return candidates[-1]

    def _save_resume_checkpoint(
        self,
        resume_state_dir,
        next_clip_idx,
        prev_last_latent,
        video_anchor_latent,
        prev_clip_latent_trajectory,
        current_input_image,
        config,
    ):
        os.makedirs(resume_state_dir, exist_ok=True)
        state = {
            "next_clip_idx": next_clip_idx,
            "prev_last_latent": prev_last_latent.detach().cpu() if prev_last_latent is not None else None,
            "video_anchor_latent": video_anchor_latent.detach().cpu() if video_anchor_latent is not None else None,
            "prev_clip_latent_trajectory": [
                latent.detach().cpu() for latent in prev_clip_latent_trajectory
            ] if prev_clip_latent_trajectory is not None else None,
            "current_input_frames": self._frames_to_uint8_tensor(
                current_input_image if isinstance(current_input_image, list) else [current_input_image]
            ),
            "config": config,
            "rng_state": self._capture_rng_state(),
        }
        path = self._resume_checkpoint_path(resume_state_dir, next_clip_idx)
        tmp_path = f"{path}.tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        print(f"Saved resume checkpoint for next clip {next_clip_idx}: {path}")
        
        # Delete older checkpoints to save storage space
        for fname in os.listdir(resume_state_dir):
            if fname.startswith("resume_next_clip_") and fname.endswith(".pt"):
                if fname != os.path.basename(path):
                    try:
                        os.remove(os.path.join(resume_state_dir, fname))
                        print(f"Deleted old checkpoint: {fname}")
                    except Exception as e:
                        print(f"Warning: Failed to delete old checkpoint {fname}: {e}")

    def initialize_pipeline(self):
        """Initialize the WanVideo Animate pipeline"""
        print("Initializing WanVideo Animate pipeline...")
        self.pipe = WanVideoSviPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="Wan2.1_VAE.pth"),
                ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            ],
            redirect_common_files=False,
        )
        
        if self.lora_path and os.path.exists(self.lora_path):
            print(f"Loading LoRA from: {self.lora_path}")
            self.pipe.load_lora(self.pipe.dit, self.lora_path, alpha=1)
        
        print("Pipeline initialized successfully!")
    
    def load_video_clip(self, video_path, start_frame, num_frames, height, width, clip_idx=0):
        """Load a clip from video starting at start_frame with optional ping-pong looping"""
        video_data = VideoData(video_path, height=height, width=width)
        total_frames = len(video_data)
        all_frames = video_data.raw_data()
        
        if not self.use_pingpong:
            # Original simple looping
            end_frame = min(start_frame + num_frames, total_frames)
            actual_frames = end_frame - start_frame
            clip = all_frames[start_frame:end_frame]
            
            if actual_frames < num_frames:
                remaining_frames = num_frames - actual_frames
                loop_clip = all_frames[:remaining_frames]
                clip = clip + loop_clip
                print(f"  Looped video: frames {start_frame}-{end_frame-1} + 0-{remaining_frames-1}")
        else:
            # Ping-pong looping: forward-backward-forward...
            # Create extended sequence: [0,1,2,...,N-1,N-2,...,1] (exclude last frame when reversing to avoid duplicate)
            forward_sequence = all_frames
            backward_sequence = all_frames[-2:0:-1]  # Reverse excluding first and last frame
            pingpong_sequence = forward_sequence + backward_sequence
            pingpong_length = len(pingpong_sequence)
            
            # Determine if this clip should play forward or backward
            is_forward = (clip_idx % 2 == 0)
            
            # Calculate actual position in ping-pong sequence
            actual_start = start_frame % pingpong_length
            
            clip = []
            for i in range(num_frames):
                frame_idx = (actual_start + i) % pingpong_length
                clip.append(pingpong_sequence[frame_idx])
            
            direction = "forward" if is_forward else "backward"
            print(f"  Ping-pong clip {clip_idx} ({direction}): frames {actual_start} to {(actual_start + num_frames - 1) % pingpong_length} in sequence (length={pingpong_length})")
        
        return clip
    
    def concatenate_frames(self, rgb_frames, pose_frames):
        """Concatenate RGB frames (left) and pose frames (right) horizontally"""
        concatenated_frames = []
        for rgb_frame, pose_frame in zip(rgb_frames, pose_frames):
            # Ensure both are RGB PIL images (supports RGBA/gray/CHW/HWC)
            rgb_frame = self._to_pil_rgb(rgb_frame)
            pose_frame = self._to_pil_rgb(pose_frame)
            
            # Resize pose frame to match RGB height
            rgb_w, rgb_h = rgb_frame.size
            pose_frame_resized = pose_frame.resize((rgb_w, rgb_h))
            
            # Create new image with double width
            combined_width = rgb_w * 2
            combined = Image.new('RGB', (combined_width, rgb_h))
            
            # Paste RGB on left, pose on right
            combined.paste(rgb_frame, (0, 0))
            combined.paste(pose_frame_resized, (rgb_w, 0))
            
            concatenated_frames.append(combined)
        
        return concatenated_frames

    def _to_pil_rgb(self, frame):
        """Convert frame in supported formats to RGB PIL image."""
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu()
            frame_np = frame.numpy()
            if frame_np.ndim == 3 and frame_np.shape[0] in (1, 3, 4) and frame_np.shape[-1] not in (1, 3, 4):
                frame_np = np.transpose(frame_np, (1, 2, 0))

            if np.issubdtype(frame_np.dtype, np.floating):
                max_val = float(np.nanmax(frame_np)) if frame_np.size > 0 else 0.0
                if max_val <= 1.0:
                    frame_np = np.clip(frame_np * 255.0, 0, 255)
                else:
                    frame_np = np.clip(frame_np, 0, 255)

            frame_np = frame_np.astype(np.uint8)
            if frame_np.ndim == 3 and frame_np.shape[2] == 1:
                frame_np = frame_np[:, :, 0]
            return Image.fromarray(frame_np).convert("RGB")
        if isinstance(frame, np.ndarray):
            frame_np = frame
            if frame_np.ndim == 3 and frame_np.shape[0] in (1, 3, 4) and frame_np.shape[-1] not in (1, 3, 4):
                frame_np = np.transpose(frame_np, (1, 2, 0))

            if np.issubdtype(frame_np.dtype, np.floating):
                max_val = float(np.nanmax(frame_np)) if frame_np.size > 0 else 0.0
                if max_val <= 1.0:
                    frame_np = np.clip(frame_np * 255.0, 0, 255)
                else:
                    frame_np = np.clip(frame_np, 0, 255)

            frame_np = frame_np.astype(np.uint8)
            if frame_np.ndim == 3 and frame_np.shape[2] == 1:
                frame_np = frame_np[:, :, 0]
            return Image.fromarray(frame_np).convert("RGB")
        raise TypeError(f"Unsupported frame type for anchor selection: {type(frame)}")

    def _pose_foreground_mask(self, frame, threshold=20):
        """Extract a binary foreground mask for colorful 2D skeleton pose frames on black background."""
        pose = np.asarray(self._to_pil_rgb(frame), dtype=np.int16)
        return np.any(pose > threshold, axis=-1)

    def _pose_difference_score(self, reference_frame, candidate_frame):
        """Measure how different a candidate pose is from the reference pose."""
        ref_rgb = np.asarray(self._to_pil_rgb(reference_frame), dtype=np.float32)
        cand_rgb = np.asarray(self._to_pil_rgb(candidate_frame), dtype=np.float32)

        ref_mask = self._pose_foreground_mask(reference_frame)
        cand_mask = self._pose_foreground_mask(candidate_frame)
        union = ref_mask | cand_mask
        if not np.any(union):
            return 0.0

        xor_ratio = float(np.logical_xor(ref_mask, cand_mask).sum()) / float(union.sum())
        color_diff = np.abs(ref_rgb - cand_rgb).mean(axis=-1)
        fg_color_diff = float(color_diff[union].mean()) / 255.0
        return xor_ratio + 0.25 * fg_color_diff

    def _select_pose_triangle_indices(self, pose_frames):
        """Select three pose indices including frame 0 whose pairwise dissimilarity is maximal."""
        total = len(pose_frames)
        if total == 0:
            return []
        if total == 1:
            return [0, 0, 0]
        if total == 2:
            return [0, 1, 1]

        best_pair = None
        best_score = -1.0
        for i in range(1, total):
            diff_0_i = self._pose_difference_score(pose_frames[0], pose_frames[i])
            for j in range(i + 1, total):
                diff_0_j = self._pose_difference_score(pose_frames[0], pose_frames[j])
                diff_i_j = self._pose_difference_score(pose_frames[i], pose_frames[j])
                pairwise_sum = diff_0_i + diff_0_j + diff_i_j
                if pairwise_sum > best_score:
                    best_score = pairwise_sum
                    best_pair = (i, j)

        if best_pair is None:
            fallback = min(2, total - 1)
            return [0, 1, fallback]
        return [0, best_pair[0], best_pair[1]]

    def _select_first_plus_random_indices(self, total):
        """Select first frame + K random unique frames from the rest."""
        if total <= 1:
            return [0]
        k = min(self.random_anchor_frames, total - 1)
        if k <= 0:
            return [0]
        random_ids = np.random.choice(np.arange(1, total), size=k, replace=False)
        return [0] + sorted(int(x) for x in random_ids.tolist())

    def build_video_anchor_latent(self, clip_frames, pose_frames, anchor_image=None):
        """Build video_anchor_latent as [pose-selected frame 1, pose-selected frame 2, user anchor, user anchor]."""
        if anchor_image is None or self.num_video_anchor_latents == 0:
            raise ValueError("anchor_image must be provided and num_video_anchor_latents must be > 0")
        if len(clip_frames) == 0:
            raise ValueError("clip_frames is empty, cannot build video anchor latent")
        if len(pose_frames) == 0:
            raise ValueError("pose_frames is empty, cannot build pose-guided video anchor latent")

        user_anchor_latent = self._encode_user_anchor_single_latent(anchor_image)
        if self.use_repeat_anchor:
            repeated = user_anchor_latent.repeat(1, self.num_video_anchor_latents, 1, 1)
            print(f"Video anchor uses repeated user anchor for all {self.num_video_anchor_latents} slots")
            return repeated

        if self.use_random_frame_anchor:
            if self.random_anchor_with_user_first:
                num_random_slots = max(self.num_video_anchor_latents - 1, 0)
                total = len(clip_frames)
                if total <= 1 or num_random_slots == 0:
                    sampled_latents = []
                    selected = []
                else:
                    candidate_ids = np.arange(1, total)
                    sample_count = min(num_random_slots, len(candidate_ids))
                    selected = sorted(int(x) for x in np.random.choice(candidate_ids, size=sample_count, replace=False).tolist())
                    sampled_latents = [self._encode_frame_to_single_latent(clip_frames[int(i)]) for i in selected]

                latent_slices = sampled_latents + [user_anchor_latent.clone()]
                latent_slices = self._normalize_anchor_latent_count(latent_slices, self.num_video_anchor_latents)
                print(
                    f"Random-anchor mode (random + user first): random indices {selected}, "
                    f"+ 1 user-anchor slot, total slots={self.num_video_anchor_latents}"
                )
            else:
                selected = self._select_first_plus_random_indices(len(clip_frames))
                sampled_latents = [self._encode_frame_to_single_latent(clip_frames[int(i)]) for i in selected]
                latent_slices = self._normalize_anchor_latent_count(sampled_latents, self.num_video_anchor_latents)
                print(f"Random-anchor mode: first+random indices {selected}, slots={self.num_video_anchor_latents}")

            video_anchor_latent = torch.concat(latent_slices, dim=1)
            return video_anchor_latent

        num_user_slots = min(2, self.num_video_anchor_latents)
        num_sample_slots = max(self.num_video_anchor_latents - num_user_slots, 0)

        selected_triplet = self._select_pose_triangle_indices(pose_frames)
        sample_indices = selected_triplet[1:1 + num_sample_slots]
        sampled_latents = [self._encode_frame_to_single_latent(clip_frames[int(idx)]) for idx in sample_indices]
        if num_sample_slots > 0:
            sampled_latents = self._normalize_anchor_latent_count(sampled_latents, num_sample_slots)

        user_latents = [user_anchor_latent.clone() for _ in range(num_user_slots)]
        latent_slices = sampled_latents + user_latents
        latent_slices = self._normalize_anchor_latent_count(latent_slices, self.num_video_anchor_latents)
        video_anchor_latent = torch.concat(latent_slices, dim=1)

        print(
            f"Pose triplet indices {selected_triplet}; using {sample_indices} for the first "
            f"anchor slot(s) and {num_user_slots} user-anchor latent slot(s) at the tail"
        )
        return video_anchor_latent

    def generate_streaming_video(self, input_image_path, pose_video_path, face_video_path, prompt, output_path, resume=False, resume_state_dir=None):
        """Generate streaming video with pose and face control"""
        print(f"\nProcessing video generation...")
        print(f"Input image: {input_image_path}")
        print(f"Pose video: {pose_video_path}")
        print(f"Face video: {face_video_path}")
        print(f"Prompt: {prompt}")
        
        # Load input image (support both image and video files)
        if input_image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            print(f"Input is a video, selecting highest-quality frame as anchor/input...")
            video_data = VideoData(input_image_path, height=self.height, width=self.width)
            input_frames = video_data.raw_data()
            best_idx = 0
            input_image = self._to_pil_rgb(input_frames[best_idx]).resize((self.width, self.height))
            print(f"Selected input frame index {best_idx} from video (first frame, no quality-based filtering)")
        else:
            input_image = Image.open(input_image_path).convert("RGB").resize((self.width, self.height))
        anchor_image = input_image if self.use_anchor else None
        
        # Generate clips
        all_video_frames = []
        all_pose_frames = []  # Store pose frames for concatenation
        current_input_image = input_image
        prev_last_latent = None
        video_anchor_latent = None
        prev_clip_latent_trajectory = None

        base_path = os.path.splitext(output_path)[0]
        ext = os.path.splitext(output_path)[1]
        clip_last_frame_dir = f"{base_path}_last_frames"
        os.makedirs(clip_last_frame_dir, exist_ok=True)
        print(f"Per-clip last frames will be saved to: {clip_last_frame_dir}")
        if resume_state_dir is None:
            resume_state_dir = f"{base_path}_resume_state"
        frame_state_dir = os.path.join(resume_state_dir, "frames")
        save_resume_state = bool(resume)
        config = self._resume_config(input_image_path, pose_video_path, face_video_path, prompt, output_path)
        
        # Calculate frames needed for pose/face videos
        frames_needed_per_clip = self.frames_per_clip

        start_clip_idx = 0
        if resume:
            latest_checkpoint = self._latest_resume_checkpoint(resume_state_dir)
            if latest_checkpoint is None:
                print(f"Resume enabled but no checkpoint found in {resume_state_dir}; starting from clip 0.")
            else:
                state = self._torch_load(latest_checkpoint)
                self._check_resume_config(state.get("config"), config)
                start_clip_idx = int(state["next_clip_idx"])
                prev_last_latent = state.get("prev_last_latent")
                video_anchor_latent = state.get("video_anchor_latent")
                prev_clip_latent_trajectory = state.get("prev_clip_latent_trajectory")
                current_input_image = self._uint8_tensor_to_frames(state["current_input_frames"])
                self._restore_rng_state(state.get("rng_state"))
                print(f"Resuming from checkpoint: {latest_checkpoint}")
                print(f"Next clip index: {start_clip_idx} (1-based clip {start_clip_idx + 1})")

                for old_clip_idx in range(start_clip_idx):
                    old_video_frames = self._load_frame_chunk(frame_state_dir, old_clip_idx)
                    old_start_frame = old_clip_idx * (frames_needed_per_clip - self.num_overlap_frame)
                    old_pose_frames = [
                        self._to_pil_rgb(frame)
                        for frame in self.load_video_clip(
                            pose_video_path,
                            old_start_frame,
                            frames_needed_per_clip,
                            self.height,
                            self.width,
                            clip_idx=old_clip_idx,
                        )
                    ]
                    if old_clip_idx == 0:
                        all_video_frames.extend(old_video_frames)
                        all_pose_frames.extend(old_pose_frames)
                    else:
                        all_video_frames.extend(old_video_frames[self.num_overlap_frame:])
                        all_pose_frames.extend(old_pose_frames[self.num_overlap_frame:])
                print(f"Restored {len(all_video_frames)} accumulated frames from saved chunks.")
        
        for clip_idx in range(start_clip_idx, self.num_clips):
            print(f"\nGenerating clip {clip_idx + 1}/{self.num_clips}...")
            
            # Calculate start frame for this clip in the pose/face videos (with overlap)
            # For clip 0: start = 0 (frames 0-80)
            # For clip 1: start = 81 - num_overlap (frames 76-156 if overlap=5)
            # This ensures the last num_overlap frames of previous clip overlap with first num_overlap frames of current clip
            start_frame = clip_idx * (frames_needed_per_clip - self.num_overlap_frame)
            end_frame = start_frame + frames_needed_per_clip - 1
            
            # Load pose and face video clips
            print(f"Loading pose frames: {start_frame}-{end_frame} (overlap with prev clip: {self.num_overlap_frame if clip_idx > 0 else 0} frames)")
            animate_pose_video = self.load_video_clip(
                pose_video_path, 
                start_frame, 
                frames_needed_per_clip,
                self.height, 
                self.width,
                clip_idx=clip_idx
            )
            
            print(f"Loading face frames: {start_frame}-{end_frame} (overlap with prev clip: {self.num_overlap_frame if clip_idx > 0 else 0} frames)")
            animate_face_video = self.load_video_clip(
                face_video_path,
                start_frame,
                frames_needed_per_clip,
                512,  # Face video uses 512x512
                512,
                clip_idx=clip_idx
            )
            
            # Store pose frames for concatenation (convert to PIL if needed)
            pose_frames_pil = []
            for pose_frame in animate_pose_video:
                pose_frames_pil.append(self._to_pil_rgb(pose_frame))
            
            # Generate video clip
            video_anchor_for_call = None
            if clip_idx > 0 and video_anchor_latent is not None:
                video_anchor_for_call = video_anchor_latent.clone()

            video_clip_dict = self.pipe(
                prompt=prompt,
                seed=clip_idx * self.seed_multiplier,
                # seed=1,
                tiled=False,
                input_image=current_input_image,
                animate_pose_video=animate_pose_video,
                animate_face_video=animate_face_video,
                anchor=anchor_image,
                # num_frames=self.frames_per_clip + 4,  # 77+4=81 → 21 latents; first anchor latent is skipped during decode → 20 latents = 77 output frames
                num_frames=self.frames_per_clip + 4 * self.num_video_anchor_latents,  # 77+4=81 → 21 latents; first anchor latent is skipped during decode → 20 latents = 77 output frames
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                cfg_scale=self.cfg_scale,
                prev_last_latent = prev_last_latent,
                video_anchor_latent = video_anchor_for_call,
                num_motion_latents=self.num_motion_latents,
                use_face_anchor = self.use_face_anchor,
                # Debug parameters
                debug_save_latents=self.debug_save_latents,
                debug_output_dir=self.debug_output_dir,
                clip_idx=clip_idx,
                num_video_anchor_latents=self.num_video_anchor_latents,
                prev_clip_latent_trajectory=prev_clip_latent_trajectory,
                # use_zero_padding=self.use_zero_padding
            )
            
            # Convert video_clip to list of PIL Images if needed
            video_clip = video_clip_dict["video"]
            if self.num_motion_latents > 0:
                prev_last_latent = video_clip_dict["prev_last_latent"]
            prev_clip_latent_trajectory = video_clip_dict.get("latent_trajectory", None)

            if isinstance(video_clip, torch.Tensor):
                video_frames = [self._to_pil_rgb(frame) for frame in video_clip]
            elif isinstance(video_clip, list):
                video_frames = [self._to_pil_rgb(frame) for frame in video_clip]
            else:
                video_frames = [self._to_pil_rgb(video_clip)]

            # Save each clip's last generated RGB frame for quick inspection.
            last_frame_path = os.path.join(clip_last_frame_dir, f"clip_{clip_idx + 1:03d}_last.png")
            video_frames[-1].save(last_frame_path)
            print(f"Saved clip {clip_idx + 1} last frame: {last_frame_path}")

            if clip_idx == 0:
                if self.use_image_anchor:
                    video_anchor_latent = self.build_video_anchor_latent(video_frames, pose_frames_pil, anchor_image=anchor_image)
                else:
                    video_anchor_latent = prev_last_latent
            
            # For first clip, save all frames; for subsequent clips, keep previous clip's last frames and skip new clip's first overlapping frames
            if clip_idx == 0:
                all_video_frames.extend(video_frames)
                all_pose_frames.extend(pose_frames_pil)
                print(f"Clip {clip_idx + 1}: Added all {len(video_frames)} frames (total: {len(all_video_frames)})")
            else:
                # Skip the first num_overlap_frame frames to avoid duplication
                all_video_frames.extend(video_frames[self.num_overlap_frame:])
                all_pose_frames.extend(pose_frames_pil[self.num_overlap_frame:])
                print(f"Clip {clip_idx + 1}: Skipped first {self.num_overlap_frame} overlapping frames, added {len(video_frames) - self.num_overlap_frame} new frames (total: {len(all_video_frames)})")
            
            # Update current input image to last N frames (for multi-frame motion)
            current_input_image = video_frames[-self.num_motion_frame:]
            
            print(f"Total frames accumulated: {len(all_video_frames)}")
            
            # Concatenate RGB and pose frames
            concatenated_frames = self.concatenate_frames(all_video_frames, all_pose_frames)
            
            # Save incremental video after each clip (frames 1 to current clip)
            incremental_output = f"{base_path}_clip1-{clip_idx + 1}{ext}"
            print(f"Saving incremental video with pose comparison: {incremental_output} ({len(concatenated_frames)} frames)")
            save_video(concatenated_frames, incremental_output, fps=self.fps, quality=8)

            if save_resume_state:
                self._save_frame_chunk(frame_state_dir, clip_idx, video_frames)
                self._save_resume_checkpoint(
                    resume_state_dir=resume_state_dir,
                    next_clip_idx=clip_idx + 1,
                    prev_last_latent=prev_last_latent,
                    video_anchor_latent=video_anchor_latent,
                    prev_clip_latent_trajectory=prev_clip_latent_trajectory,
                    current_input_image=current_input_image,
                    config=config,
                )
        
        # Save the final full video to the user-requested output path.
        final_frames = self.concatenate_frames(all_video_frames, all_pose_frames)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_video(final_frames, output_path, fps=self.fps, quality=8)
        print(f"\n✅ Final video saved: {output_path} ({len(all_video_frames)} frames at {self.fps} FPS)")
        print(f"   Video format: RGB (left) + Pose (right)")
        print(f"   Per-clip last frames: {clip_last_frame_dir}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Streaming Animate Video Generation")
    
    # Path arguments
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--pose_video", type=str, required=True, help="Path to pose video")
    parser.add_argument("--face_video", type=str, required=True, help="Path to face video")
    parser.add_argument("--output", type=str, required=True, help="Path to output video")
    parser.add_argument("--lora_path", type=str, default="", help="Path to LoRA weights")
    
    # Generation arguments
    parser.add_argument("--prompt", type=str, default="视频中的人在做动作", help="Text prompt")
    parser.add_argument("--num_clips", type=int, default=10, help="Number of clips to generate")
    parser.add_argument("--frames_per_clip", type=int, default=77, help="Output frames per clip (pipeline generates frames_per_clip+4 latents internally, skips anchor latent on decode)")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS")
    parser.add_argument("--use_face_anchor", action="store_true", help="Use face anchor image")
    parser.add_argument("--use_anchor", action="store_true", help="Use anchor image")
    parser.add_argument("--seed_multiplier", type=int, default=42, help="Seed multiplier")
    parser.add_argument("--use_pingpong", action="store_true", help="Use ping-pong (forward-backward) looping for pose/face videos")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Sigma/timestep shift used by Wan FlowMatch scheduler")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_motion_frame", type=int, default=1, help="Number of frames to look back for the next input image")
    parser.add_argument("--num_motion_latents", type=int, default=1, help="Number of latents to use from input frames")
    parser.add_argument("--num_overlap_frame", type=int, default=4, help="Number of overlapping frames between clips")
    parser.add_argument("--num_video_anchor_latents", type=int, default=4, help="Number of video anchor latents")
    parser.add_argument("--use_zero_padding", action="store_true", help="Use zero padding for video latents")
    parser.add_argument("--use_image_anchor", action="store_true", help="Use highest-quality frames from clip-0 as image-encoded video anchor")
    parser.add_argument("--use_repeat_anchor", action="store_true", help="When using image anchor, don't select reference frames, directly copy the user's provided input image.")
    parser.add_argument("--use_random_frame_anchor", action="store_true", help="Use first frame + random frames as anchors.")
    parser.add_argument("--random_anchor_frames", type=int, default=3, help="Number of random frames (in addition to first frame) for anchor selection.")
    parser.add_argument("--random_anchor_with_user_first", action="store_true", help="In random-anchor mode, reserve one anchor slot for the user-provided first frame latent (e.g., 3 random + 1 user).")
    parser.add_argument("--mask_anchor_motion", "--msk_anchor_motion", dest="mask_anchor_motion", action="store_true", help="Use highest-quality frames from clip-0 as image-encoded video anchor")
    parser.add_argument("--remove_pose", action="store_true", help="Remove pose information from the video")
    parser.add_argument("--bi_sink", action="store_true", help="Place one sink anchor at sequence tail and put the remaining anchor latents at the front.")
    parser.add_argument("--enable_anchor_key_focus", action="store_true", help="Boost anchor-token keys in late self-attention blocks without adding parameters or changing the FlashAttention path.")
    
    # Motion noise arguments
    parser.add_argument("--add_noise_to_motion_latent", action="store_true", help="Add same-sigma noise to motion_latent in y at each denoising step.")
    parser.add_argument("--motion_latent_shared_noise", action="store_true", help="Share the initial x_t noise for motion_latent noising instead of independent random noise.")
    
    # Debug arguments
    parser.add_argument("--debug_save_latents", action="store_true", help="Save latents before VAE decoding for debugging")
    parser.add_argument("--debug_output_dir", type=str, default="debug/latents", help="Directory to save debug latents")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from the latest checkpoint in --resume_state_dir, or start from clip 0 and create checkpoints. Enabled by default.")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="Disable automatic resume checkpoint loading and checkpoint creation.")
    parser.add_argument("--resume_state_dir", type=str, default=None, help="Directory for resume checkpoints and lossless per-clip frame chunks. Defaults to <output>_resume_state.")
    parser.add_argument("--pad_first_clip_with_anchor", action="store_true", help="Pad first clip motion temporal latent with anchor latent")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize processor
    processor = StreamingAnimateVideoProcessor(
        lora_path=args.lora_path,
        use_anchor=True,
        seed_multiplier=args.seed_multiplier,
        use_pingpong=args.use_pingpong,
        num_motion_frame=args.num_motion_frame,
        num_motion_latents=args.num_motion_latents,
        num_overlap_frame=args.num_overlap_frame,
        use_face_anchor=args.use_face_anchor,
        debug_save_latents=args.debug_save_latents,
        debug_output_dir=args.debug_output_dir,
        num_video_anchor_latents=args.num_video_anchor_latents,
        use_zero_padding=args.use_zero_padding,
        add_noise_to_motion_latent=args.add_noise_to_motion_latent,
        motion_latent_shared_noise=args.motion_latent_shared_noise,
        use_image_anchor=args.use_image_anchor,
        remove_pose=args.remove_pose,
        mask_anchor_motion=args.mask_anchor_motion,
        bi_sink=args.bi_sink,
        use_repeat_anchor=args.use_repeat_anchor,
        enable_anchor_key_focus=args.enable_anchor_key_focus,
        use_random_frame_anchor=args.use_random_frame_anchor,
        random_anchor_frames=args.random_anchor_frames,
        random_anchor_with_user_first=args.random_anchor_with_user_first,
    )
    
    # Update configuration
    processor.frames_per_clip = args.frames_per_clip
    processor.height = args.height
    processor.width = args.width
    processor.fps = args.fps
    processor.num_clips = args.num_clips
    processor.num_inference_steps = args.num_inference_steps
    processor.sigma_shift = args.sigma_shift
    processor.cfg_scale = args.cfg_scale
    processor.pipe.pad_first_clip_with_anchor = args.pad_first_clip_with_anchor
    
    # Generate video
    processor.generate_streaming_video(
        input_image_path=args.input_image,
        pose_video_path=args.pose_video,
        face_video_path=args.face_video,
        prompt=args.prompt,
        output_path=args.output,
        resume=args.resume,
        resume_state_dir=args.resume_state_dir,
    )
    
    print("\n🎉 Processing completed!")


if __name__ == "__main__":
    main()
