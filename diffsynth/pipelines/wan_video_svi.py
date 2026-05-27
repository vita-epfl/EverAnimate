import torch, types
import numpy as np
import os
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
from transformers import Wav2Vec2Processor
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d, compute_svi_attn_mask
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_mot import MotWanModel
from ..models.wav2vec import WanS2VAudioEncoder
from ..models.longcat_video_dit import LongCatVideoTransformer3DModel
import random
from ..utils.visualize_video_tensor import visualize_video_tensor
from dataclasses import dataclass

@dataclass
class SviConfig:
    """Configuration for SVI parameters"""
    enable_image_enhancement: bool = False  # Whether to apply enhancement to image latents (similar to motion latent enhancement)
    image_enhancement_prob: float = 0.9  # Probability of applying image enhancement
    num_video_anchor_latents: int = 4  # Number of video anchor latents to use (1 means only the first frame's latent, 4 means 4 evenly spaced anchor frames per clip)
    same_augmentation: bool = False  # Whether to apply the same random augmentation parameters across all frames in a clip (vs. per-frame randomization)

    # ── Motion latent noise during denoising ─────────────────────────────────
    # Add the same noise level as x_t to motion_latent in y at every denoising step.
    # This closes the train-test gap for motion latents when they are expected to be
    # noisy during generation (e.g. for iterative refinement).
    add_noise_to_motion_latent: bool = False   # Enable noising motion_latent in y at each denoising step
    motion_latent_shared_noise: bool = False   # True: derive noise from initial x_t noise; False: independent random noise
    mask_anchor_motion: bool = False   # True: derive noise from initial x_t noise; False: independent random noise

    use_pose_aug: bool = False
    aug_anchor: bool = False  # Apply random geometric augmentation to sampled anchor frames during training
    pad_first_clip_with_anchor: bool = False  # If True, pad first clip's motion part with first frame latent instead of zeros


class WanVideoSviPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, svi_cfg=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.audio_processor: Wav2Vec2Processor = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.vap: MotWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.audio_encoder: WanS2VAudioEncoder = None
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter", "vap")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter", "vap")
        
        # Set SVI configuration parameters
        if svi_cfg is None:
            svi_cfg = SviConfig()
        
        self.enable_image_enhancement = svi_cfg.enable_image_enhancement
        self.image_enhancement_prob = svi_cfg.image_enhancement_prob
        self.num_video_anchor_latents = svi_cfg.num_video_anchor_latents
        self.same_augmentation = svi_cfg.same_augmentation
        self.use_zero_padding = True
        self.add_noise_to_motion_latent = svi_cfg.add_noise_to_motion_latent
        self.motion_latent_shared_noise = svi_cfg.motion_latent_shared_noise
        self.remove_pose = False  # Will be set from training config
        self.use_pose_aug = svi_cfg.use_pose_aug
        self.mask_anchor_motion = svi_cfg.mask_anchor_motion
        self.aug_anchor = svi_cfg.aug_anchor
        self.pad_first_clip_with_anchor = getattr(svi_cfg, 'pad_first_clip_with_anchor', False)
        self.aug_strength_adapt = False
        self.latent_aug = False

        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            # WanVideoUnit_AnimateVideoSplit(),
            WanVideoUnit_AnimatePoseLatents(),
            WanVideoUnit_AnimateFacePixelValues(),
            WanVideoUnit_AnimateInpaint(),
            WanVideoUnit_VAP(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
            WanVideoUnit_LongCatVideo(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video

    def _add_error_to_y_buffer(self, error_sample, timestep):
        """Add error sample to buffer using specified replacement strategy based on timestep grid."""
        grid_idx = self._get_timestep_grid(timestep)
        error_cpu = error_sample.detach().cpu()
        
        if len(self.y_error_buffer[grid_idx]) < self.error_buffer_size:
            # Buffer not full, simply add
            self.y_error_buffer[grid_idx].append(error_cpu)
        else:
            # Buffer full, use specified replacement strategy
            if self.buffer_replacement_strategy == "random":
                # Random replacement - O(1), fastest
                replace_idx = random.randint(0, len(self.y_error_buffer[grid_idx]) - 1)
                self.y_error_buffer[grid_idx][replace_idx] = error_cpu

            elif self.buffer_replacement_strategy == "fifo":
                # First-in-first-out - O(1), simple queue behavior
                self.y_error_buffer[grid_idx].pop(0)
                self.y_error_buffer[grid_idx].append(error_cpu)
                
            elif self.buffer_replacement_strategy == "l2_batch":
                # Batch L2 computation - O(n) but vectorized, much faster than original
                distances = self._compute_l2_distance_batch(error_cpu, self.y_error_buffer[grid_idx])
                most_similar_idx = torch.argmin(distances).item()
                self.y_error_buffer[grid_idx][most_similar_idx] = error_cpu
                
            elif self.buffer_replacement_strategy == "l2_similarity":
                # Original L2 similarity method - O(n), slowest but most precise
                min_distance = float('inf')
                most_similar_idx = -1
                
                for i, stored_error in enumerate(self.y_error_buffer[grid_idx]):
                    distance = self._compute_l2_distance(error_cpu, stored_error)
                    if distance < min_distance:
                        min_distance = distance
                        most_similar_idx = i
                
                if most_similar_idx != -1:
                    self.y_error_buffer[grid_idx][most_similar_idx] = error_cpu

    def _add_error_to_latent_buffer(self, error_sample, timestep):
        """Add error sample to buffer using specified replacement strategy based on timestep grid."""
        grid_idx = self._get_timestep_grid(timestep)
        error_cpu = error_sample.detach().cpu()
        
        if len(self.latent_error_buffer[grid_idx]) < self.error_buffer_size:
            # Buffer not full, simply add
            self.latent_error_buffer[grid_idx].append(error_cpu)
        else:
            # Buffer full, use specified replacement strategy
            if self.buffer_replacement_strategy == "random":
                # Random replacement - O(1), fastest
                replace_idx = random.randint(0, len(self.latent_error_buffer[grid_idx]) - 1)
                self.latent_error_buffer[grid_idx][replace_idx] = error_cpu
                
            elif self.buffer_replacement_strategy == "fifo":
                # First-in-first-out - O(1), simple queue behavior
                self.latent_error_buffer[grid_idx].pop(0)
                self.latent_error_buffer[grid_idx].append(error_cpu)
                
            elif self.buffer_replacement_strategy == "l2_batch":
                # Batch L2 computation - O(n) but vectorized, much faster than original
                distances = self._compute_l2_distance_batch(error_cpu, self.latent_error_buffer[grid_idx])
                most_similar_idx = torch.argmin(distances).item()
                self.latent_error_buffer[grid_idx][most_similar_idx] = error_cpu
                
            elif self.buffer_replacement_strategy == "l2_similarity":
                # Original L2 similarity method - O(n), slowest but most precise
                min_distance = float('inf')
                most_similar_idx = -1
                
                for i, stored_error in enumerate(self.latent_error_buffer[grid_idx]):
                    distance = self._compute_l2_distance(error_cpu, stored_error)
                    if distance < min_distance:
                        min_distance = distance
                        most_similar_idx = i
                
                if most_similar_idx != -1:
                    self.latent_error_buffer[grid_idx][most_similar_idx] = error_cpu

    def _sample_latent_error_from_latent_buffer(self, latents, timestep):
        """Randomly sample an error from the buffer based on timestep grid."""
        grid_idx = self._get_timestep_grid(timestep)
        
        if not self.latent_error_buffer[grid_idx]:
            return torch.zeros_like(latents)
        
        # Randomly select one sample from the corresponding grid
        selected_sample = random.choice(self.latent_error_buffer[grid_idx])
        error_sample = selected_sample.to(self.device)

        min_mod = 1.0 - self.error_modulate_factor
        max_mod = 1.0 + self.error_modulate_factor
        intensity_mod = random.uniform(min_mod, max_mod)
        error_sample = error_sample * intensity_mod

        return error_sample

    def _sample_y_error_from_y_buffer(self, latents, timestep):
        """Specially sample y_error from buffer - can be configured to sample from all grids or custom range."""
        if self.y_error_sample_range is not None:
            # Sample from custom timestep range
            start_grid, end_grid = self.y_error_sample_range
            all_samples = []
            for grid_idx in range(start_grid, min(end_grid + 1, len(self.y_error_buffer))):
                buffer = self.y_error_buffer[grid_idx]
                if buffer:  # Only add non-empty buffers
                    all_samples.extend(buffer)
            
            if not all_samples:
                return torch.zeros_like(latents)
            
            # Randomly select one sample from the custom range
            selected_sample = random.choice(all_samples)
            
        elif self.y_error_sample_from_all_grids:
            # Sample from all grids that have data
            all_samples = []
            for grid_idx, buffer in self.y_error_buffer.items():
                if buffer:  # Only add non-empty buffers
                    all_samples.extend(buffer)
            
            if not all_samples:
                return torch.zeros_like(latents)
            
            # Randomly select one sample from all available samples
            selected_sample = random.choice(all_samples)
        else:
            # Sample from current timestep grid only (original behavior)
            grid_idx = self._get_timestep_grid(timestep)
            
            if not self.y_error_buffer[grid_idx]:
                return torch.zeros_like(latents)
            
            # Randomly select one sample from the corresponding grid
            selected_sample = random.choice(self.y_error_buffer[grid_idx])
        
        error_sample = selected_sample.to(self.device)

        min_mod = 1.0 - self.error_modulate_factor
        max_mod = 1.0 + self.error_modulate_factor
        intensity_mod = random.uniform(min_mod, max_mod)
        error_sample = error_sample * intensity_mod

        return error_sample

    def _compute_l2_distance(self, tensor1, tensor2):
        """Compute L2 distance between two tensors"""
        # Flatten tensors
        flat1 = tensor1.flatten()
        flat2 = tensor2.flatten()
        
        # Compute L2 distance (Euclidean distance)
        l2_distance = torch.norm(flat1 - flat2, p=2)
        return l2_distance.item()

    def _compute_l2_distance_batch(self, new_tensor, stored_tensors):
        """Compute L2 distances between new tensor and all stored tensors efficiently."""
        if not stored_tensors:
            return torch.tensor([])
        
        # Stack all stored tensors for batch computation
        stored_stack = torch.stack(stored_tensors)  # [num_stored, ...]
        new_flat = new_tensor.flatten()
        stored_flat = stored_stack.flatten(start_dim=1)  # [num_stored, flattened_size]
        
        # Compute L2 distances in batch
        distances = torch.norm(stored_flat - new_flat.unsqueeze(0), p=2, dim=1)
        return distances

    def _pad_or_truncate_to_shape(self, src: torch.Tensor, target_shape: tuple):
        """Pad (by repeating the last element) or truncate src so it matches target_shape.

        For each dimension i, if src.shape[i] < target_shape[i], the last slice along
        that dimension is repeated to grow src to the target size. If src.shape[i] >
        target_shape[i], src is truncated along that dimension.

        This preserves device and dtype.
        """
        # Fast path
        if src.shape == tuple(target_shape):
            return src

        s = src
        # Make sure src has at least as many dims as target
        if s.dim() < len(target_shape):
            for _ in range(len(target_shape) - s.dim()):
                s = s.unsqueeze(-1)

        for dim, t_size in enumerate(target_shape):
            cur_size = s.shape[dim]
            if cur_size == t_size:
                continue
            if cur_size > t_size:
                # truncate
                indices = torch.arange(t_size, device=s.device)
                s = torch.index_select(s, dim, indices)
            else:
                # pad by repeating the last element along this dim
                last = s.narrow(dim, cur_size - 1, 1)
                repeats = [s]
                repeats.extend([last] * (t_size - cur_size))
                s = torch.cat(repeats, dim=dim)

        # If we added extra dims at the end earlier, ensure final shape matches exactly
        if s.shape != tuple(target_shape):
            # final safety: reshape/slice as needed
            slices = [slice(0, sz) for sz in target_shape]
            s = s[tuple(slices)]

        return s
    
    def _get_timestep_grid(self, timestep):
        """Get the grid index for a given timestep."""
        # Handle different timestep formats (scalar tensor, tensor with batch dim, etc.)
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                # Single timestep value
                timestep_val = timestep.item()
            else:
                # Tensor with batch dimension, take the first element
                timestep_val = timestep.flatten()[0].item()
        else:
            # Already a scalar value
            timestep_val = timestep
        
        # Ensure timestep is within valid range and calculate grid index
        timestep_val = max(0, min(timestep_val, 999))  # Clamp to [0, 999]
        grid_idx = torch.argmin((self.inferece_timesteps - timestep_val).abs()).item()

        # Ensure grid index is within valid range
        max_grid_idx = len(self.latent_error_buffer) - 1
        grid_idx = min(grid_idx, max_grid_idx)
        
        return grid_idx

    def _resolve_aug_strength_from_timestep(self, timestep):
        """Map timestep to augmentation strength: high-noise -> strong, low-noise -> weak."""
        if timestep is None or not hasattr(self.scheduler, "timesteps") or len(self.scheduler.timesteps) == 0:
            return 1.0

        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas
        if isinstance(timestep, torch.Tensor):
            ts = timestep.detach().to(device=timesteps.device, dtype=timesteps.dtype).reshape(-1)
        else:
            ts = torch.tensor([timestep], device=timesteps.device, dtype=timesteps.dtype)

        idx = torch.argmin((timesteps - ts[0]).abs())
        sigma_t = sigmas[idx].to(dtype=torch.float32)
        sigma_min = torch.min(sigmas).to(dtype=torch.float32)
        sigma_max = torch.max(sigmas).to(dtype=torch.float32)
        progress = ((sigma_t - sigma_min) / (sigma_max - sigma_min).clamp_min(1e-8)).clamp(0.0, 1.0)

        min_strength = 0.02
        gamma = 1.6
        strength = min_strength + (1.0 - min_strength) * torch.pow(progress, gamma)
        return float(strength.item())

    def apply_augmentation_to_latents(self, latents: torch.Tensor, augmentation_strength: float = 1.0):
        """Apply latent-space corruption analogous to pixel augments (color shift / blur / oversaturation)."""
        strength = float(max(0.0, min(1.0, augmentation_strength)))
        if strength <= 0.0:
            return latents

        x = latents.clone()

        # 1) Color cast equivalent: per-channel gain/bias jitter in latent channels.
        if random.random() < 0.8:
            c = x.shape[1]
            gain = 1.0 + torch.empty((1, c, 1, 1, 1), device=x.device, dtype=x.dtype).uniform_(-0.25, 0.25) * strength
            bias = torch.empty((1, c, 1, 1, 1), device=x.device, dtype=x.dtype).uniform_(-0.08, 0.08) * strength
            x = x * gain + bias

        # 2) Blur equivalent: low-pass in spatial dimensions with strength-controlled blend.
        if random.random() < 0.7:
            k = 3 if random.random() < 0.6 else 5
            pad = k // 2
            x_blur = torch.nn.functional.avg_pool3d(x, kernel_size=(1, k, k), stride=1, padding=(0, pad, pad))
            blur_mix = 0.15 + 0.55 * strength
            x = x * (1.0 - blur_mix) + x_blur * blur_mix

        # 3) Oversaturation equivalent: amplify high-frequency residual in latent space.
        if random.random() < 0.8:
            low = torch.nn.functional.avg_pool3d(x, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
            high = x - low
            hf_alpha = 1.0 + random.uniform(0.1, 1.2) * strength
            x = low + hf_alpha * high

        return x
    
    def _update_error_buffers_local(self, noise_error, y_error, timestep):
        """Update error buffers with samples from local GPU only (post-warmup)."""
        self._add_error_to_latent_buffer(noise_error, timestep)
        self._add_error_to_y_buffer(y_error, timestep)
    
    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
        svi_cfg: SviConfig = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
        # Initialize pipeline
        pipe = WanVideoSviPipeline(device=device, torch_dtype=torch_dtype, svi_cfg=svi_cfg)
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp()
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_pool.fetch_model("wan_video_motion_controller")
        vace = model_pool.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        pipe.vap = model_pool.fetch_model("wan_video_vap")
        pipe.audio_encoder = model_pool.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_pool.fetch_model("wan_video_animate_adapter")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer and processor
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary()
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # VAP
        vap_video: Optional[list[Image.Image]] = None,
        vap_prompt: Optional[str] = " ",
        negative_vap_prompt: Optional[str] = " ",
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # LongCat-Video
        longcat_video: Optional[list[Image.Image]] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        anchor: Optional[Image.Image] = None,
        # Continuous generation
        prev_last_latent: Optional[torch.Tensor] = None,
        num_motion_latents: Optional[int] = 1,
        num_video_anchor_latents: int = 4,
        use_face_anchor: Optional[bool] = False,
        # Debug
        debug_save_latents: Optional[bool] = False,
        debug_output_dir: Optional[str] = "debug/latents",
        clip_idx: Optional[int] = None,
        video_anchor_latent: Optional[torch.Tensor] = None,
        # Trajectory from previous clip for step-aligned motion latent injection
        prev_clip_latent_trajectory: Optional[list] = None,
        
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "vap_prompt": vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_vap_prompt": negative_vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
            "vap_video": vap_video, "anchor": anchor, "inference_mode": True,
            "prev_last_latent": prev_last_latent, "video_anchor_latent": video_anchor_latent,
            "num_video_anchor_latents": num_video_anchor_latents,
        }
        
        self.num_motion_latents = num_motion_latents

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Pre-generate motion latent noise (used only when add_noise_to_motion_latent is enabled, training / first-clip fallback)
        clean_motion_latent = inputs_shared.get("clean_motion_latent", None)
        if self.add_noise_to_motion_latent and clean_motion_latent is not None:
            if self.motion_latent_shared_noise:
                # Derive from the initial x_t noise: take first num_motion_latents temporal frames
                init_noise = inputs_shared.get("noise", None)
                if init_noise is not None:
                    motion_latent_noise = init_noise[0, :, :clean_motion_latent.shape[1], :, :].to(device=self.device, dtype=self.torch_dtype)
                else:
                    motion_latent_noise = torch.randn_like(clean_motion_latent)
            else:
                motion_latent_noise = torch.randn_like(clean_motion_latent)
            inputs_shared["_motion_latent_noise"] = motion_latent_noise

        # Collect per-step latent trajectory so the next clip can reuse it
        latent_trajectory = []

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * 1000 and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Record current x_t (before denoising) for trajectory output
            if self.add_noise_to_motion_latent:
                num_motion_latents = self.num_motion_latents
                latent_trajectory.append(
                    inputs_shared["latents"][0, :, -num_motion_latents:, :, :].detach().cpu()
                )  # [C, num_motion_latents, H, W]

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
            if vace_reference_image is not None and isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, num_video_anchor_latents:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])

        # Debug: Save latents before decoding
        if debug_save_latents:
            import os
            from datetime import datetime
            os.makedirs(debug_output_dir, exist_ok=True)
            
            # Build descriptive filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            latent_shape = "x".join([str(s) for s in inputs_shared["latents"].shape])
            
            filename_parts = [f"latent_{timestamp}"]
            if clip_idx is not None:
                filename_parts.append(f"clip{clip_idx:03d}")
            if seed is not None:
                filename_parts.append(f"seed{seed}")
            filename_parts.append(f"shape_{latent_shape}")
            
            filename = "_".join(filename_parts) + ".pt"
            save_path = os.path.join(debug_output_dir, filename)
            
            torch.save(inputs_shared["latents"].cpu(), save_path)
            print(f"[DEBUG] Saved latents to: {save_path}")

        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        # Save last latent for continuous generation
        # prev_last_latent = inputs_shared["latents"][:, :, -1:].detach()[0]
        prev_last_latent = inputs_shared["latents"].detach()[0]
        # return video, prev_last_latent
        return dict(video=video, prev_last_latent=prev_last_latent, latent_trajectory=latent_trajectory if latent_trajectory else None)



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoSviPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoSviPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image", "anchor", "train_sampled_timestep"),
            output_params=("latents", "input_latents", "input_latents_aug"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoSviPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image, anchor=None, train_sampled_timestep=None):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        
        anchor = None

        input_video_frames = input_video

        if anchor is not None:
            input_video = pipe.preprocess_video(input_video[:-4])
            anchor = pipe.preprocess_image(anchor).to(pipe.device)
            anchor_latent = pipe.vae.encode(
                [anchor.transpose(0, 1).to(dtype=pipe.torch_dtype, device=pipe.device)],
                device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            )[0].to(device=pipe.device)
            input_latents = torch.concat([anchor_latent.unsqueeze(0), pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)], dim=2)

        else:
            input_video = pipe.preprocess_video(input_video)
            input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

        input_latents_aug = None
        if (
            pipe.scheduler.training
            and (getattr(pipe, "rand_aug", False) or getattr(pipe, "mixed_aug", False))
            and input_video_frames is not None
        ):
            video_aug_prob = float(getattr(pipe, "video_aug_prob", 1.0))
            rand_aug_sigma_threshold = float(getattr(pipe, "rand_aug_sigma_threshold", 0.5))
            use_input_aug = True
            if train_sampled_timestep is not None and rand_aug_sigma_threshold > 0.0:
                timesteps = pipe.scheduler.timesteps
                sigmas = pipe.scheduler.sigmas
                ts = train_sampled_timestep.detach().to(device=timesteps.device, dtype=timesteps.dtype).reshape(-1)
                idx = torch.argmin((timesteps - ts[0]).abs())
                sigma_t = sigmas[idx].to(dtype=torch.float32)
                use_input_aug = bool((sigma_t >= rand_aug_sigma_threshold).item())
            if use_input_aug and random.random() < video_aug_prob:
                aug_strength = 1.0
                if bool(getattr(pipe, "aug_strength_adapt", False)):
                    aug_strength = pipe._resolve_aug_strength_from_timestep(train_sampled_timestep)
                if bool(getattr(pipe, "latent_aug", False)):
                    input_latents_aug = pipe.apply_augmentation_to_latents(
                        input_latents,
                        augmentation_strength=aug_strength,
                    ).to(dtype=pipe.torch_dtype, device=pipe.device)
                else:
                    # aug_frames = pipe.apply_augmentation_to_images_condition(
                    aug_frames = pipe.apply_augmentation_to_images(
                        input_video_frames,
                        same_augmentation=pipe.same_augmentation,
                        k=0,
                        augmentation_strength=aug_strength,
                    )
                    augmented_video = [img.to(pipe.device).transpose(0, 1) for img in aug_frames]
                    augmented_video = torch.cat(augmented_video, dim=1)  # [3, T, H, W]
                    input_latents_aug = pipe.vae.encode(
                        [augmented_video.to(dtype=pipe.torch_dtype, device=pipe.device)],
                        device=pipe.device,
                        tiled=tiled,
                        tile_size=tile_size,
                        tile_stride=tile_stride,
                    )[0].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
            if input_latents_aug is not None:
                input_latents_aug = torch.concat([vace_reference_latents, input_latents_aug], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents, "input_latents_aug": input_latents_aug}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: WanVideoSviPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoSviPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoSviPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        input_image = input_image[0] if isinstance(input_image, list) else input_image
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image","anchor", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride","inference_mode", "prev_last_latent", "auxiliary_video", "video_anchor_latent"),
            output_params=("y", "clean_motion_latent"),
            onload_model_names=("vae",)
        )

    def _sample_anchor_aug_theta(self, frame_bchw, pipe: WanVideoSviPipeline):
        # Translation is normalized in affine_grid coordinates; keep this mild
        # because anchor latents are conditioning tokens, not supervision targets.

        scale_min = float(getattr(pipe, "anchor_aug_scale_min", 0.8))
        scale_max = float(getattr(pipe, "anchor_aug_scale_max", 1.2))
        max_translation = float(getattr(pipe, "anchor_aug_max_translation", 0.10))
        scale = random.uniform(scale_min, scale_max)
        tx = random.uniform(-max_translation, max_translation)
        ty = random.uniform(-max_translation, max_translation)
        return torch.tensor(
            [[[scale, 0.0, tx], [0.0, scale, ty]]],
            dtype=frame_bchw.dtype,
            device=frame_bchw.device,
        )

    def _augment_anchor_frame(self, frame, pipe: WanVideoSviPipeline, theta=None):
        # frame: [3, 1, H, W] in normalized range, apply shared anchor affine augmentation.
        frame_bchw = frame.transpose(0, 1)
        if theta is None:
            theta = self._sample_anchor_aug_theta(frame_bchw, pipe)
        grid = torch.nn.functional.affine_grid(theta, frame_bchw.shape, align_corners=False)
        # Use neutral padding for out-of-frame samples. The image tensor is in
        # [-1, 1], so grid_sample's zero padding corresponds to mid-gray, not black.
        aug_frame = torch.nn.functional.grid_sample(
            frame_bchw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        aug_frame = torch.nan_to_num(aug_frame, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return aug_frame.transpose(0, 1)

    def process(self, pipe: WanVideoSviPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride, anchor=None, inference_mode=False, prev_last_latent=None, auxiliary_video=None, video_anchor_latent=None):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        
        if anchor is None:
            anchor = input_image[0] if isinstance(input_image, list) else input_image
            
        pipe.load_models_to_device(self.onload_model_names)
        training_mode = not inference_mode
        num_video_anchor_latents = pipe.num_video_anchor_latents
        num_motion_latents = pipe.num_motion_latents

        is_first_clip = (training_mode and random.random() < 0.2) or (inference_mode and prev_last_latent is None)

        if inference_mode:
            num_frames = 81
        total_latents = (num_frames - 1) // 4 + 1

        anchor = anchor.resize((width, height))
        anchor_input = pipe.preprocess_image(anchor).to(pipe.device).transpose(0, 1).to(dtype=pipe.torch_dtype, device=pipe.device)  # [3, 1, H, W]    
        anchor_latent = pipe.vae.encode(
            [anchor_input],
            device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )[0].to(device=pipe.device)  
        # Build latent sequence based on clip type
        if is_first_clip:

            anchor_vid_latent = anchor_latent.repeat(1, num_video_anchor_latents, 1, 1)
            
            if getattr(pipe, "pad_first_clip_with_anchor", False):
                motion_latent = anchor_latent.repeat(1, num_motion_latents, 1, 1)
            else:
                motion_latent = anchor_latent.new_zeros(
                    anchor_latent.shape[0],
                    num_motion_latents,
                    anchor_latent.shape[2],
                    anchor_latent.shape[3],
                )
                
            padding_size = total_latents - num_motion_latents - 1
            zero_padding = anchor_latent.new_zeros(
                anchor_latent.shape[0],
                padding_size,
                anchor_latent.shape[2],
                anchor_latent.shape[3],
            )
            y = torch.concat([anchor_vid_latent, motion_latent, zero_padding], dim=1)

        else:
            # test
            if prev_last_latent is not None and video_anchor_latent is not None:
                # Reuse motion latent from previous generation (inference path)
                prev_vid_latent = prev_last_latent.to(device=pipe.device)
                anchor_vid_latent = video_anchor_latent.to(device=pipe.device)
            else:
                # for training
                prev_chunk_resized = [img.resize((width, height)) for img in auxiliary_video]
                prev_chunk = [pipe.preprocess_image(img).transpose(0, 1).to(pipe.device) for img in prev_chunk_resized]
                prev_chunk = torch.cat(prev_chunk, dim=1)  # Shape: [3, num_frames, H, W]

                if random.random() < pipe.image_enhancement_prob and pipe.enable_image_enhancement:
                    # augmented_previous_chunk = pipe.apply_augmentation_to_images_condition(prev_chunk_resized, same_augmentation=pipe.same_augmentation)
                    augmented_previous_chunk = pipe.apply_augmentation_to_images(prev_chunk_resized, same_augmentation=pipe.same_augmentation)
                    augmented_previous_chunk = [img.to(pipe.device).transpose(0, 1) for img in augmented_previous_chunk]
                    augmented_previous_chunk = torch.cat(augmented_previous_chunk, dim=1)  # Shape: [3, num_frames, H, W]
                    prev_vid_latent = pipe.vae.encode(
                        [augmented_previous_chunk.to(dtype=pipe.torch_dtype, device=pipe.device)],
                        device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                    )[0].to(device=pipe.device)
                else:
                    prev_vid_latent = pipe.vae.encode(
                        [prev_chunk.to(dtype=pipe.torch_dtype, device=pipe.device)],
                        device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                    )[0].to(device=pipe.device)

                if random.random() < 0.95:
                    
                    num_input_frames = len(auxiliary_video)
                    # num_random_frames = max(0, num_video_anchor_latents - 2)
                    num_random_frames = max(0, num_video_anchor_latents)
                    selected_indices = sorted(random.sample(range(num_input_frames), min(num_random_frames, num_input_frames)))
                    
                    encoded_frames = []
                    apply_anchor_aug = (
                        training_mode
                        and getattr(pipe, "aug_anchor", False)
                        and selected_indices
                        and random.random() < float(getattr(pipe, "anchor_aug_prob", 0.5))
                    )
                    shared_anchor_theta = None
                    if apply_anchor_aug:
                        ref_frame = prev_chunk[:, selected_indices[0]:selected_indices[0]+1, :, :]
                        shared_anchor_theta = self._sample_anchor_aug_theta(ref_frame.transpose(0, 1), pipe)
                    for idx in selected_indices:
                        frame = prev_chunk[:, idx:idx+1, :, :]  # [3, 1, H, W]
                        if apply_anchor_aug:
                            frame = self._augment_anchor_frame(frame, pipe, theta=shared_anchor_theta)
                        frame_latent = pipe.vae.encode(
                            [frame.to(dtype=pipe.torch_dtype, device=pipe.device)],
                            device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                        )[0].to(device=pipe.device)  # [C, 1, h, w]
                        encoded_frames.append(frame_latent)
        
                    anchor_vid_latent = torch.concat(encoded_frames, dim=1)
                else:
                    anchor_vid_latent = anchor_latent.repeat(1, num_video_anchor_latents, 1, 1)

            motion_latent = prev_vid_latent[:, -pipe.num_motion_latents:]
            # anchor_vid_latent = anchor_vid_latent[:, :num_video_anchor_latents]

            padding_size = total_latents - motion_latent.shape[1] - 1
            zero_padding = anchor_vid_latent.new_zeros(
                anchor_vid_latent.shape[0],
                padding_size,
                anchor_vid_latent.shape[2],
                anchor_vid_latent.shape[3],
            )
            
            y = torch.concat([anchor_vid_latent, motion_latent, zero_padding], dim=1)


        num_frames += 4 * (num_video_anchor_latents-1)    

        # Create frame mask (1 for first frame, 0 for rest)
        mask_condition = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        mask_condition[:, 1:] = 0
        mask_condition = torch.concat([torch.repeat_interleave(mask_condition[:, 0:1], repeats=4, dim=1), mask_condition[:, 1:]], dim=1)
        mask_condition = mask_condition.view(1, mask_condition.shape[1] // 4, 4, height//8, width//8)
        mask_condition = mask_condition.transpose(1, 2)[0]
        mask_condition[:, :num_video_anchor_latents] = 1 

        y = torch.concat([mask_condition, y]).unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        
        return {"y": y,  "anchor_latent_vid": anchor_vid_latent,
                "clean_motion_latent": motion_latent if not is_first_clip else None}

class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoSviPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}

class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            output_params=("clip_feature", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoSviPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            output_params=("reference_latents", "clip_feature"),
            onload_model_names=("vae", "image_encoder")
        )

    def process(self, pipe: WanVideoSviPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("control_camera_latents_input", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoSviPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            mask_condition = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            mask_condition[:, 1:] = 0
            mask_condition = torch.concat([torch.repeat_interleave(mask_condition[:, 0:1], repeats=4, dim=1), mask_condition[:, 1:]], dim=1)
            mask_condition = mask_condition.view(1, mask_condition.shape[1] // 4, 4, height//8, width//8)
            mask_condition = mask_condition.transpose(1, 2)[0]
            y = torch.cat([mask_condition,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}


class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("motion_bucket_id",),
            output_params=("motion_bucket_id",)
        )

    def process(self, pipe: WanVideoSviPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            output_params=("vace_context", "vace_scale"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoSviPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image,list):
                    vace_reference_image = [vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                bs, c, f, h, w = vace_reference_image.shape
                new_vace_ref_images = []
                for j in range(f):
                    new_vace_ref_images.append(vace_reference_image[0, :, j:j+1])
                vace_reference_image = new_vace_ref_images
                
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


class WanVideoUnit_VAP(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("text_encoder", "vae", "image_encoder"),
            input_params=("vap_video", "vap_prompt", "negative_vap_prompt", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("vap_clip_feature", "vap_hidden_state", "context_vap")
        )
        
    def encode_prompt(self, pipe: WanVideoSviPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoSviPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("vap_video") is None:
            return inputs_shared, inputs_posi, inputs_nega
        else:
            # 1. encode vap prompt
            pipe.load_models_to_device(["text_encoder"])
            vap_prompt, negative_vap_prompt = inputs_posi.get("vap_prompt", ""), inputs_nega.get("negative_vap_prompt", "")
            vap_prompt_emb = self.encode_prompt(pipe, vap_prompt)
            negative_vap_prompt_emb = self.encode_prompt(pipe, negative_vap_prompt)
            inputs_posi.update({"context_vap":vap_prompt_emb})
            inputs_nega.update({"context_vap":negative_vap_prompt_emb})
            # 2. prepare vap image clip embedding
            pipe.load_models_to_device(["vae", "image_encoder"])
            vap_video, end_image = inputs_shared.get("vap_video"), inputs_shared.get("end_image")

            num_frames, height, width = inputs_shared.get("num_frames"),inputs_shared.get("height"), inputs_shared.get("width")
            
            image_vap = pipe.preprocess_image(vap_video[0].resize((width, height))).to(pipe.device)

            vap_clip_context = pipe.image_encoder.encode_image([image_vap])
            if end_image is not None:
                vap_end_image = pipe.preprocess_image(vap_video[-1].resize((width, height))).to(pipe.device)
                if pipe.dit.has_image_pos_emb:
                    vap_clip_context = torch.concat([vap_clip_context, pipe.image_encoder.encode_image([vap_end_image])], dim=1)
            vap_clip_context = vap_clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_clip_feature":vap_clip_context})

            # 3. prepare vap latents            
            mask_condition = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            mask_condition[:, 1:] = 0
            if end_image is not None:
                mask_condition[:, -1:] = 1
                last_image_vap = pipe.preprocess_image(vap_video[-1].resize((width, height))).to(pipe.device)
                vae_input = torch.concat([image_vap.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image_vap.device), last_image_vap.transpose(0,1)],dim=1)
            else:
                vae_input = torch.concat([image_vap.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image_vap.device)], dim=1)
            
            mask_condition = torch.concat([torch.repeat_interleave(mask_condition[:, 0:1], repeats=4, dim=1), mask_condition[:, 1:]], dim=1)
            mask_condition = mask_condition.view(1, mask_condition.shape[1] // 4, 4, height//8, width//8)
            mask_condition = mask_condition.transpose(1, 2)[0]

            tiled,tile_size,tile_stride = inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")

            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.concat([mask_condition, y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_video = pipe.preprocess_video(vap_video)
            vap_latent = pipe.vae.encode(vap_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_latent = torch.concat([vap_latent,y], dim=1).to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_hidden_state":vap_latent})

            return inputs_shared, inputs_posi, inputs_nega



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: WanVideoSviPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            output_params=("tea_cache",)
        )

    def process(self, pipe: WanVideoSviPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoSviPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",),
            input_params=("input_audio", "audio_embeds", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "audio_sample_rate", "s2v_pose_video", "s2v_pose_latents", "motion_video"),
            output_params=("audio_embeds", "motion_latents", "drop_motion_frames", "s2v_pose_latents"),
        )

    def process_audio(self, pipe: WanVideoSviPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames-1, dtype=pipe.torch_dtype, device=pipe.device)
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(self, pipe: WanVideoSviPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None and len(motion_video) > 0:
            assert len(motion_video) == motion_frames, f"motion video must have {motion_frames} frames, but got {len(motion_video)}"
            motion_latents = pipe.preprocess_video(motion_video)
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: WanVideoSviPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:,:,1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoSviPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio", None), inputs_shared.pop("audio_embeds", None), inputs_shared.get("audio_sample_rate", 16000)
        s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video", None), inputs_shared.pop("s2v_pose_latents", None), inputs_shared.pop("motion_video", None)

        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: WanVideoSviPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: WanVideoSviPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class WanVideoUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video"),
            output_params=("animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video")
        )

    def process(self, pipe: WanVideoSviPipeline, input_video, animate_pose_video, animate_face_video, animate_inpaint_video, animate_mask_video):
        if input_video is None:
            return {}
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[:len(input_video) - 4]
            # animate_pose_video = animate_pose_video
        if animate_face_video is not None:
            animate_face_video = animate_face_video[:len(input_video) - 4]
            # animate_face_video = animate_face_video
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[:len(input_video) - 4]
            # animate_inpaint_video = animate_inpaint_video
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[:len(input_video) - 4]
            # animate_mask_video = animate_mask_video
            # animate_mask_video = animate_mask_video[:len(input_video) - 4]
        return {"animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video}


class WanVideoUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride", "inference_mode"),
            output_params=("pose_latents",),
            onload_model_names=("vae",)
        )

    def random_transform_pose_video(self, video_tensor, scale_range=(0.5, 1.5), translate_range=(-0.3, 0.3)):
        """
        Apply random scaling and translation to pose video with black (-1) background padding
        All frames use the SAME spatial transformation
        Uses resize + place on canvas approach to ensure no content is cropped
        video_tensor: shape [B, C, T, H, W], value range [-1, 1]
        """
        import random
        import torch.nn.functional as F
        
        B, C, T, H, W = video_tensor.shape
        
        # Random scaling factor (same for all frames)
        scale = random.uniform(scale_range[0], scale_range[1])
        
        # Random translation (same for all frames, relative to image dimensions after scaling)
        translate_x = random.uniform(translate_range[0], translate_range[1])
        translate_y = random.uniform(translate_range[0], translate_range[1])
        
        # Reshape video to [B*T, C, H, W] to process all frames at once
        video_reshaped = video_tensor.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        video_reshaped = video_reshaped.view(B * T, C, H, W)  # [B*T, C, H, W]
        
        # Calculate new dimensions after scaling
        new_h = int(H * scale)
        new_w = int(W * scale)
        
        # Resize the video
        resized = F.interpolate(video_reshaped, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Create canvas with -1 background (black for pose)
        canvas = torch.ones(B * T, C, H, W, dtype=video_tensor.dtype, device=video_tensor.device) * (-1.0)
        
        # Calculate placement position (centered + translation offset)
        # Ensure the entire resized content fits within the canvas
        offset_x = int((W - new_w) / 2 + translate_x * W)
        offset_y = int((H - new_h) / 2 + translate_y * H)
        
        # Clamp offsets to ensure resized content stays within canvas bounds
        offset_x = max(min(offset_x, W - 1), -(new_w - 1))
        offset_y = max(min(offset_y, H - 1), -(new_h - 1))
        
        # Calculate valid regions for source and destination
        src_x_start = max(0, -offset_x)
        src_y_start = max(0, -offset_y)
        src_x_end = min(new_w, W - offset_x)
        src_y_end = min(new_h, H - offset_y)
        
        dst_x_start = max(0, offset_x)
        dst_y_start = max(0, offset_y)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # Place resized content onto canvas
        canvas[:, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            resized[:, :, src_y_start:src_y_end, src_x_start:src_x_end]
        
        # Reshape back to [B, C, T, H, W]
        transformed = canvas.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        
        return transformed

    def process(self, pipe: WanVideoSviPipeline, animate_pose_video, tiled, tile_size, tile_stride, inference_mode=False):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        
        # Remove pose if remove_pose is enabled
        import random
        if not inference_mode and random.random() < 0.05:
            animate_pose_video = torch.zeros_like(animate_pose_video) - 1.0
        
        # Apply random data augmentation during training
        if not inference_mode and pipe.use_pose_aug:
            import random
            #     )
            if random.random() < 0.05:
                animate_pose_video = torch.ones_like(animate_pose_video) * (-1.0)
            else:
                animate_pose_video = self.random_transform_pose_video(
                        animate_pose_video, 
                        scale_range=(0.9, 1.1),  # Scale range: 0.5x to 1.5x
                        translate_range=(-0.1, 0.1)  # Translation range: -10% to +10%
                    )               
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

        # pose_latents = pose_latents[:,:,1:,...]
        return {"pose_latents": pose_latents}


class WanVideoUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("animate_face_video",),
            output_params=("face_pixel_values"),
        )

    def process(self, pipe: WanVideoSviPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("animate_face_video", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        inputs_posi["face_pixel_values"] = pipe.preprocess_video(inputs_shared["animate_face_video"])
        inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_inpaint_video", "animate_mask_video", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )
        
    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            mask_condition = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            mask_condition = mask_pixel_values.clone()
        mask_condition[:, :mask_len] = 1
        mask_condition = torch.concat([torch.repeat_interleave(mask_condition[:, 0:1], repeats=4, dim=1), mask_condition[:, 1:]], dim=1)
        mask_condition = mask_condition.view(1, mask_condition.shape[1] // 4, 4, lat_h, lat_w)
        mask_condition = mask_condition.transpose(1, 2)[0]
        return mask_condition

    def process(self, pipe: WanVideoSviPipeline, animate_inpaint_video, animate_mask_video, input_image, tiled, tile_size, tile_stride):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(bg_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape
        
        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(ref_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=pipe.device)
        
        mask_pixel_values = 1 - pipe.preprocess_video(animate_mask_video, max_value=1, min_value=0)
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(mask_pixel_values, size=(lat_h, lat_w), mode='nearest')
        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixel_values=mask_pixel_values, device=pipe.device)
        
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=pipe.device)
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class WanVideoUnit_LongCatVideo(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("longcat_video",),
            output_params=("longcat_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoSviPipeline, longcat_video):
        if longcat_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        longcat_video = pipe.preprocess_video(longcat_video)
        longcat_latents = pipe.vae.encode(longcat_video, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"longcat_latents": longcat_latents}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def build_anchor_token_ranges(
    num_frames: int,
    h: int,
    w: int,
    num_video_anchor_latents: int,
    ref_offset: int = 0,
):
    if num_video_anchor_latents is None or num_video_anchor_latents <= 0 or num_frames <= 0:
        return []

    hw = h * w
    anchor_frames = min(num_video_anchor_latents, num_frames)
    return [(ref_offset, ref_offset + anchor_frames * hw)]


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    vap: MotWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state = None,
    vap_clip_feature = None,
    context_vap = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    num_motion_latents: Optional[int] = 1, # new
    motion_penalty: float = 3.0,
    num_video_anchor_latents: int = 1,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # LongCat-Video
    if isinstance(dit, LongCatVideoTransformer3DModel):
        return model_fn_longcat_video(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            longcat_latents=longcat_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
        
    # Camera control
    x = dit.patchify(x, control_camera_latents_input)
    
    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(
            x,
            pose_latents,
            face_pixel_values,
            num_video_anchor_latents=num_video_anchor_latents,
        )
    
    # Patchify
    f, h, w = x.shape[2:]
    latent_frame_count = f
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    svi_attn_mask = None
    # num_motion_latents = 1 
    # num_motion_latents = 0
    num_motion_latents = 0
    
    if num_motion_latents is not None and num_motion_latents > 0:
        # ref_offset: reference-latent tokens prepended to sequence (accounted for below)
        svi_attn_mask = compute_svi_attn_mask(
            f=f, h=h, w=w,
            num_motion_latents = num_motion_latents,
            device=x.device,
            dtype=x.dtype,
            ref_offset=0,  # updated after reference_latents are prepended
            motion_penalty=motion_penalty,
        )

    anchor_token_ranges = None
    if getattr(dit, "enable_anchor_key_focus", False):
        anchor_token_ranges = build_anchor_token_ranges(
            num_frames=latent_frame_count,
            h=h,
            w=w,
            num_video_anchor_latents=num_video_anchor_latents,
            ref_offset=0,
        )

    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        ref_tokens = reference_latents.shape[1]
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
        if anchor_token_ranges is not None:
            anchor_token_ranges = build_anchor_token_ranges(
                num_frames=latent_frame_count,
                h=h,
                w=w,
                num_video_anchor_latents=num_video_anchor_latents,
                ref_offset=ref_tokens,
            )
        # Rebuild mask with correct ref_offset.
        # Use f-1 (original latent frames before the +1 reference-frame adjustment)
        # because the actual token count is ref_tokens + (f-1)*h*w.
        if num_motion_latents is not None and num_motion_latents > 0:
            svi_attn_mask = compute_svi_attn_mask(
                f=f - 1, h=h, w=w,
                num_motion_latents=num_motion_latents,
                device=x.device,
                dtype=x.dtype,
                ref_offset=ref_tokens,
                motion_penalty=motion_penalty,
            )

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # VAP 
    if vap is not None:
        # hidden state
        x_vap = vap_hidden_state
        x_vap = vap.patchify(x_vap)
        x_vap = rearrange(x_vap, 'b c f h w -> b (f h w) c').contiguous()
        # Timestep
        clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(timestep.dtype)
        t = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
        t_mod_vap = vap.time_projection(t).unflatten(1, (6, vap.dim))

        # rope
        freqs_vap = vap.compute_freqs_mot(f,h,w).to(x.device)

        # context
        vap_clip_embedding = vap.img_emb(vap_clip_feature)
        context_vap = vap.text_embedding(context_vap)
        context_vap = torch.cat([vap_clip_embedding, context_vap], dim=1)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
            anchor_token_ranges = None
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module, attn_mask=None, anchor_token_ranges=None, anchor_key_scale: float = 1.0):
            def custom_forward(*inputs):
                return module(
                    *inputs,
                    attn_mask=attn_mask,
                    anchor_token_ranges=anchor_token_ranges,
                    anchor_key_scale=anchor_key_scale,
                )
            return custom_forward
        
        def create_custom_forward_vap(block, vap):
            def custom_forward(*inputs):
                return vap(block, *inputs)
            return custom_forward
        
        anchor_focus_start_block = int(len(dit.blocks) * getattr(dit, "anchor_key_focus_start_ratio", 0.5))
        anchor_key_focus_scale = getattr(dit, "anchor_key_focus_scale", 1.0)
        for block_id, block in enumerate(dit.blocks):
            block_anchor_ranges = None
            block_anchor_scale = 1.0
            if (
                anchor_token_ranges is not None
                and block_id >= anchor_focus_start_block
                and anchor_key_focus_scale != 1.0
            ):
                block_anchor_ranges = anchor_token_ranges
                block_anchor_scale = anchor_key_focus_scale
            # Block
            if vap is not None and block_id in vap.mot_layers_mapping:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_vap = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_vap(block, vap),
                            x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x, x_vap = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_vap(block, vap),
                        x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                        use_reentrant=False,
                    )
                else:
                    x, x_vap = vap(block, x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id)
            else:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(
                                block,
                                attn_mask=svi_attn_mask,
                                anchor_token_ranges=block_anchor_ranges,
                                anchor_key_scale=block_anchor_scale,
                            ),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(
                            block,
                            attn_mask=svi_attn_mask,
                            anchor_token_ranges=block_anchor_ranges,
                            anchor_key_scale=block_anchor_scale,
                        ),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
                else:
                    x = block(
                        x,
                        context,
                        t_mod,
                        freqs,
                        attn_mask=svi_attn_mask,
                        anchor_token_ranges=block_anchor_ranges,
                        anchor_key_scale=block_anchor_scale,
                    )
            
            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


def model_fn_longcat_video(
    dit: LongCatVideoTransformer3DModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    longcat_latents: torch.Tensor = None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
):
    if longcat_latents is not None:
        latents[:, :, :longcat_latents.shape[2]] = longcat_latents
        num_cond_latents = longcat_latents.shape[2]
    else:
        num_cond_latents = 0
    context = context.unsqueeze(0)
    encoder_attention_mask = torch.any(context != 0, dim=-1)[:, 0].to(torch.int64)
    output = dit(
        latents,
        timestep,
        context,
        encoder_attention_mask,
        num_cond_latents=num_cond_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    output = -output
    output = output.to(latents.dtype)
    return output


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None)
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x_global, use_unified_sequence_parallel)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x
