from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch, os, argparse, accelerate, warnings
from datetime import datetime
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video_svi import WanVideoSviPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from diffsynth.utils.data import save_video, VideoData
import random

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        num_motion_latents=2,
        num_video_anchor_latents=4,
        online=False,
        enable_image_enhancement=False,
        same_augmentation=False,
        image_enhancement_prob=0.8,
        add_noise_to_motion_latent=False,
        motion_latent_shared_noise=False,
        use_pose_aug=False,
        remove_pose=False,
        mask_anchor_motion=False,
        enable_anchor_key_focus=False,
        rand_aug=False,
        mixed_aug=False,
        use_self_pred_aug=False,
        self_pred_aug_backprop_first=False,
        aug_anchor=False,
        pad_first_clip_with_anchor=False,
        aug_strength_adapt=False,
        latent_aug=False,
        adaptive_aug=False,
        trajectory_correction_schedule="fixed",
        trajectory_correction_weight=1.0,
        traj_correction_alpha=None,
        video_aug_prob=1.0,
        rand_aug_sigma_threshold=0.5,
        sigma_shift=5.0,
        train_timestep_mode="full",
        train_inference_num_steps=20,
        train_full_num_inference_steps=1000,
        train_timestep_mixed_inference_prob=0.8
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoSviPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            redirect_common_files=False,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.pipe.num_motion_latents = num_motion_latents
        self.pipe.num_video_anchor_latents = num_video_anchor_latents
        self.num_video_anchor_latents = num_video_anchor_latents
        self.pipe.enable_image_enhancement = enable_image_enhancement
        self.pipe.same_augmentation = same_augmentation
        self.pipe.use_zero_padding = True
        self.pipe.image_enhancement_prob = image_enhancement_prob
        self.pipe.add_noise_to_motion_latent = add_noise_to_motion_latent
        self.pipe.motion_latent_shared_noise = motion_latent_shared_noise
        self.pipe.use_pose_aug = use_pose_aug
        self.pipe.remove_pose = remove_pose
        self.pipe.mask_anchor_motion = mask_anchor_motion
        self.pipe.enable_anchor_key_focus = enable_anchor_key_focus
        self.pipe.rand_aug = rand_aug
        self.pipe.mixed_aug = mixed_aug
        self.pipe.use_self_pred_aug = use_self_pred_aug
        self.pipe.self_pred_aug_backprop_first = self_pred_aug_backprop_first
        self.pipe.aug_anchor = aug_anchor
        self.pipe.pad_first_clip_with_anchor = pad_first_clip_with_anchor
        self.pipe.aug_strength_adapt = aug_strength_adapt
        self.pipe.latent_aug = latent_aug
        self.pipe.adaptive_aug = adaptive_aug
        if traj_correction_alpha is not None:
            trajectory_correction_schedule, trajectory_correction_weight = self._parse_legacy_traj_correction_alpha(traj_correction_alpha)
        self.pipe.trajectory_correction_schedule = trajectory_correction_schedule
        self.pipe.trajectory_correction_weight = trajectory_correction_weight
        self.pipe.video_aug_prob = video_aug_prob
        self.pipe.rand_aug_sigma_threshold = rand_aug_sigma_threshold
        self.pipe.sigma_shift = sigma_shift
        self.pipe.train_timestep_mode = train_timestep_mode
        self.pipe.train_inference_num_steps = train_inference_num_steps
        self.pipe.train_full_num_inference_steps = train_full_num_inference_steps
        self.pipe.train_timestep_mixed_inference_prob = train_timestep_mixed_inference_prob
        self.online = online
        if self.pipe.dit is not None:
            self.pipe.dit.enable_anchor_key_focus = enable_anchor_key_focus
        if getattr(self.pipe, "dit2", None) is not None:
            self.pipe.dit2.enable_anchor_key_focus = enable_anchor_key_focus


        # Store online flag in pipe for loss function access
        if hasattr(self.pipe, 'online'):
            self.pipe.online = online

        # Print effective pipeline configuration
        cfg_lines = [
            "",
            "=" * 60,
            "  Effective Pipeline Configuration",
            "=" * 60,
            f"  task                        : {task}",
            f"  num_motion_latents          : {self.pipe.num_motion_latents}",
            f"  num_video_anchor_latents    : {self.pipe.num_video_anchor_latents}",
            f"  max_timestep_boundary       : {max_timestep_boundary}",
            f"  min_timestep_boundary       : {min_timestep_boundary}",
            f"  use_gradient_checkpointing  : {use_gradient_checkpointing}",
            f"  use_grad_ckpt_offload       : {use_gradient_checkpointing_offload}",
            f"  enable_image_enhancement    : {self.pipe.enable_image_enhancement}",
            f"  same_augmentation           : {self.pipe.same_augmentation}",
            f"  add_noise_to_motion_latent  : {self.pipe.add_noise_to_motion_latent}",
            f"  motion_latent_shared_noise  : {self.pipe.motion_latent_shared_noise}",
            f"  remove_pose                 : {self.pipe.remove_pose}",
            f"  enable_anchor_key_focus     : {self.pipe.enable_anchor_key_focus}",
            f"  rand_aug                    : {getattr(self.pipe, 'rand_aug', False)}",
            f"  mixed_aug                   : {getattr(self.pipe, 'mixed_aug', False)}",
            f"  use_self_pred_aug           : {getattr(self.pipe, 'use_self_pred_aug', False)}",
            f"  self_pred_aug_backprop_1st  : {getattr(self.pipe, 'self_pred_aug_backprop_first', False)}",
            f"  aug_anchor                  : {getattr(self.pipe, 'aug_anchor', False)}",
            f"  pad_first_clip_with_anchor  : {getattr(self.pipe, 'pad_first_clip_with_anchor', False)}",
            f"  aug_strength_adapt          : {getattr(self.pipe, 'aug_strength_adapt', False)}",
            f"  latent_aug                  : {getattr(self.pipe, 'latent_aug', False)}",
            f"  adaptive_aug                : {getattr(self.pipe, 'adaptive_aug', False)}",
            f"  trajectory_corr_schedule    : {self.pipe.trajectory_correction_schedule}",
            f"  trajectory_corr_weight      : {self.pipe.trajectory_correction_weight}",
            f"  video_aug_prob              : {self.pipe.video_aug_prob}",
            f"  rand_aug_sigma_threshold    : {self.pipe.rand_aug_sigma_threshold}",
            f"  sigma_shift                 : {self.pipe.sigma_shift}",
            f"  train_timestep_mode         : {self.pipe.train_timestep_mode}",
            f"  train_inference_num_steps   : {self.pipe.train_inference_num_steps}",
            f"  train_full_num_inf_steps    : {self.pipe.train_full_num_inference_steps}",
            f"  train_timestep_mixed_prob   : {self.pipe.train_timestep_mixed_inference_prob}",
            f"  online                      : {online}",
            f"  extra_inputs                : {self.extra_inputs}",
            f"  trainable_models            : {trainable_models}",
            f"  lora_base_model             : {lora_base_model}",
            f"  lora_rank                   : {lora_rank}",
            f"  device                      : {device}",
            "=" * 60,
            "",
        ]
        print("\n".join(cfg_lines))

    def _parse_legacy_traj_correction_alpha(self, value):
        value = float(value)
        if value >= 0:
            return "fixed", value
        legacy_schedule_map = {
            -1.0: "sigma_ramp",
            -2.0: "augmented_velocity",
            -3.0: "scheduler_weight",
            -4.0: "gaussian_timestep",
        }
        if value not in legacy_schedule_map:
            raise ValueError(
                "Unsupported legacy traj_correction_alpha value. "
                "Use --trajectory_correction_schedule and --trajectory_correction_weight instead."
            )
        return legacy_schedule_map[value], 1.0
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        # import ipdb; ipdb.set_trace()
        using_precomputed = "video_latent" in inputs_shared and inputs_shared["video_latent"] is not None
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                if not using_precomputed:
                    inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "input_clip":
                if not using_precomputed:
                    inputs_shared["input_image"] = data["video"][:5]
            elif extra_input == "auxiliary_video":
                # In latent mode auxiliary_latent is already in inputs_shared; skip raw video
                if not using_precomputed:
                    inputs_shared["auxiliary_video"] = data["auxiliary_video"]
            elif extra_input == "end_image":
                if not using_precomputed:
                    inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "anchor":
                # In latent mode anchor_latent_full is already in inputs_shared; skip raw PIL
                if not using_precomputed:
                    inputs_shared["anchor"] = data["anchor"]
            elif extra_input == "face_image":
                inputs_shared["face_image"] = random.choice(inputs_shared['animate_face_video'])
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Trim last 4 frames (overlap) only when raw pixel data is used
        if not using_precomputed:
            inputs_shared["input_video"] = inputs_shared["input_video"][:-4]
        if inputs_shared.get("animate_face_video") is not None:
            inputs_shared["animate_face_video"] = inputs_shared["animate_face_video"][:-4]
        if not using_precomputed and inputs_shared.get("animate_pose_video") is not None:
            inputs_shared["animate_pose_video"] = inputs_shared["animate_pose_video"][:-4]

        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # Pre-computed latent mode: video_latent is a tensor [C, T', H', W'].
        # We derive height/width/num_frames from the latent shape (×8 for spatial, ×4-1 for time).
        using_precomputed = "video_latent" in data and data["video_latent"] is not None

        if using_precomputed:
            vl = data["video_latent"]          # [C, T', H', W']
            _, t_lat, h_lat, w_lat = vl.shape
            height     = h_lat * 8
            width      = w_lat * 8
            num_frames = (t_lat - 1) * 4 + 1
            input_video = None
        else:
            input_video = data["video"]
            height      = data["video"][0].size[1]
            width       = data["video"][0].size[0]
            num_frames  = len(data["video"])

        inputs_shared = {
            "input_video": input_video,
            "height": height,
            "width":  width,
            "num_frames": num_frames,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "num_video_anchor_latents": self.num_video_anchor_latents,
            "sigma_shift": getattr(self.pipe, "sigma_shift", 5.0),
            "train_timestep_mode": getattr(self.pipe, "train_timestep_mode", "full"),
            "train_inference_num_steps": getattr(self.pipe, "train_inference_num_steps", 20),
            "train_full_num_inference_steps": getattr(self.pipe, "train_full_num_inference_steps", 1000),
            "train_timestep_mixed_inference_prob": getattr(self.pipe, "train_timestep_mixed_inference_prob", 0.8),
        }

        # Pass through all pre-computed VAE latents when available
        for latent_key in ("video_latent", "auxiliary_latent", "anchor_latent_full",
                           "anchor_size", "pose_latent"):
            if latent_key in data and data[latent_key] is not None:
                inputs_shared[latent_key] = data[latent_key]

        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)

        inputs_shared, inputs_posi, inputs_nega = inputs
        rand_aug_sigma_threshold = float(getattr(self.pipe, "rand_aug_sigma_threshold", 0.5))
        need_preset_timestep = (
            bool(getattr(self.pipe, "aug_strength_adapt", False))
            or (
                rand_aug_sigma_threshold > 0.0
                and (bool(getattr(self.pipe, "rand_aug", False)) or bool(getattr(self.pipe, "mixed_aug", False)))
            )
        )
        if need_preset_timestep:
            # Pre-sample one training timestep so augmentation gating and loss use the same noise level.
            mode = inputs_shared.get("train_timestep_mode", getattr(self.pipe, "train_timestep_mode", "full"))
            sigma_shift = float(inputs_shared.get("sigma_shift", getattr(self.pipe, "sigma_shift", 5.0)))
            full_steps = int(inputs_shared.get("train_full_num_inference_steps", getattr(self.pipe, "train_full_num_inference_steps", 1000)))
            infer_steps = int(inputs_shared.get("train_inference_num_steps", getattr(self.pipe, "train_inference_num_steps", 20)))
            mixed_prob = float(inputs_shared.get("train_timestep_mixed_inference_prob", getattr(self.pipe, "train_timestep_mixed_inference_prob", 0.8)))

            active_mode = mode
            if mode == "mixed":
                active_mode = "inference" if torch.rand(1).item() < mixed_prob else "full"
            num_steps = infer_steps if active_mode == "inference" else full_steps
            self.pipe.scheduler.set_timesteps(num_steps, training=True, shift=sigma_shift)

            max_timestep_boundary = int(inputs_shared.get("max_timestep_boundary", 1) * len(self.pipe.scheduler.timesteps))
            min_timestep_boundary = int(inputs_shared.get("min_timestep_boundary", 0) * len(self.pipe.scheduler.timesteps))
            max_timestep_boundary = max(min_timestep_boundary + 1, max_timestep_boundary)
            timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
            sampled_timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            inputs_shared["train_sampled_timestep"] = sampled_timestep

        inputs = (inputs_shared, inputs_posi, inputs_nega)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--use_aux_video", default=False, action="store_true", help="Whether to use anchor frame.")
    parser.add_argument("--enable_image_enhancement", default=False, action="store_true", help="Whether to use anchor frame.")
    parser.add_argument("--same_augmentation", default=False, action="store_true", help="Whether to apply the same random augmentation parameters across all frames in a clip.")
    parser.add_argument("--image_to_train_prob", type=float, default=0.0, help="Probability of using image to train (default: 0.2).")

    parser.add_argument("--replace_first_frame_with_anchor", default=False, action="store_true", help="Whether to replace the first frame with the anchor frame.")
    parser.add_argument("--mask_anchor_motion", "--msk_anchor_motion", dest="mask_anchor_motion", default=False, action="store_true", help="Whether to replace the first frame with the anchor frame.")
    parser.add_argument("--num_motion_latents", type=int, default=1, help="Number of motion latents to use.")
    parser.add_argument("--num_video_anchor_latents", type=int, default=4, help="Number of video anchor latents (anchor frames prepended to the latent sequence).")
    parser.add_argument("--num_overlap_frames", type=int, default=4, help="Number of overlapping frames to use.")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Sigma/timestep shift used by the Wan FlowMatch scheduler during training/inference-aligned scheduling.")
    parser.add_argument("--train_timestep_mode", type=str, default="full", choices=["full", "inference", "mixed"], help="How training samples timestep ids: full=sample from dense training schedule, inference=sample from inference-aligned schedule, mixed=probabilistically mix the two.")
    parser.add_argument("--train_inference_num_steps", type=int, default=20, help="Inference-aligned schedule length used when train_timestep_mode is inference or mixed.")
    parser.add_argument("--train_full_num_inference_steps", type=int, default=1000, help="Dense schedule length used when train_timestep_mode is full or mixed.")
    parser.add_argument("--train_timestep_mixed_inference_prob", type=float, default=0.8, help="When train_timestep_mode=mixed, probability of sampling from the inference-aligned schedule.")
    parser.add_argument("--online", default=False, action="store_true", help="Enable online training mode: each sample passes twice, first without error to compute error, then with error to compute loss.")
    parser.add_argument("--use_precomputed_latents", default=False, action="store_true",
                        help="Load pre-computed VAE latents from .pt files instead of encoding videos on-the-fly. "
                             "Requires the metadata CSV to have a 'latent' column pointing to .pt files generated "
                             "by precompute_latents.py.")
    parser.add_argument("--image_enhancement_prob", type=float, default=0.8, help="Probability of applying image enhancement.")
    parser.add_argument("--add_noise_to_motion_latent", default=False, action="store_true", help="Add same-sigma noise to motion_latent in y at each denoising step (training and inference).")
    parser.add_argument("--motion_latent_shared_noise", default=False, action="store_true", help="Share the initial x_t noise for motion_latent noising instead of using independent random noise.")
    parser.add_argument("--use_pose_aug", default=False, action="store_true", help="Use pose augmentation during training.")
    parser.add_argument("--remove_pose", default=False, action="store_true", help="Remove/zero out all pose conditioning during training. Useful for ablation studies.")
    parser.add_argument("--enable_anchor_key_focus", default=False, action="store_true", help="Boost anchor-token keys in late self-attention blocks without adding parameters or changing the FlashAttention path.")
    parser.add_argument("--rand_aug", default=False, action="store_true", help="Use VAE latents encoded from randomly augmented video frames when constructing x_t~.")
    parser.add_argument("--mixed_aug", default=False, action="store_true", help="At each sample, randomly choose one augmentation path between rand_aug and use_self_pred_aug when constructing x_t~.")
    parser.add_argument("--use_self_pred_aug", default=False, action="store_true", help="Run a first clean pass, integrate one-step x_1 prediction, and use detach(x_1_pred) as an additional augmentation source for the second pass.")
    parser.add_argument("--self_pred_aug_backprop_first", default=False, action="store_true", help="When --use_self_pred_aug is enabled, also compute loss and backprop on the first clean pass. The final loss becomes the mean of first and second pass losses.")
    parser.add_argument("--aug_anchor", default=False, action="store_true", help="Apply random geometric augmentation (translation/scale) to sampled anchor frames during training.")
    parser.add_argument("--pad_first_clip_with_anchor", default=False, action="store_true", help="Pad first clip's motion part with first frame latent instead of zeros.")
    parser.add_argument("--aug_strength_adapt", default=False, action="store_true", help="Adapt image-frame augmentation strength by sampled timestep: weaker in low-noise region, stronger in high-noise region.")
    parser.add_argument("--latent_aug", default=False, action="store_true", help="Apply augmentation directly in VAE latent space (color-cast/blur/over-saturation equivalents) instead of pixel-space augmentation.")
    parser.add_argument("--adaptive_aug", default=False, action="store_true", help="Adapt augmentation intensity by timestep/sigma: weaker near low-noise region and stronger near high-noise region.")
    parser.add_argument(
        "--trajectory_correction_schedule",
        type=str,
        default="fixed",
        choices=["fixed", "sigma_ramp", "augmented_velocity", "scheduler_weight", "gaussian_timestep"],
        help="Schedule used to weight the trajectory correction term.",
    )
    parser.add_argument(
        "--trajectory_correction_weight",
        type=float,
        default=1.0,
        help="Scalar multiplier for the selected trajectory correction schedule.",
    )
    parser.add_argument("--traj_correction_alpha", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--video_aug_prob", type=float, default=1.0, help="Probability of applying the selected latent augmentation path.")
    parser.add_argument("--rand_aug_sigma_threshold", type=float, default=0.5, help="Only use input_latents_aug when the sampled training sigma is at least this value. Set to 0 or below to disable this gate.")
    return parser


def find_latest_checkpoint(output_path):
    """Scan output_path for step-*.safetensors checkpoints and return the path
    with the highest step number, or None if no checkpoint is found."""
    if not os.path.isdir(output_path):
        return None, None
    best_step = -1
    best_path = None
    for fname in os.listdir(output_path):
        if fname.startswith("step-") and fname.endswith(".safetensors"):
            try:
                step = int(fname[len("step-"):-len(".safetensors")])
                if step > best_step:
                    best_step = step
                    best_path = os.path.join(output_path, fname)
            except ValueError:
                continue
    if best_path is None:
        return None, None
    return best_path, best_step


def backup_wan_video_svi(output_path, accelerator=None, workspace_root=None):
    """Backup wan_video_svi.py to the same folder where safetensors are saved.
    
    Args:
        output_path: Output path from training args (same folder as safetensors)
        accelerator: Accelerator instance (for distributed training check)
        workspace_root: Root directory of the workspace
    """
    # Only backup on main process in distributed training
    if accelerator is not None and not accelerator.is_main_process:
        return
    
    if workspace_root is None:
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Source file
    source_file = os.path.join(workspace_root, "diffsynth/pipelines/wan_video_svi.py")
    
    # Save directly into the safetensors output folder
    os.makedirs(output_path, exist_ok=True)
    
    # Destination file with timestamp
    dest_file = os.path.join(output_path, f"wan_video_svi_{timestamp}.py")


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        log_with="tensorboard",
        project_dir=args.output_path,
    )
    accelerator.init_trackers("wan_svi_training", config=vars(args), init_kwargs={"tensorboard": {"flush_secs": 60}})

    # Auto-resume: if no explicit checkpoint given, look for the latest one in output_path
    resumed_step = None
    # if args.lora_checkpoint is None:
    auto_ckpt, auto_step = find_latest_checkpoint(args.output_path)
    if auto_ckpt is not None:
        args.lora_checkpoint = auto_ckpt
        resumed_step = auto_step
        if accelerator.is_main_process:
            print(f"[Auto-resume] Found checkpoint at step {auto_step}: {auto_ckpt}")
    else:
        if accelerator.is_main_process:
            print("[Auto-resume] No existing checkpoint found, training from scratch.")

    # Backup wan_video_svi.py before training (only on main process)
    backup_wan_video_svi(args.output_path, accelerator=accelerator)

    # Print all arguments for record keeping
    if accelerator.is_main_process:
        print("")
        print("=" * 60)
        print("  Training Arguments")
        print("=" * 60)
        args_dict = vars(args)
        max_key_len = max(len(k) for k in args_dict.keys())
        for key in sorted(args_dict.keys()):
            print(f"  {key:<{max_key_len}} : {args_dict[key]}")
        print("=" * 60)
        print("")

    if args.use_precomputed_latents:
        # ── Pre-computed latent mode ──────────────────────────────────────────
        # The metadata CSV must have a 'latent' column with relative paths to .pt
        # files produced by precompute_latents.py.  Each .pt stores a dict with keys:
        #   video_latent, auxiliary_latent, anchor_latent_full, anchor_size, pose_latent
        # The face video is still loaded on-the-fly (small and not pre-computed).
        from diffsynth.core.data.operators import LoadTorchPickle
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=["latent", "animate_face_video"],
            image_to_train_prob=args.image_to_train_prob,
            main_data_operator=ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(
                    args.num_frames, 4, 1,
                    frame_processor=ImageCropAndResize(512, 512, None, 16, 16)
                ),
            },
        )
    else:
        # ── Default on-the-fly encoding mode ─────────────────────────────────
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            image_to_train_prob=args.image_to_train_prob,
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
                use_aux_video=args.use_aux_video,
                num_overlap_frames=args.num_overlap_frames,
                replace_first_frame_with_anchor=args.replace_first_frame_with_anchor,
            ),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(
                    args.num_frames, 4, 1,
                    frame_processor=ImageCropAndResize(512, 512, None, 16, 16)
                ),
                "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            },
        )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_motion_latents=args.num_motion_latents,
        num_video_anchor_latents=args.num_video_anchor_latents,
        enable_image_enhancement=args.enable_image_enhancement,
        same_augmentation=args.same_augmentation,
        online=args.online,
        image_enhancement_prob=args.image_enhancement_prob,
        add_noise_to_motion_latent=args.add_noise_to_motion_latent,
        motion_latent_shared_noise=args.motion_latent_shared_noise,
        use_pose_aug=args.use_pose_aug,
        remove_pose=args.remove_pose,
        mask_anchor_motion=args.mask_anchor_motion,
        enable_anchor_key_focus=args.enable_anchor_key_focus,
        rand_aug=args.rand_aug,
        mixed_aug=args.mixed_aug,
        use_self_pred_aug=args.use_self_pred_aug,
        self_pred_aug_backprop_first=args.self_pred_aug_backprop_first,
        aug_anchor=args.aug_anchor,
        pad_first_clip_with_anchor=args.pad_first_clip_with_anchor,
        aug_strength_adapt=args.aug_strength_adapt,
        latent_aug=args.latent_aug,
        adaptive_aug=args.adaptive_aug,
        trajectory_correction_schedule=args.trajectory_correction_schedule,
        trajectory_correction_weight=args.trajectory_correction_weight,
        traj_correction_alpha=args.traj_correction_alpha,
        video_aug_prob=args.video_aug_prob,
        rand_aug_sigma_threshold=args.rand_aug_sigma_threshold,
        sigma_shift=args.sigma_shift,
        train_timestep_mode=args.train_timestep_mode,
        train_inference_num_steps=args.train_inference_num_steps,
        train_full_num_inference_steps=args.train_full_num_inference_steps,
        train_timestep_mixed_inference_prob=args.train_timestep_mixed_inference_prob
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    # Restore step counter so checkpoint saving continues from the right step
    if resumed_step is not None:
        model_logger.num_steps = resumed_step
        if accelerator.is_main_process:
            print(f"[Auto-resume] Resuming ModelLogger from step {resumed_step}.")
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
