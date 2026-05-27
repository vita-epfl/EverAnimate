from .base_pipeline import BasePipeline
import torch
import random

def _resolve_sigma_shift(pipe, inputs=None, default=5.0):
    if inputs is not None and isinstance(inputs, dict) and "sigma_shift" in inputs and inputs["sigma_shift"] is not None:
        return inputs["sigma_shift"]
    return getattr(pipe, "sigma_shift", default)


def _configure_training_timesteps(pipe, inputs):
    mode = inputs.get("train_timestep_mode", getattr(pipe, "train_timestep_mode", "full"))
    sigma_shift = _resolve_sigma_shift(pipe, inputs)
    full_steps = int(inputs.get("train_full_num_inference_steps", getattr(pipe, "train_full_num_inference_steps", 1000)))
    inference_steps = int(inputs.get("train_inference_num_steps", getattr(pipe, "train_inference_num_steps", inputs.get("num_inference_steps", 20))))
    mixed_prob = float(inputs.get("train_timestep_mixed_inference_prob", getattr(pipe, "train_timestep_mixed_inference_prob", 0.8)))

    active_mode = mode
    if mode == "mixed":
        active_mode = "inference" if torch.rand(1).item() < mixed_prob else "full"

    num_steps = inference_steps if active_mode == "inference" else full_steps
    pipe.scheduler.set_timesteps(num_steps, training=True, shift=sigma_shift)
    return active_mode, num_steps


def _align_aug_latents_to_clean(clean_latents, aug_latents, inputs):
    if aug_latents is None:
        return None
    if aug_latents.shape == clean_latents.shape:
        return aug_latents

    aug_aligned = clean_latents.clone()
    t_aug = aug_latents.shape[2]
    t_clean = clean_latents.shape[2]
    anchor_latent_vid = inputs.get("anchor_latent_vid", None)
    if anchor_latent_vid is not None and anchor_latent_vid.shape[1] > 0:
        num_ancs = int(anchor_latent_vid.shape[1])
        aug_aligned[:, :, num_ancs:num_ancs + t_aug, :, :] = aug_latents
    else:
        aug_aligned[:, :, t_clean - t_aug:, :, :] = aug_latents
    return aug_aligned



def _resolve_trajectory_correction_config(pipe: BasePipeline):
    schedule = getattr(pipe, "trajectory_correction_schedule", None)
    weight = getattr(pipe, "trajectory_correction_weight", None)
    if schedule is not None:
        return schedule, float(1.0 if weight is None else weight)

    # Backward compatibility for old checkpoints/scripts that encoded schedule
    # choices as negative traj_correction_alpha values.
    legacy_alpha = float(getattr(pipe, "traj_correction_alpha", 1.0))
    if legacy_alpha >= 0:
        return "fixed", legacy_alpha
    legacy_schedule_map = {
        -1.0: "sigma_ramp",
        -2.0: "augmented_velocity",
        -3.0: "scheduler_weight",
        -4.0: "gaussian_timestep",
    }
    return legacy_schedule_map.get(legacy_alpha, "fixed"), 1.0


def _resolve_trajectory_correction_alpha(pipe: BasePipeline, timestep):
    schedule, weight = _resolve_trajectory_correction_config(pipe)
    if schedule == "fixed":
        return weight

    if isinstance(timestep, torch.Tensor):
        timestep_cpu = timestep.detach().to(device=pipe.scheduler.timesteps.device, dtype=pipe.scheduler.timesteps.dtype)
    else:
        timestep_cpu = torch.tensor([timestep], device=pipe.scheduler.timesteps.device, dtype=pipe.scheduler.timesteps.dtype)

    if schedule == "sigma_ramp":
        # Monotonic with noise level: low-noise -> smaller alpha, high-noise -> larger alpha.
        timesteps = pipe.scheduler.timesteps
        sigmas = pipe.scheduler.sigmas
        timestep_values = timestep_cpu.reshape(-1)
        timestep_index = torch.argmin((timesteps - timestep_values[0]).abs())
        sigma_t = sigmas[timestep_index].to(dtype=torch.float32)
        sigma_min = torch.min(sigmas).to(dtype=torch.float32)
        sigma_max = torch.max(sigmas).to(dtype=torch.float32)
        sigma_range = (sigma_max - sigma_min).clamp_min(1e-8)
        alpha = (sigma_t - sigma_min) / sigma_range
        return (alpha * weight).to(dtype=pipe.torch_dtype, device=pipe.device)

    if schedule == "scheduler_weight":
        alpha = pipe.scheduler.training_weight(timestep_cpu.reshape(-1)[:1])
        return (alpha * weight).to(dtype=pipe.torch_dtype, device=pipe.device)

    if schedule == "gaussian_timestep":
        timesteps = pipe.scheduler.timesteps.to(dtype=torch.float32)
        steps = float(getattr(pipe.scheduler, 'num_train_timesteps', 1000))
        gaussian = torch.exp(-2 * ((timesteps - steps / 2) / steps) ** 2)
        timestep_values = timestep_cpu.reshape(-1).to(dtype=timesteps.dtype)
        timestep_index = torch.argmin((timesteps - timestep_values[0]).abs())
        gaussian_timestep_value = gaussian[timestep_index]
        gaussian_min = torch.min(gaussian)
        gaussian_max = torch.max(gaussian)
        gaussian_range = (gaussian_max - gaussian_min).clamp_min(1e-8)
        alpha = (gaussian_timestep_value - gaussian_min) / gaussian_range
        return (alpha * weight).to(dtype=pipe.torch_dtype, device=pipe.device)

    raise ValueError(f"Unknown trajectory_correction_schedule: {schedule}")



def _resolve_adaptive_aug_strength(pipe: BasePipeline, timestep):
    if not bool(getattr(pipe, 'adaptive_aug', False)):
        return 1.0

    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas
    if isinstance(timestep, torch.Tensor):
        timestep_values = timestep.detach().to(device=timesteps.device, dtype=timesteps.dtype).reshape(-1)
    else:
        timestep_values = torch.tensor([timestep], device=timesteps.device, dtype=timesteps.dtype)

    timestep_index = torch.argmin((timesteps - timestep_values[0]).abs())
    sigma_t = sigmas[timestep_index].to(dtype=torch.float32)

    gamma = max(float(getattr(pipe, 'adaptive_aug_gamma', 2.0)), 1e-6)
    min_strength = float(getattr(pipe, 'adaptive_aug_min_strength', 0.0))
    min_strength = max(0.0, min(1.0, min_strength))

    # Low-noise repair targets are the most unstable. Fade augmentation out below
    # sigma_low, restore it by sigma_high, and shape the transition with gamma.
    sigma_low = float(getattr(pipe, 'adaptive_aug_sigma_low', 0.05))
    sigma_high = float(getattr(pipe, 'adaptive_aug_sigma_high', 0.30))
    sigma_low = max(0.0, sigma_low)
    sigma_high = max(sigma_low + 1e-6, sigma_high)

    p = ((sigma_t - sigma_low) / (sigma_high - sigma_low)).clamp(0.0, 1.0)
    strength = min_strength + (1.0 - min_strength) * torch.pow(p, gamma)
    return strength.to(dtype=pipe.torch_dtype, device=pipe.device)


def _rand_aug_allowed_for_timestep(pipe: BasePipeline, timestep):
    threshold = float(getattr(pipe, 'rand_aug_sigma_threshold', 0.5))
    if threshold <= 0.0:
        return True

    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas
    if isinstance(timestep, torch.Tensor):
        timestep_values = timestep.detach().to(device=timesteps.device, dtype=timesteps.dtype).reshape(-1)
    else:
        timestep_values = torch.tensor([timestep], device=timesteps.device, dtype=timesteps.dtype)

    timestep_index = torch.argmin((timesteps - timestep_values[0]).abs())
    sigma_t = sigmas[timestep_index].to(dtype=torch.float32)
    return bool((sigma_t >= threshold).item())


def _build_video_aug_training_pair(pipe: BasePipeline, inputs, noise, timestep, aug_latents=None):
    clean_latents = inputs["input_latents"]

    x_t_clean = pipe.scheduler.add_noise(clean_latents, noise, timestep)
    v_clean_t = pipe.scheduler.training_target(clean_latents, noise, timestep)

    if aug_latents is None:
        return x_t_clean, v_clean_t

    aug_latents = _align_aug_latents_to_clean(clean_latents, aug_latents, inputs)
    aug_strength = _resolve_adaptive_aug_strength(pipe, timestep)
    aug_latents = clean_latents + (aug_latents - clean_latents) * aug_strength
    x_t_aug = pipe.scheduler.add_noise(aug_latents, noise, timestep)

    # Special mode: construct velocity target directly from augmented noisy sample.
    trajectory_schedule, _ = _resolve_trajectory_correction_config(pipe)
    if trajectory_schedule == "augmented_velocity":
        if isinstance(timestep, torch.Tensor):
            timestep_values = timestep.detach().to(device=pipe.scheduler.timesteps.device, dtype=pipe.scheduler.timesteps.dtype).reshape(-1)
        else:
            timestep_values = torch.tensor([timestep], device=pipe.scheduler.timesteps.device, dtype=pipe.scheduler.timesteps.dtype)
        timestep_index = torch.argmin((pipe.scheduler.timesteps - timestep_values[0]).abs())
        sigma_t = pipe.scheduler.sigmas[timestep_index].to(dtype=pipe.torch_dtype, device=pipe.device)
        # sigma_t = pipe.scheduler.sigmas[timestep_index].to(dtype=pipe.torch_dtype, device=pipe.device).clamp_min(1e-8)
        v_t = (x_t_aug.float() - clean_latents.float()) / sigma_t
        return x_t_aug, v_t

    alpha = _resolve_trajectory_correction_alpha(pipe, timestep)
    v_t = v_clean_t.float() + (x_t_aug.float() - x_t_clean.float()) * alpha
    return x_t_aug, v_t



def _apply_motion_latent_noise(pipe: BasePipeline, inputs, noise, timestep, start_motion):
    if not getattr(pipe, 'add_noise_to_motion_latent', False):
        return
    if "y" not in inputs or inputs["y"] is None or inputs.get("clean_motion_latent") is None:
        return

    clean_motion_latents = inputs["clean_motion_latent"]
    timestep_cpu = timestep.cpu()
    timestep_index = torch.argmin((pipe.scheduler.timesteps - timestep_cpu).abs())
    sigma = pipe.scheduler.sigmas[timestep_index].to(dtype=pipe.torch_dtype, device=pipe.device)
    if getattr(pipe, 'motion_latent_shared_noise', False):
        motion_noise = noise[0, :, :clean_motion_latents.shape[1], :, :].to(device=pipe.device, dtype=pipe.torch_dtype)
    else:
        motion_noise = torch.randn_like(clean_motion_latents)
    noisy_motion_latents = ((1.0 - sigma) * clean_motion_latents + sigma * motion_noise).unsqueeze(0)
    num_anchor_latents = start_motion
    num_motion_latents = getattr(pipe, 'num_motion_latents', 1)
    inputs["y"][:, 4:, num_anchor_latents:num_anchor_latents + num_motion_latents, :, :] = noisy_motion_latents.to(inputs["y"].dtype)



def _run_training_pass(pipe: BasePipeline, inputs, timestep, noise, start_motion, aug_latents=None, compute_loss=True):
    run_inputs = dict(inputs)
    if run_inputs.get("y") is not None:
        run_inputs["y"] = run_inputs["y"].clone()

    run_inputs["noise"] = noise
    run_inputs["latents"], training_target = _build_video_aug_training_pair(
        pipe,
        run_inputs,
        noise,
        timestep,
        aug_latents=aug_latents,
    )

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **run_inputs, timestep=timestep)

    loss = None
    if compute_loss:
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * pipe.scheduler.training_weight(timestep)

    return run_inputs, training_target, noise_pred, loss



def _compute_x1_pred_aug_latents(pipe: BasePipeline, inputs, timestep, noise, start_motion, with_grad=False):
    if with_grad:
        run_inputs, _, noise_pred, loss = _run_training_pass(
            pipe,
            inputs,
            timestep,
            noise,
            start_motion,
            compute_loss=True,
        )
        x_1_pred = pipe.scheduler.step(noise_pred, timestep, run_inputs["latents"], to_final=True, self_corr=False)
        return x_1_pred.detach(), loss

    with torch.no_grad():
        run_inputs, _, noise_pred, _ = _run_training_pass(
            pipe,
            inputs,
            timestep,
            noise,
            start_motion,
            compute_loss=False,
        )
        x_1_pred = pipe.scheduler.step(noise_pred, timestep, run_inputs["latents"], to_final=True, self_corr=False)
    return x_1_pred.detach(), None


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    preset_timestep = inputs.get("train_sampled_timestep", None)
    if preset_timestep is None:
        _configure_training_timesteps(pipe, inputs)
    elif not hasattr(pipe.scheduler, "timesteps") or len(pipe.scheduler.timesteps) == 0:
        # Fallback for unexpected call paths that did not pre-configure scheduler.
        _configure_training_timesteps(pipe, inputs)
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    anchor_latent_vid = inputs.get("anchor_latent_vid", None)
    start_motion = 0
    if anchor_latent_vid is not None and anchor_latent_vid.shape[1] > 0:
        inputs["input_latents"] = torch.cat([anchor_latent_vid.unsqueeze(0), inputs["input_latents"]], dim=2)
        start_motion = anchor_latent_vid.shape[1]

    if preset_timestep is None:
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    else:
        if isinstance(preset_timestep, torch.Tensor):
            timestep = preset_timestep.to(dtype=pipe.torch_dtype, device=pipe.device).reshape(-1)[:1]
        else:
            timestep = torch.tensor([preset_timestep], dtype=pipe.torch_dtype, device=pipe.device)
    noise = torch.randn_like(inputs["input_latents"])
    video_aug_prob = float(getattr(pipe, "video_aug_prob", 1.0))
    mode_flags = {
        "rand_aug": bool(getattr(pipe, "rand_aug", False)),
        "mixed_aug": bool(getattr(pipe, "mixed_aug", False)),
        "use_self_pred_aug": bool(getattr(pipe, "use_self_pred_aug", False)),
    }
    enabled_modes = [name for name, enabled in mode_flags.items() if enabled]
    if len(enabled_modes) > 1:
        raise ValueError("Enable only one augmentation mode at a time: rand_aug, mixed_aug, or use_self_pred_aug.")

    selected_mode = None
    if enabled_modes and torch.rand(1).item() < video_aug_prob:
        only_mode = enabled_modes[0]
        if only_mode == "mixed_aug":
            selected_mode = random.choice(["rand_aug", "use_self_pred_aug"])
        else:
            selected_mode = only_mode

    if selected_mode == "use_self_pred_aug":
        first_backprop = bool(getattr(pipe, "self_pred_aug_backprop_first", False))
        self_pred_aug, first_loss = _compute_x1_pred_aug_latents(
            pipe,
            inputs,
            timestep,
            noise,
            start_motion,
            with_grad=first_backprop,
        )
        _, _, _, second_loss = _run_training_pass(
            pipe,
            inputs,
            timestep,
            noise,
            start_motion,
            aug_latents=self_pred_aug,
            compute_loss=True,
        )
        if first_backprop and first_loss is not None:
            return 0.5 * (first_loss + second_loss)
        return second_loss

    rand_aug_latents = None
    if selected_mode == "rand_aug" and _rand_aug_allowed_for_timestep(pipe, timestep):
        rand_aug_latents = inputs.get("input_latents_aug", None)

    _, _, _, loss = _run_training_pass(
        pipe,
        inputs,
        timestep,
        noise,
        start_motion,
        aug_latents=rand_aug_latents,
        compute_loss=True,
    )
    return loss

def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"], shift=_resolve_sigma_shift(pipe, inputs))
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student, shift=_resolve_sigma_shift(pipe, inputs_shared))
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True, shift=_resolve_sigma_shift(pipe, inputs_shared))
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps, shift=_resolve_sigma_shift(pipe, inputs_shared))
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps): 
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8, shift=_resolve_sigma_shift(pipe, inputs_shared))
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
