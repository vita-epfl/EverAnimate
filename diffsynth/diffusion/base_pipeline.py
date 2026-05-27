from PIL import Image
import torch
import numpy as np
from einops import repeat, reduce
from typing import Union
from ..core import AutoTorchModule, AutoWrappedLinear, load_state_dict, ModelConfig
from ..utils.lora import GeneralLoRALoader
from ..models.model_loader import ModelPool
from ..utils.controlnet import ControlNetInput
from typing import Optional
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import random

class MotionBlur:
    """Apply motion blur to simulate camera/object motion"""
    def __init__(self, kernel_size=15, angle_range=(-45, 45)):
        self.kernel_size = kernel_size
        self.angle_range = angle_range
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image with motion blur applied
        """
        import random
        import math
        
        # Convert to tensor
        img_tensor = T.ToTensor()(img).unsqueeze(0)  # (1, C, H, W)
        
        # Random angle for motion direction
        angle = random.uniform(*self.angle_range)
        
        # Create motion blur kernel
        kernel = torch.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        
        # Calculate motion direction
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        # Draw a line in the kernel (motion blur direction)
        for i in range(self.kernel_size):
            offset = i - center
            x = int(center + offset * cos_angle)
            y = int(center + offset * sin_angle)
            if 0 <= x < self.kernel_size and 0 <= y < self.kernel_size:
                kernel[y, x] = 1.0
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Expand kernel for each channel
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(img_tensor.shape[1], 1, 1, 1)  # (C, 1, K, K)
        
        # Apply convolution (motion blur)
        padding = self.kernel_size // 2
        blurred = F.conv2d(img_tensor, kernel, padding=padding, groups=img_tensor.shape[1])
        
        # Convert back to PIL
        blurred = blurred.squeeze(0).clamp(0, 1)
        blurred = T.ToPILImage()(blurred)
        
        return blurred


class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        output_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.output_params = output_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def fetch_input_params(self):
        params = []
        if self.input_params is not None:
            for param in self.input_params:
                params.append(param)
        if self.input_params_posi is not None:
            for _, param in self.input_params_posi.items():
                params.append(param)
        if self.input_params_nega is not None:
            for _, param in self.input_params_nega.items():
                params.append(param)
        params = sorted(list(set(params)))
        return params
    
    def fetch_output_params(self):
        params = []
        if self.output_params is not None:
            for param in self.output_params:
                params.append(param)
        return params

    def process(self, pipe, **kwargs) -> dict:
        return {}
    
    def post_process(self, pipe, **kwargs) -> dict:
        return {}


class BasePipeline(torch.nn.Module):

    def __init__(
        self,
        device="cuda", torch_dtype=torch.float16,
        height_division_factor=64, width_division_factor=64,
        time_division_factor=None, time_division_remainder=None,
    ):
        super().__init__()
        # The device and torch_dtype is used for the storage of intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # VRAM management
        self.vram_management_enabled = False
        # Pipeline Unit Runner
        self.unit_runner = PipelineUnitRunner()
        # LoRA Loader
        self.lora_loader = GeneralLoRALoader
        
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self


    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
            return height, width, num_frames



    def apply_augmentation_to_images_condition(self, images, same_augmentation=True, k=50, augmentation_strength=1.0):
        def sample_params():
            return {
                "apply_color_cast": random.random() < 0.8,
                "color_cast": [
                    random.uniform(-0.1, 0.1),
                    random.uniform(-0.1, 0.1),
                    random.uniform(-0.1, 0.1),
                ],
                "apply_blur": random.random() < 0.5,
                "blur_kernel_size": random.choice([3, 5]),
                "blur_sigma": random.uniform(0.15, 0.8),
            }

        def apply_color_cast(img_tensor, params, strength):
            if not params.get("apply_color_cast", False) or strength <= 0:
                return img_tensor

            # Mild RGB bias to simulate gentle color shift.
            cast = torch.tensor(
                params["color_cast"], dtype=img_tensor.dtype, device=img_tensor.device
            ).view(3, 1, 1)
            cast = cast * strength
            return (img_tensor + cast).clamp(0, 1)

        def apply_blur(img_tensor, params, strength):
            if not params.get("apply_blur", False) or strength <= 0:
                return img_tensor
            kernel_size = params["blur_kernel_size"]
            sigma = max(1e-6, params["blur_sigma"] * strength)
            return TF.gaussian_blur(
                img_tensor,
                kernel_size=[kernel_size, kernel_size],
                sigma=[sigma, sigma],
            )

        def apply_params(img, params, strength):
            img_tensor = TF.to_tensor(img)
            try:
                if img_tensor.ndim == 3 and img_tensor.shape[0] == 3:
                    img_tensor = apply_color_cast(img_tensor, params, strength)
                    img_tensor = apply_blur(img_tensor, params, strength)
            except Exception as e:
                print(f"Video augmentation failed: {e}")
            return img_tensor.clamp(0, 1)

        shared_params = sample_params() if same_augmentation else None

        augmented_images = []
        start_augmentation_idx = max(0, int(k))
        strength = float(max(0.0, min(1.0, augmentation_strength)))
        for idx, image in enumerate(images):
            if idx >= start_augmentation_idx:
                params = shared_params if same_augmentation else sample_params()
                augmented_tensor = apply_params(image, params, strength)
            else:
                augmented_tensor = TF.to_tensor(image)
            tensor_image = augmented_tensor.mul(2.0).sub(1.0).unsqueeze(0)
            augmented_images.append(tensor_image)

        return augmented_images


    def apply_augmentation_to_images(self, images, same_augmentation=True, k=50, augmentation_strength=1.0):
        def sample_patch_specs(params):
            min_ratio, max_ratio = params["erase_patch_ratio_range"]
            min_downsample, max_downsample = params["erase_downsample_factor_range"]
            specs = []
            for _ in range(params["erase_patch_count"]):
                specs.append({
                    "height_ratio": random.uniform(min_ratio, max_ratio),
                    "width_ratio": random.uniform(min_ratio, max_ratio),
                    "top_ratio": random.uniform(0.0, 1.0),
                    "left_ratio": random.uniform(0.0, 1.0),
                    "downsample_factor": random.randint(min_downsample, max_downsample),
                })
            return specs

        def sample_params():
            params = {
                "apply_clahe": random.random() < 0.9,
                "clahe_clip_limit": random.uniform(1.0, 64.0),
                "clahe_tile_grid_size": random.randint(1, 32),
                "apply_oversaturation": random.random() < 0.3,
                "saturation_boost": random.uniform(1.3, 3.0),
                "apply_brightness": random.random() < 0.1,
                "brightness_scale": random.uniform(0.8, 1.2),

                "apply_blur": random.random() < 0.0,
                "blur_kernel_size": random.choice([3, 5, 7]),
                "blur_sigma": random.uniform(0.2, 1.2),

                "apply_detail_erase": random.random() < 0.9,
                "erase_patch_count": random.randint(1, 4),
                "erase_patch_ratio_range": (0.1, 0.3),
                "erase_downsample_factor_range": (8, 32),
            }
            params["erase_patch_specs"] = sample_patch_specs(params)
            return params

        def apply_clahe(img_tensor, params):
            if not params.get("apply_clahe", False):
                return img_tensor
            img_np = (img_tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            tile_grid_size = params["clahe_tile_grid_size"]
            clahe = cv2.createCLAHE(
                clipLimit=params["clahe_clip_limit"],
                tileGridSize=(tile_grid_size, tile_grid_size),
            )
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            out_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            out_tensor = torch.from_numpy(out_np).to(dtype=torch.float32) / 255.0
            return out_tensor.permute(2, 0, 1)

        def apply_color_jitter(img_tensor, params, strength):
            apply_oversaturation = params.get("apply_oversaturation", False)
            apply_brightness = params.get("apply_brightness", False)
            if not apply_oversaturation and not apply_brightness:
                return img_tensor
            out = img_tensor
            if apply_oversaturation:
                saturation_factor = 1.0 + (params["saturation_boost"] - 1.0) * strength
                out = TF.adjust_saturation(out, saturation_factor)
            if apply_brightness:
                brightness_factor = 1.0 + (params["brightness_scale"] - 1.0) * strength
                out = TF.adjust_brightness(out, brightness_factor)
            return out.clamp(0, 1)

        def apply_blur(img_tensor, params, strength):
            if not params.get("apply_blur", False) or strength <= 0:
                return img_tensor
            kernel_size = params["blur_kernel_size"]
            sigma = max(1e-6, params["blur_sigma"] * strength)
            return TF.gaussian_blur(img_tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        def apply_detail_erase(img_tensor, params, strength):
            if not params.get("apply_detail_erase", False):
                return img_tensor
            if strength <= 0:
                return img_tensor
            out = img_tensor.clone()
            _, h, w = out.shape
            patch_specs = params["erase_patch_specs"]
            patch_count = max(1, int(round(params["erase_patch_count"] * strength)))
            for patch_spec in patch_specs[:patch_count]:
                patch_h = max(8, int(h * patch_spec["height_ratio"]))
                patch_w = max(8, int(w * patch_spec["width_ratio"]))
                patch_h = min(patch_h, h)
                patch_w = min(patch_w, w)
                max_top = max(0, h - patch_h)
                max_left = max(0, w - patch_w)
                y1 = min(max_top, int(max_top * patch_spec["top_ratio"]))
                x1 = min(max_left, int(max_left * patch_spec["left_ratio"]))
                y2 = y1 + patch_h
                x2 = x1 + patch_w
                patch = out[:, y1:y2, x1:x2].unsqueeze(0)
                downsample_factor = patch_spec["downsample_factor"]
                small_h = max(1, patch_h // downsample_factor)
                small_w = max(1, patch_w // downsample_factor)
                patch_small = torch.nn.functional.interpolate(patch, size=(small_h, small_w), mode="area")
                patch_restore = torch.nn.functional.interpolate(patch_small, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
                out[:, y1:y2, x1:x2] = patch_restore.squeeze(0)
            return out.clamp(0, 1)

        def apply_params(img, params, strength):
            img_tensor = TF.to_tensor(img)
            try:
                if img_tensor.ndim == 3 and img_tensor.shape[0] == 3:
                    img_tensor = apply_clahe(img_tensor, params)
                    img_tensor = apply_color_jitter(img_tensor, params, strength)
                    img_tensor = apply_blur(img_tensor, params, strength)
                    img_tensor = apply_detail_erase(img_tensor, params, strength)
            except Exception as e:
                print(f"Video augmentation failed: {e}")
            return img_tensor.clamp(0, 1)

        shared_params = sample_params() if same_augmentation else None

        augmented_images = []
        start_augmentation_idx = max(0, int(k))
        strength = float(max(0.0, min(1.0, augmentation_strength)))
        for idx, image in enumerate(images):
            if idx >= start_augmentation_idx:
                params = shared_params if same_augmentation else sample_params()
                augmented_tensor = apply_params(image, params, strength)
            else:
                augmented_tensor = TF.to_tensor(image)
            tensor_image = augmented_tensor.mul(2.0).sub(1.0).unsqueeze(0)
            augmented_images.append(tensor_image)

        return augmented_images



    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image


    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a list of PIL.Image to torch.Tensor
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video


    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image


    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]
        return video


    def load_models_to_device(self, model_names):
        if self.vram_management_enabled:
            # offload models
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        if hasattr(model, "offload"):
                            model.offload()
                        else:
                            for module in model.modules():
                                if hasattr(module, "offload"):
                                    module.offload()
            torch.cuda.empty_cache()
            # onload models
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        if hasattr(model, "onload"):
                            model.onload()
                        else:
                            for module in model.modules():
                                if hasattr(module, "onload"):
                                    module.onload()


    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise

        
    def get_vram(self):
        return torch.cuda.mem_get_info(self.device)[1] / (1024 ** 3)
    
    def get_module(self, model, name):
        if "." in name:
            name, suffix = name[:name.index(".")], name[name.index(".") + 1:]
            if name.isdigit():
                return self.get_module(model[int(name)], suffix)
            else:
                return self.get_module(getattr(model, name), suffix)
        else:
            return getattr(model, name)
    
    def freeze_except(self, model_names):
        self.eval()
        self.requires_grad_(False)
        for name in model_names:
            module = self.get_module(self, name)
            if module is None:
                print(f"No {name} models in the pipeline. We cannot enable training on the model. If this occurs during the data processing stage, it is normal.")
                continue
            module.train()
            module.requires_grad_(True)
                
    
    def blend_with_mask(self, base, addition, mask):
        return base * (1 - mask) + addition * mask
    
    
    def step(self, scheduler, latents, progress_id, noise_pred, input_latents=None, inpaint_mask=None, **kwargs):
        timestep = scheduler.timesteps[progress_id]
        if inpaint_mask is not None:
            noise_pred_expected = scheduler.return_to_timestep(scheduler.timesteps[progress_id], latents, input_latents)
            noise_pred = self.blend_with_mask(noise_pred_expected, noise_pred, inpaint_mask)
        latents_next = scheduler.step(noise_pred, timestep, latents)
        return latents_next
    
    
    def split_pipeline_units(self, model_names: list[str]):
        return PipelineUnitGraph().split_pipeline_units(self.units, model_names)
    
    
    def flush_vram_management_device(self, device):
        for module in self.modules():
            if isinstance(module, AutoTorchModule):
                module.offload_device = device
                module.onload_device = device
                module.preparing_device = device
                module.computation_device = device
                
    
    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=None,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        lora_loader = self.lora_loader(torch_dtype=self.torch_dtype, device=self.device)
        lora = lora_loader.convert_state_dict(lora)
        if hotload is None:
            hotload = hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled")
        if hotload:
            if not (hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled")):
                raise ValueError("VRAM Management is not enabled. LoRA hotloading is not supported.")
            updated_num = 0
            for _, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    name = module.name
                    lora_a_name = f'{name}.lora_A.weight'
                    lora_b_name = f'{name}.lora_B.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        updated_num += 1
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
            print(f"{updated_num} tensors are patched by LoRA. You can use `pipe.clear_lora()` to clear all LoRA layers.")
        else:
            lora_loader.fuse_lora_to_base_model(module, lora, alpha=alpha)
            
            
    def clear_lora(self):
        cleared_num = 0
        for name, module in self.named_modules():
            if isinstance(module, AutoWrappedLinear):
                if hasattr(module, "lora_A_weights"):
                    if len(module.lora_A_weights) > 0:
                        cleared_num += 1
                    module.lora_A_weights.clear()
                if hasattr(module, "lora_B_weights"):
                    module.lora_B_weights.clear()
        print(f"{cleared_num} LoRA layers are cleared.")
        
    
    def download_and_load_models(self, model_configs: list[ModelConfig] = [], vram_limit: float = None):
        model_pool = ModelPool()
        for model_config in model_configs:
            model_config.download_if_necessary()
            vram_config = model_config.vram_config()
            vram_config["computation_dtype"] = vram_config["computation_dtype"] or self.torch_dtype
            vram_config["computation_device"] = vram_config["computation_device"] or self.device
            model_pool.auto_load_model(
                model_config.path,
                vram_config=vram_config,
                vram_limit=vram_limit,
                clear_parameters=model_config.clear_parameters,
            )
        return model_pool
    
    
    def check_vram_management_state(self):
        vram_management_enabled = False
        for module in self.children():
            if hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled"):
                vram_management_enabled = True
        return vram_management_enabled
    
    
    def cfg_guided_model_fn(self, model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega, **inputs_others):
        noise_pred_posi = model_fn(**inputs_posi, **inputs_shared, **inputs_others)
        if cfg_scale != 1.0:
            noise_pred_nega = model_fn(**inputs_nega, **inputs_shared, **inputs_others)
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi
        return noise_pred


class PipelineUnitGraph:
    def __init__(self):
        pass
    
    def build_edges(self, units: list[PipelineUnit]):
        # Establish dependencies between units
        # to search for subsequent related computation units.
        last_compute_unit_id = {}
        edges = []
        for unit_id, unit in enumerate(units):
            for input_param in unit.fetch_input_params():
                if input_param in last_compute_unit_id:
                    edges.append((last_compute_unit_id[input_param], unit_id))
            for output_param in unit.fetch_output_params():
                last_compute_unit_id[output_param] = unit_id
        return edges
    
    def build_chains(self, units: list[PipelineUnit]):
        # Establish updating chains for each variable
        # to track their computation process.
        params = sum([unit.fetch_input_params() + unit.fetch_output_params() for unit in units], [])
        params = sorted(list(set(params)))
        chains = {param: [] for param in params}
        for unit_id, unit in enumerate(units):
            for param in unit.fetch_output_params():
                chains[param].append(unit_id)
        return chains
    
    def search_direct_unit_ids(self, units: list[PipelineUnit], model_names: list[str]):
        # Search for units that directly participate in the model's computation.
        related_unit_ids = []
        for unit_id, unit in enumerate(units):
            for model_name in model_names:
                if unit.onload_model_names is not None and model_name in unit.onload_model_names:
                    related_unit_ids.append(unit_id)
                    break
        return related_unit_ids
    
    def search_related_unit_ids(self, edges, start_unit_ids, direction="target"):
        # Search for subsequent related computation units.
        related_unit_ids = [unit_id for unit_id in start_unit_ids]
        while True:
            neighbors = []
            for source, target in edges:
                if direction == "target" and source in related_unit_ids and target not in related_unit_ids:
                    neighbors.append(target)
                elif direction == "source" and source not in related_unit_ids and target in related_unit_ids:
                    neighbors.append(source)
            neighbors = sorted(list(set(neighbors)))
            if len(neighbors) == 0:
                break
            else:
                related_unit_ids.extend(neighbors)
        related_unit_ids = sorted(list(set(related_unit_ids)))
        return related_unit_ids
    
    def search_updating_unit_ids(self, units: list[PipelineUnit], chains, related_unit_ids):
        # If the input parameters of this subgraph are updated outside the subgraph,
        # search for the units where these updates occur.
        first_compute_unit_id = {}
        for unit_id in related_unit_ids:
            for param in units[unit_id].fetch_input_params():
                if param not in first_compute_unit_id:
                    first_compute_unit_id[param] = unit_id
        updating_unit_ids = []
        for param in first_compute_unit_id:
            unit_id = first_compute_unit_id[param]
            chain = chains[param]
            if unit_id in chain and chain.index(unit_id) != len(chain) - 1:
                for unit_id_ in chain[chain.index(unit_id) + 1:]:
                    if unit_id_ not in related_unit_ids:
                        updating_unit_ids.append(unit_id_)
        related_unit_ids.extend(updating_unit_ids)
        related_unit_ids = sorted(list(set(related_unit_ids)))
        return related_unit_ids
    
    def split_pipeline_units(self, units: list[PipelineUnit], model_names: list[str]):
        # Split the computation graph,
        # separating all model-related computations.
        related_unit_ids = self.search_direct_unit_ids(units, model_names)
        edges = self.build_edges(units)
        chains = self.build_chains(units)
        while True:
            num_related_unit_ids = len(related_unit_ids)
            related_unit_ids = self.search_related_unit_ids(edges, related_unit_ids, "target")
            related_unit_ids = self.search_updating_unit_ids(units, chains, related_unit_ids)
            if len(related_unit_ids) == num_related_unit_ids:
                break
            else:
                num_related_unit_ids = len(related_unit_ids)
        related_units = [units[i] for i in related_unit_ids]
        unrelated_units = [units[i] for i in range(len(units)) if i not in related_unit_ids]
        return related_units, unrelated_units


class PipelineUnitRunner:
    def __init__(self):
        pass

    def __call__(self, unit: PipelineUnit, pipe: BasePipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict) -> tuple[dict, dict]:
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega)
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega