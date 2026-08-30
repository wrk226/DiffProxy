# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import PIL
import math
import torch
import torch.nn.functional as F
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, T2IAdapter, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
 
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ..loaders import CustomAdapterMixin
from ..models.attention_processor import (
    DecoupledMVRowSelfAttnProcessor2_0,
    set_unet_2d_condition_attn_processor,
)
from mvadapter.utils.epi_spad_cached import build_block_epipolar_mask_and_plucker_cached
from mvadapter.utils.epi import get_epi_projection
from mvadapter.utils.view_bbox import (
    expand_bbox_norm,
    compute_affine_from_main_bbox,
    map_child_bbox_to_main_canvas,
    rect_to_mask_on_grid,
    estimate_patch_grid_dims,
)
XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class MVAdapterI2MVSDPipeline(StableDiffusionPipeline, CustomAdapterMixin):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

        # 默认开关：便于直接使用 pipeline 推理时无需依赖 system 注入
        self.sync_noise_across_views = False
        self.cd_num_tasks = 2
        

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.prepare_latents
    def prepare_image_latents(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
        add_noise=True,
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(
                        self.vae.encode(image[i : i + 1]), generator=generator[i]
                    )
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(
                    self.vae.encode(image), generator=generator
                )

            init_latents = self.vae.config.scaling_factor * init_latents

        if (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] == 0
        ):
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt, dim=0
            )
        elif (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # get latents
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents

    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        num_empty_images=0,  # for concat in batch like ImageDream
    ):
        assert hasattr(
            self, "control_image_processor"
        ), "control_image_processor is not initialized"

        image = self.control_image_processor.preprocess(
            image, height=height, width=width
        ).to(dtype=torch.float32)

        if num_empty_images > 0:
            image = torch.cat(
                [image, torch.zeros_like(image[:num_empty_images])], dim=0
            )

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt  # always 1 for control image

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # NEW
        num_views: Optional[int] = 4,
        mv_scale: float = 1.0,
        # Camera or geometry condition
        source_image: Optional[PipelineImageInput] = None, # unet input and output
        control_image: Optional[PipelineImageInput] = None, # unet residual
        control_conditioning_scale: Optional[float] = 1.0,
        control_conditioning_factor: float = 1.0,
        # Image condition
        reference_image: Optional[PipelineImageInput] = None, # unet cross attention
        reference_conditioning_scale: Optional[float] = 1.0,
        reference_mode: str = "residual",  # "residual" or "cross_attention"
        # Optional. controlnet
        controlnet_image: Optional[PipelineImageInput] = None,
        controlnet_conditioning_scale: Optional[float] = 1.0,
        # NEW: Save intermediate steps
        save_intermediate_steps: bool = False,
        intermediate_output_dir: Optional[str] = None,
        intermediate_save_frequency: int = 5,  # Save every N steps
        # NEW: SDEdit support - 从图像或 latent 初始化而非纯噪声
        sdedit_image: Optional[torch.Tensor] = None,  # (B, 3, H, W) in [-1,1]，会先 encode 再加噪
        sdedit_latent: Optional[torch.Tensor] = None,  # (B, 4, H/8, W/8)，已经 encode+scaled 的 latent，直接加噪
        sdedit_strength: float = 0.3,  # 噪声强度，0=不加噪，1=等价于正常生成
        batch=None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # === 读取可选的数据集相机参数 ===


        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        if sdedit_image is not None or sdedit_latent is not None:
            # SDEdit 模式
            sdedit_start_step = int(len(timesteps) * (1 - sdedit_strength))
            sdedit_start_step = max(0, min(sdedit_start_step, len(timesteps) - 1))
            sdedit_timestep = timesteps[sdedit_start_step]
            timesteps = timesteps[sdedit_start_step:]

            if sdedit_latent is not None:
                # 直接使用预计算的 latent（已经 encode + scaled）
                init_latents = sdedit_latent.to(device=device, dtype=prompt_embeds.dtype)
                if sdedit_strength > 0:
                    noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
                    init_latents = self.scheduler.add_noise(init_latents, noise, sdedit_timestep)
                latents = init_latents
            else:
                # 从图像 encode 再加噪
                latents = self.prepare_image_latents(
                    sdedit_image,
                    sdedit_timestep,
                    batch_size,
                    num_images_per_prompt,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    add_noise=(sdedit_strength > 0),
                )

            num_inference_steps = len(timesteps)
            print(f"[SDEdit] strength={sdedit_strength}, start_step={sdedit_start_step}, "
                  f"timestep={sdedit_timestep}, denoising {num_inference_steps} steps"
                  f", mode={'latent' if sdedit_latent is not None else 'image'}")
        else:
            # 正常模式：从纯噪声开始
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )


        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 动态同步本批次视角数到 attention processors（避免固定为初始化的 num_views）
        current_views = int(num_views)
        for _name, _proc in self.unet.attn_processors.items():
            if hasattr(_proc, 'set_num_views'):
                _proc.set_num_views(current_views)

        # 推理期：永不 drop condition/reference
        B = batch_size

        # Always define ref_hidden_states to avoid unbound when reference is dropped or missing
        ref_hidden_states = {}

        # Preprocess reference image
        if reference_image is not None:
            reference_image = self.image_processor.preprocess(reference_image)
            reference_latents = self.prepare_image_latents(
                reference_image,
                timesteps[:1].repeat(batch_size * num_images_per_prompt),  # no use
                batch_size,
                1,
                prompt_embeds.dtype,
                device,
                generator,
                add_noise=False,
            )

            # Process reference features based on mode
            if reference_mode == "residual":
                # 支持多张 reference 图像的 residual 模式
                # num_ref_imgs = max(1, reference_latents.shape[0] // batch_size)
                # num_tasks = 2
                # num_ref_imgs = reference_latents.shape[0] // num_tasks
                ref_timesteps = torch.zeros(reference_latents.shape[0], device=device, dtype=torch.long)
                # 仅使用最后一个 prompt embedding，并重复到与 reference 数量一致
                prompt_embeds_for_ref = prompt_embeds[reference_latents.shape[0]:]#[-1:].repeat(num_ref_imgs*num_tasks, 1, 1)

                ref_hidden_states_raw = {}
                with torch.no_grad():
                    self.unet(
                        reference_latents,
                        ref_timesteps,
                        encoder_hidden_states=prompt_embeds_for_ref,
                        cross_attention_kwargs={
                            "cache_hidden_states": ref_hidden_states_raw,
                            "use_ref": False,
                            "use_mv": False,
                        },
                        return_dict=False,
                    )

                ref_hidden_states = ref_hidden_states_raw

                if self.do_classifier_free_guidance:
                    ref_hidden_states = {
                        k: torch.cat([torch.zeros_like(v), v], dim=0)
                        for k, v in ref_hidden_states.items()
                    }
            elif reference_mode in ["sapiens", "dinov2"]:
                # ---------------- ViT-based feature extraction branch -----------------
                assert hasattr(self, "vit") and hasattr(self, "layer_projectors"), (
                    "ViT components are not properly initialized in the pipeline"
                )

                # VaeImageProcessor.preprocess() 将像素缩放到 [-1, 1]，但 ViT 期待 [0, 1]
                # 因此在送入 ViT 之前再映射回 [0, 1]
                # num_task = 2
                ref_imgs = ((reference_image + 1) / 2).clamp(0, 1).to(device, dtype=torch.float32)

                B, C, H_img, W_img = ref_imgs.shape

                # 根据特征提取器类型设置输入尺寸
                feature_extractor = self.extractor_type
                if feature_extractor == "sapiens":
                    tgt_h = tgt_w = 1024
                elif feature_extractor == "dinov2":
                    tgt_h = tgt_w = 518
                elif feature_extractor == "vggt":
                    tgt_h = tgt_w = 518
                else:
                    raise NotImplementedError

                scale = min(tgt_h / H_img, tgt_w / W_img)
                new_h = int(round(H_img * scale))
                new_w = int(round(W_img * scale))

                resized = F.interpolate(ref_imgs, size=(new_h, new_w), mode="bilinear", align_corners=False)
                pad_top = (tgt_h - new_h) // 2
                pad_bottom = tgt_h - new_h - pad_top
                pad_left = (tgt_w - new_w) // 2
                pad_right = tgt_w - new_w - pad_left

                padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

                mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
                std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
                normed = (padded - mean) / std
                num_tasks = self.cd_num_tasks
                if normed.shape[0] % num_tasks != 0:
                    raise ValueError(
                        f"The number of reference images ({normed.shape[0]}) must be divisible by the number of tasks ({num_tasks}) for Epi-Attention.")

                num_ref_views_per_task = normed.shape[0] // num_tasks
                with torch.no_grad():
                    if feature_extractor == "dinov2":
                        vit_out = self.vit(normed)
                        # DINOv2 returns dict with 'x_norm_patchtokens' (patch tokens) and other features
                        if isinstance(vit_out, dict):
                            if 'x_norm_patchtokens' in vit_out:
                                vit_feats = vit_out['x_norm_patchtokens']
                            else:
                                vit_feats = vit_out.get('last_hidden_state', next(iter(vit_out.values())))
                        else:
                            vit_feats = vit_out
                        vit_feats = vit_feats.float()
                    else:
                        raise ValueError(f"Unsupported feature_extractor: {feature_extractor}")

                # Extract secondary DINOv2 features for semantic fusion if available
                dino_global = vit_feats[:, :1]
                dino_patch  = vit_feats[:, 1:]

                # Layer-wise projection merges VGGT and DINOv2 features via fusion MLP
                assert hasattr(self, "layer_projectors"), "layer_projectors missing in pipeline"
                layer_proj_cache_dino = {}
                def _hs(layer_name):
                    if layer_name.startswith("mid_block"):
                        return self.unet.config.block_out_channels[-1]
                    elif layer_name.startswith("up_blocks"):
                        # 提取 up_blocks.{idx} 中的 idx
                        parts = layer_name.split(".", 2)
                        bid = int(parts[1])
                        return list(reversed(self.unet.config.block_out_channels))[bid]
                    elif layer_name.startswith("down_blocks"):
                        # 提取 down_blocks.{idx} 中的 idx
                        parts = layer_name.split(".", 2)
                        bid = int(parts[1])
                        return self.unet.config.block_out_channels[bid]
                    else:
                        return self.unet.config.cross_attention_dim

                ref_hidden_states = {}
                for name in self.unet.attn_processors.keys():
                    hs = _hs(name)
                    key_sz = str(hs)
                    if key_sz not in layer_proj_cache_dino:
                        # Project patch tokens first
                        layer_proj_cache_dino[key_sz] = self.layer_projectors[key_sz](dino_patch)

                        # global tokens
                        proj_dino_global = self.layer_projectors[key_sz](dino_global.mean(dim=1, keepdim=True))
                        global_fused = proj_dino_global
                        layer_proj_cache_dino[f"global_{key_sz}"] = global_fused

                    # create fused patch
                    v_patch = layer_proj_cache_dino[key_sz]
                    fused_patch = v_patch

                    # fused_global = layer_proj_cache_dino[f"global_{key_sz}"]
                    # fused_tokens = torch.cat([fused_global, fused_patch], dim=1)
                    fused_tokens = fused_patch
                    feat = fused_tokens.repeat_interleave(num_images_per_prompt, dim=0)

                    if self.do_classifier_free_guidance:
                        feat = torch.cat([torch.zeros_like(feat), feat], dim=0)
                    ref_hidden_states[name] = feat

        # Build cross_attention_kwargs
        use_ref_flag = bool(self.use_ref_attn and (len(ref_hidden_states) > 0))
        cross_attention_kwargs = {
                # "num_views": num_images_per_prompt,
                "mv_scale": self.mv_scale,
                "use_mv": self.use_mv_attn,
                "densemv_scale": self.densemv_scale,
                "use_densemv": self.use_densemv_attn,
                "use_cd": self.use_cd_attn,
                "cd_scale": self.cd_scale,
                "use_ref": use_ref_flag,
                "ref_hidden_states": {k: v.clone() for k, v in ref_hidden_states.items()},
                "ref_scale": reference_conditioning_scale,
                **(self.cross_attention_kwargs or {}),
        }

        # 已移除参考特征掩码逻辑（use_ref_bbox_mask）
        # ---- 初始化变量，避免作用域问题 ----
        pixel_locations = pixel_mask = sample_coordinates = None
        spad_block_mask = spad_plucker = None
        # 匹配/相机预测（VGGT）相关逻辑已移除
        k_near_indices = None
        sample_coordinates = None
        need_camera = self.use_spad_epi
        if need_camera:

            # --- 检查是否使用数据集相机参数 ---
            use_dataset_cam = self.use_dataset_cam
            if use_dataset_cam:
                dataset_extrinsics = batch["dataset_extrinsics"]
                dataset_intrinsics = batch["dataset_intrinsics"]
                c2w4 = dataset_extrinsics.to(device)
                intr = dataset_intrinsics.to(device)
                R = c2w4[..., :3, :3]
                t = c2w4[..., :3, 3:]
                centers = t.squeeze(-1) # c2w's t is the center
            # --- 否则不再支持 VGGT 预测相机 ---
            else:
                raise ValueError("use_dataset_cam=False 时已移除VGGT相机路径")


            # --- 准备相机矩阵 ---
            Bv = intr.shape[0]
            intr4 = torch.zeros(Bv, num_views, 4, 4, device=device, dtype=intr.dtype)
            intr4[..., 3, 3] = 1.0
            intr4[..., :3, :3] = intr
            cameras = torch.cat([intr4.view(Bv, num_views, -1), c2w4.view(Bv, num_views, -1)], dim=-1)
 
        # SPAD-style epipolar mask + plucker (PyTorch3D)
        if self.use_spad_epi:
            spad_block_mask, spad_plucker = build_block_epipolar_mask_and_plucker_cached(
                intrinsics=intr.to(torch.float32),
                extrinsics_w2c=torch.linalg.inv(c2w4.to(torch.float64))[..., :3, :4].float(),
                image_size=batch['target_size'][0],
                dialate_mask=True,
                use_cache=True,
                cam_files=batch['cam_files'],
                # 无父/子拆分
            )
            # 训练时随机决定使用SPAD(1)还是不用(0)，推理时使用配置值
            current_spad_scale = self.spad_epi_scale
            if self.use_random_spad_epi_scale and self.unet.training:
                current_spad_scale = float(torch.randint(0, 2, (1,), device=self._execution_device).item())
            # 永不因 drop 而关闭
            
            cross_attention_kwargs.update({
                "use_spad_epi": self.use_spad_epi,
                "spad_epi_scale": current_spad_scale,
                "spad_block_epi_mask": spad_block_mask,
                "spad_plucker_embed": spad_plucker,
            })

        # 匹配注意力已移除

        # Preprocess control image
        adapter_state = None
        if control_image is not None:
            control_image_feature = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=1,  # NOTE: always 1 for control images
                device=device,
                dtype=latents.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            control_image_feature = control_image_feature.to(
                device=device, dtype=latents.dtype
            )
            if hasattr(self, 'cond_encoder') and self.cond_encoder is not None:
                adapter_state = self.cond_encoder(control_image_feature)
                for i, state in enumerate(adapter_state):
                    adapter_state[i] = state * control_conditioning_scale
        

        # ===== Dual-condition support for inference (cond_encoder2) =====
        # If dataset provides low-res guidance (cond2_rgb) and cond_encoder2 exists, fuse it with main branch
        if batch is not None and hasattr(self, 'cond_encoder2') and self.cond_encoder2 is not None and ('cond2_rgb' in batch):
            cond2_flat = batch['cond2_rgb'].to(device=device, dtype=latents.dtype)

            # Extract low-res adapter features
            lowres_state = self.cond_encoder2(cond2_flat)

            # If main branch missing, create zero features with same shapes
            if adapter_state is None:
                adapter_state = [torch.zeros_like(s) for s in lowres_state]

            # Build per-sample mask from use_lowres_flags if available
            T = int(self.cd_num_tasks)
            V = int(num_views) if num_views is not None else 1
            N0 = cond2_flat.shape[0]
            B = max(1, N0 // max(1, T * V))
            if 'use_lowres_flags' in batch and batch['use_lowres_flags'] is not None:
                use_lowres_flags = batch['use_lowres_flags'].to(device=device)
                if use_lowres_flags.ndim == 0:
                    use_lowres_flags = use_lowres_flags[None]
            else:
                use_lowres_flags = torch.ones(B, device=device, dtype=torch.bool)
            use_lowres_flat = use_lowres_flags.view(B, 1, 1).expand(B, T, V).reshape(-1)

            # Align low-res branch and mask batch size with main branch (handles CFG doubling)
            target_len = adapter_state[0].shape[0] if adapter_state is not None else lowres_state[0].shape[0]
            current_len = lowres_state[0].shape[0]
            if current_len != target_len:
                if target_len % current_len == 0:
                    rep = target_len // current_len
                    lowres_state = [s.repeat_interleave(rep, dim=0) for s in lowres_state]
                else:
                    # fallback: tile then crop
                    rep = (target_len + current_len - 1) // current_len
                    lowres_state = [torch.cat([s] * rep, dim=0)[:target_len] for s in lowres_state]

            # Prepare fusion weights
            w_main = float(self.cond_fusion_weight_main)
            w_low = float(self.cond_fusion_weight_lowres)

            # Build mask aligned to target_len
            mask_flat = use_lowres_flat
            if mask_flat.shape[0] != target_len:
                if target_len % mask_flat.shape[0] == 0:
                    k = target_len // mask_flat.shape[0]
                    mask_flat = mask_flat.repeat(k)
                else:
                    k = (target_len + mask_flat.shape[0] - 1) // mask_flat.shape[0]
                    mask_flat = mask_flat.repeat(k)[:target_len]
            mask = mask_flat.view(-1, 1, 1, 1).to(device=device, dtype=adapter_state[0].dtype)

            # Fuse features layer-wise
            fused_state = []
            for f_main, f_low in zip(adapter_state, lowres_state):
                if f_main is None:
                    fused_state.append(f_low)
                    continue
                if f_low is None:
                    fused_state.append(f_main)
                    continue
                fused = ((1.0 - mask) * f_main) + mask * (w_main * f_main + w_low * f_low)
                fused_state.append(fused)

            adapter_state = fused_state

                # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        # Store intermediate results if requested
        intermediate_latents = []
        intermediate_images = []
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if adapter_state is not None and i < int(num_inference_steps * control_conditioning_factor):
                    down_intrablock_additional_residuals = [
                        state.clone() for state in adapter_state
                    ]
                else:
                    down_intrablock_additional_residuals = None

                unet_add_kwargs = {}

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    added_cond_kwargs=added_cond_kwargs,
                    **unet_add_kwargs,
                ).sample

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # Save intermediate steps if requested
                if save_intermediate_steps and i % intermediate_save_frequency == 0:
                    # Store latents for later decoding
                    intermediate_latents.append(latents.clone())
                    
                    # Optionally decode and save image at this step
                    if intermediate_output_dir is not None:
                        with torch.no_grad():
                            intermediate_image = self.vae.decode(
                                latents / self.vae.config.scaling_factor, 
                                return_dict=False,
                                generator=generator
                            )[0]
                            
                            # Process image for saving
                            do_denormalize = [True] * intermediate_image.shape[0]
                            intermediate_image_processed = self.image_processor.postprocess(
                                intermediate_image, 
                                output_type="pt", 
                                do_denormalize=do_denormalize
                            )
                            
                            # Save intermediate image using torchvision
                            import torchvision.utils as vutils
                            import os
                            os.makedirs(intermediate_output_dir, exist_ok=True)
                            step_output_path = os.path.join(
                                intermediate_output_dir, 
                                f"step_{i:04d}_timestep_{t.item():04d}.png"
                            )
                            vutils.save_image(
                                intermediate_image_processed.clamp(0, 1),
                                step_output_path,
                                normalize=False,
                                nrow=num_views if hasattr(self, 'num_views') else 2,
                                padding=2,
                            )
 
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                # XLA support removed

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            # image, has_nsfw_concept = self.run_safety_checker(
            #     image, device, prompt_embeds.dtype
            # )
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    ### NEW: adapters ###
    def _init_custom_adapter(
        self,
        # Multi-view adapter
        num_views: int = 1,
        num_ref_views: int = 1,
        self_attn_processor: Any = DecoupledMVRowSelfAttnProcessor2_0,
        # Condition encoder
        cond_in_channels: int = 6,
        enable_dual_cond: bool = False,
        cond2_in_channels: int = 3,
        # For training
        copy_attn_weights: bool = True,
        zero_init_module_keys: List[str] = [],
        # Reference projector
        reference_mode: str = "residual",
        projector_hidden_dim: int = 1024,
        projector_num_layers: int = 2,
        # Dense Multi-View attention
        use_densemv_attn: bool = False,
        densemv_attn_layers: List[str] = None,
        densemv_scale: float = 1.0,
        # Multi-View attention
        use_mv_attn: bool = False,
        use_ref_attn: bool = True,
        ref_attn_mode: str = "multi",  # 新增：ref attention模式
        mv_attn_layers: List[str] = None,  # None means all layers
        mv_scale: float = 1.0,
        ref_scale: float = 1.0,
        # Cross-Domain IPCD Attention
        use_cd_attn: bool = False,
        num_tasks: int = 2,
        cd_scale: float = 1.0,
        # SPAD-style epi attention  
        spad_epi_scale: float = 1.0,
        use_spad_epi: bool = False,
        use_random_spad_epi_scale: bool = False,  # 训练时是否随机选择0或1
        # 取消 EPI 父/子分裂相关开关
        
        # Camera parameter options
        use_dataset_cam: bool = True,
        # Runtime flags propagated from config (optional)
        sync_noise_across_views: bool = False,
        cd_num_tasks: int = 2,
    ):
        # Condition encoder - 只有当需要时才创建
        if cond_in_channels > 0:
            self.cond_encoder = T2IAdapter(
                in_channels=cond_in_channels,
                channels=self.unet.config.block_out_channels,
                num_res_blocks=self.unet.config.layers_per_block,
                downscale_factor=8,
            )
        else:
            self.cond_encoder = None

        # 第二个条件编码器：与第一个结构一致，但独立权重
        if enable_dual_cond and cond2_in_channels > 0:
            self.cond_encoder2 = T2IAdapter(
                in_channels=cond2_in_channels,
                channels=self.unet.config.block_out_channels,
                num_res_blocks=self.unet.config.layers_per_block,
                downscale_factor=8,
            )
        else:
            self.cond_encoder2 = None

        # set custom attn processor for multi-view attention
        self.unet: UNet2DConditionModel

        # Original multi-view processor
        set_unet_2d_condition_attn_processor(
            self.unet,
            set_self_attn_proc_func=lambda name, hs, cad, ap: self_attn_processor(
                query_dim=hs,
                inner_dim=hs,
                num_views=num_views,
                name=name,
                use_mv=use_mv_attn,
                use_ref=use_ref_attn,
                ref_attn_mode=ref_attn_mode,  # 新增：传递ref attention模式
                use_densemv=use_densemv_attn,
                use_cd=use_cd_attn,
                num_tasks=num_tasks,
                use_spad_epi=use_spad_epi,
                
                # 无父/子拆分
            ),
            set_cross_attn_proc_func=lambda name, hs, cad, ap: self_attn_processor(
                query_dim=hs,
                inner_dim=hs,
                num_views=num_views,
                name=name,
                use_mv=False,
                use_ref=False,
                ref_attn_mode=ref_attn_mode,  # 新增：传递ref attention模式（虽然cross attn中不使用）
                use_densemv=False,  # Dense Multi-View attention not used for cross attention
                use_cd=False,
                num_tasks=num_tasks,
                use_spad_epi=False,
                
            ),
        )

        # copy decoupled attention weights from original unet
        if copy_attn_weights:
            # 从 zero_init_module_keys 自动推导出需要处理的模块后缀
            module_suffixes = []
            for module_key in zero_init_module_keys:
                if module_key.startswith("to_out_"):
                    suffix = "_" + module_key[7:]  # 去掉 "to_out_" 前缀，加上 "_"
                    module_suffixes.append(suffix)
            
            state_dict = self.unet.state_dict()
            for key in state_dict.keys():
                compatible_key = key
                
                # 检查并处理模块后缀
                for suffix in module_suffixes:
                    if suffix in key:
                        compatible_key = key.replace(suffix, "").replace("processor.", "")
                        break

                is_zero_init_key = any([k in key for k in zero_init_module_keys])
                if is_zero_init_key:
                    state_dict[key] = torch.zeros_like(state_dict[compatible_key])
                else:
                    state_dict[key] = state_dict[compatible_key].clone()
            self.unet.load_state_dict(state_dict)
            
        # Save Dense Multi-View attention scale for forward pass

        self.densemv_scale = densemv_scale
        self.use_densemv_attn = use_densemv_attn
        
        # Save MV attention scale for forward pass
        self.mv_scale = mv_scale
        self.use_mv_attn = use_mv_attn
        
        # Save Ref attention parameters for forward pass
        self.ref_scale = ref_scale
        self.use_ref_attn = use_ref_attn

        self.cd_scale = cd_scale
        self.use_cd_attn = use_cd_attn

        # Save SPAD-style epi parameters for forward pass
        self.spad_epi_scale = spad_epi_scale
        self.use_spad_epi = use_spad_epi
        self.use_random_spad_epi_scale = use_random_spad_epi_scale
        # 取消 EPI 父/子视角拆分相关状态存储
        
        # Save Match-style attention parameters for forward pass
        
        
        # Save camera parameter options
        self.use_dataset_cam = use_dataset_cam

        # 保存运行时开关（便于不经过 system 直接测试时生效）
        self.sync_noise_across_views = bool(sync_noise_across_views)
        self.cd_num_tasks = int(cd_num_tasks)



    def _load_custom_adapter(self, state_dict):
        self.unet.load_state_dict(state_dict, strict=False)
        # 只有当 cond_encoder 存在时才加载
        if hasattr(self, 'cond_encoder') and self.cond_encoder is not None:
            self.cond_encoder.load_state_dict(state_dict, strict=False)
        # 新增：可选加载第二个 cond encoder
        if hasattr(self, 'cond_encoder2') and self.cond_encoder2 is not None:
            self.cond_encoder2.load_state_dict(state_dict, strict=False)

    def _save_custom_adapter(
        self,
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        def include_fn(k):
            is_included = False

            if include_keys is not None:
                is_included = is_included or any([key in k for key in include_keys])
            if exclude_keys is not None:
                is_included = is_included and not any(
                    [key in k for key in exclude_keys]
                )

            return is_included

        state_dict = {k: v for k, v in self.unet.state_dict().items() if include_fn(k)}
        # 只有当 cond_encoder 存在时才保存
        if hasattr(self, 'cond_encoder') and self.cond_encoder is not None:
            state_dict.update(self.cond_encoder.state_dict())
        # 新增：保存第二个 cond encoder（若存在）
        if hasattr(self, 'cond_encoder2') and self.cond_encoder2 is not None:
            state_dict.update(self.cond_encoder2.state_dict())

        return state_dict
