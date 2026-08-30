import os
from dataclasses import dataclass, field

import torch
from diffusers import DDPMScheduler
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter

from mvadapter.models.custom_unet import CustomUNet2DConditionModel

from ..pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline
from ..schedulers.scheduling_shift_snr import ShiftSNRScheduler
from ..utils.core import find
from ..utils.typing import *
from .base import BaseSystem


class MVAdapterImageSDSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):

        # Model / Adapter
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        pretrained_vae_name_or_path: Optional[str] = None
        pretrained_adapter_name_or_path: Optional[str] = None
        pretrained_unet_name_or_path: Optional[str] = None
        init_adapter_kwargs: Dict[str, Any] = field(default_factory=dict)

        unet_out: int = 3
        unet_in: int = 3

        use_fp16_vae: bool = True
        use_fp16_clip: bool = True

        projector_hidden_dim: int = 512
        projector_num_layers: int = 2

        # Noise sampler
        noise_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
        prediction_type: Optional[str] = None
        shift_noise: bool = False
        shift_noise_mode: str = "interpolated"
        shift_noise_scale: float = 1.0

        # Evaluation
        eval_seed: int = 0
        eval_num_inference_steps: int = 30
        eval_guidance_scale: float = 1.0
        eval_height: int = 512
        eval_width: int = 512
        split_image: bool = False

        # Sapiens ViT
        sapiens_torchscript_path: Optional[str] = None

        # Feature extractor settings
        feature_extractor: str = ""  # "sapiens"  # "sapiens" or "dinov2"
        dinov2_model_name: str = ""  # "dinov2_vitb14"  # dinov2 model variant

        # Secondary feature extractor settings for semantic features
        secondary_feature_extractor: str = ""  # e.g., "dinov2"
        secondary_dinov2_model_name: str = ""  # DINOv2 variant, e.g., "dinov2_vitb14"
        # Feature fusion settings
        fused_feature_dim: int = 1024  # 融合后特征维度，用于 concat+MLP
        pe_num_freqs: int = 6  # 位置编码频率数量（暂未使用）

        use_dataset_cam: bool = True
        cond_fusion_weight_main: float = 0.5
        cond_fusion_weight_lowres: float = 0.5

    cfg: Config

    def _replace_unet_conv_in(self, unet):
        r = self.cfg.unet_in // 4
        assert self.cfg.unet_in % 4 == 0 and r >= 1, "unet_in 必须是 4 的倍数"
        # replace the first layer to accept 8 in_channels
        _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, r, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight[:, 4:] *= 0
        # new conv_in channel
        _n_convin_out_channel = unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            self.cfg.unet_in, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        unet.conv_in = _new_conv_in
        unet.config["in_channels"] = self.cfg.unet_in
        unet.config.in_channels = self.cfg.unet_in
        return unet

    def _replace_unet_conv_out(self, unet):
        r = self.cfg.unet_in // 4
        assert self.cfg.unet_out % 4 == 0 and r >= 1, "unet_out 必须是 4 的倍数"
        # replace the first layer to accept 8 in_channels
        _weight = unet.conv_out.weight.clone()  # [320, 4, 3, 3]
        _bias = unet.conv_out.bias.clone()  # [320]
        _weight = _weight.repeat((r, 1, 1, 1))  # Keep selected channel(s)
        _bias = _bias.repeat((r))
        # half the activation magnitude
        # _weight *= (0.5**(self.cfg.unet_in//4))
        # new conv_in channel
        _n_convout_in_channel = unet.conv_out.in_channels
        _new_conv_out = Conv2d(
            _n_convout_in_channel, self.cfg.unet_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_out.weight = Parameter(_weight)
        _new_conv_out.bias = Parameter(_bias)
        unet.conv_out = _new_conv_out
        unet.config["out_channels"] = self.cfg.unet_out
        unet.config.out_channels = self.cfg.unet_out
        return unet

    def configure(self):
        super().configure()
        import torch
        # Prepare pipeline
        pipeline_kwargs = {}
        if self.cfg.pretrained_vae_name_or_path is not None:
            pipeline_kwargs["vae"] = AutoencoderKL.from_pretrained(
                self.cfg.pretrained_vae_name_or_path
            )
        if self.cfg.pretrained_unet_name_or_path is not None:
            pipeline_kwargs["unet"] = CustomUNet2DConditionModel.from_pretrained(
                self.cfg.pretrained_unet_name_or_path, subfolder='unet'
            )

        pipeline: MVAdapterI2MVSDPipeline
        pipeline = MVAdapterI2MVSDPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, **pipeline_kwargs
        )
        # 禁用 NSFW 安全检测
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
        # 运行期开关以 init_adapter_kwargs 为准（避免与 system 重复）

        # 设置随机SPAD权重训练参数（在pipeline初始化时已设置）

        init_adapter_kwargs = OmegaConf.to_container(self.cfg.init_adapter_kwargs)
        if "self_attn_processor" in init_adapter_kwargs:
            self_attn_processor = init_adapter_kwargs["self_attn_processor"]
            if self_attn_processor is not None and isinstance(self_attn_processor, str):
                self_attn_processor = find(self_attn_processor)
                init_adapter_kwargs["self_attn_processor"] = self_attn_processor

        # 根据 feature_extractor 推断 reference_mode，并写入 adapter kwargs
        reference_mode = (
            self.cfg.feature_extractor
            if self.cfg.feature_extractor in ["sapiens", "dinov2"]
            else "residual"
        )
        init_adapter_kwargs["reference_mode"] = reference_mode
        init_adapter_kwargs["projector_hidden_dim"] = self.cfg.projector_hidden_dim

        # 传递相机参数配置 + 双分支cond设置（在pipeline中创建cond_encoder2）
        init_adapter_kwargs["use_dataset_cam"] = self.cfg.use_dataset_cam
        init_adapter_kwargs["projector_num_layers"] = self.cfg.projector_num_layers
        # 从 init_adapter_kwargs 读取第二分支配置；若未提供则给出默认
        if "enable_dual_cond" not in init_adapter_kwargs:
            init_adapter_kwargs["enable_dual_cond"] = False
        if "cond2_in_channels" not in init_adapter_kwargs:
            init_adapter_kwargs["cond2_in_channels"] = 3

        # 确保运行期开关存在（若 YAML 中未写，则给默认）。以 init_adapter_kwargs 为唯一来源
        if "sync_noise_across_views" not in init_adapter_kwargs:
            init_adapter_kwargs["sync_noise_across_views"] = False
        if "cd_num_tasks" not in init_adapter_kwargs:
            init_adapter_kwargs["cd_num_tasks"] = 2

        pipeline.init_custom_adapter(**init_adapter_kwargs)

        if self.cfg.pretrained_adapter_name_or_path:
            pretrained_path = os.path.dirname(self.cfg.pretrained_adapter_name_or_path)
            adapter_name = os.path.basename(self.cfg.pretrained_adapter_name_or_path)
            pipeline.load_custom_adapter(pretrained_path, weight_name=adapter_name)

        noise_scheduler = DDPMScheduler.from_config(
            pipeline.scheduler.config, **self.cfg.noise_scheduler_kwargs
        )
        if self.cfg.shift_noise:
            noise_scheduler = ShiftSNRScheduler.from_scheduler(
                noise_scheduler,
                shift_mode=self.cfg.shift_noise_mode,
                shift_scale=self.cfg.shift_noise_scale,
                scheduler_class=DDPMScheduler,
            )
        pipeline.scheduler = noise_scheduler

        # Prepare models
        self.pipeline: MVAdapterI2MVSDPipeline = pipeline
        # 在推理阶段启用随机 drop（使用与训练一致的概率）
        # 不再使用随机 drop 配置，推理期 drop 由 cond/ref 是否存在决定
        

        self.vae = self.pipeline.vae.to(
            dtype=torch.float16 if self.cfg.use_fp16_vae else torch.float32
        )
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder.to(
            dtype=torch.float16 if self.cfg.use_fp16_clip else torch.float32
        )
        self.feature_extractor = self.pipeline.feature_extractor

        self.cond_encoder = getattr(self.pipeline, 'cond_encoder', None)
        # 从pipeline获取第二个cond encoder，并设置训练标志
        self.cond_encoder2 = getattr(self.pipeline, 'cond_encoder2', None)
        self.unet = self.pipeline.unet
        self.noise_scheduler = self.pipeline.scheduler
        self.inference_scheduler = DDPMScheduler.from_config(
            self.noise_scheduler.config
        )
        self.pipeline.scheduler = self.inference_scheduler
        if self.cfg.prediction_type is not None:
            self.noise_scheduler.register_to_config(
                prediction_type=self.cfg.prediction_type
            )

        # --------------------------------------------------
        #  VGGT / Pi3 匹配注意力相关加载已移除
        # --------------------------------------------------

        # --------------------------------------------------
        #  Feature extractor setup (reference_mode == "sapiens" or "dinov2")
        # --------------------------------------------------
        if self.cfg.feature_extractor in ["sapiens", "dinov2"]:

            if self.cfg.feature_extractor == "sapiens":
                # Sapiens ViT setup
                ckpt_path = (
                    self.cfg.sapiens_torchscript_path
                    if hasattr(self.cfg, "sapiens_torchscript_path") and self.cfg.sapiens_torchscript_path is not None
                    else None
                )
                if ckpt_path is None:
                    raise ValueError(
                        "Please set `sapiens_torchscript_path` in the yaml to a Sapiens TorchScript weight file")

                self.vit = torch.jit.load(ckpt_path).eval().to(self.device, dtype=torch.float32)

                for p in self.vit.parameters():
                    p.requires_grad_(False)

                # preprocessing mean / std (Sapiens uses ImageNet like)
                self.vit_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
                self.vit_std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
                self.vit_input_size = 1024

                # Test model output to get feature dimensions
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 1024, 1024, device=self.device, dtype=torch.float32)
                    vit_out = self.vit(dummy)
                    # TorchScript may return tensor or tuple/dict
                    if isinstance(vit_out, torch.Tensor):
                        vit_feats = vit_out
                    elif isinstance(vit_out, (list, tuple)):
                        vit_feats = vit_out[0]
                    elif isinstance(vit_out, dict):
                        vit_feats = vit_out.get("last_hidden_state", next(iter(vit_out.values())))
                    else:
                        raise TypeError("Unexpected output type from Sapiens model")

                    vit_feats = vit_feats.float()
                    vit_feats = vit_feats.flatten(2).transpose(1, 2).contiguous()
                    self.vit_feature_dim = vit_feats.shape[-1]

            elif self.cfg.feature_extractor == "dinov2":
                # DINOv2 setup
                print("loading dinov2")

                # 从Hugging Face加载DINOv2
                from transformers import AutoModel
                model_id = os.environ.get(
                    "DINO_BASE_MODEL", "pretrained/dinov2_large"
                )
                print(f"加载 DINOv2: {model_id}")
                self.vit = AutoModel.from_pretrained(model_id)

                print("successfully loaded")
                self.vit = self.vit.eval()

                # 检查设备兼容性，优先使用CUDA，但确保在CPU上也能运行
                self.vit = self.vit.to(self.device, dtype=torch.float16)  # 使用float16以兼容xformers

                for p in self.vit.parameters():
                    p.requires_grad_(False)

                # DINOv2 preprocessing (ImageNet normalization)
                self.vit_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
                self.vit_std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

                # DINOv2 expects 518x518 for optimal performance, but can handle different sizes
                self.vit_input_size = 518 #224 maybe better

                # 使用预定义的特征维度作为fallback
                model_dims = {
                    'dinov2_vits14': 384,
                    'dinov2_vitb14': 768,
                    'dinov2_vitl14': 1024,
                    'dinov2_vitg14': 1536,
                }
                self.vit_feature_dim = model_dims.get(self.cfg.dinov2_model_name, 768)
                print(f"使用默认特征维度: {self.vit_feature_dim}")

                print(f"DINOv2 {self.cfg.dinov2_model_name} 初始化完成，特征维度: {self.vit_feature_dim}")


            else:
                raise ValueError(f"Unsupported feature_extractor: {self.cfg.feature_extractor}")

            # --------- Layer-wise projectors to match each UNet hidden_size ---------
            # Build mapping layer_hidden_size -> projector
            hidden_sizes = {}
            for name in self.unet.attn_processors.keys():
                if name.startswith("mid_block"):
                    hs = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hs = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hs = self.unet.config.block_out_channels[block_id]
                else:
                    continue
                hidden_sizes[name] = hs

            unique_sizes = sorted(list(set(hidden_sizes.values())))
            from torch import nn
            from mvadapter.models.projector import SimpleMLP
            self.layer_projectors = nn.ModuleDict()
            for sz in unique_sizes:
                self.layer_projectors[str(sz)] = SimpleMLP(
                    input_dim=self.vit_feature_dim,
                    output_dim=sz,
                    hidden_dim=max(sz, self.cfg.projector_hidden_dim),
                    num_layers=self.cfg.projector_num_layers,
                )
            # Setup secondary projectors and fusion MLPs for DINOv2 semantic features
            if self.cfg.secondary_feature_extractor == "dinov2":
                self.layer_projectors2 = nn.ModuleDict()
                self.layer_fusion_mlps = nn.ModuleDict()
                for sz in unique_sizes:
                    self.layer_projectors2[str(sz)] = SimpleMLP(
                        input_dim=self.vit2_feature_dim,
                        output_dim=sz,
                        hidden_dim=max(sz, self.cfg.projector_hidden_dim),
                        num_layers=self.cfg.projector_num_layers,
                    )
                    self.layer_fusion_mlps[str(sz)] = SimpleMLP(
                        input_dim=sz * 2,
                        output_dim=sz,
                        hidden_dim=max(sz * 2, self.cfg.projector_hidden_dim),
                        num_layers=self.cfg.projector_num_layers,
                    )

            # expose to pipeline
            self.pipeline.layer_projectors = self.layer_projectors
            self.pipeline.vit = self.vit
            self.pipeline.extractor_type = self.cfg.feature_extractor
            self.pipeline.dinov2_model_name = self.cfg.dinov2_model_name


            # Propagate dual-cond fusion weights for inference path
            if hasattr(self.cfg, 'cond_fusion_weight_main'):
                self.pipeline.cond_fusion_weight_main = float(self.cfg.cond_fusion_weight_main)
            if hasattr(self.cfg, 'cond_fusion_weight_lowres'):
                self.pipeline.cond_fusion_weight_lowres = float(self.cfg.cond_fusion_weight_lowres)


        # Public release is inference-only: freeze every constructed component.
        self.unet.requires_grad_(False)
        self.unet.eval()
        if self.cond_encoder is not None:
            self.cond_encoder.requires_grad_(False)
            self.cond_encoder.eval()
        if getattr(self, "cond_encoder2", None) is not None:
            self.cond_encoder2.requires_grad_(False)
            self.cond_encoder2.eval()
        if hasattr(self, "layer_projectors"):
            self.layer_projectors.requires_grad_(False)
            self.layer_projectors.eval()
        if hasattr(self, "layer_projectors2"):
            self.layer_projectors2.requires_grad_(False)
            self.layer_projectors2.eval()
        if hasattr(self, "layer_fusion_mlps"):
            self.layer_fusion_mlps.requires_grad_(False)
            self.layer_fusion_mlps.eval()
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

    



    

    

    

    

    def generate_images(self, batch, save_intermediate_steps=False, intermediate_output_dir=None,
                        intermediate_save_frequency=5,
                        sdedit_image=None, sdedit_latent=None, sdedit_strength=0.3, **kwargs):
        self.pipeline.to(self.device)
        return self.pipeline(
            num_views=batch["num_views"],
            prompt=batch["prompts"],
            control_image=batch.get("condition_rgb", None),
            source_image=batch.get("source_rgb", None),
            num_images_per_prompt=1,  # batch["num_views"],
            generator=torch.Generator(device=self.device).manual_seed(
                self.cfg.eval_seed
            ),
            num_inference_steps=self.cfg.eval_num_inference_steps,
            guidance_scale=self.cfg.eval_guidance_scale,
            height=self.cfg.eval_height,
            width=self.cfg.eval_width,
            reference_image=batch.get("reference_rgb", None),
            reference_mode=(self.cfg.feature_extractor if self.cfg.feature_extractor in ["sapiens", "dinov2",
                                                                                         "vggt"] else "residual"),
            # 将数据集提供的相机参数传入 pipeline
            dataset_extrinsics=batch.get("dataset_extrinsics", None),
            dataset_intrinsics=batch.get("dataset_intrinsics", None),
            output_type="pt",
            safety_checker=None,
            batch=batch,
            # 新增：中间步骤保存参数
            save_intermediate_steps=save_intermediate_steps,
            intermediate_output_dir=intermediate_output_dir,
            intermediate_save_frequency=intermediate_save_frequency,
            # SDEdit 支持
            sdedit_image=sdedit_image,
            sdedit_latent=sdedit_latent,
            sdedit_strength=sdedit_strength,
        ).images

    

    

    

    

    

    # 删除 mode1 构建函数

    
