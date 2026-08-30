import math
from typing import Callable, List, Optional, Union, Dict

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.unets import UNet2DConditionModel
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from einops import rearrange, repeat
from torch import nn


def default_set_attn_proc_func(
    name: str,
    hidden_size: int,
    cross_attention_dim: Optional[int],
    ori_attn_proc: object,
) -> object:
    return ori_attn_proc


def set_unet_2d_condition_attn_processor(
    unet: UNet2DConditionModel,
    set_self_attn_proc_func: Callable = default_set_attn_proc_func,
    set_cross_attn_proc_func: Callable = default_set_attn_proc_func,
    set_custom_attn_proc_func: Callable = default_set_attn_proc_func,
    set_self_attn_module_names: Optional[List[str]] = None,
    set_cross_attn_module_names: Optional[List[str]] = None,
    set_custom_attn_module_names: Optional[List[str]] = None,
) -> None:
    do_set_processor = lambda name, module_names: (
        any([name.startswith(module_name) for module_name in module_names])
        if module_names is not None
        else True
    )  # prefix match

    attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        # set attn_processor by default, if module_names is None
        set_self_attn_processor = do_set_processor(name, set_self_attn_module_names)
        set_cross_attn_processor = do_set_processor(name, set_cross_attn_module_names)
        set_custom_attn_processor = do_set_processor(name, set_custom_attn_module_names)

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        is_custom = "attn_mid_blocks" in name or "attn_post_blocks" in name
        if is_custom:
            attn_procs[name] = (
                set_custom_attn_proc_func(name, hidden_size, None, attn_processor)
                if set_custom_attn_processor
                else attn_processor
            )
        else:
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if cross_attention_dim is None or "motion_modules" in name:
                # self attention
                attn_procs[name] = (
                    set_self_attn_proc_func(
                        name, hidden_size, cross_attention_dim, attn_processor
                    )
                    if set_self_attn_processor
                    else attn_processor
                )
            else:
                # cross attention
                attn_procs[name] = (
                    set_cross_attn_proc_func(
                        name, hidden_size, cross_attention_dim, attn_processor
                    )
                    if set_cross_attn_processor
                    else attn_processor
                )

    unet.set_attn_processor(attn_procs)

#
class DecoupledMVRowSelfAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for Decoupled Row-wise Self-Attention and Image Cross-Attention for PyTorch 2.0.
    """

    def __init__(
        self,
        query_dim: int,
        inner_dim: int,
        num_views: int = 1,
        name: Optional[str] = None,
        use_mv: bool = True,
        use_ref: bool = False,
        ref_attn_mode: str = "multi",  # 新增：ref attention模式，"multi"或"single"
        use_densemv: bool = False,
        use_cd: bool = False,  # 新增：是否启用跨域(任务)交互
        num_tasks: int = 2,    # 新增：任务数量（如 uv / seg 两个任务）
        use_spad_epi: bool = False,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "DecoupledMVRowSelfAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        super().__init__()

        self.num_views = num_views
        self.name = name  # NOTE: need for image cross-attention
        self.use_mv = use_mv
        self.use_ref = use_ref
        self.ref_attn_mode = ref_attn_mode  # 新增：存储ref attention模式
        self.use_densemv = use_densemv
        self.use_cd = use_cd
        self.num_tasks = num_tasks
        self.use_spad_epi = use_spad_epi
        # 取消父/子视角拆分策略



        # --- SPAD-style epipolar attention (mask-based across concatenated views) ---
        if self.use_spad_epi:
            self.to_q_spad = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_k_spad = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_v_spad = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_out_spad = nn.ModuleList([
                nn.Linear(inner_dim, query_dim, bias=True), nn.Dropout(0.0)
            ])
            # project plucker (6 channels) to token dim and inject
            self.to_plucker_proj = nn.Linear(6, inner_dim, bias=False)

        # Match-based attention 已移除

        if self.use_mv:
            self.to_q_mv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_k_mv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_v_mv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_out_mv = nn.ModuleList(
                [
                    nn.Linear(in_features=inner_dim, out_features=query_dim, bias=True),
                    nn.Dropout(0.0),
                ]
            )

        if self.use_densemv:
            self.to_q_densemv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_k_densemv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_v_densemv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_out_densemv = nn.ModuleList(
                [
                    nn.Linear(in_features=inner_dim, out_features=query_dim, bias=True),
                    nn.Dropout(0.0),
                ]
            )

        if self.use_ref:
            self.to_q_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_k_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_v_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_out_ref = nn.ModuleList(
                [
                    nn.Linear(in_features=inner_dim, out_features=query_dim, bias=True),
                    nn.Dropout(0.0),
                ]
            )

        # ==== Cross-Domain (IPCD) Attention 専用投影 ====
        if self.use_cd:
            self.to_q_cd = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_k_cd = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_v_cd = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_out_cd = nn.ModuleList(
                [
                    nn.Linear(in_features=inner_dim, out_features=query_dim, bias=True),
                    nn.Dropout(0.0),
                ]
            )

    def get_sinusoidal_positional_embedding(self, positions: torch.Tensor, num_freqs: int):
        """
        Generates sinusoidal positional embeddings.
        Args:
            positions: Tensor of shape (..., D)
            num_freqs: Number of frequency bands.
        Returns:
            Tensor of shape (..., D * 2 * num_freqs)
        """
        log_space = torch.logspace(
            0.0,
            math.log(1000.0) / math.log(10.0),
            num_freqs,
            device=positions.device,
        ) * math.pi

        # (..., D, 1) * (F) -> (..., D, F)
        scaled_positions = positions.unsqueeze(-1) * log_space

        # (..., D, 2*F)
        sincos = torch.cat([torch.sin(scaled_positions), torch.cos(scaled_positions)], dim=-1)

        # (..., D * 2*F)
        return sincos.flatten(-2)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        mv_scale: float = 1.0,
        densemv_scale: float = 1.0,
        ref_hidden_states: Optional[torch.FloatTensor] = None,
        ref_scale: float = 1.0,
        cache_hidden_states: Optional[List[torch.FloatTensor]] = None,
        use_mv: bool = True,
        use_densemv: bool = False,
        use_ref: bool = True,
        use_cd: Optional[bool] = None,
        cd_scale: float = 1.0,
        # 参考可见域掩码（局部视角使用），支持来自 cross_attention_kwargs 的直传
        ref_patch_masks: Optional[torch.Tensor] = None,
        # for SPAD-style epi
        use_spad_epi: bool = False,
        spad_block_epi_mask: Optional[torch.Tensor] = None,  # (B, V*S, V*S)
        spad_plucker_embed: Optional[torch.Tensor] = None,    # (B, V, 6, H, W)
        spad_epi_scale: float = 1.0,
        # 取消父/子视角拆分相关参数
        # Match-based attention 已移除
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        New args:
            mv_scale (float): scale for multi-view self-attention.
            densemv_scale (float): scale for dense multi-view attention.
            ref_hidden_states (torch.FloatTensor): reference encoder hidden states for image cross-attention.
            ref_scale (float): scale for image cross-attention.
            cache_hidden_states (List[torch.FloatTensor]): cache hidden states from reference unet.

        """
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        # NEW: cache hidden states for reference unet
        if cache_hidden_states is not None:
            cache_hidden_states[self.name] = hidden_states.detach() #.clone()

        use_mv = self.use_mv and use_mv
        use_ref = self.use_ref and use_ref
        use_densemv = self.use_densemv and use_densemv
        use_cd = self.use_cd and use_cd
        use_spad_epi = self.use_spad_epi and use_spad_epi
        # Match-based attention 已移除


        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if use_mv:
            query_mv = self.to_q_mv(hidden_states)
            key_mv = self.to_k_mv(encoder_hidden_states)
            value_mv = self.to_v_mv(encoder_hidden_states)

            query_mv = query_mv.view(batch_size, -1, attn.heads, head_dim)
            key_mv = key_mv.view(batch_size, -1, attn.heads, head_dim)
            value_mv = value_mv.view(batch_size, -1, attn.heads, head_dim)

            height = width = math.isqrt(sequence_length)

            # row self-attention
            query_mv = rearrange(
                query_mv,
                "(b nv) (ih iw) h c -> (b nv ih) iw h c",
                nv=self.num_views,
                ih=height,
                iw=width,
            ).transpose(1, 2)
            key_mv = rearrange(
                key_mv,
                "(b nv) (ih iw) h c -> b ih (nv iw) h c",
                nv=self.num_views,
                ih=height,
                iw=width,
            )
            key_mv = (
                key_mv.repeat_interleave(self.num_views, dim=0)
                .view(batch_size * height, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )
            value_mv = rearrange(
                value_mv,
                "(b nv) (ih iw) h c -> b ih (nv iw) h c",
                nv=self.num_views,
                ih=height,
                iw=width,
            )
            value_mv = (
                value_mv.repeat_interleave(self.num_views, dim=0)
                .view(batch_size * height, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )

            hidden_states_mv = F.scaled_dot_product_attention(
                query_mv,
                key_mv,
                value_mv,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_mv = rearrange(
                hidden_states_mv,
                "(b nv ih) h iw c -> (b nv) (ih iw) (h c)",
                nv=self.num_views,
                ih=height,
            )
            hidden_states_mv = hidden_states_mv.to(query.dtype)

            # linear proj
            hidden_states_mv = self.to_out_mv[0](hidden_states_mv)
            # dropout
            hidden_states_mv = self.to_out_mv[1](hidden_states_mv)

        if use_densemv:
            query_densemv = self.to_q_densemv(hidden_states)
            key_densemv = self.to_k_densemv(encoder_hidden_states)
            value_densemv = self.to_v_densemv(encoder_hidden_states)
            # b:batch t:task v:view s:sequence h:head c:channel
            query_densemv = rearrange(query_densemv, "(b t v) s (h c) -> (b t) h (v s) c",
                                  t=self.num_tasks, v=self.num_views, h=attn.heads)
            key_densemv = rearrange(key_densemv, "(b t v) s (h c) -> (b t) h (v s) c",
                                t=self.num_tasks, v=self.num_views, h=attn.heads)
            value_densemv = rearrange(value_densemv, "(b t v) s (h c) -> (b t) h (v s) c",
                                  t=self.num_tasks, v=self.num_views, h=attn.heads)

            # Perform full dense multi-view attention across all views and spatial dimensions
            hidden_states_densemv = F.scaled_dot_product_attention(
                query_densemv,
                key_densemv,
                value_densemv,
                dropout_p=0.0,
                is_causal=False,
            )
            
            # Reshape back to original format
            hidden_states_densemv = rearrange(hidden_states_densemv, "(b t) h (v s) c -> (b t v) s (h c)",
                                  t=self.num_tasks, v=self.num_views, h=attn.heads)
            
            hidden_states_densemv = hidden_states_densemv.to(query.dtype)
            
            # linear proj
            hidden_states_densemv = self.to_out_densemv[0](hidden_states_densemv)
            # dropout
            hidden_states_densemv = self.to_out_densemv[1](hidden_states_densemv)

        if use_ref:
            reference_hidden_states = ref_hidden_states[self.name] # (B*T*V, S, C)
            query_ref = self.to_q_ref(hidden_states)
            key_ref = self.to_k_ref(reference_hidden_states)
            value_ref = self.to_v_ref(reference_hidden_states)

            query_ref = rearrange(query_ref, "(b t v) s (h c) -> (b t v) h s c",
                                  t=self.num_tasks, v=self.num_views, h=attn.heads)
            
            # 根据ref_attn_mode选择不同的处理方式
            if self.ref_attn_mode == "single":
                # single模式：每个视角仅使用当前视角的ref image
                key_ref = rearrange(key_ref, "(b t v) s (h c) -> (b t v) h s c",
                                    t=self.num_tasks, v=self.num_views, h=attn.heads)
                value_ref = rearrange(value_ref, "(b t v) s (h c) -> (b t v) h s c",
                                      t=self.num_tasks, v=self.num_views, h=attn.heads)
            else:
                # multi模式（默认）：每个视角使用所有视角的ref image
                key_ref = rearrange(key_ref, "(b t v) s (h c) -> (b t) h (v s) c",
                                    t=self.num_tasks, v=self.num_views, h=attn.heads)
                key_ref = repeat(key_ref, "bt h vs c -> (bt v) h vs c", v=self.num_views)
                value_ref = rearrange(value_ref, "(b t v) s (h c) -> (b t) h (v s) c",
                                      t=self.num_tasks, v=self.num_views, h=attn.heads)
                value_ref = repeat(value_ref, "bt h vs c -> (bt v) h vs c", v=self.num_views)

            # 可选：参考 token 掩码（用于局部视角限制可见域）
            # 优先使用显式参数；为兼容旧路径，也接受 kwargs 中的同名字段
            if ref_patch_masks is None:
                ref_patch_masks = kwargs.get("ref_patch_masks", None)
            attn_mask_bias_ref = None
            if ref_patch_masks is not None:
                # 推断 B、T、V 和 cf_repeat
                if self.ref_attn_mode == "single":
                    # key_ref: (B*T*V) h s c
                    B = ref_patch_masks.shape[0]
                    V = ref_patch_masks.shape[1]
                    T = self.num_tasks
                    cf_repeat = key_ref.shape[0] // (B * T * V)
                    S_k = key_ref.shape[2]
                    S_m = ref_patch_masks.shape[2]
                    # 若 mask 的 S 与 key 的 S 不同，做最近邻插值到一致
                    if S_m != S_k:
                        Hm = Wm = int(S_m ** 0.5)
                        Hk = Wk = int(S_k ** 0.5)
                        m2d = ref_patch_masks.view(B * V, 1, Hm, Wm)
                        m2d = F.interpolate(m2d, size=(Hk, Wk), mode='nearest')
                        ref_patch_masks_use = m2d.view(B, V, Hk * Wk)
                    else:
                        ref_patch_masks_use = ref_patch_masks
                    # 展开到 (B*T*V, 1, 1, S)
                    m = ref_patch_masks_use[:, None].repeat(1, T, 1, 1)  # (B,T,V,S)
                    m = m.view(B * T * V, S_k)
                    # 0->-inf, 1->0 bias
                    attn_mask_bias_ref = torch.where(
                        m > 0,
                        torch.zeros_like(m),
                        torch.full_like(m, -1e9),
                    ).view(B * T * V, 1, 1, S_k)
                    # 考虑 CFG 倍数：若 query/key 在 batch 维存在 cf_repeat，则掩码需同步复制
                    if cf_repeat > 1:
                        attn_mask_bias_ref = attn_mask_bias_ref.repeat_interleave(cf_repeat, dim=0)
                else:
                    # multi: key_ref: (B*T) h (V*S) c
                    B = ref_patch_masks.shape[0]
                    V = ref_patch_masks.shape[1]
                    T = self.num_tasks
                    cf_repeat = key_ref.shape[0] // (B * T)
                    S_k = key_ref.shape[2]
                    S_m = ref_patch_masks.shape[2]
                    VSk = S_k
                    # 展平 V 维
                    masks_flat = ref_patch_masks.reshape(B, V * S_m)
                    if V * S_m != VSk:
                        # 近邻插值到 V*S_k
                        Hm = Wm = int(S_m ** 0.5)
                        Hk = Wk = int((S_k // V) ** 0.5)
                        m2d = ref_patch_masks.view(B * V, 1, Hm, Wm)
                        m2d = F.interpolate(m2d, size=(Hk, Wk), mode='nearest')
                        masks_flat = m2d.view(B, V, Hk * Wk).reshape(B, V * Hk * Wk)
                    m = masks_flat[:, None].repeat(1, T, 1)  # (B,T,V*S)
                    m = m.view(B * T, S_k)
                    attn_mask_bias_ref = torch.where(
                        m > 0,
                        torch.zeros_like(m),
                        torch.full_like(m, -1e9),
                    ).view(B * T, 1, 1, S_k)
                    # 考虑 CFG 倍数
                    if cf_repeat > 1:
                        attn_mask_bias_ref = attn_mask_bias_ref.repeat_interleave(cf_repeat, dim=0)

            hidden_states_ref = F.scaled_dot_product_attention(
                query_ref, key_ref, value_ref, attn_mask=attn_mask_bias_ref, dropout_p=0.0, is_causal=False
            )

            hidden_states_ref = rearrange(hidden_states_ref, "(b t v) h s c -> (b t v) s (h c)",
                                  t=self.num_tasks, v=self.num_views, h=attn.heads)
            hidden_states_ref = hidden_states_ref.to(query.dtype)

            # linear proj
            hidden_states_ref = self.to_out_ref[0](hidden_states_ref)
            # dropout
            hidden_states_ref = self.to_out_ref[1](hidden_states_ref)

        if use_cd:
            query_cd = self.to_q_cd(hidden_states)
            key_cd = self.to_k_cd(encoder_hidden_states)
            value_cd = self.to_v_cd(encoder_hidden_states)

            query_cd = rearrange(query_cd, "(b t v) s (h c) -> (b v) h (t s) c",
                                  t=self.num_tasks, v=self.num_views, h=attn.heads)
            key_cd = rearrange(key_cd, "(b t v) s (h c) -> (b v) h (t s) c",
                                t=self.num_tasks, v=self.num_views, h=attn.heads)
            value_cd = rearrange(value_cd, "(b t v) s (h c) -> (b v) h (t s) c",
                                  t=self.num_tasks, v=self.num_views, h=attn.heads)

            # Perform full dense multi-view attention across all views and spatial dimensions
            hidden_states_cd = F.scaled_dot_product_attention(
                query_cd,
                key_cd,
                value_cd,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_cd = rearrange(hidden_states_cd, "(b v) h (t s) c -> (b t v) s (h c)",
                                  t=self.num_tasks, v=self.num_views, h=attn.heads)

            hidden_states_cd = hidden_states_cd.to(query.dtype)

            # linear proj
            hidden_states_cd = self.to_out_cd[0](hidden_states_cd)
            # dropout
            hidden_states_cd = self.to_out_cd[1](hidden_states_cd)

        # ================== SPAD-style Epi Attention (mask-based) ==================
        if use_spad_epi:
            # Token dims
            S_total = hidden_states.shape[1]
            H_map = W_map = int(S_total ** 0.5)
            B_mask, VW_q, VW_k = spad_block_epi_mask.shape

            # Determine batch/task/CFG factors
            B = B_mask
            T = self.num_tasks
            V = self.num_views
            cf_repeat = hidden_states.shape[0] // (B * T * V)

            # Build tokens grouped by (B, cf, T, V, S, C)
            # 修正：CFG 在最前面，所以维度顺序应该是 (cf, b, t, v)
            tokens = rearrange(hidden_states, '(cf b t v) s c -> b cf t v s c', b=B, t=T, v=V, cf=cf_repeat)

            # Inject plucker as additive token bias
            # spad_plucker_embed: (B, V, 6, H, W) -> (B, V, S, 6)
            b_plk, v_plk, _, h0, w0 = spad_plucker_embed.shape
            assert b_plk == B and v_plk == V
            
            # 处理 Plücker 特征的分辨率匹配
            s0 = h0 * w0
            sm = H_map * W_map
            
            if s0 != sm:
                # 需要对 Plücker 特征进行分辨率调整
                if s0 > sm:
                    # 下采样：使用平均池化
                    k_h = h0 // H_map
                    k_w = w0 // W_map
                    # Reshape to (B*V, 6, h0, w0)
                    plk_reshaped = spad_plucker_embed.view(B * V, 6, h0, w0)
                    # 使用平均池化进行下采样
                    plk_downsampled = F.avg_pool2d(plk_reshaped, kernel_size=(k_h, k_w), stride=(k_h, k_w))
                    # Reshape back to (B, V, 6, H_map, W_map)
                    plk_adjusted = plk_downsampled.view(B, V, 6, H_map, W_map)
                else:
                    # 上采样：使用双线性插值
                    # Reshape to (B*V, 6, h0, w0)
                    plk_reshaped = spad_plucker_embed.view(B * V, 6, h0, w0)
                    # 使用双线性插值进行上采样
                    plk_upsampled = F.interpolate(plk_reshaped, size=(H_map, W_map), mode='bilinear', align_corners=False)
                    # Reshape back to (B, V, 6, H_map, W_map)
                    plk_adjusted = plk_upsampled.view(B, V, 6, H_map, W_map)
                
                plk = rearrange(plk_adjusted, 'b v c h w -> b v (h w) c')
            else:
                plk = rearrange(spad_plucker_embed, 'b v c h w -> b v (h w) c')
            
            plk_proj = self.to_plucker_proj(plk)  # (B, V, sm, inner_dim)
            # Broadcast to (B, cf, T, V, sm, inner_dim)
            plk_proj = plk_proj[:, None, None].expand(B, cf_repeat, T, V, sm, plk_proj.shape[-1])

            # --- Build/resize SPAD block mask to current resolution ---
            # 修复双线性插值：正确处理 batch 和 V*H*W 的语义结构
            target_size = (V * sm, V * sm)  # (V * H_map * W_map, V * H_map * W_map)
            current_size = spad_block_epi_mask.shape[-2:]
            
            if current_size != target_size:
                # 当前大小 - spad_block_epi_mask: (B, V*H_old*W_old, V*H_old*W_old)
                B_mask = spad_block_epi_mask.shape[0]
                current_VHW = current_size[0]  # V * H_old * W_old
                target_VHW = target_size[0]    # V * sm = V * H_new * W_new
                
                # 计算原始和目标的空间尺寸
                H_old = W_old = int((current_VHW // V) ** 0.5)
                H_new = W_new = int((target_VHW // V) ** 0.5)
                
                # 第一维插值：(B, V*H_old*W_old, V*H_old*W_old) -> (B, V*H_new*W_new, V*H_old*W_old)
                # 将第一维 reshape 成 (B*V, H_old, W_old, V*H_old*W_old)
                mask_reshaped_dim1 = spad_block_epi_mask.view(B_mask * V, H_old, W_old, current_VHW)
                # 对 B*V 个 (H_old, W_old) 并行进行空间插值
                mask_interp_dim1 = F.interpolate(
                    mask_reshaped_dim1.permute(0, 3, 1, 2).float(),  # (B*V, current_VHW, H_old, W_old)
                    size=(H_new, W_new), 
                    mode='bilinear', 
                    align_corners=False
                ).permute(0, 2, 3, 1)  # (B*V, H_new, W_new, current_VHW)
                # Reshape 回 (B, V*H_new*W_new, V*H_old*W_old)
                mask_after_dim1 = mask_interp_dim1.contiguous().view(B_mask, target_VHW, current_VHW)
                
                # 第二维插值：(B, V*H_new*W_new, V*H_old*W_old) -> (B, V*H_new*W_new, V*H_new*W_new)
                # 转置并将第二维 reshape 成 (B*V, H_old, W_old, V*H_new*W_new)
                mask_reshaped_dim2 = mask_after_dim1.transpose(-1, -2).reshape(B_mask * V, H_old, W_old, target_VHW)
                # 对 B*V 个 (H_old, W_old) 并行进行空间插值
                mask_interp_dim2 = F.interpolate(
                    mask_reshaped_dim2.permute(0, 3, 1, 2).float(),  # (B*V, target_VHW, H_old, W_old)
                    size=(H_new, W_new),
                    mode='bilinear', 
                    align_corners=False
                ).permute(0, 2, 3, 1)  # (B*V, H_new, W_new, target_VHW)
                # Reshape 回 (B, V*H_new*W_new, V*H_new*W_new) 并转置回来
                mask_bool = mask_interp_dim2.contiguous().view(B_mask, target_VHW, target_VHW).transpose(-1, -2).bool()
            else:
                mask_bool = spad_block_epi_mask

            # Prepare inputs for QKV
            x_spad = tokens
            x_spad_qkv = x_spad  # use same for q/k/v
            # Project to inner dim via shared linear before head split
            q_spad = self.to_q_spad(rearrange(x_spad_qkv, 'b cf t v s c -> (b cf t v) s c'))
            k_spad = self.to_k_spad(rearrange(x_spad_qkv, 'b cf t v s c -> (b cf t v) s c'))
            v_spad = self.to_v_spad(rearrange(x_spad_qkv, 'b cf t v s c -> (b cf t v) s c'))

            # Add plucker after projection to match inner_dim
            plk_flat = rearrange(plk_proj, 'b cf t v s d -> (b cf t v) s d')
            q_spad = q_spad + plk_flat
            k_spad = k_spad + plk_flat

            head_dim = q_spad.shape[-1] // attn.heads
            q_spad = q_spad.view(B * cf_repeat * T * V, -1, attn.heads, head_dim).transpose(1, 2)
            k_spad = k_spad.view(B * cf_repeat * T * V, -1, attn.heads, head_dim).transpose(1, 2)
            v_spad = v_spad.view(B * cf_repeat * T * V, -1, attn.heads, head_dim).transpose(1, 2)
 
            # Build attention bias from boolean mask
            mask_bias = mask_bool.float()
            mask_bias[mask_bias == 0] = -1e9
            mask_bias[mask_bias == 1] = 0
            # Expand to (B*cf*T, heads, q_len, k_len)
            # mask_bool shape: (B, V*S, V*S), need to expand to (B, cf_repeat, T, heads, V*S, V*S)
            mask_bias = mask_bias.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(B, cf_repeat, T, attn.heads, V * H_map * W_map, V * H_map * W_map)
            mask_bias = rearrange(mask_bias, 'b cf t h q k -> (b cf t) h q k')

            # 将注意力拆分为：
            # 1) 主视角<->主视角（所有batch共享同一顺序）
            # 2) 子视角->其父主视角（拼接后一次性 masked attention）

            # reshape 为 (B, cf, T, V, h, s, c)
            q_b = rearrange(q_spad, '(b cf t v) h s c -> b cf t v h s c', b=B, cf=cf_repeat, t=T, v=V)
            k_b = rearrange(k_spad, '(b cf t v) h s c -> b cf t v h s c', b=B, cf=cf_repeat, t=T, v=V)
            v_b = rearrange(v_spad, '(b cf t v) h s c -> b cf t v h s c', b=B, cf=cf_repeat, t=T, v=V)

            # 输出容器
            out_b = torch.zeros_like(q_b)

            # 不区分父/子：保持全视角联通的 epi 注意力
            q_cat = rearrange(q_b, 'b cf t v h s c -> (b cf t) h (v s) c')
            k_cat = rearrange(k_b, 'b cf t v h s c -> (b cf t) h (v s) c')
            v_cat = rearrange(v_b, 'b cf t v h s c -> (b cf t) h (v s) c')
            out_cat = F.scaled_dot_product_attention(q_cat, k_cat, v_cat, attn_mask=mask_bias, dropout_p=0.0, is_causal=False)
            out_b = rearrange(out_cat, '(b cf t) h (v s) c -> b cf t v h s c', b=B, cf=cf_repeat, t=T, v=V, s=H_map * W_map)

            # 投影回输出格式
            out_spad = rearrange(out_b, 'b cf t v h s c -> b cf t v s (h c)')
            out_spad = self.to_out_spad[0](rearrange(out_spad, 'b cf t v s c -> (b cf t v) s c'))
            out_spad = self.to_out_spad[1](out_spad)
            hidden_states_spad = out_spad

        # Match-based attention 已移除







        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if use_mv:
            hidden_states = hidden_states + hidden_states_mv * mv_scale

        if use_densemv:
            hidden_states = hidden_states + hidden_states_densemv * densemv_scale

        if use_ref:
            hidden_states = hidden_states + hidden_states_ref * ref_scale

        if use_cd:
            hidden_states = hidden_states + hidden_states_cd * cd_scale

        if use_spad_epi:
            hidden_states = hidden_states + hidden_states_spad * spad_epi_scale
            # print(spad_epi_scale)

        # Match-based attention 已移除

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor


        return hidden_states

    def set_num_views(self, num_views: int) -> None:
        self.num_views = num_views
