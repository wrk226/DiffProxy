#!/usr/bin/env python
# coding: utf-8
"""Reload the merged proxy VAE after the main Lightning checkpoint."""

import os

import torch
from safetensors.torch import load_file


def reload_proxy_vae(system, checkpoint_path: str):
    """Restore the canonical proxy VAE used for UV and segmentation outputs.

    The main DiffProxy checkpoint is loaded after model construction and may
    restore its historical VAE state. Reapply the merged proxy VAE afterwards
    so the final inference state matches the validated segmentation decoder.
    """
    model_path = os.path.join(
        checkpoint_path,
        "diffusion_pytorch_model.safetensors",
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"merged proxy VAE weights not found: {model_path}"
        )

    checkpoint = load_file(model_path)
    system.pipeline.vae.load_state_dict(checkpoint, strict=True)
    system.pipeline.vae = system.pipeline.vae.to(
        device=system.device,
        dtype=torch.float32,
    )
    return system
