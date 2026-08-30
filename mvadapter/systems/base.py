from dataclasses import dataclass

import pytorch_lightning as pl
import torch

from ..utils.config import parse_structured
from ..utils.typing import *


class BaseSystem(pl.LightningModule):
    """Minimal Lightning wrapper required for inference checkpoint loading."""

    @dataclass
    class Config:
        allow_tf32: bool = True
        target_image_modality: Optional[List[str]] = None

    cfg: Config

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        torch.backends.cuda.matmul.allow_tf32 = self.cfg.allow_tf32
        self.configure()
        self.post_configure()

    def configure(self) -> None:
        pass

    def post_configure(self) -> None:
        pass
