import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ReferenceImageProjector(nn.Module):
    """
    Projects reference image features from UNet intermediate layers to cross-attention compatible features.
    Similar to IP-Adapter's image projection module.
    """
    
    def __init__(
        self,
        hidden_dims: List[int],  # UNet's block_out_channels
        cross_attention_dim: int = 768,  # Text encoder hidden size
        num_layers: int = 2,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.cross_attention_dim = cross_attention_dim
        self.output_dim = output_dim or cross_attention_dim
        
        # Create projectors for each UNet block
        self.projectors = nn.ModuleDict()
        
        # For each resolution level in UNet
        for i, hidden_dim in enumerate(hidden_dims):
            layers = []
            in_dim = hidden_dim
            
            # Multi-layer MLP
            for j in range(num_layers):
                out_dim = self.output_dim if j == num_layers - 1 else hidden_dim
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim) if j < num_layers - 1 else nn.Identity(),
                    nn.GELU() if j < num_layers - 1 else nn.Identity(),
                ])
                in_dim = out_dim
            
            self.projectors[f"block_{i}"] = nn.Sequential(*layers)
    
    def forward(self, reference_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Projects reference features from each UNet block.
        
        Args:
            reference_features: Dictionary mapping layer names to feature tensors
                                Each tensor has shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            projected_features: Dictionary with same keys but projected features
        """
        projected_features = {}
        
        for name, features in reference_features.items():
            # Determine which block this feature belongs to
            block_idx = self._get_block_index(name)
            if block_idx is not None and f"block_{block_idx}" in self.projectors:
                # Project features
                projected = self.projectors[f"block_{block_idx}"](features)
                projected_features[name] = projected
            else:
                # If no specific projector, use the last one as default
                projected = self.projectors[f"block_{len(self.hidden_dims)-1}"](features)
                projected_features[name] = projected
        
        return projected_features
    
    def _get_block_index(self, layer_name: str) -> Optional[int]:
        """Extract block index from layer name."""
        if "down_blocks" in layer_name:
            # Extract index from names like "down_blocks.0.attentions.0"
            parts = layer_name.split(".")
            for i, part in enumerate(parts):
                if part == "down_blocks" and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        pass
        elif "mid_block" in layer_name:
            # Mid block corresponds to the last hidden dim
            return len(self.hidden_dims) - 1
        elif "up_blocks" in layer_name:
            # Extract index and reverse it for up blocks
            parts = layer_name.split(".")
            for i, part in enumerate(parts):
                if part == "up_blocks" and i + 1 < len(parts):
                    try:
                        up_idx = int(parts[i + 1])
                        # Reverse the index for up blocks
                        return len(self.hidden_dims) - 2 - up_idx
                    except ValueError:
                        pass
        return None


class SimpleMLP(nn.Module):
    """
    Simple MLP projector for reference features.
    Maps from any input dimension to output dimension.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or max(input_dim, output_dim)
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < num_layers - 1 else nn.Identity(),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) 