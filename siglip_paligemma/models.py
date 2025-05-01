import torch
import torch.nn as nn
from transformers import SiglipVisionConfig, PaliGemmaForConditionalGeneration, SiglipVisionModel, AutoConfig
from transformers.activations import PytorchGELUTanh # activation used in base siglip
from .configs import ModelConfig

# Copied from modeling_siglip.py with self.activation_fn=PytorchGELUTanh()
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = PytorchGELUTanh()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

# Copied from modeling_siglip.py
class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]

class PoolingHead(nn.Module):
    def __init__(self, vision_config: SiglipVisionConfig, num_classes = 1000):
        super().__init__()
        
        self.attention_head = SiglipMultiheadAttentionPoolingHead(vision_config)
        self.classification_head = nn.Linear(vision_config.hidden_size, num_classes)
        
    def forward(self, last_hidden_state: torch.Tensor):
        self.pooler_out = self.attention_head(last_hidden_state)
        self.pooler_out = self.classification_head(self.pooler_out)
        
        return self.pooler_out

class SiglipWithPoolingHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        if "paligemma" in config.model_name:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(config.model_name).vision_tower
        else:
            vision_config = AutoConfig.from_pretrained("google/paligemma-3b-pt-224").vision_config
            self.model = SiglipVisionModel(vision_config)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.pooling_head = PoolingHead(vision_config, config.num_classes)
        
        # logging
        model_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in self.pooling_head.parameters()) / 1e6
        print(f"{model_params=}M, {trainable_params=}M")
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self
    
    def forward(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            out = self.model(pixel_values=pixel_values)
        out = self.pooling_head(out.last_hidden_state)
        return out
        
