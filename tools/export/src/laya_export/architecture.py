"""Reviewed Laya decision architecture used only by the offline exporter.

Decision-head semantics are adapted from Apache-2.0 upstream commit
573e5b62696ba441230cd6be71d593331b5d23af. The encoder implementation comes
from the locked Transformers dependency and is constructed from local config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_model(config: dict[str, Any], encoder_directory: Path):
    """Construct the full encoder and decision heads without remote lookup."""

    import torch
    from transformers import AutoConfig, AutoModel

    encoder_config = AutoConfig.from_pretrained(
        encoder_directory,
        local_files_only=True,
        trust_remote_code=False,
    )
    encoder_config.reference_compile = False
    encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")

    class DecisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            hidden_size = encoder.config.hidden_size
            attention_heads = max(1, hidden_size // 64)
            layer = torch.nn.TransformerEncoderLayer(
                hidden_size,
                attention_heads,
                4 * hidden_size,
                0.1,
                batch_first=True,
                norm_first=True,
            )
            head_layers = int(config.get("head_layers", 2))
            self.head = torch.nn.TransformerEncoder(
                layer,
                head_layers,
                enable_nested_tensor=False,
            )
            self.type_emb = torch.nn.Embedding(3, hidden_size)
            self.scorer = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_size),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_size, 1),
            )
            action_count = len(config.get("act_costs", {})) + 1
            self.act_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size + 4, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, action_count),
            )
            self.register_buffer("temperature", torch.ones(3))

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            hidden = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            hidden = hidden + self.type_emb(qtype)[:, None, :]
            padding = ~attention_mask.bool()
            for layer in self.head.layers:
                hidden = layer(hidden, src_key_padding_mask=padding)
            indexes = marker_pos.clamp(min=0)[:, :, None].expand(
                -1, -1, hidden.size(-1)
            )
            markers = torch.gather(hidden, 1, indexes)
            logits = self.scorer(markers).squeeze(-1).float()
            logits = logits.masked_fill(~marker_mask, -1e4)
            probabilities = torch.softmax(logits.detach(), -1)
            option_count = marker_mask.sum(-1).clamp(min=2).float()
            entropy = -(
                probabilities * torch.log(probabilities.clamp_min(1e-9))
            ).sum(-1) / torch.log(option_count)
            if probabilities.size(-1) >= 2:
                top_two = probabilities.topk(2, -1).values
            else:
                top_one = probabilities.topk(1, -1).values
                top_two = torch.cat([top_one, torch.zeros_like(top_one)], dim=-1)
            features = torch.stack(
                [
                    top_two[:, 0],
                    top_two[:, 0] - top_two[:, 1],
                    entropy,
                    option_count / 255.0,
                ],
                -1,
            )
            pooled = hidden[:, 0].float()
            action_logits = self.act_head(torch.cat([pooled, features], -1))
            return logits, action_logits

    return DecisionModel()


def load_exact_weights(model, checkpoint: Path) -> dict[str, Any]:
    """Load Safetensors only and require exhaustive key and shape coverage."""

    from safetensors.torch import load_file

    weights = load_file(str(checkpoint), device="cpu")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(weights)
        if tuple(expected[name].shape) != tuple(weights[name].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "state-dict coverage failure: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={mismatched[:5]}"
        )
    model.load_state_dict(weights, strict=True)
    model.float().cpu().eval()
    return {
        "tensor_count": len(weights),
        "parameter_count": sum(value.numel() for value in model.parameters()),
    }
