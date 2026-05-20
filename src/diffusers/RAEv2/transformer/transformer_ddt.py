from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class ConfigMixin:
        config_name = "config.json"

    class ModelMixin(nn.Module):
        pass

    def register_to_config(init):
        return init


from transformer_layers import ConditionEmbedder, GaussianFourierEmbedding, RoPE, RMSNorm, NormAttention, SwiGLUFFN


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


@dataclass
class RAEV2Transformer2DOutput(BaseOutput):
    sample: torch.FloatTensor
    projection_states: Optional[torch.FloatTensor] = None


class DDTEncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * hidden_size * mlp_ratio))

    def forward(self, x, rope, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DDTDecoderBlock(DDTEncoderBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__(hidden_size, num_heads, mlp_ratio)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c, rope, attn_mask=None):
        modulation = self.adaln_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope, attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiTwDDTHead(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        input_size=16,
        in_channels=768,
        patch_size=None,
        hidden_size=None,
        depth=None,
        num_heads=None,
        mlp_ratio=4.0,
        enable_repa=False,
        repa_layer_depth=8,
        z_dim=None,
        num_classes=1000,
        condition_type="label",
        context_dim=768,
        cond_arch=None,
        use_cfg_conds=False,
    ):
        super().__init__()
        if patch_size is None:
            patch_size = [1, 1]
        if hidden_size is None:
            hidden_size = [1152, 2048]
        if depth is None:
            depth = [28, 2]
        if num_heads is None:
            num_heads = [16, 16]

        self.in_channels = in_channels
        self.enc_hidden_size, dec_hidden_size = hidden_size
        self.num_enc_blocks, self.num_dec_blocks = depth
        self.s_patch_size, self.x_patch_size = patch_size
        enc_num_heads, dec_num_heads = num_heads

        self.repa_layer_depth = repa_layer_depth
        self.use_cfg_conds = use_cfg_conds

        self.s_embedder = PatchEmbed(input_size, self.s_patch_size, in_channels, self.enc_hidden_size)
        self.x_embedder = PatchEmbed(input_size, self.x_patch_size, in_channels, dec_hidden_size)
        self.s_projector = (
            nn.Linear(self.enc_hidden_size, dec_hidden_size)
            if self.enc_hidden_size != dec_hidden_size
            else nn.Identity()
        )

        self.num_cond_tokens = cond_arch.num_t_tokens + cond_arch.num_c_tokens
        self.t_embedder = GaussianFourierEmbedding(self.enc_hidden_size, cond_arch.num_t_tokens)
        self.ctx_embedder = ConditionEmbedder(
            self.enc_hidden_size,
            num_classes,
            context_dim,
            condition_type,
            cond_arch.num_c_tokens,
            latent_in_channels=in_channels,
            latent_patch_size=self.s_patch_size,
            n_action_tokens=getattr(cond_arch, "n_action_tokens", 4),
        )
        if self.use_cfg_conds:
            self.num_cond_tokens += cond_arch.num_cfg_omega_tokens
            self.cfg_w_embedder = GaussianFourierEmbedding(self.enc_hidden_size, cond_arch.num_cfg_omega_tokens)

        blocks = []
        for _ in range(self.num_enc_blocks):
            blocks.append(DDTEncoderBlock(self.enc_hidden_size, enc_num_heads, mlp_ratio))
        for _ in range(self.num_dec_blocks):
            blocks.append(DDTDecoderBlock(dec_hidden_size, dec_num_heads, mlp_ratio))
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = DDTFinalLayer(dec_hidden_size, self.x_patch_size, in_channels)
        self.enc_rope = RoPE(self.enc_hidden_size // enc_num_heads, self.s_embedder.num_patches, self.num_cond_tokens)
        self.dec_rope = RoPE(dec_hidden_size // dec_num_heads, self.x_embedder.num_patches)
        if enable_repa:
            self.repa_projector = nn.Linear(self.enc_hidden_size, z_dim)

        self.initialize_weights()

    def initialize_weights(self):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        if hasattr(self.ctx_embedder, "mlp"):
            nn.init.normal_(self.ctx_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.ctx_embedder.mlp[2].weight, std=0.02)
        if hasattr(self.ctx_embedder, "embedding_table"):
            nn.init.normal_(self.ctx_embedder.embedding_table.weight, std=0.02)

        for block in self.blocks:
            if hasattr(block, "adaln_modulation"):
                nn.init.constant_(block.adaln_modulation[-1].weight, 0)
                nn.init.constant_(block.adaln_modulation[-1].bias, 0)

        for t_embedder in ("t_embedder", "cfg_w_embedder"):
            if hasattr(self, t_embedder):
                nn.init.normal_(getattr(self, t_embedder).mlp[0].weight, std=0.02)
                nn.init.normal_(getattr(self, t_embedder).mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        h, c = int(x.shape[1] ** 0.5), self.in_channels
        return x.reshape(x.shape[0], h, h, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h * p, h * p)

    def _build_sequence(self, x, t, condition_kwargs):
        seq = [self.s_embedder(x)]
        t_emb_base, t_emb = self.t_embedder(t, return_base_embed=True)
        seq.append(t_emb)
        if self.use_cfg_conds:
            seq.append(self.cfg_w_embedder(condition_kwargs["omega"]))
        seq.append(self.ctx_embedder(condition_kwargs["context"]))
        return torch.cat(seq, dim=1), t_emb_base

    def _build_attn_mask(self, seq, condition_kwargs):
        attn_mask = torch.ones((seq.shape[0], seq.shape[1]), device=seq.device)
        cond_mask = condition_kwargs.get("attn_mask")
        if cond_mask is not None:
            attn_mask[:, -cond_mask.shape[1] :] = cond_mask
        return (1.0 - attn_mask[:, None, None, :]) * torch.finfo(seq.dtype).min

    def forward(
        self,
        x,
        t,
        return_dict: bool = False,
        return_intermediate: bool = False,
        **condition_kwargs,
    ) -> Union[RAEV2Transformer2DOutput, Tuple[Any, ...]]:
        zt_intermediate = None
        seq, t_emb_base = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(seq, condition_kwargs)
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, self.enc_rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(seq[:, : self.s_embedder.num_patches, :])
        seq = self.s_projector(F.silu(t_emb_base + seq[:, : self.s_embedder.num_patches, :]))

        x = self.x_embedder(x)
        for i in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + i](x, seq, self.dec_rope)

        x = self.final_layer(x, seq)
        x = self.unpatchify(x, self.x_patch_size)

        if not return_dict:
            if return_intermediate:
                return x, zt_intermediate
            return x

        return RAEV2Transformer2DOutput(sample=x, projection_states=zt_intermediate)


class DiTwDDTHeadIG(DiTwDDTHead):
    @register_to_config
    def __init__(self, base_model_depth=8, **kwargs):
        super().__init__(**kwargs)
        self.base_model_depth = base_model_depth
        self.base_final_layer = DDTFinalLayer(self.enc_hidden_size, self.s_patch_size, self.in_channels)
        nn.init.constant_(self.base_final_layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.base_final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.base_final_layer.linear.weight, 0)
        nn.init.constant_(self.base_final_layer.linear.bias, 0)

    def forward(
        self,
        x,
        t,
        return_dict: bool = False,
        return_intermediate: bool = False,
        **condition_kwargs,
    ) -> Union[RAEV2Transformer2DOutput, Tuple[Any, ...]]:
        zt_intermediate = None
        x_base = None
        seq, t_emb_base = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(seq, condition_kwargs)
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, self.enc_rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(seq[:, : self.s_embedder.num_patches, :])
            if (i + 1) == self.base_model_depth:
                x_base = seq[:, : self.s_embedder.num_patches, :]
        seq = self.s_projector(F.silu(t_emb_base + seq[:, : self.s_embedder.num_patches, :]))

        x = self.x_embedder(x)
        for i in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + i](x, seq, self.dec_rope)

        x = self.final_layer(x, seq)
        x = self.unpatchify(x, self.x_patch_size)

        x_base = F.silu(t_emb_base + x_base)
        x_base = self.base_final_layer(x_base, x_base)
        x_base = self.unpatchify(x_base, self.s_patch_size)

        if not return_dict:
            if return_intermediate:
                return (x, x_base), zt_intermediate
            return (x, x_base)

        return RAEV2Transformer2DOutput(sample=(x, x_base), projection_states=zt_intermediate)
