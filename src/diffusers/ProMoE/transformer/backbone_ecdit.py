import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_promoe_common import (
    Attention,
    FinalLayer,
    LabelEmbedder,
    Mlp,
    MoeMLP_DiffMoE as MoeMLP,
    PatchEmbed,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    modulate,
)


class SparseMoEBlock(nn.Module):
    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.empty((num_experts, hidden_dim)))
        nn.init.normal_(self.gate_weight, std=0.006)
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts
        self.n_shared_experts = n_shared_experts
        if self.n_shared_experts > 0:
            intermediate_size = hidden_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(hidden_size=hidden_dim, intermediate_size=intermediate_size, pretraining_tp=2)

    def forward(self, x):
        identity = x
        batch_size, seq_len, _ = x.shape
        logits = F.linear(x, self.gate_weight, None)
        affinity = logits.softmax(dim=-1)
        affinity = torch.einsum("b s e -> b e s", affinity)
        k = int((seq_len / self.num_experts) * self.capacity)
        gating, index = torch.topk(affinity, k=k, dim=-1, sorted=False)
        dispatch = F.one_hot(index, num_classes=seq_len).to(device=x.device, dtype=x.dtype)
        x_in = torch.einsum("b e c s, b s d -> b e c d", dispatch, x)
        x_e = [self.experts[e](x_in[:, e]) for e in range(self.num_experts)]
        x_e = torch.stack(x_e, dim=1)
        x_out = torch.einsum("b e c s, b e c, b e c d -> b s d", dispatch, gating, x_e)
        if self.n_shared_experts > 0:
            x_out = x_out + self.shared_experts(identity)
        return x_out


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim=None,
        mlp_ratio=4.0,
        use_swiglu=False,
        MoE_config=None,
        use_moe=False,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_moe:
            if not use_swiglu:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                experts = [
                    Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
                    for _ in range(MoE_config.num_experts)
                ]
            else:
                experts = [MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim) for _ in range(MoE_config.num_experts)]
            self.mlp = SparseMoEBlock(
                experts=experts,
                hidden_dim=hidden_size,
                num_experts=MoE_config.num_experts,
                capacity=MoE_config.capacity,
                n_shared_experts=MoE_config.n_shared_experts,
            )
        else:
            if not use_swiglu:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
            else:
                self.mlp = MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        qk_norm=False,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        use_swiglu=False,
        MoE_config=None,
        head_dim=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.MoE_config = MoE_config
        use_moe_flag = [i % 2 == 1 for i in range(depth)] if self.MoE_config.interleave else [True] * depth

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    use_swiglu=use_swiglu,
                    MoE_config=MoE_config,
                    use_moe=use_moe_flag[i],
                )
                for i in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.init_MoeMLP = MoE_config.init_MoeMLP
        self.initialize_weights()
        self.capacity_schedule = MoE_config.get("capacity_schedule", None)
        if self.capacity_schedule:
            self.training_iters = -1

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        def init_moe_mlp(module, std=0.006):
            nn.init.normal_(module.gate_proj.weight, std=std)
            nn.init.normal_(module.up_proj.weight, std=std)
            nn.init.normal_(module.down_proj.weight, std=std)

        if self.init_MoeMLP:
            for block in self.blocks:
                if hasattr(block.mlp, "experts"):
                    for expert in block.mlp.experts:
                        if hasattr(expert, "gate_proj"):
                            init_moe_mlp(expert)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, context, **kwargs):
        y = context
        if len(x.shape) != 4:
            x = x.squeeze(2)
        if self.training and self.capacity_schedule:
            num_experts = self.MoE_config.num_experts
            capacity = self.MoE_config.capacity
            stage_i = self.MoE_config.capacity_schedule.capacity_schedule_stage_I_iters
            stage_ii = self.MoE_config.capacity_schedule.capacity_schedule_stage_II_iters
            if self.training_iters <= stage_i:
                capacity = num_experts
            elif self.training_iters <= stage_ii:
                capacity = capacity + (num_experts - capacity) * (stage_ii - self.training_iters) / (stage_ii - stage_i)
            for block in self.blocks:
                if hasattr(block.mlp, "capacity"):
                    block.mlp.capacity = capacity

        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)
