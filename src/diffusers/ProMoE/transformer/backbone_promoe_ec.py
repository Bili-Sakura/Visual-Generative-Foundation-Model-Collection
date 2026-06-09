import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_promoe_common import (
    Attention,
    FinalLayer,
    LabelEmbedder,
    Mlp,
    MoeMLP,
    PatchEmbed,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    modulate,
)


class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device) if ctx.required_aux_loss else None
        return grad_output, grad_loss


class SparseMoeBlock(nn.Module):
    def __init__(
        self,
        num_routed_experts,
        hidden_size,
        moe_intermediate_size,
        shared_expert_intermediate_size,
        top_k=1,
        load_balance_loss_coef=0,
        norm_topk_prob=False,
        seq_aux=False,
        use_shared_expert=True,
        use_uncond_expert=True,
        router_weight_mode="softmax",
        routing_contrastive_lam=0,
        use_top_k_for_routing_contrastive=False,
        routing_contrastive_temperature=0.1,
        **kwargs,
    ):
        super().__init__()
        del load_balance_loss_coef, norm_topk_prob, seq_aux, use_top_k_for_routing_contrastive
        self.num_experts = num_routed_experts + 1 if use_uncond_expert else num_routed_experts
        self.num_routed_experts = num_routed_experts
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.cluster_centers = nn.Parameter(torch.randn(num_routed_experts, hidden_size))
        self.use_shared_expert = use_shared_expert
        self.use_uncond_expert = use_uncond_expert
        self.router_weight_mode = router_weight_mode
        self.routing_contrastive_lam = routing_contrastive_lam
        self.routing_contrastive_temperature = routing_contrastive_temperature
        self.experts = nn.ModuleList(
            [MoeMLP(hidden_size=hidden_size, intermediate_size=moe_intermediate_size) for _ in range(self.num_experts)]
        )
        if use_shared_expert:
            self.shared_expert = MoeMLP(hidden_size=hidden_size, intermediate_size=shared_expert_intermediate_size)
        self._init_weights()

    def compute_router(self, cond_hidden_states):
        b_cond, seq_len, _ = cond_hidden_states.shape
        num_cond_experts = self.num_routed_experts
        input_norm = F.normalize(cond_hidden_states, p=2, dim=-1)
        cluster_norm = F.normalize(self.cluster_centers, p=2, dim=-1)
        cos_sim = input_norm @ cluster_norm.T
        cos_sim_expert_view = cos_sim.transpose(1, 2)
        if self.router_weight_mode == "softmax":
            cond_weights = F.softmax(cos_sim_expert_view, dim=-1)
        elif self.router_weight_mode == "sigmoid":
            cond_weights = torch.sigmoid(cos_sim_expert_view)
        elif self.router_weight_mode == "identity":
            cond_weights = cos_sim_expert_view
        else:
            raise ValueError(f"Unsupported router_weight_mode: {self.router_weight_mode}")
        k = max(1, min(int((seq_len / num_cond_experts) * self.top_k), seq_len))
        router_weights, indices = torch.topk(cond_weights, k=k, dim=-1, sorted=False)
        dispatch_mask = F.one_hot(indices, num_classes=seq_len).to(dtype=cond_hidden_states.dtype)
        expert_inputs = torch.einsum("becs,bsd->becd", dispatch_mask, cond_hidden_states)
        return dispatch_mask, router_weights, expert_inputs

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor):
        identity = hidden_states
        batch_size, _, hidden_dim = hidden_states.shape
        final_output = torch.zeros_like(hidden_states)
        loss = None
        cond_batch_mask = (
            labels.view(-1) != 1000
        ) if self.use_uncond_expert else torch.ones(batch_size, dtype=torch.bool, device=hidden_states.device)
        uncond_batch_mask = ~cond_batch_mask
        cond_experts = self.experts[:-1] if self.use_uncond_expert else self.experts

        if cond_batch_mask.any():
            cond_hidden_states = hidden_states[cond_batch_mask]
            dispatch_mask, gating_scores, expert_inputs = self.compute_router(cond_hidden_states)
            num_cond_experts = len(cond_experts)
            expert_outputs = torch.stack([cond_experts[e](expert_inputs[:, e]) for e in range(num_cond_experts)], dim=1)
            cond_output = torch.einsum("becs,bec,becd->bsd", dispatch_mask, gating_scores, expert_outputs).to(hidden_states.dtype)
            final_output[cond_batch_mask] = cond_output
            if self.training and self.routing_contrastive_lam > 0 and num_cond_experts > 1:
                expert_token_means = expert_inputs.mean(dim=2)
                routing_contrastive_loss = self.compute_routing_contrastive_loss(expert_token_means)
                loss = routing_contrastive_loss * self.routing_contrastive_lam
        else:
            dummy_input = torch.zeros(1, 1, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            for expert in cond_experts:
                final_output = final_output + expert(dummy_input).sum() * 0

        if self.use_uncond_expert:
            if uncond_batch_mask.any():
                uncond_hidden_states = hidden_states[uncond_batch_mask]
                final_output[uncond_batch_mask] = self.experts[-1](uncond_hidden_states).to(final_output.dtype)
            else:
                dummy_input = torch.zeros(1, 1, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
                final_output = final_output + self.experts[-1](dummy_input).sum() * 0

        if self.use_shared_expert:
            final_output += self.shared_expert(identity).to(hidden_states.dtype)
        return final_output, loss

    def compute_routing_contrastive_loss(self, expert_token_means):
        batch_size, num_cond_experts, _ = expert_token_means.shape
        if num_cond_experts < 2:
            return torch.tensor(0.0, device=expert_token_means.device)
        centers_norm = F.normalize(self.cluster_centers, p=2, dim=1)
        means_norm = F.normalize(expert_token_means, p=2, dim=2)
        sim_matrix = torch.einsum("id,bjd->bij", centers_norm, means_norm)
        logits = sim_matrix / self.routing_contrastive_temperature
        labels = torch.arange(num_cond_experts, device=logits.device).unsqueeze(0).expand(batch_size, -1)
        return F.cross_entropy(logits.reshape(batch_size * num_cond_experts, -1), labels.reshape(-1))

    def _init_weights(self):
        nn.init.normal_(self.cluster_centers, mean=0.0, std=0.02)


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
        self.use_moe = use_moe
        if use_moe:
            self.mlp = SparseMoeBlock(hidden_size=hidden_size, **MoE_config)
        else:
            if not use_swiglu:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
            else:
                self.mlp = MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, label):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_moe:
            x_mlp, aux_loss = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), label)
            if aux_loss is not None:
                x_mlp = AddAuxiliaryLoss.apply(x_mlp, aux_loss)
            return x + gate_mlp.unsqueeze(1) * x_mlp
        return x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


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
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob, return_labels=True)
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
            nn.init.normal_(module.up_proj.weight, std=std)
            nn.init.normal_(module.down_proj.weight, std=std)

        if self.init_MoeMLP:
            for block in self.blocks:
                if hasattr(block.mlp, "experts"):
                    for expert in block.mlp.experts:
                        init_moe_mlp(expert)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, timestep, context, **kwargs):
        y = context
        if len(x.shape) != 4:
            x = x.squeeze(2)
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(timestep)
        y, labels = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c, labels)
        x = self.final_layer(x, c)
        return self.unpatchify(x)
