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
        top_k=2,
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
        del norm_topk_prob
        self.num_experts = num_routed_experts + 1 if use_uncond_expert else num_routed_experts
        self.num_routed_experts = num_routed_experts
        self.seq_aux = seq_aux
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.cluster_centers = nn.Parameter(torch.randn(num_routed_experts, hidden_size))
        self.alpha = load_balance_loss_coef
        self.use_shared_expert = use_shared_expert
        self.use_uncond_expert = use_uncond_expert
        self.router_weight_mode = router_weight_mode
        self.routing_contrastive_lam = routing_contrastive_lam
        self.use_top_k_for_routing_contrastive = use_top_k_for_routing_contrastive
        self.routing_contrastive_temperature = routing_contrastive_temperature
        self.experts = nn.ModuleList(
            [MoeMLP(hidden_size=hidden_size, intermediate_size=moe_intermediate_size) for _ in range(self.num_experts)]
        )
        if use_shared_expert:
            self.shared_expert = MoeMLP(hidden_size=hidden_size, intermediate_size=shared_expert_intermediate_size)
        self._init_weights()

    def compute_router(self, hidden_states, labels):
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        flat_input = hidden_states.view(-1, self.hidden_size)
        flat_labels = labels.view(batch_size, 1).expand(-1, seq_len).reshape(-1)
        if self.use_uncond_expert and flat_labels is not None:
            uncond_mask = flat_labels == 1000
            cond_mask = ~uncond_mask
        else:
            uncond_mask = None
            cond_mask = torch.ones_like(flat_labels, dtype=torch.bool)

        router_weights = torch.zeros(batch_size * seq_len, self.top_k, device=device, dtype=hidden_states.dtype)
        expert_indices = torch.zeros(batch_size * seq_len, self.top_k, device=device, dtype=torch.long)

        if uncond_mask is not None and uncond_mask.any():
            uncond_positions = torch.where(uncond_mask)[0]
            router_weights[uncond_positions, 0] = 1.0
            expert_indices[uncond_positions] = self.num_experts - 1

        cond_weights = None
        topk_idx = None
        if cond_mask.any():
            cond_positions = torch.where(cond_mask)[0]
            cond_input = flat_input[cond_positions]
            input_norm = F.normalize(cond_input, p=2, dim=1)
            cluster_norm = F.normalize(self.cluster_centers, p=2, dim=1)
            cos_sim = input_norm @ cluster_norm.T
            if self.router_weight_mode == "softmax":
                cond_weights = F.softmax(cos_sim, dim=1)
            elif self.router_weight_mode == "sigmoid":
                cond_weights = torch.sigmoid(cos_sim)
            elif self.router_weight_mode == "identity":
                cond_weights = cos_sim
            else:
                raise ValueError(f"Unsupported router_weight_mode: {self.router_weight_mode}")
            topk_scores, topk_idx = torch.topk(cond_weights, k=self.top_k, dim=1)
            router_weights[cond_positions] = topk_scores.to(router_weights.dtype)
            expert_indices[cond_positions] = topk_idx

        router_weights = router_weights.view(batch_size, seq_len, self.top_k)
        expert_indices = expert_indices.view(batch_size, seq_len, self.top_k)

        load_balance_loss = None
        if self.training and self.alpha > 0.0 and cond_weights is not None and topk_idx is not None:
            cond_batch_size = (labels != 1000).sum()
            scores_for_aux = F.softmax(cond_weights, dim=1) if self.router_weight_mode != "softmax" else cond_weights
            topk_idx_for_aux_loss = topk_idx.view(cond_batch_size, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(cond_batch_size, seq_len, -1)
                ce = torch.zeros(cond_batch_size, self.num_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(cond_batch_size, seq_len * self.top_k, device=hidden_states.device),
                ).div_(seq_len * self.top_k / self.num_routed_experts)
                load_balance_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.num_routed_experts)
                ce = mask_ce.float().mean(0)
                pi = scores_for_aux.mean(0)
                fi = ce * self.num_routed_experts
                load_balance_loss = (pi * fi).sum() * self.alpha
        return router_weights, expert_indices, load_balance_loss

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor):
        router_weights, expert_indices, load_balance_loss = self.compute_router(hidden_states, labels)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_input = hidden_states.view(-1, hidden_dim)
        flat_weights = router_weights.view(-1, self.top_k)
        flat_indices = expert_indices.view(-1, self.top_k)
        total_tokens = batch_size * seq_len
        final_output = torch.zeros(total_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

        for expert_id in range(self.num_experts):
            expert_mask = (flat_indices == expert_id).any(dim=1)
            token_ids = torch.where(expert_mask)[0]
            if token_ids.numel() > 0:
                expert_input = flat_input[token_ids]
                expert_weight_mask = flat_indices[token_ids] == expert_id
                expert_weights = flat_weights[token_ids] * expert_weight_mask.to(dtype=flat_weights.dtype)
                combined_weights = expert_weights.sum(dim=1)
                expert_output = self.experts[expert_id](expert_input)
                weighted_output = expert_output * combined_weights.unsqueeze(1)
                final_output.index_add_(0, token_ids, weighted_output)
            else:
                dummy_input = torch.zeros(1, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
                final_output[0] += self.experts[expert_id](dummy_input)[0] * 0

        final_output = final_output.view(batch_size, seq_len, hidden_dim)
        if self.use_shared_expert:
            final_output += self.shared_expert(hidden_states)

        loss = load_balance_loss
        if self.training and self.routing_contrastive_lam > 0:
            flat_labels = labels.view(batch_size, 1).expand(-1, seq_len).reshape(-1)
            cond_mask = ~(
                flat_labels == 1000
            ) if self.use_uncond_expert else torch.ones(batch_size * seq_len, dtype=torch.bool, device=hidden_states.device)
            cond_token_embeddings = flat_input[cond_mask]
            if self.use_top_k_for_routing_contrastive:
                cond_cluster_assignments = expert_indices.view(batch_size * seq_len, self.top_k)[cond_mask]
            else:
                top1_expert_indices = expert_indices.view(batch_size * seq_len, self.top_k)[:, 0]
                cond_cluster_assignments = top1_expert_indices[cond_mask]
            routing_contrastive_loss = self.compute_routing_contrastive_loss(
                cond_token_embeddings,
                cond_cluster_assignments,
                use_top_k=self.use_top_k_for_routing_contrastive,
            )
            routing_contrastive_loss = routing_contrastive_loss * self.routing_contrastive_lam
            loss = routing_contrastive_loss if loss is None else loss + routing_contrastive_loss

        return final_output, loss

    def compute_routing_contrastive_loss(self, token_embeddings, cluster_assignments, use_top_k=False):
        cluster_centers = self.cluster_centers
        num_clusters = cluster_centers.size(0)
        device = cluster_centers.device
        cluster_means = []
        valid_clusters = []
        for cluster_id in range(num_clusters):
            mask = (cluster_assignments == cluster_id).any(dim=1) if use_top_k else cluster_assignments == cluster_id
            if mask.sum() > 0:
                cluster_means.append(token_embeddings[mask].mean(dim=0, keepdim=True))
                valid_clusters.append(cluster_id)
        if len(valid_clusters) < 2:
            return torch.tensor(0.0, device=device)
        cluster_means = torch.cat(cluster_means, dim=0)
        valid_centers = cluster_centers[valid_clusters]
        centers_norm = F.normalize(valid_centers, p=2, dim=1)
        means_norm = F.normalize(cluster_means, p=2, dim=1)
        sim_matrix = centers_norm @ means_norm.T
        logits = sim_matrix / self.routing_contrastive_temperature
        labels = torch.arange(sim_matrix.size(0), device=device)
        return F.cross_entropy(logits, labels)

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
