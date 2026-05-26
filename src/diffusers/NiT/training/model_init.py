# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch.nn as nn


def initialize_nit_weights(model) -> None:
    """Weight initialization matching the official NiT repository."""

    def _basic_init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    model.apply(_basic_init)

    w = model.x_embedder.proj.weight.data
    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    nn.init.constant_(model.x_embedder.proj.bias, 0)

    nn.init.normal_(model.y_embedder.embedding_table.weight, std=0.02)
    nn.init.normal_(model.t_embedder.mlp[0].weight, std=0.02)
    nn.init.normal_(model.t_embedder.mlp[2].weight, std=0.02)

    for block in model.blocks:
        nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    nn.init.constant_(model.final_layer.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(model.final_layer.adaLN_modulation[-1].bias, 0)
    nn.init.constant_(model.final_layer.linear.weight, 0)
    nn.init.constant_(model.final_layer.linear.bias, 0)
