# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
from torch import nn, Tensor

import math
import torch.nn.functional as F
from torch import nn


def gen_encoder_output_proposals(memory:Tensor, spatial_shapes:Tensor):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []

    for lvl, (H_, W_) in enumerate(spatial_shapes):

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

        scale = torch.tensor([W_, H_], dtype=torch.float32, device=memory.device).view(1, 1, 1, 2).repeat(N_, 1, 1, 1)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        # wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

        # proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposal = torch.cat((grid, grid), -1).view(N_, -1, 4)
        proposals.append(proposal)

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory, output_proposals


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob =  inputs.sigmoid()  
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1-alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor, hidden_dim):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    hidden_dim_ = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(hidden_dim_, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / hidden_dim_)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    w_embed = pos_tensor[:, :, 2] * scale
    pos_w = w_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    h_embed = pos_tensor[:, :, 3] * scale
    pos_h = h_embed[:, :, None] / dim_t
    pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    return pos

