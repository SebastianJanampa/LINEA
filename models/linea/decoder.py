"""
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import math
from typing import Optional
from collections import OrderedDict

import torch 
from torch import nn, Tensor
import torch.nn.functional as F 
import torch.nn.init as init 

from .dn_components import prepare_for_cdn
from .attention_mechanism import MSDeformAttn, MSDeformLineAttn
from .linea_utils import weighting_function, distance2bbox, inverse_sigmoid, get_activation

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

def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 ):
        super().__init__()
        # cross attention
        self.cross_attn = MSDeformLineAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_modelmemory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # self attention
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1),
                               memory,
                               memory_spatial_shapes, 
                               ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # feed forward network
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max+1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class TransformerDecoder(nn.Module):
    def __init__(
    	self, 
    	decoder_layer, 
    	num_layers, 
    	norm=None, 
        d_model=256, 
        query_dim=4, 
        num_feature_levels=1,
        aux_loss=False,
        eval_idx=5,
        # from D-FINE
        reg_max=32,
        reg_scale=4,
        ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        self.num_layers = num_layers
        # self.norm = norm
        self.query_dim = query_dim
        self.num_feature_levels = num_feature_levels
        
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)

        self.reg_max = reg_max
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        self.d_model = d_model

        # prediction layers
        _class_embed = nn.Linear(d_model, 2)
        _enc_bbox_embed = MLP(d_model, d_model, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(2) * bias_value
        nn.init.constant_(_enc_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_enc_bbox_embed.layers[-1].bias.data, 0)

        _bbox_embed = MLP(d_model, d_model, 4 * (self.reg_max + 1), 3)

        self.bbox_embed = nn.ModuleList([copy.deepcopy(_bbox_embed) for i in range(num_layers)])
        self.class_embed = nn.ModuleList([copy.deepcopy(_class_embed) for i in range(num_layers)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max)) for _ in range(num_layers)])
        self.integral = Integral(self.reg_max)

        self.aux_loss = aux_loss

        # inference
        self.eval_idx = eval_idx

    def sine_embedding(self, tensor, hidden_dim):
        hidden_dim_ = hidden_dim // 2
        scale = 2 * math.pi
        dim_t = torch.arange(hidden_dim_, dtype=torch.float32, device=tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / hidden_dim_)
        x_embed = tensor[:, :, 0] * scale
        y_embed = tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        w_embed = tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        return pos

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)

    def forward(self, 
    	tgt, 
    	memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
        # for memory
        spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
        ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
        """
        output = tgt
        output_detach = pred_corners_undetach = 0

        intermediate = []
        ref_points_detach = refpoints_unsigmoid.sigmoid()

        ref_points_initial = ref_points_detach

        dec_out_bboxes = []
        dec_out_logits = []

        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, self.up, self.reg_scale)
        else:
            project = self.project

        for layer_id, layer in enumerate(self.layers):
            query_sine_embed = self.sine_embedding(ref_points_detach, self.d_model) # nq, bs, 256*2 
            
            ref_points_input = ref_points_detach[:, :, None]  # nq, bs, nlevel, 4

            query_pos = self.ref_point_head(query_sine_embed) # nq, bs, 256

            output = layer(
                tgt = output,
                tgt_query_pos = query_pos,
                tgt_query_sine_embed = query_sine_embed,
                tgt_key_padding_mask = tgt_key_padding_mask,
                tgt_reference_points = ref_points_input,

                memory = memory,
                memory_spatial_shapes = spatial_shapes,
                memory_pos = pos,

                self_attn_mask = tgt_mask,
                cross_attn_mask = memory_mask
            )

            pred_corners = self.bbox_embed[layer_id](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, self.integral(pred_corners, project), self.reg_scale) 

            if self.training or layer_id == self.eval_idx:
            	scores = self.class_embed[layer_id](output)
            	scores = self.lqe_layers[layer_id](scores, pred_corners)
            	dec_out_logits.append(scores)
            	dec_out_bboxes.append(inter_ref_bbox)

            pred_corners_undetach = pred_corners
            if self.training:
            	ref_points_detach = inter_ref_bbox.detach() 
            	output_detach = output.detach()
            else:
            	ref_points_detach = inter_ref_bbox
            	output_detach = output

        return torch.stack(dec_out_bboxes).permute(0, 2, 1, 3), torch.stack(dec_out_logits).permute(0, 2, 1, 3), 

class LINEATransformer(nn.Module):
    def __init__(
        self,
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        d_model=256, 
        num_classes=2,
        nhead=8, 
        num_queries=300, 
        num_decoder_layers=6, 
        dim_feedforward=2048, 
        dropout=0.0,
        activation="relu", 
        normalize_before=False, 
        query_dim=4,
        aux_loss=False,
        # for deformable encoder
        num_feature_levels=1,
        dec_n_points=4,
        # from D-FINE
        reg_max=32,
        reg_scale=4,
        # denoising
        dn_number=100,
        dn_label_noise_ratio=0.5,
        dn_line_noise_scale=0.5,
        # for inference
        eval_spatial_size=None,
        eval_idx=5
        ):
        super().__init__()

        # init learnable queries
        self.tgt_embed = nn.Embedding(num_queries, d_model)
        nn.init.normal_(self.tgt_embed.weight.data)

        # line segment detection parameters
        self.num_classes = num_classes
        self.num_queries = num_queries

        # anchor selection at the output of encoder
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model) 
        self._reset_parameters()

        # prediction layers
        _class_embed = nn.Linear(d_model, num_classes)
        _bbox_embed = MLP(d_model, d_model, 4, 3)

        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(2) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        self.enc_out_bbox_embed  = copy.deepcopy(_bbox_embed)
        self.enc_out_class_embed  = copy.deepcopy(_class_embed)

        # decoder parameters
        self.d_model = d_model
        self.n_heads = nhead
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        d_model=d_model, query_dim=query_dim, 
                                        num_feature_levels=num_feature_levels, 
                                        eval_idx=eval_idx, aux_loss=aux_loss,
                                        reg_max=reg_max, reg_scale=reg_scale)

        # for inference mode
        self.eval_spatial_size = eval_spatial_size
        if eval_spatial_size is not None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in feat_strides
            ]
            output_proposals, output_proposals_valid = self.generate_anchors(spatial_shapes)
            self.register_buffer('output_proposals', output_proposals)
            self.register_buffer('output_proposals_mask', ~output_proposals_valid)


        # denoising parameters
        self.dn_number = dn_number
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_line_noise_scale = dn_line_noise_scale
        self.label_enc = nn.Embedding(90 + 1, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn): # or isinstance(m, MSDeformLineAttn):
                m._reset_parameters()

    def generate_anchors(self, spatial_shapes):
        proposals = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32), indexing='ij')
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

            scale = torch.tensor([W_, H_], dtype=torch.float32,).view(1, 1, 1, 2)
            grid = (grid.unsqueeze(0) + 0.5) / scale

            proposal = torch.cat((grid, grid), -1).view(1, -1, 4)
            proposals.append(proposal)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        return output_proposals, output_proposals_valid


    def forward(self, feats, targets):
        # flatten feature maps 
        memory = []
        spatial_shapes = []
        split_sizes = []
        for feat in feats:
            bs, c, h, w = feat.shape
            memory.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            split_sizes.append(h*w)

        # spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feats[0].device)
        memory = torch.cat(memory, 1) # bs, \sum{hxw}, c 

        # two-stage
        if self.training:
            output_proposals, output_proposals_valid = self.generate_anchors(spatial_shapes)
            output_proposals = output_proposals.to(memory.device).repeat(bs, 1, 1)
            output_memory = memory.masked_fill(~output_proposals_valid.to(memory.device), float(0))
        else:
            output_proposals = self.output_proposals.repeat(bs, 1, 1)
            output_memory = memory.masked_fill(self.output_proposals_mask, float(0))

        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
        enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory) + output_proposals # (bs, \sum{hw}, 4) unsigmoid
        topk = self.num_queries
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1] # bs, nq

        # gather boxes
        refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4) ) # unsigmoid
        refpoint_embed = refpoint_embed_undetach.detach()
        init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid() # sigmoid

        # gather tgt
        tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
        tgt = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1) # nq, bs, d_model

        # denoise (only for training)
        if self.training and targets is not None:
            dn_tgt, dn_refpoint_embed, dn_attn_mask, dn_meta =\
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_line_noise_scale),
                                training=self.training,num_queries=self.num_queries, num_classes=self.num_classes,
                                hidden_dim=self.d_model, label_enc=self.label_enc)
            tgt = torch.cat([dn_tgt, tgt.transpose(0, 1)], dim=1).transpose(0, 1)
            refpoint_embed = torch.cat([dn_refpoint_embed, refpoint_embed], dim=1)
        else:
            dn_attn_mask = dn_meta = None

        # preprocess memory for MSDeformableLineAttention
        value = memory.unflatten(2, (self.n_heads, -1)) # (bs, \sum{hxw}, n_heads, d_model//n_heads)
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_sizes, dim=-1)
        out_coords, out_class = self.decoder(
                tgt=tgt, 
                memory=value, #memory.transpose(0, 1), 
                pos=None,
                refpoints_unsigmoid=refpoint_embed.transpose(0, 1), 
                spatial_shapes=spatial_shapes,
                tgt_mask=dn_attn_mask)

        # output
        if self.training:
            if dn_meta is not None:
                dn_out_coords, out_coords = torch.split(out_coords, [dn_meta['pad_size'], self.num_queries], dim=2)
                dn_out_class, out_class = torch.split(out_class, [dn_meta['pad_size'], self.num_queries], dim=2)

            out = {'pred_logits': out_class[-1], 'pred_lines': out_coords[-1]}

            if self.decoder.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(out_class[:-1], out_coords[:-1])

            # for encoder output
            out_coords_enc = refpoint_embed_undetach.sigmoid()
            out_class_enc = self.enc_out_class_embed(tgt_undetach)
            out['aux_interm_outputs'] = {'pred_logits': out_class_enc, 'pred_lines': out_coords_enc}

            if dn_meta is not None:
                dn_out = {}
                dn_out['aux_outputs'] = self._set_aux_loss(dn_out_class, dn_out_coords)
                out['aux_denoise'] = dn_out
        else:
            out = {'pred_logits': out_class[0], 'pred_lines': out_coords[0]}

        out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_lines': b}
                for a, b in zip(outputs_class, outputs_coord)]

    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
                      teacher_corners=None, teacher_class=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_lines': b, 'pred_corners': c, 'ref_points': d,
                     'teacher_corners': teacher_corners, 'teacher_logits': teacher_class}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]


def build_decoder(args):
    return LINEATransformer(
            feat_channels = args.feat_channels_decoder,
            feat_strides=args.feat_strides,
            num_classes=args.num_classes,
            d_model=args.hidden_dim, 
            nhead=args.nheads, 
            num_queries=args.num_queries, 
            num_decoder_layers=args.dec_layers, 
            dim_feedforward=args.dim_feedforward, 
            dropout=args.dropout,
            activation=args.transformer_activation, 
            normalize_before=args.pre_norm, 
            query_dim=args.query_dim,
            aux_loss=True,
            # for deformable encoder
            num_feature_levels=args.num_feature_levels,
            dec_n_points=args.dec_n_points,
            # for D-FINE layers
            reg_max=args.reg_max,
            reg_scale=args.reg_scale,
            # for inference
            eval_spatial_size=args.eval_spatial_size,
            eval_idx=args.eval_idx,
            # for denoising
            dn_number=args.dn_number,
            dn_label_noise_ratio=args.dn_label_noise_ratio,
            dn_line_noise_scale=args.dn_line_noise_scale,
            )


