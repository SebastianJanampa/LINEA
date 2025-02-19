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

import torch 
from torch import nn, Tensor
import torch.nn.functional as F 


def get_activation(act: str, inpace: bool=True):
    """get activation
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act 

    act = act.lower()
    
    if act == 'silu' or act == 'swish':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()

    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 

class CBLinear(nn.Module):
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, 0, groups=g, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs

class CBFuse(nn.Module):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out

class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.convH = ConvNormLayer(ch_in, ch_out, (3, 1), 1, padding=(1, 0), act=None)
        self.convW = ConvNormLayer(ch_in, ch_out, (1, 3), 1, padding=(0, 1), act=None)
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y_vertical = self.convH(x) 
            y_horizontal = self.convW(x) 
            y = self.conv1(x) + self.conv2(x) + y_horizontal + y_vertical

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        self.__delattr__('conv1')
        self.__delattr__('conv2') 
        self.__delattr__('convH')
        self.__delattr__('convW')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernel3x1, bias3x1 = self._fuse_bn_tensor(self.convH)
        kernel1x3, bias1x3 = self._fuse_bn_tensor(self.convW)
        
        kernel = kernel3x3  + self._pad_1x1_to_3x3_tensor(kernel1x1) + self._pad_1x1_to_3x3_tensor(kernel3x1, 'Vertical') + self._pad_1x1_to_3x3_tensor(kernel1x3, 'Horizontal')
        bias = bias3x3 + bias1x1 + bias3x1 + bias1x3
        return kernel, bias

    def _pad_1x1_to_3x3_tensor(self, kernel1x1, assymetric='None'):
        if kernel1x1 is None:
            return 0
        else:
            if assymetric == 'None':
                return F.pad(kernel1x1, [1, 1, 1, 1])
            elif assymetric == 'Vertical':
                return F.pad(kernel1x1, [1, 1, 0, 0])
            elif assymetric == 'Horizontal':
                return F.pad(kernel1x1, [0, 0, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu"):
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, 
        src, 
        src_mask=None, 
        src_key_padding_mask=None,
        pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, 
            value=src, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, 
        src, 
        src_mask=None, 
        src_key_padding_mask=None,
        pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, 
                src_mask=src_mask, 
                src_key_padding_mask=src_key_padding_mask, 
                pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class HybridEncoderAsymConv(nn.Module):
    def __init__(
            self,
            in_channels=[512, 1024, 2048],
            feat_strides=[8, 16, 32],
            n_levels=3,
            hidden_dim=256,
            nhead=8,
            dim_feedforward = 1024,
            dropout=0.0,
            enc_act='gelu',
            use_encoder_idx=[2],
            num_encoder_layers=1,
            expansion=1.0,
            depth_mult=1.0,
            act='silu',
            eval_spatial_size=None,
            # position embedding
            temperatureH=20,
            temperatureW=20,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.n_levels = n_levels
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.eval_spatial_size = eval_spatial_size

        self.temperatureW = temperatureW
        self.temperatureH = temperatureH

        # channel projection
        input_proj_list = []
        for in_channel in in_channels:
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(32, hidden_dim)
                )
            )
        self.input_proj = nn.ModuleList(input_proj_list)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(n_levels - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult))
                # CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(n_levels - 1):
            self.downsample_convs.append(nn.Sequential(
                SCDown(hidden_dim, hidden_dim, 3, 2),
                )
            )
            self.pan_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult))
                # CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            # nn.init.constant_(proj[0].bias, 0)

        if self.eval_spatial_size is not None:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.create_sinehw_position_embedding(
                self.eval_spatial_size[1] // stride, 
                self.eval_spatial_size[0] // stride,
                self.hidden_dim // 2
                )
                setattr(self, f'pos_embed{idx}', pos_embed)

    def create_sinehw_position_embedding(self, w, h, hidden_dim, scale=None, device='cpu'):
        """
        """
        grid_w = torch.arange(1, int(w)+1, dtype=torch.float32, device=device)
        grid_h = torch.arange(1, int(h)+1, dtype=torch.float32, device=device)

        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')

        if scale is None:
            scale = 2 * math.pi

        eps = 1e-6
        grid_w = grid_w / (int(w) + eps) * scale
        grid_h = grid_h / (int(h) + eps) * scale

        dim_tx = torch.arange(hidden_dim, dtype=torch.float32, device=device)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / hidden_dim)
        pos_x = grid_w[..., None] / dim_tx

        dim_ty = torch.arange(hidden_dim, dtype=torch.float32, device=device)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / hidden_dim)
        pos_y = grid_h[..., None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        pos = pos[None].flatten(2).permute(0, 2, 1).contiguous()

        return pos

    def forward(self, feats):
        """
        Input:
            - feats: List of features from the backbone
        Outpus: 
            - output: List of enhanced features
        """
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        for i, enc_idx in enumerate(self.use_encoder_idx):
            N_, C_, H_, W_ = proj_feats[enc_idx].shape
            # flatten [B, C, H, W] to [B, HxW, C]
            src_flatten = proj_feats[enc_idx].flatten(2).permute(0, 2, 1)
            if self.training or self.eval_spatial_size is None:
                pos_embed = self.create_sinehw_position_embedding(
                    H_, W_, self.hidden_dim//2, device=src_flatten.device)
            else:
                pos_embed = getattr(self, f'pos_embed{enc_idx}', None).to(src_flatten.device)

            proj_feats[enc_idx] = self.encoder[i](
                src_flatten, 
                pos_embed=pos_embed
                ).permute(0, 2, 1).reshape(N_, C_, H_, W_).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(self.n_levels - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[self.n_levels - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[self.n_levels-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(self.n_levels - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)
        return outs
        
def build_hybrid_encoder_with_asymmetric_conv(args):
    return HybridEncoderAsymConv(
        in_channels=args.in_channels_encoder,
        feat_strides=args.feat_strides,
        n_levels=args.num_feature_levels,
        hidden_dim=args.hidden_dim,
        nhead=args.nheads,
        dim_feedforward = args.dim_feedforward,
        dropout=args.dropout,
        enc_act='gelu',
        expansion=args.expansion,
        depth_mult=args.depth_mult,
        act='silu',
        temperatureH=args.pe_temperatureH,
        temperatureW=args.pe_temperatureW,
        eval_spatial_size= args.eval_spatial_size,
        )
