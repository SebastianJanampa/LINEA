import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.init import xavier_uniform_, constant_

import math

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights, total_num_points):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, P_, _ = sampling_locations[0].shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = (2 * sampling_locations[lid_] - 1).transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, total_num_points)
    output = (torch.cat(sampling_value_list, dim=-1) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()

def ms_deform_attn_core_pytorchv2(value, value_spatial_shapes, sampling_locations, attention_weights, num_points_list):
    # for debug and test only,
    # need to use cuda version instead
    _, D_ , _= value[0].shape
    N_, Lq_, M_, _, _ = sampling_locations.shape

    sampling_grids = 2 * sampling_locations - 1
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_* M_, D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value[lid_].unflatten(2, (H_, W_)) 
        # N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_locations_list[lid_]
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, sum(num_points_list))
    output = (torch.cat(sampling_value_list, dim=-1) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

class MSDeformLineAttn(nn.Module):
    def __init__(
        self, 
        d_model=256, 
        n_levels=4, 
        n_heads=8, 
        n_points=4
        ):
        """
        This version is inspired from DFine. We removed the following layers:
        -   value_proj
        -   output_proj
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads

        if isinstance(n_points, list):
            assert len(n_points) == n_levels, ''
            num_points_list = n_points
        else:
            num_points_list = [n_points for _ in range(n_levels)]
        self.num_points_list = num_points_list
        self.total_num_points = sum(num_points_list)

        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32).reshape(-1, 1))

        self.sampling_ratios = nn.Linear(d_model, n_heads * sum(num_points_list))
        self.attention_weights = nn.Linear(d_model, n_heads * sum(num_points_list))

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_ratios.weight.data, 0.)
        with torch.no_grad():
            self.sampling_ratios.bias = nn.Parameter(torch.linspace(-1, 1, self.n_heads * self.total_num_points))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param value               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param value_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]

        :return output                     (N, Length_{query}, C)

        ####################################################################
        # Difference respect to MSDeformAttn 
        # The query already stores the line's junctions
        # :param reference_points is not needed. We keep it to make both 
                MSDeformAttn and MSDeformLineAttn interchangebale 
                between different frameworks
        # MSDeformLineAttn does not generate offsets. Instead, it samples 
                n_points equally-spaced points from the line segment
        ####################################################################
        """
        N, Len_q, _ = query.shape

        sampling_ratios = self.sampling_ratios(query).view(N, Len_q, self.n_heads, self.total_num_points, 1)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.total_num_points)
        attention_weights = F.softmax(attention_weights, -1)

        num_points_scale = self.num_points_scale.to(dtype=query.dtype)
        
        vector = reference_points[:, :, None, :, :2] - reference_points[:, :, None, :, 2:]
        center = 0.5 * (reference_points[:, :, None, :, :2] + reference_points[:, :, None, :, 2:])

        sampling_locations = center + sampling_ratios * num_points_scale * vector * 0.5

        output = ms_deform_attn_core_pytorchv2(
            value, 
            value_spatial_shapes, 
            sampling_locations, 
            attention_weights, 
            self.num_points_list
        )
        return output


#######################
## Previous versions ##
#######################

# class MSDeformLineAttn(nn.Module):
#     def __init__(
#         self, 
#         d_model=256, 
#         n_levels=4, 
#         n_heads=8, 
#         n_points=4
#     ):
#         """
#         Multi-Scale Deformable Attention Module
#         :param d_model      hidden dimension
#         :param n_levels     number of feature levels
#         :param n_heads      number of attention heads
#         :param n_points     number of sampling points per attention head per feature level
#         """
#         super().__init__()
#         if d_model % n_heads != 0:
#             raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
#         _d_per_head = d_model // n_heads
#         # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
#         if not _is_power_of_2(_d_per_head):
#             warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
#                           "which is more efficient in our CUDA implementation.")

#         self.d_model = d_model
#         self.n_levels = n_levels
#         self.n_heads = n_heads

#         if isinstance(n_points, list):
#             assert len(n_points) == n_levels, ''
#             num_points_list = n_points
#         else:
#             num_points_list = [n_points for _ in range(n_levels)]
#         self.num_points_list = num_points_list
#         self.total_num_points = sum(num_points_list)

#         num_points_scale = [1/n for n in num_points_list]
#         self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32).reshape(-1, 1, 1))

#         self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * 4)

#         self.attention_weights = nn.Linear(d_model, n_heads * sum(num_points_list))
#         self.value_proj = nn.Linear(d_model, d_model)
#         self.output_proj = nn.Linear(d_model, d_model)

#         for i in range(len(num_points_list)):
#             if num_points_list[i] == 1:
#                 lambda_  = torch.linspace(0.5, 0.5, num_points_list[i])[:, None]
#             else:
#                 lambda_  = torch.linspace(0, 1, num_points_list[i])[:, None]
#             self.register_buffer(f"lambda_{i}", lambda_)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         constant_(self.sampling_offsets.weight.data, 0.)
#         thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, 2, 1)
#         for i in range(1):
#             grid_init[:, :, 2*i, :] *= i + 1
#             grid_init[:, :, 2*i+1, :] *= i + 1
#         with torch.no_grad():
#             self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
#         constant_(self.attention_weights.weight.data, 0.)
#         constant_(self.attention_weights.bias.data, 0.)
#         xavier_uniform_(self.value_proj.weight.data)
#         constant_(self.value_proj.bias.data, 0.)
#         xavier_uniform_(self.output_proj.weight.data)
#         constant_(self.output_proj.bias.data, 0.)

#     def forward(self, query, reference_points, input_flatten, input_spatial_shapes):
#         """
#         :param query                       (N, Length_{query}, C)
#         :param reference_points            (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
#         :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
#         :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
#         :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]

#         :return output                     (N, Length_{query}, C)

#         ####################################################################
#         # Difference respect to MSDeformAttn 
#         # The query already stores the line's junctions
#         # :param reference_points is not needed. We keep it to make both 
#                 MSDeformAttn and MSDeformLineAttn interchangebale 
#                 between different frameworks
#         # MSDeformLineAttn does not generate offsets. Instead, it samples 
#                 n_points equally-spaced points from the line segment
#         ####################################################################
#         """
#         N, Len_q, _ = query.shape
#         N, Len_in, _ = input_flatten.shape

#         value = self.value_proj(input_flatten)

#         value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
#         sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, 1, 4)
#         attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.total_num_points)
#         attention_weights = F.softmax(attention_weights, -1)

#         num_points_scale = self.num_points_scale.to(dtype=query.dtype)
        
#         wh = reference_points[:, :, None, :, None, :2] - reference_points[:, :, None, :, None, 2:]
#         center = 0.5 * (reference_points[:, :, None, :, None, :2] + reference_points[:, :, None, :, None, 2:])

#         sampling_junctions = torch.cat((center, center), dim=-1) \
#                                 + sampling_offsets * num_points_scale * torch.cat([wh, wh], -1) * 0.5

#         sampling_locations = []

#         # sampling_junctions_level = torch.split(sampling_junctions, self.num_points_list, dim=-2)
#         for i in range(len(self.num_points_list)):
#             lambda_ = getattr(self, f'lambda_{i}')
#             junctions = sampling_junctions[:, :, :, i]
#             locations = junctions[..., :2] * lambda_ + junctions[..., 2:] * (1 - lambda_)
#             sampling_locations.append(locations)

#         output = ms_deform_attn_core_pytorch(
#             value, 
#             input_spatial_shapes, 
#             sampling_locations, 
#             attention_weights, 
#             self.total_num_points
#         )
#         output = self.output_proj(output)
#         return output


# class MSDeformLineAttnV2(nn.Module):
#     def __init__(
#         self, 
#         d_model=256, 
#         n_levels=4, 
#         n_heads=8, 
#         n_points=4
#     ):
#         """
#         Multi-Scale Deformable Attention Module
#         :param d_model      hidden dimension
#         :param n_levels     number of feature levels
#         :param n_heads      number of attention heads
#         :param n_points     number of sampling points per attention head per feature level
#         """
#         super().__init__()
#         if d_model % n_heads != 0:
#             raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
#         _d_per_head = d_model // n_heads
#         # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
#         if not _is_power_of_2(_d_per_head):
#             warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
#                           "which is more efficient in our CUDA implementation.")

#         self.d_model = d_model
#         self.n_levels = n_levels
#         self.n_heads = n_heads

#         if isinstance(n_points, list):
#             assert len(n_points) == n_levels, ''
#             num_points_list = n_points
#         else:
#             num_points_list = [n_points for _ in range(n_levels)]
#         self.num_points_list = num_points_list
#         self.total_num_points = sum(num_points_list)

#         num_points_scale = [1/n for n in num_points_list]
#         self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32).reshape(-1, 1, 1))

#         self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * 4)
#         self.sampling_ratios = nn.Linear(d_model, n_heads * sum(num_points_list))

#         self.attention_weights = nn.Linear(d_model, n_heads * sum(num_points_list))
#         self.value_proj = nn.Linear(d_model, d_model)
#         self.output_proj = nn.Linear(d_model, d_model)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         constant_(self.sampling_offsets.weight.data, 0.)
#         thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, 2, 1)
#         for i in range(1):
#             grid_init[:, :, 2*i, :] *= i + 1
#             grid_init[:, :, 2*i+1, :] *= i + 1
#         with torch.no_grad():
#             self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
#         constant_(self.attention_weights.weight.data, 0.)
#         constant_(self.attention_weights.bias.data, 0.)
#         xavier_uniform_(self.value_proj.weight.data)
#         constant_(self.value_proj.bias.data, 0.)
#         xavier_uniform_(self.output_proj.weight.data)
#         constant_(self.output_proj.bias.data, 0.)

#     def forward(self, query, reference_points, input_flatten, input_spatial_shapes):
#         """
#         :param query                       (N, Length_{query}, C)
#         :param reference_points            (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
#         :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
#         :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
#         :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]

#         :return output                     (N, Length_{query}, C)

#         ####################################################################
#         # Difference respect to MSDeformAttn 
#         # The query already stores the line's junctions
#         # :param reference_points is not needed. We keep it to make both 
#                 MSDeformAttn and MSDeformLineAttn interchangebale 
#                 between different frameworks
#         # MSDeformLineAttn does not generate offsets. Instead, it samples 
#                 n_points equally-spaced points from the line segment
#         ####################################################################
#         """
#         N, Len_q, _ = query.shape
#         N, Len_in, _ = input_flatten.shape

#         value = self.value_proj(input_flatten)

#         value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
#         sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, 1, 4)
#         sampling_ratios = self.sampling_ratios(query).view(N, Len_q, self.n_heads, self.total_num_points).sigmoid()
#         attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.total_num_points)
#         attention_weights = F.softmax(attention_weights, -1)

#         num_points_scale = self.num_points_scale.to(dtype=query.dtype)
        
#         wh = reference_points[:, :, None, :, None, :2] - reference_points[:, :, None, :, None, 2:]
#         center = 0.5 * (reference_points[:, :, None, :, None, :2] + reference_points[:, :, None, :, None, 2:])

#         sampling_junctions = torch.cat((center, center), dim=-1) \
#                                 + sampling_offsets * num_points_scale * torch.cat([wh, wh], -1) * 0.5

#         sampling_locations = [] 

#         for i, lambda_ in enumerate(torch.split(sampling_ratios, self.num_points_list, dim=-1)):
#             lambda_ = lambda_[..., None]
#             junctions = sampling_junctions[:, :, :, i]
#             locations = junctions[..., :2] * lambda_ + junctions[..., 2:] * (1 - lambda_)
#             sampling_locations.append(locations)

#         output = ms_deform_attn_core_pytorch(
#             value, 
#             input_spatial_shapes, 
#             sampling_locations, 
#             attention_weights, 
#             self.total_num_points
#         )
#         output = self.output_proj(output)
#         return output


# class MSDeformLineAttnV3(nn.Module):
#     def __init__(
#         self, 
#         d_model=256, 
#         n_levels=4, 
#         n_heads=8, 
#         n_points=4
#     ):
#         """
#         This version is inspired from DFine. We removed the following layers:
#         -   value_proj
#         -   output_proj
#         """
#         super().__init__()
#         if d_model % n_heads != 0:
#             raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
#         _d_per_head = d_model // n_heads
#         # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
#         if not _is_power_of_2(_d_per_head):
#             warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
#                           "which is more efficient in our CUDA implementation.")

#         self.d_model = d_model
#         self.n_levels = n_levels
#         self.n_heads = n_heads

#         if isinstance(n_points, list):
#             assert len(n_points) == n_levels, ''
#             num_points_list = n_points
#         else:
#             num_points_list = [n_points for _ in range(n_levels)]
#         self.num_points_list = num_points_list
#         self.total_num_points = sum(num_points_list)

#         num_points_scale = [1/n for n in num_points_list]
#         self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32).reshape(-1, 1, 1))

#         self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * 4)
#         self.sampling_ratios = nn.Linear(d_model, n_heads * sum(num_points_list))

#         self.attention_weights = nn.Linear(d_model, n_heads * sum(num_points_list))

#         self._reset_parameters()

#     def _reset_parameters(self):
#         constant_(self.sampling_offsets.weight.data, 0.)
#         thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, 2, 1)
#         for i in range(1):
#             grid_init[:, :, 2*i, :] *= i + 1
#             grid_init[:, :, 2*i+1, :] *= i + 1
#         with torch.no_grad():
#             self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
#         constant_(self.attention_weights.weight.data, 0.)
#         constant_(self.attention_weights.bias.data, 0.)

#     def forward(self, query, reference_points, value, value_spatial_shapes):
#         """
#         :param query                       (N, Length_{query}, C)
#         :param reference_points            (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
#         :param value               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
#         :param value_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
#         :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]

#         :return output                     (N, Length_{query}, C)

#         ####################################################################
#         # Difference respect to MSDeformAttn 
#         # The query already stores the line's junctions
#         # :param reference_points is not needed. We keep it to make both 
#                 MSDeformAttn and MSDeformLineAttn interchangebale 
#                 between different frameworks
#         # MSDeformLineAttn does not generate offsets. Instead, it samples 
#                 n_points equally-spaced points from the line segment
#         ####################################################################
#         """
#         N, Len_q, _ = query.shape

#         sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, 1, 4)
#         sampling_ratios = self.sampling_ratios(query).view(N, Len_q, self.n_heads, self.total_num_points).sigmoid()
#         attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.total_num_points)
#         attention_weights = F.softmax(attention_weights, -1)

#         num_points_scale = self.num_points_scale.to(dtype=query.dtype)
        
#         wh = reference_points[:, :, None, :, None, :2] - reference_points[:, :, None, :, None, 2:]
#         center = 0.5 * (reference_points[:, :, None, :, None, :2] + reference_points[:, :, None, :, None, 2:])

#         sampling_junctions = torch.cat((center, center), dim=-1) \
#                                 + sampling_offsets * num_points_scale * torch.cat([wh, wh], -1) * 0.5

#         sampling_locations = [] 

#         for i, lambda_ in enumerate(torch.split(sampling_ratios, self.num_points_list, dim=-1)):
#             lambda_ = lambda_[..., None]
#             junctions = sampling_junctions[:, :, :, i]
#             locations = junctions[..., :2] * lambda_ + junctions[..., 2:] * (1 - lambda_)
#             sampling_locations.append(locations)

#         output = ms_deform_attn_core_pytorchv2(
#             value, 
#             value_spatial_shapes, 
#             sampling_locations, 
#             attention_weights, 
#             self.total_num_points
#         )
#         return output
