_base_ = [
	'./include/dataset.py',
	'./include/optimizer.py',
	'./include/dfine.py'
	]

# backbone
backbone = 'HGNetv2_B4'
param_dict_type = backbone.lower()
use_lab = False


# transformer
feat_strides = [8, 16, 32]
hidden_dim = 256
dim_feedforward = 1024
nheads = 8
use_lmap = False

## encoder
hybrid_encoder = 'hybrid_encoder_asymmetric_conv'
in_channels_encoder = [512, 1024, 2048]
pe_temperatureH = 20
pe_temperatureW = 20
expansion = 1.0
depth_mult = 1.0

## decoder
anchor_generation_type = 'dfine'
cross_attn_type = 'mda'
dec_n_points = [4, 4, 4]
feat_channels_decoder = [256, 256, 256]
dec_layers = 6
num_queries = 500
num_select = 300
reg_max = 16
reg_scale = 4

# criterion
criterion_type = 'default'
weight_dict = {'loss_logits': 1, 'loss_line': 5}
losses = ['labels', 'lines'] #, 'lmap']

# optimizer params
model_parameters = [
	{
	'params': '^(?=.*backbone)(?!.*norm|bn).*$',
    'lr': 0.0000125
    },

    {
    'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$',
    'weight_decay': 0.
    }
]
lr = 0.00025
betas = [0.9, 0.999]
weight_decay = 0.000125
