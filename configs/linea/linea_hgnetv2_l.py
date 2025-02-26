_base_ = [
	'./include/dataset.py',
	'./include/optimizer.py',
	'./include/linea.py'
	]

output_dir = 'output/linea_hgnetv2_l'

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
expansion = 0.5
depth_mult = 1.0

## decoder
feat_channels_decoder = [256, 256, 256]
dec_layers = 6
num_queries = 1100
num_select = 300
reg_max = 16
reg_scale = 4

# criterion
weight_dict = {'loss_logits': 4, 'loss_line': 5}
use_warmup = False

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
