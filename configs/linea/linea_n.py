_base_ = [
	'./include/dataset.py',
	'./include/optimizer.py',
	'./include/linea.py'
	]

# backbone
backbone = 'HGNetv2_B0'
use_lab = True
freeze_norm = False
freeze_stem_only = True

# transformer
feat_strides = [8, 16, 32]
hidden_dim = 128
dim_feedforward = 512
nheads = 8
use_lmap = False

## encoder
hybrid_encoder = 'hybrid_encoder_asymmetric_conv'
in_channels_encoder = [256, 512, 1024]
pe_temperatureH = 20
pe_temperatureW = 20
expansion = 0.34
depth_mult = 0.5

## decoder
feat_channels_decoder = [hidden_dim, hidden_dim, hidden_dim]
dec_layers = 3
num_queries = 1100
num_select = 300
reg_max = 16
reg_scale = 4
eval_idx = 2

# criterion
epochs = 72
lr_drop_list = [60]
weight_dict = {'loss_logits': 2, 'loss_line': 5}
use_warmup = False

# optimizer params
model_parameters = [
	{
    'params': '^(?=.*backbone)(?!.*norm|bn).*$',
    'lr': 0.0004
    },
    {
    'params': '^(?=.*backbone)(?=.*norm|bn).*$',
      'lr': 0.0004,
      'weight_decay': 0.
    },
    {
    'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$',
    'weight_decay': 0.
    }
]
lr = 0.0008
betas = [0.9, 0.999]
weight_decay = 0.0001
