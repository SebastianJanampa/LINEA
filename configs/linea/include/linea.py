# model
modelname = 'linea'
eval_spatial_size = (640, 640)
eval_idx = 5 # 6 decoder layers
num_classes = 2

## backbone
pretrained = True
use_checkpoint = False
return_interm_indices = [1, 2, 3]
freeze_norm = True
freeze_stem_only = True

## encoder
hybrid_encoder = 'hybrid_encoder_asymmetric_conv'
in_channels_encoder = [512, 1024, 2048]
pe_temperatureH = 20
pe_temperatureW = 20

## encoder
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True

## decoder
num_queries = 1100
query_dim = 4
num_feature_levels = 3
dec_n_points = [4, 1, 1] 
dropout = 0.0
pre_norm = False

# denoise
use_dn = True
dn_number = 300
dn_line_noise_scale = 1.0
dn_label_noise_ratio = 0.5
embed_init_tgt = True
dn_labelbook_size = 2
match_unstable_error = True

# matcher
set_cost_class = 2.0
set_cost_lines = 5.0

# criterion
criterion_type = 'default'
weight_dict = {'loss_logits': 1, 'loss_line': 5}
losses = ['labels', 'lines'] 
focal_alpha = 0.1

matcher_type = 'HungarianMatcher' # or SimpleMinsumMatcher
nms_iou_threshold = -1

# for ema
use_ema = False
ema_decay = 0.9997
ema_epoch = 0


