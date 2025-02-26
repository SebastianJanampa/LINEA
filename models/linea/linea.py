import torch
from torch import nn

from .hgnetv2 import build_hgnetv2
from .hybrid_encoder import build_hybrid_encoder
from .decoder import build_decoder

from ..registry import MODULE_BUILD_FUNCS


class LINEA(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self,
        backbone,
        encoder,
        decoder,
        ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, samples, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features = self.backbone(samples)

        features = self.encoder(features)

        out = self.decoder(features, targets)

        return out

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self) -> None:
        super().__init__()
        self.deploy_mode = False

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_line = outputs['pred_logits'], outputs['pred_lines']

        scores = out_logits[..., 0].sigmoid()

        # convert to [x0, y0, x1, y1] format
        lines = out_line * target_sizes.repeat(1, 2).unsqueeze(1)

        if self.deploy_mode:
            return lines, scores

        results = [{'lines': l, 'scores': s} for s, l in zip(scores, lines)]

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self

@MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA')
def build_linea(args):
    num_classes = args.num_classes

    backbone = build_hgnetv2(args)
    encoder = build_hybrid_encoder(args)
    decoder = build_decoder(args)

    model = LINEA(
        backbone,
        encoder,
        decoder,
    )

    postprocessors = PostProcess()

    return model, postprocessors
