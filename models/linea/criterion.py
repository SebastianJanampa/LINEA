import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import resize

from .utils import sigmoid_focal_loss

from .matcher import build_matcher

from .linea_utils import weighting_function, bbox2distance

from ..registry import MODULE_BUILD_FUNCS

# TODO. Quick solution to make the model run on GoogleColab
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from util.misc import get_world_size, is_dist_avail_and_initialized


class LINEACriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:,:,:-1]
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_logits': loss_ce}

        return losses

    def loss_lines(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_lines' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_lines = outputs['pred_lines'][idx]
        target_lines = torch.cat([t['lines'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_line = F.l1_loss(src_lines, target_lines, reduction='none')

        losses = {}
        losses['loss_line'] = loss_line.sum() / num_boxes

        return losses

    def loss_lmap(self, outputs, targets, indices, num_boxes):
        losses = {}
        if 'aux_lmap' in outputs:
            src_lmap = outputs['aux_lmap']
            size = src_lmap[0].size(2)
            target_lmap = []
            for t in targets:
                lmaps_flatten = []
                for lmap, downsampling in zip(t['lmap'], [1, 2, 4]):
                    lmap_ = resize(lmap, (size//downsampling, size//downsampling))
                    lmaps_flatten.append(lmap_.flatten(1))
                target_lmap.append(torch.cat(lmaps_flatten, dim=1))
            target_lmap = torch.cat(target_lmap, dim=0)

            src_lmap = torch.cat([lmap_.flatten(1) for lmap_ in src_lmap], dim=1)

            loss_lmap = F.binary_cross_entropy_with_logits(src_lmap, target_lmap, reduction='mean')

            losses['loss_lmap'] = loss_lmap

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'lines': self.loss_lines,
            'lmap': self.loss_lmap,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            return indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        for loss in self.losses:
            indices_in = indices
            num_boxes_in = num_boxes
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:      
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'aux_interm_outputs' in outputs:
            interm_outputs = outputs['aux_interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # pre output loss
        if 'aux_pre_outputs' in outputs:
            pre_outputs = outputs['aux_pre_outputs']
            indices = self.matcher(pre_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, pre_outputs, targets, indices, num_boxes)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + f'_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'aux_denoise' in outputs:
            single_pad, scalar = self.prep_for_dn(dn_meta)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(len(targets[i]['labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            dn_outputs = outputs['aux_denoise']

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in dn_outputs:
                for idx, aux_outputs in enumerate(dn_outputs['aux_outputs']):
                    for loss in self.losses:
                        l_dict = self.get_loss(loss, aux_outputs, targets, dn_pos_idx, num_boxes*scalar)
                        l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                        l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            if 'aux_pre_outputs' in dn_outputs:
                aux_outputs_known = dn_outputs['aux_pre_outputs']
                l_dict={}
                for loss in self.losses:
                    l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar))
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}  
                l_dict = {k + f'_pre_dn': v for k, v in l_dict.items()}
                losses.update(l_dict)

        losses = {k: v for k, v in sorted(losses.items(), key=lambda item: item[0])}

        return losses

    def prep_for_dn(self,dn_meta):
        # output_known_lbs_lines = dn_meta['output_known_lbs_lines']
        num_dn_groups, pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return single_pad,num_dn_groups

class DFINESetCriterion(LINEACriterion):
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, reg_max, losses):
        super().__init__(num_classes, matcher, weight_dict, focal_alpha, losses)
        self.reg_max = reg_max

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_lines = torch.cat([t['lines'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()

            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, target_lines,
                                                        self.reg_max, outputs['reg_scale'], 
                                                        outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, target_lines,
                                                        self.reg_max, outputs['reg_scale'], 
                                                        outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, None, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if torch.equal(pred_corners, target_corners):
                    losses['loss_ddf'] = pred_corners.sum() * 0
                else:
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_lines'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'lines': self.loss_lines,
            'lmap': self.loss_lmap,
            'local': self.loss_local,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for i, aux_outputs in enumerate(outputs['aux_outputs'] + [outputs['aux_pre_outputs']]):
                indices_aux = self.matcher(aux_outputs, targets)
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate([outputs['aux_interm_outputs']]):
                indices_enc = self.matcher(aux_outputs, targets)
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            # assert 'aux_outputs' in outputs, ''
            indices_go = indices

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        for loss in self.losses:
            indices_in = indices_go if loss in ['lines', 'local'] else indices
            num_boxes_in = num_boxes_go if loss in ['lines', 'local'] else num_boxes
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:      
                    indices_in = indices_go if loss in ['lines', 'local'] else cached_indices[idx]
                    num_boxes_in = num_boxes_go if loss in ['lines', 'local'] else num_boxes
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'aux_interm_outputs' in outputs:
            interm_outputs = outputs['aux_interm_outputs']
            # indices = self.matcher(interm_outputs, targets)
            for loss in self.losses:
                indices_in = indices_go if loss in ['lines', 'local'] else cached_indices_enc[0]
                num_boxes_in = num_boxes_go if loss in ['lines', 'local'] else num_boxes
                l_dict = self.get_loss(loss, interm_outputs, targets, indices_in, num_boxes_in)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # pre output loss
        if 'aux_pre_outputs' in outputs:
            pre_outputs = outputs['aux_pre_outputs']
            # indices = self.matcher(pre_outputs, targets)
            for loss in self.losses:
                indices_in = indices_go if loss in ['lines', 'local'] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ['lines', 'local'] else num_boxes
                l_dict = self.get_loss(loss, pre_outputs, targets, indices_in, num_boxes_in)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + f'_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)


        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'aux_denoise' in outputs:
            single_pad, scalar = self.prep_for_dn(dn_meta)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(len(targets[i]['labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            dn_outputs = outputs['aux_denoise']

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in dn_outputs:
                for idx, aux_outputs in enumerate(dn_outputs['aux_outputs']):
                    aux_outputs['is_dn'] = True
                    aux_outputs['reg_scale'] = outputs['reg_scale']
                    aux_outputs['up'] = outputs['up']
                    # indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        l_dict = self.get_loss(loss, aux_outputs, targets, dn_pos_idx, num_boxes*scalar)
                        l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                        l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            if 'aux_pre_outputs' in dn_outputs:
                aux_outputs_known = dn_outputs['aux_pre_outputs']
                l_dict={}
                for loss in self.losses:
                    l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar))
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}  
                l_dict = {k + f'_pre_dn': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'aux_lmap' in outputs:
            l_dict = self.get_loss('lmap', outputs, targets, indices, num_boxes, **kwargs)
            l_dict = {k: v for k, v in l_dict.items()}
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}  
            losses.update(l_dict)

        losses = {k: v for k, v in sorted(losses.items(), key=lambda item: item[0])}

        return losses

@MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEACRITERION')
def build_criterion(args):
    num_classes = args.num_classes
    
    matcher = build_matcher(args)

    if args.criterion_type == 'default':
        criterion = LINEACriterion(num_classes, matcher=matcher, weight_dict=args.weight_dict,
                             focal_alpha=args.focal_alpha, losses=args.losses)
    elif args.criterion_type == 'dfine':
        criterion = DFINESetCriterion(num_classes, matcher=matcher, weight_dict=args.weight_dict,
                             focal_alpha=args.focal_alpha, reg_max=args.reg_max, losses=args.losses)
    else:
        raise Exception(f"Criterion type: {args.criterion_type}.We only support two classes: 'default' and 'dfine'. ")

    return criterion
