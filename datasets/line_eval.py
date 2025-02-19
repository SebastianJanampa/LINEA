import torch

__all__ = ['LineEvaluator']


class LineEvaluator(object):
    def __init__(self):
        self.n_gt = []
        self.distances = []
        self.choices = []
        self.spa_metrics = {'sap5': 5, 'sap10': 10, 'sap15': 15}
        self.tp = {'sap5': [], 'sap10': [], 'sap15': []}
        self.fp = {'sap5': [], 'sap10': [], 'sap15': []}
        self.scores = []

    def prepare(self, lines, scores=None):
        lines = lines.unflatten(-1, (2, 2)).flip([-1])

        if scores is not None:
            scores = scores[..., 0].sigmoid()
            scores_idx = torch.argsort(scores, descending=True, dim=-1)
            scores = torch.gather(scores, 1, scores_idx) 
            lines = torch.gather(lines, 1, scores_idx[:, :, None, None].repeat(1, 1, 2, 2))
            return lines * 128., scores

        return lines * 128.

    def compute_distances(self, pred_lines, gt_lines):
        dist =  ((pred_lines[:, None, :, None] - gt_lines[:, None]) ** 2).sum(-1)
        dist = torch.minimum(
            dist[:, :, 0, 0] + dist[:, :, 1, 1], 
            dist[:, :, 0, 1] + dist[:, :, 1, 0]
            )

        dist, choice = torch.min(dist, 1)
        return dist, choice

    def msTPFP(self, distances, choice, threshold):
        hit = torch.zeros_like(distances)
        tp = torch.zeros_like(distances)
        fp = torch.zeros_like(distances)

        for i in range(len(distances)):
            if distances[i] < threshold and not hit[choice[i]]:
                hit[choice[i]] = True
                tp[i] = 1
            else:
                fp[i] = 1
        return tp, fp

    def update(self, predictions, ground_truth):
        pred_lines, pred_scores = self.prepare(predictions['pred_lines'], predictions['pred_logits'])
        self.scores.append(pred_scores.flatten(0, 1))

        for pred_l, gt in zip(pred_lines, ground_truth):
            gt_l = self.prepare(gt['lines'])
            self.n_gt.append(len(gt_l))

            distances, choices = self.compute_distances(pred_l, gt_l)
            for k, v in self.spa_metrics.items():
                tp, fp = self.msTPFP(distances, choices, v)
                self.tp[k].append(tp)
                self.fp[k].append(fp)

    def ap(self, tp, fp):
        recall = tp
        precision = tp / torch.maximum(tp + fp, torch.tensor(1e-9, dtype=tp.dtype, device=tp.device))

        zero = torch.tensor([0.0], dtype=torch.float, device=tp.device)
        one = torch.tensor([1.0], dtype=torch.float, device=tp.device)
        recall = torch.cat((zero, recall, one))
        precision = torch.cat((zero, precision, zero))

        for i in range(precision.size()[0] - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])

        idx = torch.where(recall[1:] != recall[:-1])[0]
        return torch.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])

    # TODO
    # def ap(self, tp, fp):
    #     recall = tp
    #     precision = tp / torch.maximum(tp + fp, torch.tensor(1e-9, dtype=tp.dtype, device=tp.device))

    #     zero = torch.tensor([[0.0]], dtype=torch.float, device=tp.device).repeat(len(recall), 1)
    #     one = torch.tensor([[1.0]], dtype=torch.float, device=tp.device).repeat(len(recall), 1)
    #     recall = torch.cat((zero, recall, one), dim=1)
    #     precision = torch.cat((zero, precision, zero), dim=1)

    #     for i in range(precision.size()[1] - 1, 0, -1):
    #         precision[:, i - 1] = torch.maximum(precision[:, i - 1], precision[:, i])

    #     idx = torch.where(recall[:, 1:] != recall[:, :-1])[0]

    #     return torch.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])

    def accumulate(self,):
        self.sap_results = {}
        scores = torch.cat(self.scores)
        n_gt = sum(self.n_gt)

        tp, fp = [], []
        for k in self.spa_metrics:
            tp_ = torch.cat(self.tp[k])
            fp_ = torch.cat(self.fp[k])

            index = torch.argsort(scores, descending=True)
            tp_ = torch.cumsum(tp_[index], dim=0) / n_gt
            fp_ = torch.cumsum(fp_[index], dim=0) / n_gt

            # tp.append(tp_)
            # fp.append(fp_)

        # tp = torch.stack(tp)
        # fp = torch.stack(fp)
        # self.ap(tp, fp)

            self.sap_results[k] = self.ap(tp_, fp_).item() * 100

    def summarize(self,):
        for sap, results in self.sap_results.items():
            print(f'{sap}:\t{results:.1f}')

    def cleanup(self, ):
        del self.n_gt, self.tp, self.fp, self.scores
        torch.cuda.empty_cache()
        self.n_gt = []
        self.tp = {'sap5': [], 'sap10': [], 'sap15': []}
        self.fp = {'sap5': [], 'sap10': [], 'sap15': []}
        self.scores = []
