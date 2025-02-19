import json
import torch
import torch.nn as nn

import re


def get_optim_params(cfg: list, model: nn.Module):
    """
    E.g.:
        ^(?=.*a)(?=.*b).*$  means including a and b
        ^(?=.*(?:a|b)).*$   means including a or b
        ^(?=.*a)(?!.*b).*$  means including a, but not b
    """

    param_groups = []
    visited = []
    for pg in cfg:
        pattern = pg['params']
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
        pg['params'] = params.values()
        param_groups.append(pg)
        visited.extend(list(params.keys()))

    names = [k for k, v in model.named_parameters() if v.requires_grad]

    if len(visited) < len(names):
        unseen = set(names) - set(visited)
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
        param_groups.append({'params': params.values()})
        visited.extend(list(params.keys()))

    assert len(visited) == len(names), ''

    return param_groups
