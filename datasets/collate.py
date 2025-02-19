import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
import random

# This only for printing
RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"

# __all__ = [
#     BatchImageCollateFunction,
# ]

class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self,
        stop_epoch=None,
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        if self.scales is not None:
            self.scales.sort()
            print(GREEN  + "Multi-scaling uses the following size: " + RESET, self.scales)
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        # self.interpolation = interpolation

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None: # and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images = resize(images, [sz, sz])

        return images, targets
